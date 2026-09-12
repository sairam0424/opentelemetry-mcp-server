"""Datadog backend implementation using the Spans API (v2).

Unlike Jaeger/Tempo/Traceloop, Datadog's public API is span-centric rather than
trace-centric: there is no "get trace by ID" endpoint. A trace is reconstructed
by searching spans filtered by ``trace_id`` and grouping the results. Search
uses Datadog's span search query syntax (the same syntax that powers the Logs
Explorer - see https://docs.datadoghq.com/logs/explorer/search_syntax/),
not TraceQL or Jaeger-style tag params.

Schema note: this implementation is grounded in Datadog's published OpenAPI
spec (SpansListRequest/SpansListResponse/SpansAttributes in
https://github.com/DataDog/datadog-api-client-python/blob/master/.generator/schemas/v2/openapi.yaml)
rather than guessed field names. Two things the spec does not pin down, and
that could not be verified without a live Datadog account:

1. Where OTel ``gen_ai.*``/error attributes land in the response - the
   documented ``SpansAttributes`` has a generic ``attributes`` (custom
   attributes) object and a ``tags`` array, but doesn't document exactly how
   OTLP-ingested span attributes are split between the two. This
   implementation checks both.
2. There's no documented first-class ``status``/``error`` field on a span
   resource. Error state is inferred from common signals (an ``error`` custom
   attribute, an ``error:*``-prefixed tag, or standard ``error.type``/
   ``error.message`` OTel attributes).

Both are flagged again at the call sites below. Someone with a live Datadog
account and real gen_ai-instrumented traces should verify against actual
payloads before merge.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from opentelemetry_mcp.attributes import HealthCheckResponse, SpanAttributes, SpanEvent
from opentelemetry_mcp.backends.base import BaseBackend
from opentelemetry_mcp.backends.filter_engine import FilterEngine
from opentelemetry_mcp.constants import Fields
from opentelemetry_mcp.models import (
    Filter,
    FilterOperator,
    SpanData,
    SpanQuery,
    TraceData,
    TraceQuery,
)

logger = logging.getLogger(__name__)

# Datadog search syntax fields that are query facets rather than custom
# attributes - these are queried bare (`service:foo`), everything else is
# queried with an `@` prefix (`@gen_ai.system:foo`), matching Datadog's Logs
# Explorer / span search convention for custom vs. reserved attributes.
_FACET_FIELDS = {
    Fields.SERVICE_NAME: "service",
    Fields.OPERATION_NAME: "resource_name",
}

# Default lookback window used when a query has no time range and there is no
# narrower signal (e.g. get_trace, list_services). Datadog's span search
# requires filter.from/filter.to; APM span retention is commonly far shorter
# than this, so it errs toward "wide enough to find something" rather than
# being a claim about actual retention.
_DEFAULT_LOOKBACK = timedelta(days=7)

_MAX_TRACES_TO_HYDRATE = 50


class DatadogBackend(BaseBackend):
    """Datadog Spans API (v2) backend."""

    def __init__(
        self,
        url: str,
        api_key: str | None = None,
        app_key: str | None = None,
        timeout: float = 30.0,
    ):
        """Initialize Datadog backend.

        Args:
            url: Datadog site API base URL (e.g. https://api.datadoghq.com or
                https://api.datadoghq.eu for the EU site)
            api_key: Datadog API key (required)
            app_key: Datadog Application key (required - trace/span queries
                need both, unlike simple ingestion which only needs api_key)
            timeout: Request timeout in seconds
        """
        super().__init__(url, api_key, timeout)

        if not self.api_key:
            raise ValueError("Datadog backend requires an API key (BACKEND_API_KEY)")
        if not app_key:
            raise ValueError(
                "Datadog backend requires an Application key (BACKEND_APP_KEY) "
                "in addition to the API key"
            )

        self.app_key = app_key

    def _create_headers(self) -> dict[str, str]:
        """Create headers for Datadog API requests.

        Returns:
            Dictionary with DD-API-KEY, DD-APPLICATION-KEY, and Content-Type
        """
        return {
            "DD-API-KEY": self.api_key or "",
            "DD-APPLICATION-KEY": self.app_key,
            "Content-Type": "application/json",
        }

    def get_supported_operators(self) -> set[FilterOperator]:
        """Get natively supported operators via Datadog span search syntax.

        Returns:
            Set of supported FilterOperator values
        """
        return {
            FilterOperator.EQUALS,
            FilterOperator.NOT_EQUALS,
            FilterOperator.GT,
            FilterOperator.LT,
            FilterOperator.GTE,
            FilterOperator.LTE,
            FilterOperator.EXISTS,
            FilterOperator.NOT_EXISTS,
            FilterOperator.IN,
        }

    async def search_traces(self, query: TraceQuery) -> list[TraceData]:
        """Search traces by searching spans and hydrating each matching trace.

        Datadog has no trace-level search endpoint, so this searches spans
        matching the query, then re-fetches the full span set for each
        distinct trace_id found (mirroring the Tempo backend's
        search-then-hydrate approach) so a partial/filtered span match still
        yields a complete trace.

        Args:
            query: Trace query parameters

        Returns:
            List of matching traces with all spans

        Raises:
            httpx.HTTPError: If the API request fails
        """
        all_filters = query.get_all_filters()
        supported_operators = self.get_supported_operators()
        native_filters = [f for f in all_filters if f.operator in supported_operators]
        client_filters = [f for f in all_filters if f.operator not in supported_operators]

        if client_filters:
            logger.info(
                f"Will apply {len(client_filters)} filters client-side: "
                f"{[f.operator.value for f in client_filters]}"
            )

        dd_query = self._build_dd_query(native_filters)
        start, end = self._time_range(query.start_time, query.end_time)

        spans_data = await self._search_spans_raw(dd_query, start, end, query.limit * 5)

        trace_ids: list[str] = []
        seen: set[str] = set()
        for span_obj in spans_data:
            trace_id = span_obj.get("attributes", {}).get("trace_id")
            if trace_id and trace_id not in seen:
                seen.add(trace_id)
                trace_ids.append(trace_id)

        max_to_fetch = min(len(trace_ids), _MAX_TRACES_TO_HYDRATE)
        if len(trace_ids) > max_to_fetch:
            logger.warning(
                f"Limiting trace fetch to {max_to_fetch} out of {len(trace_ids)} "
                f"results to avoid excessive API calls"
            )

        traces: list[TraceData] = []
        for trace_id in trace_ids[:max_to_fetch]:
            try:
                traces.append(await self.get_trace(trace_id))
            except Exception as e:
                logger.warning(f"Failed to fetch trace {trace_id}: {e}")

        if client_filters:
            traces = FilterEngine.apply_filters(traces, client_filters)

        return traces[: query.limit]

    async def search_spans(self, query: SpanQuery) -> list[SpanData]:
        """Search for individual spans matching the query.

        Args:
            query: Span query parameters

        Returns:
            List of matching spans

        Raises:
            httpx.HTTPError: If the API request fails
        """
        all_filters = query.get_all_filters()
        supported_operators = self.get_supported_operators()
        native_filters = [f for f in all_filters if f.operator in supported_operators]
        client_filters = [f for f in all_filters if f.operator not in supported_operators]

        if client_filters:
            logger.info(
                f"Will apply {len(client_filters)} span filters client-side: "
                f"{[(f.field, f.operator.value) for f in client_filters]}"
            )

        dd_query = self._build_dd_query(native_filters)
        start, end = self._time_range(query.start_time, query.end_time)

        spans_data = await self._search_spans_raw(dd_query, start, end, query.limit * 2)

        spans: list[SpanData] = []
        for span_obj in spans_data:
            span = self._parse_dd_span(span_obj)
            if span:
                spans.append(span)

        if client_filters:
            spans = FilterEngine.apply_filters(spans, client_filters)

        return spans[: query.limit]

    async def get_trace(self, trace_id: str) -> TraceData:
        """Get a specific trace by ID by searching all spans sharing it.

        Args:
            trace_id: Trace identifier

        Returns:
            Complete trace data with all spans

        Raises:
            ValueError: If no spans are found for the trace ID
            httpx.HTTPError: If the API request fails
        """
        start, end = self._time_range(None, None, lookback=_DEFAULT_LOOKBACK)
        dd_query = f"trace_id:{trace_id}"

        spans_data = await self._search_spans_raw(dd_query, start, end, limit=1000)

        spans: list[SpanData] = []
        for span_obj in spans_data:
            span = self._parse_dd_span(span_obj)
            if span:
                spans.append(span)

        if not spans:
            raise ValueError(f"No spans found for trace {trace_id}")

        return self._group_into_trace(trace_id, spans)

    async def list_services(self) -> list[str]:
        """List all available services by sampling recent spans.

        Datadog's Spans API has no dedicated "list services" endpoint, so
        this searches a broad recent window and extracts unique service
        names client-side, matching the Tempo backend's fallback approach.

        Returns:
            List of service names

        Raises:
            httpx.HTTPError: If the API request fails
        """
        logger.debug("Listing services")
        start, end = self._time_range(None, None)

        spans_data = await self._search_spans_raw("*", start, end, limit=1000)

        services: set[str] = set()
        for span_obj in spans_data:
            service = span_obj.get("attributes", {}).get("service")
            if service:
                services.add(service)

        result = sorted(services)
        logger.debug(f"Found {len(result)} unique services from {len(spans_data)} spans")
        return result

    async def get_service_operations(self, service_name: str) -> list[str]:
        """Get operations (resource names) for a service.

        Args:
            service_name: Service name

        Returns:
            List of operation names

        Raises:
            httpx.HTTPError: If the API request fails
        """
        logger.debug(f"Getting operations for service: {service_name}")
        start, end = self._time_range(None, None)

        dd_query = f"service:{self._quote_if_needed(service_name)}"
        spans_data = await self._search_spans_raw(dd_query, start, end, limit=1000)

        operations: set[str] = set()
        for span_obj in spans_data:
            resource_name = span_obj.get("attributes", {}).get("resource_name")
            if resource_name:
                operations.add(resource_name)

        return sorted(operations)

    async def health_check(self) -> HealthCheckResponse:
        """Check Datadog backend health via a minimal span search.

        Returns:
            Health status information
        """
        logger.debug("Checking backend health")

        try:
            start, end = self._time_range(None, None)
            await self._search_spans_raw("*", start, end, limit=1)

            return HealthCheckResponse(
                status="healthy",
                backend="datadog",
                url=self.url,
            )
        except Exception as e:
            return HealthCheckResponse(
                status="unhealthy",
                backend="datadog",
                url=self.url,
                error=str(e),
            )

    # -- internal helpers ---------------------------------------------------

    async def _search_spans_raw(
        self, dd_query: str, start: datetime, end: datetime, limit: int
    ) -> list[dict[str, Any]]:
        """Call the Datadog Spans search endpoint and return raw span objects.

        Args:
            dd_query: Datadog span search query string
            start: Start of the search window
            end: End of the search window
            limit: Maximum spans to request (capped at Datadog's own 1000 max)

        Returns:
            List of raw ``Span`` objects (each with ``id``/``type``/``attributes``)

        Raises:
            httpx.HTTPError: If the API request fails
        """
        body = {
            "data": {
                "type": "search_request",
                "attributes": {
                    "filter": {
                        "query": dd_query,
                        "from": start.isoformat(),
                        "to": end.isoformat(),
                    },
                    "page": {"limit": min(max(limit, 1), 1000)},
                    "sort": "-timestamp",
                },
            }
        }

        response = await self.client.post("/api/v2/spans/events/search", json=body)
        response.raise_for_status()

        data = response.json()
        result: list[dict[str, Any]] = data.get("data", [])
        return result

    def _time_range(
        self,
        start_time: datetime | None,
        end_time: datetime | None,
        lookback: timedelta = _DEFAULT_LOOKBACK,
    ) -> tuple[datetime, datetime]:
        """Resolve a query's time range, defaulting to a lookback window.

        Args:
            start_time: Explicit start time, if any
            end_time: Explicit end time, if any
            lookback: Default window size when start_time is not given

        Returns:
            (start, end) tuple, both timezone-aware
        """
        end = end_time or datetime.now(UTC)
        start = start_time or (end - lookback)
        return start, end

    def _build_dd_query(self, filters: list[Filter]) -> str:
        """Build a Datadog span search query string from Filter objects.

        Args:
            filters: List of Filter conditions

        Returns:
            Datadog span search query string (defaults to "*" if no filters)
        """
        raw_conditions = [self._filter_to_dd_query(f) for f in filters]
        conditions = [c for c in raw_conditions if c is not None]

        if not conditions:
            return "*"

        return " AND ".join(conditions)

    def _dd_field(self, field: str) -> str:
        """Map an internal field name to its Datadog search syntax name.

        Args:
            field: Internal field name (e.g. "service.name", "gen_ai.system")

        Returns:
            Datadog facet name (bare) or custom attribute name (`@`-prefixed)
        """
        if field in _FACET_FIELDS:
            return _FACET_FIELDS[field]
        if field == Fields.STATUS:
            return "status"
        if field == Fields.DURATION:
            return "@duration"
        # Custom/OTel attributes (gen_ai.*, arbitrary tags) are queried with
        # the `@` prefix per Datadog's search syntax for non-facet attributes.
        return f"@{field}"

    def _quote_if_needed(self, value: str) -> str:
        """Quote a search term if it contains whitespace."""
        if " " in value:
            return f'"{value}"'
        return value

    def _filter_to_dd_query(self, filter_obj: Filter) -> str | None:
        """Convert a single Filter to a Datadog span search condition.

        Args:
            filter_obj: Filter to convert

        Returns:
            Datadog search condition string, or None if unsupported
        """
        field = self._dd_field(filter_obj.field)
        operator = filter_obj.operator
        value = filter_obj.value
        values = filter_obj.values

        is_duration = field == "@duration"
        # Datadog's @duration facet is in nanoseconds; the rest of this
        # codebase works in milliseconds.
        scale = 1_000_000 if is_duration else 1

        if operator == FilterOperator.EQUALS:
            if field == "status" and value == "ERROR":
                return "status:error"
            if field == "status" and value == "OK":
                return "status:ok"
            v = value * scale if isinstance(value, int | float) else value
            return f"{field}:{self._quote_if_needed(str(v))}"

        elif operator == FilterOperator.NOT_EQUALS:
            if field == "status" and value == "ERROR":
                return "-status:error"
            v = value * scale if isinstance(value, int | float) else value
            return f"-{field}:{self._quote_if_needed(str(v))}"

        elif operator == FilterOperator.GT:
            v = value * scale if isinstance(value, int | float) else value
            return f"{field}:{{{v} TO *}}"

        elif operator == FilterOperator.GTE:
            v = value * scale if isinstance(value, int | float) else value
            return f"{field}:[{v} TO *]"

        elif operator == FilterOperator.LT:
            v = value * scale if isinstance(value, int | float) else value
            return f"{field}:{{* TO {v}}}"

        elif operator == FilterOperator.LTE:
            v = value * scale if isinstance(value, int | float) else value
            return f"{field}:[* TO {v}]"

        elif operator == FilterOperator.EXISTS:
            return f"{field}:*"

        elif operator == FilterOperator.NOT_EXISTS:
            return f"-{field}:*"

        elif operator == FilterOperator.IN:
            if not values:
                return None
            or_terms = [f"{field}:{self._quote_if_needed(str(v))}" for v in values]
            return "(" + " OR ".join(or_terms) + ")"

        logger.warning(f"Unsupported operator for Datadog query: {operator}")
        return None

    def _parse_dd_span(self, span_obj: dict[str, Any]) -> SpanData | None:
        """Parse a raw Datadog Span resource into SpanData.

        Args:
            span_obj: Raw ``Span`` object from the search response

        Returns:
            Parsed SpanData, or None if required fields are missing
        """
        try:
            attrs = span_obj.get("attributes", {})
            trace_id = attrs.get("trace_id")
            span_id = attrs.get("span_id")
            if not trace_id or not span_id:
                return None

            # Datadog uses "0" as the sentinel parent_id for root spans.
            parent_id_raw = attrs.get("parent_id")
            parent_span_id = parent_id_raw if parent_id_raw and parent_id_raw != "0" else None

            start_time = self._parse_dd_timestamp(attrs.get("start_timestamp"))
            end_time = self._parse_dd_timestamp(attrs.get("end_timestamp"))
            duration_ms = (
                (end_time - start_time).total_seconds() * 1000 if start_time and end_time else 0.0
            )

            custom_attrs = attrs.get("attributes", {}) or {}
            tags = attrs.get("tags", []) or []

            status = self._infer_status(custom_attrs, tags)
            span_attributes = SpanAttributes(**self._extract_semconv_attributes(custom_attrs))

            events: list[SpanEvent] = []

            return SpanData(
                trace_id=str(trace_id),
                span_id=str(span_id),
                parent_span_id=str(parent_span_id) if parent_span_id else None,
                operation_name=attrs.get("resource_name") or attrs.get("service", "unknown"),
                service_name=attrs.get("service", "unknown"),
                start_time=start_time or datetime.now(UTC),
                duration_ms=duration_ms,
                status=status,
                attributes=span_attributes,
                events=events,
            )
        except Exception as e:
            logger.error(f"Error parsing Datadog span: {e}")
            return None

    def _parse_dd_timestamp(self, value: str | None) -> datetime | None:
        """Parse a Datadog ISO8601 timestamp string.

        Args:
            value: Timestamp string (e.g. "2023-01-02T09:42:36.420Z")

        Returns:
            Parsed timezone-aware datetime, or None
        """
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            logger.warning(f"Could not parse Datadog timestamp: {value}")
            return None

    def _infer_status(self, custom_attrs: dict[str, Any], tags: list[str]) -> Any:
        """Best-effort error/status inference.

        Datadog's documented Span resource has no first-class status/error
        field (see module docstring) - this checks the signals actually
        documented elsewhere in Datadog's product (an `error` custom
        attribute, an `error:*`-prefixed tag, or `error.type`/`error.message`
        OTel attributes) and defaults to UNSET rather than guessing OK, since
        an unrecognized shape should not silently read as a passing span.

        Args:
            custom_attrs: The span's custom/OTel attributes object
            tags: The span's tag list

        Returns:
            "ERROR", "OK", or "UNSET"
        """
        if custom_attrs.get("error") in (True, "true", 1):
            return "ERROR"
        if any(k in custom_attrs for k in ("error.type", "error.message", "error.stack")):
            return "ERROR"
        for tag in tags:
            if tag == "error:true" or tag.startswith("error.type:"):
                return "ERROR"
        if custom_attrs.get("error") in (False, "false", 0):
            return "OK"
        return "UNSET"

    def _extract_semconv_attributes(self, custom_attrs: dict[str, Any]) -> dict[str, Any]:
        """Extract known gen_ai.*/llm.* semantic-convention keys for SpanAttributes.

        Handles both a flat dotted-key shape (`{"gen_ai.system": "openai"}`)
        and a nested-object shape (`{"gen_ai": {"system": "openai"}}`), since
        which one Datadog's OTLP intake produces for a given attribute is not
        pinned down by the public schema (see module docstring).

        Args:
            custom_attrs: The span's custom/OTel attributes object

        Returns:
            Dict of dotted-key attributes, suitable for SpanAttributes(**...)
        """
        flat: dict[str, Any] = {}

        def walk(prefix: str, obj: Any) -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    key = f"{prefix}.{k}" if prefix else k
                    walk(key, v)
            else:
                flat[prefix] = obj

        walk("", custom_attrs)
        return flat

    def _group_into_trace(self, trace_id: str, spans: list[SpanData]) -> TraceData:
        """Group a flat list of spans belonging to one trace into TraceData.

        Args:
            trace_id: The trace ID all spans share
            spans: All spans for this trace

        Returns:
            Assembled TraceData
        """
        root_spans = [s for s in spans if not s.parent_span_id]
        root_span = root_spans[0] if root_spans else spans[0]

        start_times = [s.start_time for s in spans]
        end_times = [
            datetime.fromtimestamp(
                s.start_time.timestamp() + (s.duration_ms / 1000), tz=s.start_time.tzinfo
            )
            for s in spans
        ]
        trace_start = min(start_times)
        trace_end = max(end_times)
        trace_duration_ms = (trace_end - trace_start).total_seconds() * 1000

        trace_status: Any = "ERROR" if any(s.has_error for s in spans) else "OK"

        return TraceData(
            trace_id=trace_id,
            spans=spans,
            start_time=trace_start,
            duration_ms=trace_duration_ms,
            service_name=root_span.service_name,
            root_operation=root_span.operation_name,
            status=trace_status,
        )
