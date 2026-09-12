"""Tests for Datadog backend.

Fixtures below are hand-built from Datadog's published OpenAPI spec for the
Spans API (v2) - specifically the `Span`/`SpansAttributes` schema in
https://github.com/DataDog/datadog-api-client-python/blob/master/.generator/schemas/v2/openapi.yaml
- not against a live account (none was available). See the module docstring
in `opentelemetry_mcp/backends/datadog.py` for the two things the public
schema does not pin down (where gen_ai.* attributes land, and how error
status is represented) and how this implementation handles both possibilities
it could find documented.
"""

from datetime import UTC, datetime

import pytest

from opentelemetry_mcp.backends.datadog import DatadogBackend
from opentelemetry_mcp.models import Filter, FilterOperator, FilterType

FAKE_API_KEY = "dd-api1"
FAKE_APP_KEY = "dd-app1"


def test_datadog_backend_requires_api_key() -> None:
    """Test that Datadog backend requires an API key."""
    with pytest.raises(ValueError, match="requires an API key"):
        DatadogBackend(url="https://api.datadoghq.com", api_key=None, app_key=FAKE_APP_KEY)


def test_datadog_backend_requires_app_key() -> None:
    """Test that Datadog backend requires an Application key even with an API key."""
    with pytest.raises(ValueError, match="Application key"):
        DatadogBackend(url="https://api.datadoghq.com", api_key=FAKE_API_KEY, app_key=None)


def test_datadog_backend_initialization() -> None:
    """Test Datadog backend initializes correctly with both keys."""
    backend = DatadogBackend(
        url="https://api.datadoghq.com",
        api_key=FAKE_API_KEY,
        app_key=FAKE_APP_KEY,
        timeout=15.0,
    )

    assert backend.url == "https://api.datadoghq.com"
    assert backend.api_key == FAKE_API_KEY
    assert backend.app_key == FAKE_APP_KEY
    assert backend.timeout == 15.0


def test_datadog_client_headers() -> None:
    """Test that Datadog client sends DD-API-KEY and DD-APPLICATION-KEY headers."""
    backend = DatadogBackend(
        url="https://api.datadoghq.com",
        api_key=FAKE_API_KEY,
        app_key=FAKE_APP_KEY,
    )

    client = backend.client
    assert client.headers["DD-API-KEY"] == FAKE_API_KEY
    assert client.headers["DD-APPLICATION-KEY"] == FAKE_APP_KEY
    assert client.headers["Content-Type"] == "application/json"


def _backend() -> DatadogBackend:
    return DatadogBackend(
        url="https://api.datadoghq.com", api_key=FAKE_API_KEY, app_key=FAKE_APP_KEY
    )


class TestBuildDatadogQuery:
    """Test Filter -> Datadog span search query string conversion."""

    def test_equals_facet_field(self) -> None:
        backend = _backend()
        f = Filter(
            field="service.name",
            operator=FilterOperator.EQUALS,
            value="my-service",
            value_type=FilterType.STRING,
        )
        assert backend._filter_to_dd_query(f) == "service:my-service"

    def test_equals_custom_attribute_is_at_prefixed(self) -> None:
        backend = _backend()
        f = Filter(
            field="gen_ai.system",
            operator=FilterOperator.EQUALS,
            value="openai",
            value_type=FilterType.STRING,
        )
        assert backend._filter_to_dd_query(f) == "@gen_ai.system:openai"

    def test_not_equals(self) -> None:
        backend = _backend()
        f = Filter(
            field="gen_ai.system",
            operator=FilterOperator.NOT_EQUALS,
            value="openai",
            value_type=FilterType.STRING,
        )
        assert backend._filter_to_dd_query(f) == "-@gen_ai.system:openai"

    def test_status_error_equals(self) -> None:
        backend = _backend()
        f = Filter(
            field="status",
            operator=FilterOperator.EQUALS,
            value="ERROR",
            value_type=FilterType.STRING,
        )
        assert backend._filter_to_dd_query(f) == "status:error"

    def test_duration_gte_converts_ms_to_ns(self) -> None:
        backend = _backend()
        f = Filter(
            field="duration", operator=FilterOperator.GTE, value=1000, value_type=FilterType.NUMBER
        )
        # 1000ms -> 1_000_000_000ns
        assert backend._filter_to_dd_query(f) == "@duration:[1000000000 TO *]"

    def test_duration_lt_converts_ms_to_ns(self) -> None:
        backend = _backend()
        f = Filter(
            field="duration", operator=FilterOperator.LT, value=5000, value_type=FilterType.NUMBER
        )
        assert backend._filter_to_dd_query(f) == "@duration:{* TO 5000000000}"

    def test_exists(self) -> None:
        backend = _backend()
        f = Filter(
            field="gen_ai.system", operator=FilterOperator.EXISTS, value_type=FilterType.STRING
        )
        assert backend._filter_to_dd_query(f) == "@gen_ai.system:*"

    def test_not_exists(self) -> None:
        backend = _backend()
        f = Filter(
            field="gen_ai.system", operator=FilterOperator.NOT_EXISTS, value_type=FilterType.STRING
        )
        assert backend._filter_to_dd_query(f) == "-@gen_ai.system:*"

    def test_in_builds_or(self) -> None:
        backend = _backend()
        f = Filter(
            field="gen_ai.system",
            operator=FilterOperator.IN,
            values=["openai", "anthropic"],
            value_type=FilterType.STRING,
        )
        assert (
            backend._filter_to_dd_query(f) == "(@gen_ai.system:openai OR @gen_ai.system:anthropic)"
        )

    def test_build_dd_query_empty_defaults_to_wildcard(self) -> None:
        backend = _backend()
        assert backend._build_dd_query([]) == "*"

    def test_build_dd_query_joins_with_and(self) -> None:
        backend = _backend()
        filters = [
            Filter(
                field="service.name",
                operator=FilterOperator.EQUALS,
                value="svc",
                value_type=FilterType.STRING,
            ),
            Filter(
                field="gen_ai.system",
                operator=FilterOperator.EQUALS,
                value="openai",
                value_type=FilterType.STRING,
            ),
        ]
        assert backend._build_dd_query(filters) == "service:svc AND @gen_ai.system:openai"


class TestParseDatadogSpan:
    """Test parsing raw Datadog Span resources into SpanData."""

    def test_parse_root_span(self) -> None:
        backend = _backend()
        span_obj = {
            "id": "AAAAAWgN8Xwgr1vKDQAAAABBV2dOOFh3ZzZobm1mWXJFYTR0OA",
            "type": "spans",
            "attributes": {
                "trace_id": "1234567890987654321",
                "span_id": "1234567890987654321",
                "parent_id": "0",
                "service": "my-llm-service",
                "resource_name": "chat_completion",
                "start_timestamp": "2023-01-02T09:42:36.320Z",
                "end_timestamp": "2023-01-02T09:42:36.420Z",
                "tags": ["env:prod", "team:A"],
                "attributes": {
                    "gen_ai.system": "openai",
                    "gen_ai.request.model": "gpt-4",
                    "gen_ai.usage.total_tokens": 150,
                },
            },
        }

        span = backend._parse_dd_span(span_obj)

        assert span is not None
        assert span.trace_id == "1234567890987654321"
        assert span.span_id == "1234567890987654321"
        assert span.parent_span_id is None  # "0" sentinel -> no parent
        assert span.service_name == "my-llm-service"
        assert span.operation_name == "chat_completion"
        assert span.duration_ms == pytest.approx(100.0)
        assert span.status == "UNSET"
        assert span.attributes.gen_ai_system == "openai"
        assert span.attributes.gen_ai_request_model == "gpt-4"

    def test_parse_child_span_with_real_parent(self) -> None:
        backend = _backend()
        span_obj = {
            "id": "child",
            "attributes": {
                "trace_id": "trace1",
                "span_id": "span2",
                "parent_id": "span1",
                "service": "my-service",
                "resource_name": "op",
                "start_timestamp": "2023-01-02T09:42:36.320Z",
                "end_timestamp": "2023-01-02T09:42:36.420Z",
            },
        }

        span = backend._parse_dd_span(span_obj)

        assert span is not None
        assert span.parent_span_id == "span1"

    def test_parse_span_missing_required_ids_returns_none(self) -> None:
        backend = _backend()
        assert backend._parse_dd_span({"attributes": {"service": "svc"}}) is None

    def test_error_inferred_from_custom_attribute(self) -> None:
        backend = _backend()
        span_obj = {
            "attributes": {
                "trace_id": "t1",
                "span_id": "s1",
                "service": "svc",
                "resource_name": "op",
                "start_timestamp": "2023-01-02T09:42:36.320Z",
                "end_timestamp": "2023-01-02T09:42:36.420Z",
                "attributes": {"error": True, "error.message": "boom"},
            },
        }

        span = backend._parse_dd_span(span_obj)

        assert span is not None
        assert span.status == "ERROR"

    def test_error_inferred_from_tag(self) -> None:
        backend = _backend()
        span_obj = {
            "attributes": {
                "trace_id": "t1",
                "span_id": "s1",
                "service": "svc",
                "resource_name": "op",
                "start_timestamp": "2023-01-02T09:42:36.320Z",
                "end_timestamp": "2023-01-02T09:42:36.420Z",
                "tags": ["error:true"],
            },
        }

        span = backend._parse_dd_span(span_obj)

        assert span is not None
        assert span.status == "ERROR"

    def test_nested_gen_ai_attributes_are_flattened(self) -> None:
        """Handles the case where Datadog ingest nests OTel attrs as objects
        rather than flat dotted keys - see module docstring."""
        backend = _backend()
        span_obj = {
            "attributes": {
                "trace_id": "t1",
                "span_id": "s1",
                "service": "svc",
                "resource_name": "op",
                "start_timestamp": "2023-01-02T09:42:36.320Z",
                "end_timestamp": "2023-01-02T09:42:36.420Z",
                "attributes": {"gen_ai": {"system": "anthropic", "request": {"model": "claude"}}},
            },
        }

        span = backend._parse_dd_span(span_obj)

        assert span is not None
        assert span.attributes.gen_ai_system == "anthropic"
        assert span.attributes.gen_ai_request_model == "claude"


class TestGroupIntoTrace:
    """Test grouping a flat span list into TraceData."""

    def test_group_selects_root_and_aggregates_status(self) -> None:
        backend = _backend()
        now = datetime(2023, 1, 2, 9, 42, 36, tzinfo=UTC)

        root = backend._parse_dd_span(
            {
                "attributes": {
                    "trace_id": "t1",
                    "span_id": "root",
                    "parent_id": "0",
                    "service": "svc",
                    "resource_name": "root-op",
                    "start_timestamp": now.isoformat().replace("+00:00", "Z"),
                    "end_timestamp": now.isoformat().replace("+00:00", "Z"),
                }
            }
        )
        child = backend._parse_dd_span(
            {
                "attributes": {
                    "trace_id": "t1",
                    "span_id": "child",
                    "parent_id": "root",
                    "service": "svc",
                    "resource_name": "child-op",
                    "start_timestamp": now.isoformat().replace("+00:00", "Z"),
                    "end_timestamp": now.isoformat().replace("+00:00", "Z"),
                    "attributes": {"error": True},
                }
            }
        )
        assert root is not None and child is not None

        trace = backend._group_into_trace("t1", [root, child])

        assert trace.trace_id == "t1"
        assert trace.service_name == "svc"
        assert trace.root_operation == "root-op"
        assert trace.status == "ERROR"  # child span error propagates to trace status
        assert len(trace.spans) == 2
