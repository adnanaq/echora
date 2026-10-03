"""Unit tests for the search route handler."""

from __future__ import annotations

from unittest.mock import create_autospec, patch

import pytest
from google.protobuf import struct_pb2
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from qdrant_client import AsyncQdrantClient
from qdrant_db import QdrantClient
from qdrant_db.contracts import SearchRange
from vector_db_interface import SearchHit
from vector_processing import (
    MultiVectorEmbeddingManager,
    TextProcessor,
    VisionProcessor,
)
from vector_proto.v1 import vector_search_pb2
from vector_service.routes import search as search_route
from vector_service.routes.search import (
    InvalidFiltersPayloadError,
    _map_filter_conditions,
    _proto_value_to_python,
    _validate_filter_fields,
)
from vector_service.runtime import VectorRuntime

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TRACER = trace.get_tracer("test")

_INDEXED_FIELDS: frozenset[str] = frozenset(
    {"type", "status", "year", "genres", "entity_type", "score.mean", "score.weighted"}
)


def _runtime(
    *,
    text_embedding: list[float] | None = None,
    search_results: list | None = None,
    record_query_text: bool = False,
) -> VectorRuntime:
    text_processor = create_autospec(TextProcessor, instance=True)
    text_processor.encode_text_with_sparse.return_value = (
        text_embedding or [0.1] * 4,
        None,
    )
    vision_processor = create_autospec(VisionProcessor, instance=True)
    vision_processor.encode_image.return_value = None
    qdrant_client = create_autospec(QdrantClient, instance=True)
    qdrant_client.search.return_value = search_results or []
    qdrant_client.indexed_fields = _INDEXED_FIELDS
    return VectorRuntime(
        qdrant_client=qdrant_client,
        async_qdrant_client=create_autospec(AsyncQdrantClient, instance=True),
        text_processor=text_processor,
        vision_processor=vision_processor,
        embedding_manager=create_autospec(MultiVectorEmbeddingManager, instance=True),
        embedding_cache=None,
        record_query_text=record_query_text,
    )


def _make_condition(
    field: str,
    operator: vector_search_pb2.FilterOperator,
    value: struct_pb2.Value,
    clause: vector_search_pb2.FilterClause = vector_search_pb2.FILTER_CLAUSE_UNSPECIFIED,
) -> vector_search_pb2.FilterCondition:
    return vector_search_pb2.FilterCondition(
        field=field, operator=operator, value=value, clause=clause
    )


def _str_value(s: str) -> struct_pb2.Value:
    return struct_pb2.Value(string_value=s)


def _num_value(n: float) -> struct_pb2.Value:
    return struct_pb2.Value(number_value=n)


def _list_value(*items: str) -> struct_pb2.Value:
    lv = struct_pb2.ListValue(values=[struct_pb2.Value(string_value=i) for i in items])
    return struct_pb2.Value(list_value=lv)


def _range_value(**bounds: float) -> struct_pb2.Value:
    fields = {k: struct_pb2.Value(number_value=v) for k, v in bounds.items()}
    sv = struct_pb2.Struct(fields=fields)
    return struct_pb2.Value(struct_value=sv)


# ---------------------------------------------------------------------------
# _proto_value_to_python
# ---------------------------------------------------------------------------


def test_proto_value_to_python_string_returns_str() -> None:
    assert _proto_value_to_python(_str_value("TV")) == "TV"


def test_proto_value_to_python_whole_number_returns_int() -> None:
    assert _proto_value_to_python(_num_value(2020.0)) == 2020
    assert isinstance(_proto_value_to_python(_num_value(2020.0)), int)


def test_proto_value_to_python_fractional_number_returns_float() -> None:
    assert _proto_value_to_python(_num_value(8.5)) == 8.5
    assert isinstance(_proto_value_to_python(_num_value(8.5)), float)


def test_proto_value_to_python_bool_returns_bool() -> None:
    assert _proto_value_to_python(struct_pb2.Value(bool_value=True)) is True


def test_proto_value_to_python_list_returns_list() -> None:
    assert _proto_value_to_python(_list_value("Action", "Drama")) == ["Action", "Drama"]


def test_proto_value_to_python_range_struct_returns_bounds() -> None:
    result = _proto_value_to_python(_range_value(gte=2020.0, lte=2023.0))
    assert result == {"gte": 2020, "lte": 2023}


# ---------------------------------------------------------------------------
# _validate_filter_fields
# ---------------------------------------------------------------------------


def test_validate_filter_fields_known_field_passes() -> None:
    cond = _make_condition(
        "status", vector_search_pb2.FILTER_OPERATOR_EQ, _str_value("FINISHED")
    )
    _validate_filter_fields([cond], _INDEXED_FIELDS)  # must not raise


def test_validate_filter_fields_unknown_field_raises_invalid_filters() -> None:
    cond = _make_condition(
        "unknown_field", vector_search_pb2.FILTER_OPERATOR_EQ, _str_value("x")
    )
    with pytest.raises(InvalidFiltersPayloadError, match="not indexed"):
        _validate_filter_fields([cond], _INDEXED_FIELDS)


def test_validate_filter_fields_several_unknown_fields_raises_invalid_filters() -> None:
    conditions = [
        _make_condition(
            "status", vector_search_pb2.FILTER_OPERATOR_EQ, _str_value("FINISHED")
        ),
        _make_condition(
            "bad_field", vector_search_pb2.FILTER_OPERATOR_EQ, _str_value("x")
        ),
    ]
    with pytest.raises(InvalidFiltersPayloadError):
        _validate_filter_fields(conditions, _INDEXED_FIELDS)


# ---------------------------------------------------------------------------
# _map_filter_conditions
# ---------------------------------------------------------------------------


def test_map_filter_conditions_eq_operator_returns_eq_condition() -> None:
    cond = _make_condition(
        "status", vector_search_pb2.FILTER_OPERATOR_EQ, _str_value("FINISHED")
    )
    result = _map_filter_conditions([cond])
    assert len(result) == 1
    assert result[0].field == "status"
    assert result[0].operator == "eq"
    assert result[0].value == "FINISHED"
    assert result[0].clause == "must"


def test_map_filter_conditions_ne_operator_returns_ne_condition() -> None:
    cond = _make_condition(
        "status", vector_search_pb2.FILTER_OPERATOR_NE, _str_value("CANCELLED")
    )
    result = _map_filter_conditions([cond])
    assert result[0].operator == "ne"
    assert result[0].value == "CANCELLED"


def test_map_filter_conditions_in_operator_returns_in_condition() -> None:
    cond = _make_condition(
        "genres", vector_search_pb2.FILTER_OPERATOR_IN, _list_value("Action", "Drama")
    )
    result = _map_filter_conditions([cond])
    assert result[0].operator == "in"
    assert result[0].value == ["Action", "Drama"]


def test_map_filter_conditions_not_in_operator_returns_not_in_condition() -> None:
    cond = _make_condition(
        "type", vector_search_pb2.FILTER_OPERATOR_NOT_IN, _list_value("MUSIC", "CM")
    )
    result = _map_filter_conditions([cond])
    assert result[0].operator == "not_in"
    assert result[0].value == ["MUSIC", "CM"]


def test_map_filter_conditions_range_operator_returns_search_range() -> None:
    cond = _make_condition(
        "year", vector_search_pb2.FILTER_OPERATOR_RANGE, _range_value(gte=2020.0)
    )
    result = _map_filter_conditions([cond])
    assert result[0].operator == "range"
    # SearchFilterCondition validator converts the raw dict to SearchRange
    assert isinstance(result[0].value, SearchRange)
    assert result[0].value.gte == 2020


def test_map_filter_conditions_must_not_clause_keeps_must_not() -> None:
    cond = _make_condition(
        "status",
        vector_search_pb2.FILTER_OPERATOR_NE,
        _str_value("CANCELLED"),
        clause=vector_search_pb2.FILTER_CLAUSE_MUST_NOT,
    )
    result = _map_filter_conditions([cond])
    assert result[0].clause == "must_not"


def test_map_filter_conditions_should_clause_keeps_should() -> None:
    cond = _make_condition(
        "type",
        vector_search_pb2.FILTER_OPERATOR_EQ,
        _str_value("TV"),
        clause=vector_search_pb2.FILTER_CLAUSE_SHOULD,
    )
    result = _map_filter_conditions([cond])
    assert result[0].clause == "should"


def test_map_filter_conditions_unspecified_clause_returns_must() -> None:
    cond = _make_condition(
        "status", vector_search_pb2.FILTER_OPERATOR_EQ, _str_value("FINISHED")
    )
    result = _map_filter_conditions([cond])
    assert result[0].clause == "must"


def test_map_filter_conditions_unspecified_operator_raises_invalid_filters() -> None:
    cond = _make_condition(
        "status", vector_search_pb2.FILTER_OPERATOR_UNSPECIFIED, _str_value("x")
    )
    with pytest.raises(InvalidFiltersPayloadError):
        _map_filter_conditions([cond])


# ---------------------------------------------------------------------------
# search() handler — integration with route
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_unknown_filter_field_returns_invalid_filters_error() -> None:
    runtime = _runtime()
    request = vector_search_pb2.SearchRequest(
        query_text="action anime",
        filters=[
            _make_condition(
                "bad_field", vector_search_pb2.FILTER_OPERATOR_EQ, _str_value("x")
            )
        ],
    )
    response = await search_route.search(runtime, request, context=None)
    assert response.error.code == "INVALID_FILTERS"
    assert response.error.retryable is False


@pytest.mark.asyncio
async def test_search_valid_filters_returns_results() -> None:
    hits = [SearchHit(id="1", score=0.9, payload={"title": "Bebop"})]
    runtime = _runtime(search_results=hits)
    request = vector_search_pb2.SearchRequest(
        query_text="space western",
        filters=[
            _make_condition(
                "status", vector_search_pb2.FILTER_OPERATOR_EQ, _str_value("FINISHED")
            )
        ],
    )
    response = await search_route.search(runtime, request, context=None)
    assert not response.HasField("error")
    assert len(response.data) == 1


@pytest.mark.asyncio
async def test_search_unreadable_image_returns_invalid_image_input_error() -> None:
    runtime = _runtime()
    request = vector_search_pb2.SearchRequest(image=b"not-a-real-image")
    with patch(
        "vector_service.routes.search._encode_image_bytes",
        autospec=True,
        side_effect=ValueError("unsupported image format"),
    ):
        response = await search_route.search(runtime, request, context=None)
    assert response.error.code == "INVALID_IMAGE_INPUT"
    assert response.error.retryable is False


@pytest.mark.asyncio
async def test_search_qdrant_value_error_returns_search_failed_error() -> None:
    runtime = _runtime()
    runtime.qdrant_client.search.side_effect = ValueError("qdrant value error")
    request = vector_search_pb2.SearchRequest(query_text="action anime")
    response = await search_route.search(runtime, request, context=None)
    assert response.error.code == "SEARCH_FAILED"


@pytest.mark.asyncio
async def test_search_text_query_returns_results() -> None:
    hits = [SearchHit(id="1", score=0.95, payload={"title": "Cowboy Bebop"})]
    runtime = _runtime(search_results=hits)
    request = vector_search_pb2.SearchRequest(query_text="space western")
    response = await search_route.search(runtime, request, context=None)
    assert len(response.data) == 1
    assert response.data[0].id == "1"
    assert abs(response.data[0].similarity_score - 0.95) < 1e-6
    assert not response.HasField("error")


@pytest.mark.asyncio
async def test_search_payload_choice_unset_returns_payloads() -> None:
    hits = [SearchHit(id="1", score=0.95, payload={"title": "Cowboy Bebop"})]
    runtime = _runtime(search_results=hits)
    request = vector_search_pb2.SearchRequest(query_text="space western")

    response = await search_route.search(runtime, request, context=None)

    assert runtime.qdrant_client.search.call_args.args[0].with_payload is True
    assert response.data[0].payload_json == '{"title": "Cowboy Bebop"}'


@pytest.mark.asyncio
async def test_search_payload_turned_off_returns_ids_and_scores_only() -> None:
    hits = [SearchHit(id="1", score=0.95, payload={})]
    runtime = _runtime(search_results=hits)
    request = vector_search_pb2.SearchRequest(
        query_text="space western", with_payload=False
    )

    response = await search_route.search(runtime, request, context=None)

    assert runtime.qdrant_client.search.call_args.args[0].with_payload is False
    assert response.data[0].id == "1"
    assert response.data[0].payload_json == ""


@pytest.mark.asyncio
async def test_search_no_text_or_image_returns_missing_query_input_error() -> None:
    runtime = _runtime()
    request = vector_search_pb2.SearchRequest()
    response = await search_route.search(runtime, request, context=None)
    assert response.error.code == "MISSING_QUERY_INPUT"


async def _search_in_span(
    spans: InMemorySpanExporter,
    runtime: VectorRuntime,
    request: vector_search_pb2.SearchRequest,
) -> ReadableSpan:
    with _TRACER.start_as_current_span("rpc.server.Search"):
        await search_route.search(runtime, request, context=None)
    (span,) = spans.get_finished_spans()
    return span


@pytest.mark.asyncio
async def test_search_filtered_request_records_parameters_on_span(
    span_exporter: InMemorySpanExporter,
) -> None:
    request = vector_search_pb2.SearchRequest(
        query_text="space western",
        entity_type="anime",
        limit=5,
        with_payload=False,
        filters=[
            _make_condition(
                "status", vector_search_pb2.FILTER_OPERATOR_EQ, _str_value("FINISHED")
            ),
            _make_condition(
                "year",
                vector_search_pb2.FILTER_OPERATOR_RANGE,
                _range_value(gte=1998.0),
            ),
        ],
    )

    span = await _search_in_span(span_exporter, _runtime(), request)

    assert span.attributes["search.has_text"] is True
    assert span.attributes["search.has_image"] is False
    assert span.attributes["search.entity_type"] == "anime"
    assert span.attributes["search.limit"] == 5
    assert span.attributes["search.with_payload"] is False
    assert span.attributes["search.filter_fields"] == ("status", "year")


@pytest.mark.asyncio
async def test_search_query_text_recording_off_omits_query_text_from_span(
    span_exporter: InMemorySpanExporter,
) -> None:
    request = vector_search_pb2.SearchRequest(query_text="space western")

    span = await _search_in_span(span_exporter, _runtime(), request)

    assert "search.query_text" not in span.attributes


@pytest.mark.asyncio
async def test_search_query_text_recording_on_records_trimmed_query_text(
    span_exporter: InMemorySpanExporter,
) -> None:
    request = vector_search_pb2.SearchRequest(query_text="  space western  ")

    span = await _search_in_span(
        span_exporter, _runtime(record_query_text=True), request
    )

    assert span.attributes["search.query_text"] == "space western"


@pytest.mark.asyncio
async def test_search_rejected_request_records_parameters_on_span(
    span_exporter: InMemorySpanExporter,
) -> None:
    request = vector_search_pb2.SearchRequest(limit=3)

    span = await _search_in_span(
        span_exporter, _runtime(record_query_text=True), request
    )

    assert span.attributes["search.has_text"] is False
    assert span.attributes["search.has_image"] is False
    assert span.attributes["search.limit"] == 3
    assert "search.query_text" not in span.attributes
