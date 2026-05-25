"""Unit tests for the search route handler."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from google.protobuf import struct_pb2
from vector_proto.v1 import vector_search_pb2
from vector_service.routes import search as search_route
from qdrant_db.contracts import SearchRange
from vector_service.routes.search import (
    InvalidFiltersPayloadError,
    _map_filter_conditions,
    _proto_value_to_python,
    _validate_filter_fields,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_INDEXED_FIELDS: frozenset[str] = frozenset(
    {"type", "status", "year", "genres", "entity_type", "score.arithmetic_mean"}
)


def _runtime(
    *,
    text_embedding: list[float] | None = None,
    search_results: list | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        text_processor=SimpleNamespace(
            encode_text_with_sparse=AsyncMock(
                return_value=(text_embedding or [0.1] * 4, None)
            )
        ),
        vision_processor=SimpleNamespace(encode_image=AsyncMock(return_value=None)),
        qdrant_client=SimpleNamespace(
            search=AsyncMock(return_value=search_results or []),
            indexed_fields=_INDEXED_FIELDS,
        ),
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
    lv = struct_pb2.ListValue(
        values=[struct_pb2.Value(string_value=i) for i in items]
    )
    return struct_pb2.Value(list_value=lv)


def _range_value(**bounds: float) -> struct_pb2.Value:
    fields = {k: struct_pb2.Value(number_value=v) for k, v in bounds.items()}
    sv = struct_pb2.Struct(fields=fields)
    return struct_pb2.Value(struct_value=sv)


# ---------------------------------------------------------------------------
# _proto_value_to_python
# ---------------------------------------------------------------------------


def test_proto_value_string() -> None:
    assert _proto_value_to_python(_str_value("TV")) == "TV"


def test_proto_value_whole_number_becomes_int() -> None:
    assert _proto_value_to_python(_num_value(2020.0)) == 2020
    assert isinstance(_proto_value_to_python(_num_value(2020.0)), int)


def test_proto_value_fractional_stays_float() -> None:
    assert _proto_value_to_python(_num_value(8.5)) == 8.5
    assert isinstance(_proto_value_to_python(_num_value(8.5)), float)


def test_proto_value_bool() -> None:
    assert _proto_value_to_python(struct_pb2.Value(bool_value=True)) is True


def test_proto_value_list() -> None:
    assert _proto_value_to_python(_list_value("Action", "Drama")) == ["Action", "Drama"]


def test_proto_value_struct_range() -> None:
    result = _proto_value_to_python(_range_value(gte=2020.0, lte=2023.0))
    assert result == {"gte": 2020, "lte": 2023}


# ---------------------------------------------------------------------------
# _validate_filter_fields
# ---------------------------------------------------------------------------


def test_validate_filter_fields_passes_known_field() -> None:
    cond = _make_condition("status", vector_search_pb2.FILTER_OPERATOR_EQ, _str_value("FINISHED"))
    _validate_filter_fields([cond], _INDEXED_FIELDS)  # must not raise


def test_validate_filter_fields_rejects_unknown_field() -> None:
    cond = _make_condition("unknown_field", vector_search_pb2.FILTER_OPERATOR_EQ, _str_value("x"))
    with pytest.raises(InvalidFiltersPayloadError, match="not indexed"):
        _validate_filter_fields([cond], _INDEXED_FIELDS)


def test_validate_filter_fields_rejects_on_first_unknown() -> None:
    conditions = [
        _make_condition("status", vector_search_pb2.FILTER_OPERATOR_EQ, _str_value("FINISHED")),
        _make_condition("bad_field", vector_search_pb2.FILTER_OPERATOR_EQ, _str_value("x")),
    ]
    with pytest.raises(InvalidFiltersPayloadError):
        _validate_filter_fields(conditions, _INDEXED_FIELDS)


# ---------------------------------------------------------------------------
# _map_filter_conditions
# ---------------------------------------------------------------------------


def test_map_eq_condition() -> None:
    cond = _make_condition("status", vector_search_pb2.FILTER_OPERATOR_EQ, _str_value("FINISHED"))
    result = _map_filter_conditions([cond])
    assert len(result) == 1
    assert result[0].field == "status"
    assert result[0].operator == "eq"
    assert result[0].value == "FINISHED"
    assert result[0].clause == "must"


def test_map_ne_condition() -> None:
    cond = _make_condition("status", vector_search_pb2.FILTER_OPERATOR_NE, _str_value("CANCELLED"))
    result = _map_filter_conditions([cond])
    assert result[0].operator == "ne"
    assert result[0].value == "CANCELLED"


def test_map_in_condition() -> None:
    cond = _make_condition("genres", vector_search_pb2.FILTER_OPERATOR_IN, _list_value("Action", "Drama"))
    result = _map_filter_conditions([cond])
    assert result[0].operator == "in"
    assert result[0].value == ["Action", "Drama"]


def test_map_not_in_condition() -> None:
    cond = _make_condition("type", vector_search_pb2.FILTER_OPERATOR_NOT_IN, _list_value("MUSIC", "CM"))
    result = _map_filter_conditions([cond])
    assert result[0].operator == "not_in"
    assert result[0].value == ["MUSIC", "CM"]


def test_map_range_condition() -> None:
    cond = _make_condition("year", vector_search_pb2.FILTER_OPERATOR_RANGE, _range_value(gte=2020.0))
    result = _map_filter_conditions([cond])
    assert result[0].operator == "range"
    # SearchFilterCondition validator converts the raw dict to SearchRange
    assert isinstance(result[0].value, SearchRange)
    assert result[0].value.gte == 2020


def test_map_must_not_clause() -> None:
    cond = _make_condition(
        "status",
        vector_search_pb2.FILTER_OPERATOR_NE,
        _str_value("CANCELLED"),
        clause=vector_search_pb2.FILTER_CLAUSE_MUST_NOT,
    )
    result = _map_filter_conditions([cond])
    assert result[0].clause == "must_not"


def test_map_should_clause() -> None:
    cond = _make_condition(
        "type",
        vector_search_pb2.FILTER_OPERATOR_EQ,
        _str_value("TV"),
        clause=vector_search_pb2.FILTER_CLAUSE_SHOULD,
    )
    result = _map_filter_conditions([cond])
    assert result[0].clause == "should"


def test_map_unspecified_clause_defaults_to_must() -> None:
    cond = _make_condition("status", vector_search_pb2.FILTER_OPERATOR_EQ, _str_value("FINISHED"))
    result = _map_filter_conditions([cond])
    assert result[0].clause == "must"


def test_map_unspecified_operator_raises() -> None:
    cond = _make_condition("status", vector_search_pb2.FILTER_OPERATOR_UNSPECIFIED, _str_value("x"))
    with pytest.raises(InvalidFiltersPayloadError):
        _map_filter_conditions([cond])


# ---------------------------------------------------------------------------
# search() handler — integration with route
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_with_unknown_filter_field_returns_invalid_filters() -> None:
    runtime = _runtime()
    request = vector_search_pb2.SearchRequest(
        query_text="action anime",
        filters=[
            _make_condition("bad_field", vector_search_pb2.FILTER_OPERATOR_EQ, _str_value("x"))
        ],
    )
    response = await search_route.search(runtime, request, context=None)
    assert response.error.code == "INVALID_FILTERS"
    assert response.error.retryable is False


@pytest.mark.asyncio
async def test_search_with_valid_filters_passes_through() -> None:
    hits = [SimpleNamespace(id="1", score=0.9, payload={"title": "Bebop"})]
    runtime = _runtime(search_results=hits)
    request = vector_search_pb2.SearchRequest(
        query_text="space western",
        filters=[
            _make_condition("status", vector_search_pb2.FILTER_OPERATOR_EQ, _str_value("FINISHED"))
        ],
    )
    response = await search_route.search(runtime, request, context=None)
    assert not response.HasField("error")
    assert len(response.data) == 1


@pytest.mark.asyncio
async def test_image_value_error_returns_invalid_image_input() -> None:
    runtime = _runtime()
    request = vector_search_pb2.SearchRequest(image=b"not-a-real-image")
    with patch(
        "vector_service.routes.search._encode_image_bytes",
        side_effect=ValueError("unsupported image format"),
    ):
        response = await search_route.search(runtime, request, context=None)
    assert response.error.code == "INVALID_IMAGE_INPUT"
    assert response.error.retryable is False


@pytest.mark.asyncio
async def test_qdrant_value_error_not_labeled_invalid_image_input() -> None:
    runtime = _runtime()
    runtime.qdrant_client.search = AsyncMock(side_effect=ValueError("qdrant value error"))
    request = vector_search_pb2.SearchRequest(query_text="action anime")
    response = await search_route.search(runtime, request, context=None)
    assert response.error.code == "SEARCH_FAILED"


@pytest.mark.asyncio
async def test_successful_text_search_returns_data() -> None:
    hits = [SimpleNamespace(id="1", score=0.95, payload={"title": "Cowboy Bebop"})]
    runtime = _runtime(search_results=hits)
    request = vector_search_pb2.SearchRequest(query_text="space western")
    response = await search_route.search(runtime, request, context=None)
    assert len(response.data) == 1
    assert response.data[0].id == "1"
    assert abs(response.data[0].similarity_score - 0.95) < 1e-6
    assert not response.HasField("error")


@pytest.mark.asyncio
async def test_missing_query_input_returns_error() -> None:
    runtime = _runtime()
    request = vector_search_pb2.SearchRequest()
    response = await search_route.search(runtime, request, context=None)
    assert response.error.code == "MISSING_QUERY_INPUT"
