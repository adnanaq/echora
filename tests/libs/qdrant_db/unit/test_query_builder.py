"""Unit tests for query_builder pure functions."""

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    Range,
    SparseVector,
)
from qdrant_db.contracts import SearchFilterCondition, SearchRequest, SparseVectorData
from qdrant_db.query_builder import (
    build_filter,
    build_prefetch_queries,
    build_sparse_query,
)

# ---------------------------------------------------------------------------
# build_filter
# ---------------------------------------------------------------------------


def test_build_filter_returns_none_for_empty_list() -> None:
    assert build_filter([]) is None


def test_build_filter_eq_operator() -> None:
    result = build_filter(
        [SearchFilterCondition(field="type", operator="eq", value="anime")]
    )
    assert isinstance(result, Filter)
    assert len(result.must) == 1
    cond = result.must[0]
    assert isinstance(cond, FieldCondition)
    assert cond.key == "type"
    assert isinstance(cond.match, MatchValue)
    assert cond.match.value == "anime"


def test_build_filter_in_operator() -> None:
    result = build_filter(
        [SearchFilterCondition(field="genre", operator="in", value=["action", "drama"])]
    )
    assert isinstance(result, Filter)
    cond = result.must[0]
    assert isinstance(cond, FieldCondition)
    assert isinstance(cond.match, MatchAny)
    assert cond.match.any == ["action", "drama"]


def test_build_filter_range_operator() -> None:
    result = build_filter(
        [
            SearchFilterCondition(
                field="year", operator="range", value={"gte": 2000, "lte": 2020}
            )
        ]
    )
    assert isinstance(result, Filter)
    cond = result.must[0]
    assert isinstance(cond, FieldCondition)
    assert isinstance(cond.range, Range)
    assert cond.range.gte == 2000
    assert cond.range.lte == 2020


def test_build_filter_multiple_conditions() -> None:
    result = build_filter(
        [
            SearchFilterCondition(field="type", operator="eq", value="anime"),
            SearchFilterCondition(field="genre", operator="in", value=["action"]),
        ]
    )
    assert isinstance(result, Filter)
    assert len(result.must) == 2


# ---------------------------------------------------------------------------
# build_sparse_query
# ---------------------------------------------------------------------------


def test_build_sparse_query_returns_sparse_vector() -> None:
    sparse = SparseVectorData(indices=[0, 3], values=[0.8, 0.2])
    result = build_sparse_query(sparse)
    assert isinstance(result, SparseVector)
    assert result.indices == [0, 3]
    assert result.values == [0.8, 0.2]


# ---------------------------------------------------------------------------
# build_prefetch_queries
# ---------------------------------------------------------------------------


def _base_request(**kwargs) -> SearchRequest:  # type: ignore[no-untyped-def]
    return SearchRequest(text_embedding=[0.1] * 1024, limit=10, **kwargs)


def test_build_prefetch_text_only() -> None:
    request = _base_request()
    result = build_prefetch_queries(
        request, "text_vec", "image_vec", "sparse_vec", None, prefetch_limit=20
    )
    assert len(result) == 1
    assert result[0].using == "text_vec"
    assert result[0].limit == 20
    assert result[0].filter is None


def test_build_prefetch_image_only() -> None:
    request = SearchRequest(image_embedding=[0.2] * 768, limit=5)
    result = build_prefetch_queries(
        request, "text_vec", "image_vec", "sparse_vec", None, prefetch_limit=20
    )
    assert len(result) == 1
    assert result[0].using == "image_vec"
    assert result[0].limit == 20


def test_build_prefetch_sparse_only() -> None:
    request = SearchRequest(
        sparse_embedding={"indices": [1, 2], "values": [0.5, 0.3]}, limit=5
    )
    result = build_prefetch_queries(
        request, "text_vec", "image_vec", "sparse_vec", None, prefetch_limit=20
    )
    assert len(result) == 1
    assert result[0].using == "sparse_vec"
    assert isinstance(result[0].query, SparseVector)


def test_build_prefetch_text_and_image() -> None:
    request = SearchRequest(
        text_embedding=[0.1] * 1024, image_embedding=[0.2] * 768, limit=10
    )
    result = build_prefetch_queries(
        request, "text_vec", "image_vec", "sparse_vec", None, prefetch_limit=20
    )
    assert len(result) == 2
    assert {p.using for p in result} == {"text_vec", "image_vec"}


def test_build_prefetch_all_three_embeddings() -> None:
    request = SearchRequest(
        text_embedding=[0.1] * 1024,
        image_embedding=[0.2] * 768,
        sparse_embedding={"indices": [0], "values": [1.0]},
        limit=10,
    )
    result = build_prefetch_queries(
        request, "text_vec", "image_vec", "sparse_vec", None, prefetch_limit=20
    )
    assert len(result) == 3


def test_build_prefetch_passes_filter() -> None:
    qdrant_filter = Filter(must=[])
    request = _base_request()
    result = build_prefetch_queries(
        request, "text_vec", "image_vec", "sparse_vec", qdrant_filter, prefetch_limit=20
    )
    assert result[0].filter is qdrant_filter


def test_build_prefetch_expanded_text_embeddings() -> None:
    request = SearchRequest(
        text_embedding=[0.1] * 1024,
        expanded_text_embeddings=[[0.2] * 1024, [0.3] * 1024],
        limit=10,
    )
    result = build_prefetch_queries(
        request, "text_vec", "image_vec", "sparse_vec", None, prefetch_limit=20
    )
    assert len(result) == 3
    assert all(p.using == "text_vec" for p in result)
    assert result[0].limit == 20


def test_build_prefetch_expanded_with_other_modalities() -> None:
    request = SearchRequest(
        text_embedding=[0.1] * 1024,
        image_embedding=[0.2] * 768,
        expanded_text_embeddings=[[0.3] * 1024],
        limit=10,
    )
    result = build_prefetch_queries(
        request, "text_vec", "image_vec", "sparse_vec", None, prefetch_limit=20
    )
    assert len(result) == 3
    text_branches = [p for p in result if p.using == "text_vec"]
    image_branches = [p for p in result if p.using == "image_vec"]
    assert len(text_branches) == 2
    assert len(image_branches) == 1
