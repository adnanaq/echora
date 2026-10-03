from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchExcept,
    MatchValue,
    QuantizationSearchParams,
    Range,
    SearchParams,
    SparseVector,
)
from qdrant_db.contracts import SearchFilterCondition, SearchRequest, SparseVectorData
from qdrant_db.query_builder import (
    build_filter,
    build_prefetch_queries,
    build_sparse_query,
    build_text_search_params,
)


def test_build_filter_empty_list_returns_none() -> None:
    assert build_filter([]) is None


def test_build_filter_eq_operator_returns_must_match_value() -> None:
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


def test_build_filter_in_operator_returns_match_any() -> None:
    result = build_filter(
        [SearchFilterCondition(field="genre", operator="in", value=["action", "drama"])]
    )
    assert isinstance(result, Filter)
    cond = result.must[0]
    assert isinstance(cond, FieldCondition)
    assert isinstance(cond.match, MatchAny)
    assert cond.match.any == ["action", "drama"]


def test_build_filter_range_operator_returns_range() -> None:
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


def test_build_filter_multiple_conditions_returns_one_must_per_condition() -> None:
    result = build_filter(
        [
            SearchFilterCondition(field="type", operator="eq", value="anime"),
            SearchFilterCondition(field="genre", operator="in", value=["action"]),
        ]
    )
    assert isinstance(result, Filter)
    assert len(result.must) == 2


def test_build_filter_ne_operator_returns_match_except() -> None:
    result = build_filter(
        [SearchFilterCondition(field="status", operator="ne", value="CANCELLED")]
    )
    assert isinstance(result, Filter)
    cond = result.must[0]
    assert isinstance(cond, FieldCondition)
    assert cond.key == "status"
    assert isinstance(cond.match, MatchExcept)
    assert cond.match.except_ == ["CANCELLED"]


def test_build_filter_not_in_operator_returns_match_except() -> None:
    result = build_filter(
        [SearchFilterCondition(field="type", operator="not_in", value=["MUSIC", "CM"])]
    )
    assert isinstance(result, Filter)
    cond = result.must[0]
    assert isinstance(cond, FieldCondition)
    assert isinstance(cond.match, MatchExcept)
    assert cond.match.except_ == ["MUSIC", "CM"]


def test_build_filter_must_not_clause_fills_only_must_not() -> None:
    result = build_filter(
        [
            SearchFilterCondition(
                field="status", operator="eq", value="CANCELLED", clause="must_not"
            ),
        ]
    )
    assert isinstance(result, Filter)
    assert result.must is None
    assert result.should is None
    assert len(result.must_not) == 1
    cond = result.must_not[0]
    assert isinstance(cond, FieldCondition)
    assert cond.key == "status"


def test_build_filter_should_clause_fills_only_should() -> None:
    result = build_filter(
        [
            SearchFilterCondition(
                field="type", operator="eq", value="TV", clause="should"
            ),
            SearchFilterCondition(
                field="type", operator="eq", value="MOVIE", clause="should"
            ),
        ]
    )
    assert isinstance(result, Filter)
    assert result.must is None
    assert result.must_not is None
    assert len(result.should) == 2


def test_build_filter_mixed_clauses_fills_each_clause() -> None:
    result = build_filter(
        [
            SearchFilterCondition(field="year", operator="range", value={"gte": 2020}),
            SearchFilterCondition(
                field="status", operator="ne", value="CANCELLED", clause="must_not"
            ),
            SearchFilterCondition(
                field="type", operator="eq", value="TV", clause="should"
            ),
            SearchFilterCondition(
                field="type", operator="eq", value="OVA", clause="should"
            ),
        ]
    )
    assert isinstance(result, Filter)
    assert len(result.must) == 1
    assert len(result.must_not) == 1
    assert len(result.should) == 2


def test_build_sparse_query_returns_sparse_vector() -> None:
    sparse = SparseVectorData(indices=[0, 3], values=[0.8, 0.2])
    result = build_sparse_query(sparse)
    assert isinstance(result, SparseVector)
    assert result.indices == [0, 3]
    assert result.values == [0.8, 0.2]


def _base_request(**kwargs) -> SearchRequest:  # type: ignore[no-untyped-def]
    return SearchRequest(text_embedding=[0.1] * 1024, limit=10, **kwargs)


def test_build_prefetch_queries_text_only_returns_text_branch() -> None:
    request = _base_request()
    result = build_prefetch_queries(
        request, "text_vec", "image_vec", "sparse_vec", None, prefetch_limit=20
    )
    assert len(result) == 1
    assert result[0].using == "text_vec"
    assert result[0].limit == 20
    assert result[0].filter is None


def test_build_prefetch_queries_image_only_returns_image_branch() -> None:
    request = SearchRequest(image_embedding=[0.2] * 768, limit=5)
    result = build_prefetch_queries(
        request, "text_vec", "image_vec", "sparse_vec", None, prefetch_limit=20
    )
    assert len(result) == 1
    assert result[0].using == "image_vec"
    assert result[0].limit == 20


def test_build_prefetch_queries_sparse_only_returns_sparse_branch() -> None:
    request = SearchRequest(
        sparse_embedding={"indices": [1, 2], "values": [0.5, 0.3]}, limit=5
    )
    result = build_prefetch_queries(
        request, "text_vec", "image_vec", "sparse_vec", None, prefetch_limit=20
    )
    assert len(result) == 1
    assert result[0].using == "sparse_vec"
    assert isinstance(result[0].query, SparseVector)


def test_build_prefetch_queries_text_and_image_returns_both_branches() -> None:
    request = SearchRequest(
        text_embedding=[0.1] * 1024, image_embedding=[0.2] * 768, limit=10
    )
    result = build_prefetch_queries(
        request, "text_vec", "image_vec", "sparse_vec", None, prefetch_limit=20
    )
    assert len(result) == 2
    assert {branch.using for branch in result} == {"text_vec", "image_vec"}


def test_build_prefetch_queries_all_embeddings_returns_three_branches() -> None:
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


def test_build_prefetch_queries_filter_given_sets_branch_filter() -> None:
    qdrant_filter = Filter(must=[])
    request = _base_request()
    result = build_prefetch_queries(
        request, "text_vec", "image_vec", "sparse_vec", qdrant_filter, prefetch_limit=20
    )
    assert result[0].filter is qdrant_filter


def test_build_prefetch_queries_expanded_text_embeddings_returns_text_branch_each() -> (
    None
):
    request = SearchRequest(
        text_embedding=[0.1] * 1024,
        expanded_text_embeddings=[[0.2] * 1024, [0.3] * 1024],
        limit=10,
    )
    result = build_prefetch_queries(
        request, "text_vec", "image_vec", "sparse_vec", None, prefetch_limit=20
    )
    assert len(result) == 3
    assert all(branch.using == "text_vec" for branch in result)
    assert result[0].limit == 20


def test_build_prefetch_queries_expanded_text_with_image_returns_text_and_image_branches() -> (
    None
):
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
    text_branches = [branch for branch in result if branch.using == "text_vec"]
    image_branches = [branch for branch in result if branch.using == "image_vec"]
    assert len(text_branches) == 2
    assert len(image_branches) == 1


def test_build_text_search_params_nothing_set_returns_none() -> None:
    assert (
        build_text_search_params(hnsw_ef=None, rescore=None, oversampling=None) is None
    )


def test_build_text_search_params_ef_and_rescoring_set_returns_both() -> None:
    assert build_text_search_params(
        hnsw_ef=512, rescore=True, oversampling=4.0
    ) == SearchParams(
        hnsw_ef=512,
        quantization=QuantizationSearchParams(rescore=True, oversampling=4.0),
    )


def test_build_text_search_params_ef_only_returns_ef() -> None:
    assert build_text_search_params(
        hnsw_ef=128, rescore=None, oversampling=None
    ) == SearchParams(hnsw_ef=128)


def test_build_prefetch_queries_search_params_given_applies_to_text_branches_only() -> (
    None
):
    search_params = SearchParams(
        quantization=QuantizationSearchParams(rescore=True, oversampling=4.0)
    )
    request = SearchRequest(
        text_embedding=[0.1] * 1024,
        expanded_text_embeddings=[[0.3] * 1024],
        image_embedding=[0.2] * 768,
        sparse_embedding={"indices": [0], "values": [1.0]},
        limit=10,
    )
    result = build_prefetch_queries(
        request,
        "text_vec",
        "image_vec",
        "sparse_vec",
        None,
        prefetch_limit=20,
        text_search_params=search_params,
    )
    params_by_branch = [(branch.using, branch.params) for branch in result]
    assert params_by_branch == [
        ("text_vec", search_params),
        ("text_vec", search_params),
        ("image_vec", None),
        ("sparse_vec", None),
    ]
