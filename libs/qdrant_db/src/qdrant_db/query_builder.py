"""Search query construction — contract types to Qdrant query model translation.

Stateless pure functions, no I/O, no async.
"""

from typing import Any, cast

from qdrant_client.http.models import QueryResponse
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchAny,
    MatchExcept,
    MatchValue,
    Prefetch,
    QuantizationSearchParams,
    Range,
    Rrf,
    RrfQuery,
    SearchParams,
    SparseVector,
)
from qdrant_db.qdrant_models import (
    SearchFilterCondition,
    SearchRange,
    SearchRequest,
    SparseVectorPayload,
)
from vector_db_interface import SearchHit


def _build_field_condition(condition: SearchFilterCondition) -> FieldCondition:
    """Translate a single contract condition to a Qdrant FieldCondition.

    Args:
        condition: Typed filter condition.

    Returns:
        Qdrant field condition model.
    """
    if condition.operator == "eq":
        return FieldCondition(
            key=condition.field,
            match=MatchValue(value=condition.value),
        )

    if condition.operator == "ne":
        return FieldCondition(
            key=condition.field,
            match=MatchExcept(**{"except": [condition.value]}),
        )

    if condition.operator == "in":
        return FieldCondition(
            key=condition.field,
            match=MatchAny(any=cast(list[Any], condition.value)),
        )

    if condition.operator == "not_in":
        return FieldCondition(
            key=condition.field,
            match=MatchExcept(**{"except": cast(list[Any], condition.value)}),
        )

    # range
    range_value = cast(SearchRange, condition.value)
    return FieldCondition(
        key=condition.field,
        range=Range(**range_value.model_dump(exclude_none=True)),
    )


def build_filter(
    filters: list[SearchFilterCondition],
    entity_type: str | None = None,
) -> Filter | None:
    """Convert contract filter conditions into a Qdrant filter model.

    Conditions are grouped by their ``clause`` field into ``must`` (AND),
    ``must_not`` (NOT), and ``should`` (OR) buckets. Only non-empty buckets
    are passed to Qdrant.

    Args:
        filters: Typed filter conditions.
        entity_type: Optional entity type filter appended as an ``eq`` condition.

    Returns:
        Qdrant filter model or ``None`` when no filters are provided.
    """
    all_filters = list(filters)
    if entity_type:
        all_filters.append(
            SearchFilterCondition(
                field="entity_type",
                operator="eq",
                value=entity_type,
            )
        )
    if not all_filters:
        return None

    must: list[FieldCondition] = []
    must_not: list[FieldCondition] = []
    should: list[FieldCondition] = []

    for condition in all_filters:
        fc = _build_field_condition(condition)
        if condition.clause == "must_not":
            must_not.append(fc)
        elif condition.clause == "should":
            should.append(fc)
        else:
            must.append(fc)

    return Filter(
        must=cast(Any, must) or None,
        must_not=cast(Any, must_not) or None,
        should=cast(Any, should) or None,
    )


def build_sparse_query(sparse_embedding: SparseVectorPayload) -> SparseVector:
    """Convert typed sparse contract model to a Qdrant sparse query payload.

    Args:
        sparse_embedding: Typed sparse vector data contract.

    Returns:
        :class:`SparseVector` query model.
    """
    return SparseVector(
        indices=sparse_embedding.indices,
        values=sparse_embedding.values,
    )


def build_text_search_params(
    hnsw_ef: int | None, rescore: bool | None, oversampling: float | None
) -> SearchParams | None:
    """Build the search parameters for the dense text vector.

    Args:
        hnsw_ef: HNSW candidates kept during graph search, or None for Qdrant's default.
        rescore: Re-rank quantized candidates with the original vectors.
        oversampling: Extra quantized candidates fetched for rescoring.

    Returns:
        Search parameters, or None when nothing is set so Qdrant's defaults apply.
    """
    quantization = None
    if rescore is not None or oversampling is not None:
        quantization = QuantizationSearchParams(
            rescore=rescore, oversampling=oversampling
        )
    if hnsw_ef is None and quantization is None:
        return None
    return SearchParams(hnsw_ef=hnsw_ef, quantization=quantization)


def build_prefetch_queries(
    request: SearchRequest,
    text_vector_name: str,
    image_vector_name: str,
    sparse_vector_name: str,
    qdrant_filter: Filter | None,
    prefetch_limit: int,
    text_search_params: SearchParams | None = None,
    sparse_query: SparseVector | None = None,
) -> list[Prefetch]:
    """Assemble prefetch branches for fusion search.

    Args:
        request: Typed search request.
        text_vector_name: Named text vector field.
        image_vector_name: Named image vector field.
        sparse_vector_name: Named sparse vector field.
        qdrant_filter: Optional pre-built Qdrant filter.
        prefetch_limit: Candidate pool size per branch fed into fusion.
        text_search_params: Search parameters for the dense text branches.
        sparse_query: Pre-built sparse query vector to reuse when available.

    Returns:
        List of prefetch query branches (one per active embedding signal).
    """
    candidates: list[tuple[str, Any, SearchParams | None]] = []

    if request.text_embedding is not None:
        candidates.append(
            (text_vector_name, request.text_embedding, text_search_params)
        )
        for expansion in request.expanded_text_embeddings or []:
            candidates.append((text_vector_name, expansion, text_search_params))

    if request.image_embedding is not None:
        candidates.append((image_vector_name, request.image_embedding, None))

    if request.sparse_embedding is not None:
        candidates.append(
            (
                sparse_vector_name,
                sparse_query or build_sparse_query(request.sparse_embedding),
                None,
            )
        )

    return [
        Prefetch(
            using=vector_name,
            query=query,
            limit=prefetch_limit,
            filter=qdrant_filter,
            params=search_params,
        )
        for vector_name, query, search_params in candidates
    ]


def collect_active_embeddings(
    request: SearchRequest,
    text_vector_name: str,
    image_vector_name: str,
    sparse_vector_name: str,
) -> list[tuple[str, list[float] | SparseVector]]:
    """Collect active query embedding signals from a search request.

    Args:
        request: Typed search request model.
        text_vector_name: Named text vector field.
        image_vector_name: Named image vector field.
        sparse_vector_name: Named sparse vector field.

    Returns:
        List of ``(vector_name, query_vector)`` pairs for populated signals.
    """
    return [
        (name, vec)
        for name, vec in (
            (text_vector_name, request.text_embedding),
            (image_vector_name, request.image_embedding),
            (
                sparse_vector_name,
                build_sparse_query(request.sparse_embedding)
                if request.sparse_embedding
                else None,
            ),
        )
        if vec is not None
    ]


def build_fusion_query(fusion_method: str, rrf_k: int) -> FusionQuery | RrfQuery:
    """Build Qdrant fusion query model for multi-branch prefetch search.

    Args:
        fusion_method: Fusion mode (``rrf`` or ``dbsf``).
        rrf_k: Reciprocal rank fusion constant.

    Returns:
        Configured ``FusionQuery`` or ``RrfQuery`` model.
    """
    if fusion_method == "dbsf":
        return FusionQuery(fusion=Fusion.DBSF)
    return RrfQuery(rrf=Rrf(k=rrf_k))


def build_search_hits(response: QueryResponse) -> list[SearchHit]:
    """Normalize Qdrant points into search hits.

    Args:
        response: Raw Qdrant query response.

    Returns:
        List of normalized ``SearchHit`` models.
    """
    return [
        SearchHit(
            id=str(point.id),
            payload=dict(point.payload) if point.payload else {},
            score=float(point.score),
        )
        for point in response.points
    ]
