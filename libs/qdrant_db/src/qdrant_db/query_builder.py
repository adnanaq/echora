"""Search query construction — contract types to Qdrant query model translation.

Stateless pure functions, no I/O, no async.
"""

from typing import Any, cast

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchExcept,
    MatchValue,
    Prefetch,
    Range,
    SparseVector,
)

from qdrant_db.contracts import (
    SearchFilterCondition,
    SearchRange,
    SearchRequest,
    SparseVectorData,
)


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


def build_filter(filters: list[SearchFilterCondition]) -> Filter | None:
    """Convert contract filter conditions into a Qdrant filter model.

    Conditions are grouped by their ``clause`` field into ``must`` (AND),
    ``must_not`` (NOT), and ``should`` (OR) buckets. Only non-empty buckets
    are passed to Qdrant.

    Args:
        filters: Typed filter conditions.

    Returns:
        Qdrant filter model or ``None`` when no filters are provided.
    """
    if not filters:
        return None

    must: list[FieldCondition] = []
    must_not: list[FieldCondition] = []
    should: list[FieldCondition] = []

    for condition in filters:
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


def build_sparse_query(sparse_embedding: SparseVectorData) -> SparseVector:
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


def build_prefetch_queries(
    request: SearchRequest,
    text_vector_name: str,
    image_vector_name: str,
    sparse_vector_name: str,
    qdrant_filter: Filter | None,
    prefetch_limit: int,
) -> list[Prefetch]:
    """Assemble prefetch branches for fusion search.

    Args:
        request: Typed search request.
        text_vector_name: Named text vector field.
        image_vector_name: Named image vector field.
        sparse_vector_name: Named sparse vector field.
        qdrant_filter: Optional pre-built Qdrant filter.
        prefetch_limit: Candidate pool size per branch fed into fusion.

    Returns:
        List of prefetch query branches (one per active embedding signal).
    """
    prefetch: list[Prefetch] = []

    if request.text_embedding is not None:
        prefetch.append(
            Prefetch(
                using=text_vector_name,
                query=request.text_embedding,
                limit=prefetch_limit,
                filter=qdrant_filter,
            )
        )
        for expansion in request.expanded_text_embeddings or []:
            prefetch.append(
                Prefetch(
                    using=text_vector_name,
                    query=expansion,
                    limit=prefetch_limit,
                    filter=qdrant_filter,
                )
            )

    if request.image_embedding is not None:
        prefetch.append(
            Prefetch(
                using=image_vector_name,
                query=request.image_embedding,
                limit=prefetch_limit,
                filter=qdrant_filter,
            )
        )

    if request.sparse_embedding is not None:
        prefetch.append(
            Prefetch(
                using=sparse_vector_name,
                query=build_sparse_query(request.sparse_embedding),
                limit=prefetch_limit,
                filter=qdrant_filter,
            )
        )

    return prefetch
