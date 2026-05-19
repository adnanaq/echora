"""Search query construction — contract types to Qdrant query model translation.

Stateless pure functions, no I/O, no async.
"""

from typing import Any, cast

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
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


def build_filter(filters: list[SearchFilterCondition]) -> Filter | None:
    """Convert contract filter conditions into a Qdrant filter model.

    Args:
        filters: Typed filter conditions.

    Returns:
        Qdrant filter model or ``None`` when no filters are provided.
    """
    if not filters:
        return None

    conditions: list[FieldCondition] = []
    for condition in filters:
        if condition.operator == "eq":
            conditions.append(
                FieldCondition(
                    key=condition.field,
                    match=MatchValue(value=condition.value),
                )
            )
            continue

        if condition.operator == "in":
            values = cast(list[Any], condition.value)
            conditions.append(
                FieldCondition(
                    key=condition.field,
                    match=MatchAny(any=values),
                )
            )
            continue

        if condition.operator == "range":
            range_value = cast(SearchRange, condition.value)
            conditions.append(
                FieldCondition(
                    key=condition.field,
                    range=Range(**range_value.model_dump(exclude_none=True)),
                )
            )

    if not conditions:
        return None  # pragma: no cover
    return Filter(must=cast(Any, conditions))


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
