"""Handle vector_service search RPC.

This module validates typed filter conditions, executes Qdrant search calls,
and maps success or failure states into proto response objects.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any

import grpc
from common.grpc.error_details import build_error_details as error
from observability import registry
from opentelemetry import trace
from qdrant_db.contracts import FilterClause, FilterOperator, SearchFilterCondition, SearchRequest, SparseVectorData
from vector_proto.v1 import vector_search_pb2

from ..runtime import VectorRuntime

logger = logging.getLogger(__name__)

# Entity types that carry known cardinality.  Any value outside this set is
# normalised to "unknown" before being used as a metric attribute, preventing
# cardinality explosions if callers pass arbitrary free-text entity types.
_KNOWN_ENTITY_TYPES = frozenset({"anime", "manga", "character", ""})

# Maps proto FilterOperator enum values to qdrant_db contract operator strings.
_OPERATOR_MAP: dict[int, FilterOperator] = {
    vector_search_pb2.FILTER_OPERATOR_EQ: "eq",
    vector_search_pb2.FILTER_OPERATOR_NE: "ne",
    vector_search_pb2.FILTER_OPERATOR_IN: "in",
    vector_search_pb2.FILTER_OPERATOR_NOT_IN: "not_in",
    vector_search_pb2.FILTER_OPERATOR_RANGE: "range",
}

# Maps proto FilterClause enum values to qdrant_db contract clause strings.
# UNSPECIFIED defaults to "must" — matches proto3 zero-value convention.
_CLAUSE_MAP: dict[int, FilterClause] = {
    vector_search_pb2.FILTER_CLAUSE_UNSPECIFIED: "must",
    vector_search_pb2.FILTER_CLAUSE_MUST: "must",
    vector_search_pb2.FILTER_CLAUSE_MUST_NOT: "must_not",
    vector_search_pb2.FILTER_CLAUSE_SHOULD: "should",
}


class InvalidFiltersPayloadError(ValueError):
    """Raised when incoming gRPC filters cannot be represented safely."""

    def __init__(self, message: str = "Invalid filter payload.") -> None:
        super().__init__(message)


def _raise_invalid_filters(message: str = "Invalid filter payload.") -> None:
    """Raise the canonical invalid-filters exception.

    Raises:
        InvalidFiltersPayloadError: Always raised to signal invalid filters.
    """
    raise InvalidFiltersPayloadError(message)


def _proto_value_to_python(v: Any) -> Any:
    """Convert a google.protobuf.Value to a native Python value.

    Handles all Value kinds: bool, number (int/float), string, list, struct.
    Whole-number floats are converted to int to preserve type fidelity across
    the protobuf boundary (protobuf encodes all numbers as float64).

    Args:
        v: A google.protobuf.Value instance.

    Returns:
        Equivalent Python scalar, list, or dict.
    """
    kind = v.WhichOneof("kind")
    if kind == "bool_value":
        return v.bool_value
    if kind == "number_value":
        n = v.number_value
        return int(n) if isinstance(n, float) and n.is_integer() else n
    if kind == "string_value":
        return v.string_value
    if kind == "list_value":
        return [_proto_value_to_python(item) for item in v.list_value.values]
    if kind == "struct_value":
        return {k: _proto_value_to_python(val) for k, val in v.struct_value.fields.items()}
    return None


def _validate_filter_fields(
    proto_conditions: Any,
    allowed_fields: frozenset[str],
) -> None:
    """Reject any condition whose field is not in the indexed payload whitelist.

    Filtering on non-indexed fields silently triggers full collection scans in
    Qdrant. Rejecting them early surfaces misconfigured clients immediately.

    Args:
        proto_conditions: Repeated FilterCondition from the gRPC request.
        allowed_fields: Set of indexed payload field names from QdrantClient.

    Raises:
        InvalidFiltersPayloadError: When a condition references an unknown field.
    """
    for cond in proto_conditions:
        if cond.field not in allowed_fields:
            _raise_invalid_filters(
                f"Field '{cond.field}' is not indexed and cannot be filtered."
            )


def _map_filter_conditions(
    proto_conditions: Any,
) -> list[SearchFilterCondition]:
    """Map typed proto FilterCondition objects to SearchFilterCondition contracts.

    Args:
        proto_conditions: Repeated FilterCondition from the gRPC request.

    Returns:
        List of SearchFilterCondition ready for SearchRequest.

    Raises:
        InvalidFiltersPayloadError: When an operator is UNSPECIFIED.
    """
    conditions: list[SearchFilterCondition] = []
    for proto_cond in proto_conditions:
        operator = _OPERATOR_MAP.get(proto_cond.operator)
        if operator is None:
            _raise_invalid_filters(
                f"Unknown filter operator value: {proto_cond.operator}"
            )
        clause = _CLAUSE_MAP.get(proto_cond.clause, "must")
        value = _proto_value_to_python(proto_cond.value)
        conditions.append(
            SearchFilterCondition(
                field=proto_cond.field,
                operator=operator,  # type: ignore[arg-type]
                value=value,
                clause=clause,
            )
        )
    return conditions


def _normalize_limit(raw_limit: int) -> int:
    """Normalize requested limit for safe query execution.

    Args:
        raw_limit: Requested result limit from gRPC request.

    Returns:
        Limit constrained to a safe range.
    """
    if raw_limit <= 0:
        return 10
    return min(raw_limit, 100)


async def _encode_image_bytes(
    runtime: VectorRuntime, image_bytes: bytes
) -> list[float] | None:
    """Generate image embedding from raw bytes.

    Args:
        runtime: Initialized runtime dependencies.
        image_bytes: Raw image bytes.

    Returns:
        Image embedding vector or None when encoding fails.
    """
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(image_bytes)
            temp_path = tmp.name
        return await runtime.vision_processor.encode_image(temp_path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


async def search(
    runtime: VectorRuntime,
    request: vector_search_pb2.SearchRequest,
    context: grpc.aio.ServicerContext,
) -> vector_search_pb2.SearchResponse:
    """Execute vector search and map output into gRPC response shape.

    Args:
        runtime: Initialized service runtime dependencies.
        request: Search RPC request payload.
        context: gRPC request context.

    Returns:
        Search results or structured error payload.
    """
    del context
    # AioServerInterceptor handles RPC-level tracing, duration, and error metrics.
    # This handler records finer-grained embedding and search-quality signals.
    current_span = trace.get_current_span()
    try:
        query_text = (
            request.query_text.strip() if request.HasField("query_text") else ""
        )
        has_image = bool(request.image)
        if not query_text and not has_image:
            return vector_search_pb2.SearchResponse(
                error=error(
                    "MISSING_QUERY_INPUT",
                    "Provide query_text and/or image.",
                    retryable=False,
                )
            )

        filter_conditions: list[SearchFilterCondition] = []
        if request.filters:
            try:
                _validate_filter_fields(
                    request.filters, runtime.qdrant_client.indexed_fields
                )
                filter_conditions = _map_filter_conditions(request.filters)
            except InvalidFiltersPayloadError:
                raise
            except Exception as exc:
                logger.debug("Filter mapping failed: %s", exc)
                _raise_invalid_filters()

        entity_type = (
            request.entity_type.strip() if request.HasField("entity_type") else ""
        )
        # Normalise entity_type to a bounded set of known values before using
        # it as a metric attribute — prevents cardinality explosion.
        safe_entity_type = (
            entity_type if entity_type in _KNOWN_ENTITY_TYPES else "unknown"
        )
        raw_limit = request.limit if request.HasField("limit") else 10

        current_span.add_event("validation.complete")

        text_embedding: list[float] | None = None
        sparse_embedding: SparseVectorData | None = None
        if query_text:
            (
                text_embedding,
                _sparse_dict,
            ) = await runtime.text_processor.encode_text_with_sparse(query_text)
            if _sparse_dict is not None:
                sparse_embedding = SparseVectorData(**_sparse_dict)
            if not text_embedding:
                return vector_search_pb2.SearchResponse(
                    error=error(
                        "TEXT_EMBEDDING_FAILED",
                        "Failed to generate text embedding for query_text.",
                        retryable=True,
                    )
                )
            current_span.add_event(
                "embedding.text.complete", {"embedding_dim": len(text_embedding)}
            )

        image_embedding: list[float] | None = None
        if has_image:
            try:
                image_embedding = await _encode_image_bytes(runtime, request.image)
            except ValueError as exc:
                return vector_search_pb2.SearchResponse(
                    error=error("INVALID_IMAGE_INPUT", str(exc), retryable=False)
                )
            if not image_embedding:
                return vector_search_pb2.SearchResponse(
                    error=error(
                        "IMAGE_EMBEDDING_FAILED",
                        "Failed to generate image embedding from request image data.",
                        retryable=True,
                    )
                )
            current_span.add_event(
                "embedding.image.complete", {"embedding_dim": len(image_embedding)}
            )

        raw_hits = await runtime.qdrant_client.search(
            SearchRequest(
                text_embedding=text_embedding,
                image_embedding=image_embedding,
                sparse_embedding=sparse_embedding,
                entity_type=entity_type or None,
                limit=_normalize_limit(raw_limit),
                filters=filter_conditions,
            )
        )

        result_count = len(raw_hits)
        registry.SEARCH_RESULTS_COUNT.record(
            result_count, {"entity_type": safe_entity_type}
        )
        if result_count == 0:
            registry.SEARCH_EMPTY_RESULTS.add(1, {"entity_type": safe_entity_type})
        current_span.add_event("search.complete", {"result_count": result_count})

        data = [
            vector_search_pb2.SearchData(
                id=hit.id,
                similarity_score=hit.score,
                payload_json=json.dumps(hit.payload, ensure_ascii=False),
            )
            for hit in raw_hits
        ]
        return vector_search_pb2.SearchResponse(data=data)
    except InvalidFiltersPayloadError as exc:
        return vector_search_pb2.SearchResponse(
            error=error("INVALID_FILTERS", str(exc), retryable=False)
        )
    except Exception:
        logger.exception("Search RPC failed")
        return vector_search_pb2.SearchResponse(
            error=error(
                "SEARCH_FAILED",
                "An internal error occurred.",
                retryable=True,
            )
        )
