"""Handle vector_service search RPC.

This module parses search request filters, executes Qdrant search calls,
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
from google.protobuf import json_format
from vector_proto.v1 import vector_search_pb2

from ..runtime import VectorRuntime

logger = logging.getLogger(__name__)


class InvalidFiltersPayloadError(ValueError):
    """Raised when incoming gRPC filters cannot be represented safely."""

    def __init__(self, message: str = "Invalid filter payload.") -> None:
        super().__init__(message)


def _raise_invalid_filters() -> None:
    """Raise canonical invalid-filters exception."""
    raise InvalidFiltersPayloadError()


def _normalize_struct_numbers(value: Any) -> Any:
    """Normalize numbers parsed from protobuf Struct payloads.

    Protobuf Struct uses floating-point representation for all numeric values.
    This converts whole-number floats (for example `2006.0`) back to integers.
    """
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {k: _normalize_struct_numbers(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_struct_numbers(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_normalize_struct_numbers(v) for v in value)
    return value


def _is_valid_filter_payload(filters: dict[str, Any]) -> bool:
    """Return whether filter payload shape is supported."""
    for key, value in filters.items():
        if value is None:
            continue

        if isinstance(value, dict):
            is_range = any(bound in value for bound in ("gte", "lte", "gt", "lt"))
            if is_range:
                continue
            if "any" in value:
                any_values = value["any"]
                if not isinstance(any_values, list | tuple):
                    return False
                continue
            return False

        if isinstance(value, list | tuple):
            continue

        if isinstance(value, str | int | float | bool):
            continue

        return False
    return True


def _clean_filter_payload(raw_filters: Any) -> dict[str, Any]:
    """Normalize filter payload before forwarding to Qdrant.

    Args:
        raw_filters: Raw parsed JSON object from request filters.

    Returns:
        Filter map supported by qdrant_db search helper.
    """
    if not isinstance(raw_filters, dict):
        return {}
    return raw_filters


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

        filters: dict[str, Any] | None = None
        if request.HasField("filters"):
            invalid_filters = False
            try:
                parsed = json_format.MessageToDict(
                    request.filters, preserving_proto_field_name=True
                )
                normalized = _normalize_struct_numbers(parsed)
                filters = _clean_filter_payload(normalized)
                invalid_filters = not _is_valid_filter_payload(filters)
            except Exception:
                invalid_filters = True

            if invalid_filters:
                _raise_invalid_filters()

        entity_type = (
            request.entity_type.strip() if request.HasField("entity_type") else ""
        )
        raw_limit = request.limit if request.HasField("limit") else 10

        text_embedding: list[float] | None = None
        if query_text:
            text_embedding = await runtime.text_processor.encode_text(query_text)
            if not text_embedding:
                return vector_search_pb2.SearchResponse(
                    error=error(
                        "TEXT_EMBEDDING_FAILED",
                        "Failed to generate text embedding for query_text.",
                        retryable=True,
                    )
                )

        image_embedding: list[float] | None = None
        if has_image:
            image_embedding = await _encode_image_bytes(runtime, request.image)
            if not image_embedding:
                return vector_search_pb2.SearchResponse(
                    error=error(
                        "IMAGE_EMBEDDING_FAILED",
                        "Failed to generate image embedding from request image data.",
                        retryable=True,
                    )
                )

        raw_hits = await runtime.qdrant_client.search(
            text_embedding=text_embedding,
            image_embedding=image_embedding,
            entity_type=entity_type or None,
            limit=_normalize_limit(raw_limit),
            filters=filters,
        )

        data = [
            vector_search_pb2.SearchData(
                id=str(hit.get("id", "")),
                similarity_score=float(
                    hit.get("score", hit.get("similarity_score", 0.0))
                ),
                payload_json=json.dumps(hit.get("payload", hit), ensure_ascii=False),
            )
            for hit in raw_hits
        ]
        return vector_search_pb2.SearchResponse(data=data)
    except InvalidFiltersPayloadError as exc:
        return vector_search_pb2.SearchResponse(
            error=error("INVALID_FILTERS", str(exc), retryable=False)
        )
    except ValueError as exc:
        return vector_search_pb2.SearchResponse(
            error=error("INVALID_IMAGE_INPUT", str(exc), retryable=False)
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Search RPC failed")
        return vector_search_pb2.SearchResponse(
            error=error("SEARCH_FAILED", str(exc), retryable=True)
        )
