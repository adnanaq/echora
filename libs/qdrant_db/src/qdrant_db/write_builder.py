"""Pure write-operation builders for Qdrant batch document, vector, and payload updates."""

from collections.abc import Callable, Hashable
from typing import Any, TypeVar, cast

from common.utils.transient_errors import is_transient_service_error
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import (
    OverwritePayloadOperation,
    PointVectors,
    SetPayload,
    SetPayloadOperation,
    SparseVector,
)
from qdrant_db.qdrant_models import (
    BatchPayloadUpdate,
    BatchVectorUpdate,
    DeduplicationPolicy,
    PayloadUpdateMode,
)
from qdrant_db.errors import DuplicateUpdateError
from qdrant_db.normalizer import VectorNormalizer
from qdrant_db.utils import DuplicateKeyError, deduplicate_items

UpdateItem = TypeVar("UpdateItem")
UpdateKey = TypeVar("UpdateKey", bound=Hashable)


def is_transient_qdrant_error(error: Exception) -> bool:
    """Check whether a Qdrant client error is worth another attempt.

    The general rules of ``is_transient_service_error`` apply, and
    ``ResponseHandlingException`` is retried as well because Qdrant's client
    raises it when a response arrives broken, which is usually a network fault.

    Args:
        error: The error the Qdrant operation raised.

    Returns:
        True when the error is transient.
    """
    return is_transient_service_error(
        error, transient_error_types=(ResponseHandlingException,)
    )


def deduplicate_updates(
    updates: list[UpdateItem],
    key_fn: Callable[[UpdateItem], UpdateKey],
    dedup_policy: DeduplicationPolicy,
    duplicate_message: Callable[[UpdateKey], str],
) -> tuple[list[UpdateItem], int]:
    """Deduplicate batch update items and translate duplicate key errors.

    Args:
        updates: Incoming batch update items.
        key_fn: Function deriving a hashable deduplication key per item.
        dedup_policy: Duplicate handling strategy.
        duplicate_message: Formatter for ``DuplicateUpdateError`` messages.

    Returns:
        Tuple of ``(deduplicated_updates, duplicates_removed_count)``.

    Raises:
        DuplicateUpdateError: If duplicates exist under ``fail`` policy.
    """
    try:
        return deduplicate_items(
            items=updates,
            key_fn=key_fn,
            dedup_policy=dedup_policy,
        )
    except DuplicateKeyError as error:
        raise DuplicateUpdateError(
            duplicate_message(cast(UpdateKey, error.key))
        ) from error


def normalize_document_vectors(
    normalizer: VectorNormalizer, vectors: dict[str, Any]
) -> dict[str, list[float] | list[list[float]] | SparseVector]:
    """Validate and normalize all named vectors for one document.

    Args:
        normalizer: Configured vector payload normalizer.
        vectors: Mapping from vector name to raw vector payload.

    Returns:
        Normalized vector payloads keyed by vector name.
    """
    return {
        vector_name: normalizer.normalize_vector_payload(
            vector_name=vector_name,
            vector_data=vector_data,
        )
        for vector_name, vector_data in vectors.items()
    }


def group_vector_updates(
    normalizer: VectorNormalizer, updates: list[BatchVectorUpdate]
) -> list[PointVectors]:
    """Group deduplicated vector updates into per-point Qdrant structures.

    Args:
        normalizer: Configured vector payload normalizer.
        updates: Deduplicated vector update items.

    Returns:
        List of ``PointVectors`` grouped by point ID.
    """
    grouped: dict[str, dict[str, list[float] | list[list[float]] | SparseVector]] = {}
    for update in updates:
        grouped.setdefault(update.point_id, {})[update.vector_name] = (
            normalizer.normalize_vector_payload(
                vector_name=update.vector_name,
                vector_data=update.vector_data,
            )
        )
    return [
        PointVectors(id=point_id, vector=cast(dict[str, Any], vectors))
        for point_id, vectors in grouped.items()
    ]


def build_payload_operations(
    updates: list[BatchPayloadUpdate], mode: PayloadUpdateMode
) -> list[SetPayloadOperation | OverwritePayloadOperation]:
    """Convert deduplicated payload updates into Qdrant batch operations.

    Args:
        updates: Deduplicated payload update items.
        mode: ``merge`` or ``overwrite`` update mode.

    Returns:
        Ordered Qdrant payload update operations.
    """
    operations: list[SetPayloadOperation | OverwritePayloadOperation] = []
    for update in updates:
        payload_update = SetPayload(
            payload=update.payload,
            points=[update.point_id],
            key=update.key if mode == "merge" else None,
        )
        if mode == "merge":
            operations.append(SetPayloadOperation(set_payload=payload_update))
        else:
            operations.append(
                OverwritePayloadOperation(overwrite_payload=payload_update)
            )
    return operations
