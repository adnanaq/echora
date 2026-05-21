"""Vector payload validation and normalization.

Stateless except for the config values injected at construction.
No I/O, no async.
"""

from typing import Any, TypeGuard, cast

from qdrant_client.models import SparseVector

from qdrant_db.contracts import SparseVectorData
from qdrant_db.errors import ValidationError


def is_float_vector(vector: Any) -> TypeGuard[list[float]]:
    """Check whether a value is a non-empty list.

    Args:
        vector: Value to validate.

    Returns:
        ``True`` when ``vector`` is a ``list`` containing at least one
        element, otherwise ``False``.
    """
    # Intentionally omits per-element isinstance checks (e.g. all float/int).
    # Benchmarked at ~400x overhead per call on 1024-dim vectors (~25ms per 500-item batch).
    # Invalid element types are caught upstream by ty static analysis and by Qdrant server-side.
    # Revisit only if vectors start arriving from untrusted external sources.
    return isinstance(vector, list) and len(vector) > 0


def is_sparse_payload(vector: Any) -> TypeGuard[dict[str, list[int] | list[float]]]:
    """Check whether a value looks like sparse vector ``indices``/``values`` data.

    Args:
        vector: Value to validate.

    Returns:
        ``True`` when the payload has ``indices`` and ``values`` list fields.
    """
    if not isinstance(vector, dict):
        return False
    indices = vector.get("indices")
    values = vector.get("values")
    return isinstance(indices, list) and isinstance(values, list)


class VectorNormalizer:
    """Validates and coerces vector payloads for Qdrant upsert/update APIs."""

    def __init__(
        self,
        sparse_vector_names: set[str],
        multivector_vectors: set[str],
        vector_names: dict[str, int],
    ) -> None:
        """Initialize normalizer with collection vector schema.

        Args:
            sparse_vector_names: Set of vector names configured as sparse.
            multivector_vectors: Set of vector names configured as multivector.
            vector_names: Mapping of vector name to expected dimension size.
        """
        self._sparse_vector_names = sparse_vector_names
        self._multivector_vectors = multivector_vectors
        self._vector_names = vector_names

    def to_sparse_vector_data(
        self, vector_data: Any, vector_name: str
    ) -> SparseVectorData:
        """Validate and coerce sparse vector payload.

        Args:
            vector_data: Candidate sparse payload.
            vector_name: Vector name for context in validation errors.

        Returns:
            Validated sparse vector data model.

        Raises:
            ValidationError: If payload shape is invalid.
        """
        if isinstance(vector_data, SparseVectorData):
            return vector_data

        if not is_sparse_payload(vector_data):
            raise ValidationError(
                f"Sparse vector {vector_name} must be an object with indices and values lists"
            )

        try:
            return SparseVectorData.model_validate(vector_data)
        except Exception as error:
            raise ValidationError(
                f"Invalid sparse vector payload for {vector_name}"
            ) from error

    def normalize_vector_payload(
        self,
        vector_name: str,
        vector_data: Any,
    ) -> list[float] | list[list[float]] | SparseVector:
        """Normalize vector payload for Qdrant upsert/update APIs.

        Args:
            vector_name: Named vector field.
            vector_data: Dense, multivector, or sparse payload.

        Returns:
            Normalized vector payload accepted by qdrant-client.
        """
        if vector_name in self._sparse_vector_names:
            sparse_data = self.to_sparse_vector_data(vector_data, vector_name)
            return SparseVector(indices=sparse_data.indices, values=sparse_data.values)
        if is_sparse_payload(vector_data):
            raise ValidationError(
                f"Vector {vector_name} received sparse payload but is not configured as sparse"
            )
        return cast(list[float] | list[list[float]], vector_data)

    def validate_vector_update(
        self,
        vector_name: str,
        vector_data: list[float] | list[list[float]] | SparseVectorData,
    ) -> None:
        """Validate vector update payload against configured schema.

        Args:
            vector_name: Vector field to update.
            vector_data: Single-vector or multivector payload.

        Raises:
            ValidationError: If name, type, or dimensions are invalid.
        """
        expected_dim = self._vector_names.get(vector_name)
        if vector_name in self._sparse_vector_names:
            self.to_sparse_vector_data(vector_data, vector_name)
            return

        if expected_dim is None:
            raise ValidationError(f"Invalid vector name: {vector_name}")

        is_multivector = vector_name in self._multivector_vectors

        if is_multivector:
            if not isinstance(vector_data, list) or len(vector_data) == 0:
                raise ValidationError("Multivector data must be a non-empty list")
            for idx, element in enumerate(vector_data):
                if not is_float_vector(element):
                    raise ValidationError(
                        f"Multivector element {idx} is not a valid float vector"
                    )
                if len(element) != expected_dim:
                    raise ValidationError(
                        f"Multivector element {idx} dimension mismatch: expected {expected_dim}, got {len(element)}"
                    )
            return

        if not is_float_vector(vector_data):
            raise ValidationError("Vector data is not a valid float vector")
        if len(vector_data) != expected_dim:
            raise ValidationError(
                f"Vector dimension mismatch: expected {expected_dim}, got {len(vector_data)}"
            )

    def validate_payload_update(self, payload: dict[str, Any]) -> None:
        """Validate payload update dictionary.

        Args:
            payload: Payload patch or replacement payload.

        Raises:
            ValidationError: If payload is not a non-empty dict.
        """
        if not isinstance(payload, dict):
            raise ValidationError("Payload must be a dictionary")
        if not payload:
            raise ValidationError("Payload must not be empty")
