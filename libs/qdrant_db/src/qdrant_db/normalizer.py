"""Vector payload validation and normalization.

Stateless except for the config values injected at construction.
No I/O, no async.
"""

from typing import Any, TypeGuard, cast

import pydantic
from qdrant_client.models import SparseVector
from qdrant_db.errors import ValidationError
from qdrant_db.qdrant_models import (
    DenseVector,
    MultiVector,
    SparseVectorPayload,
    Uint8DenseVector,
    Uint8MultiVector,
)

_DENSE_VECTOR = pydantic.TypeAdapter(DenseVector)
_MULTI_VECTOR = pydantic.TypeAdapter(MultiVector)
_UINT8_DENSE_VECTOR = pydantic.TypeAdapter(Uint8DenseVector)
_UINT8_MULTI_VECTOR = pydantic.TypeAdapter(Uint8MultiVector)


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
        vector_datatype: str | None = None,
    ) -> None:
        """Initialize normalizer with collection vector schema.

        Args:
            sparse_vector_names: Set of vector names configured as sparse.
            multivector_vectors: Set of vector names configured as multivector.
            vector_names: Mapping of vector name to expected dimension size.
            vector_datatype: Configured Qdrant storage datatype for dense vectors.
        """
        self._sparse_vector_names = sparse_vector_names
        self._multivector_vectors = multivector_vectors
        self._vector_names = vector_names
        self._vector_datatype = vector_datatype

    def to_sparse_vector_data(
        self, vector_data: Any, vector_name: str
    ) -> SparseVectorPayload:
        """Validate and coerce sparse vector payload.

        Args:
            vector_data: Candidate sparse payload.
            vector_name: Vector name for context in validation errors.

        Returns:
            Validated sparse vector data model.

        Raises:
            ValidationError: If payload shape is invalid.
        """
        if isinstance(vector_data, SparseVectorPayload):
            return vector_data
        try:
            return SparseVectorPayload.model_validate(vector_data)
        except pydantic.ValidationError as error:
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

        Raises:
            ValidationError: If the name is not in the collection schema, the
                payload shape is wrong, or a dense vector has the wrong size.
        """
        if vector_name in self._sparse_vector_names:
            sparse_data = self.to_sparse_vector_data(vector_data, vector_name)
            return SparseVector(indices=sparse_data.indices, values=sparse_data.values)
        if vector_name not in self._vector_names:
            raise ValidationError(f"Invalid vector name: {vector_name}")
        if is_sparse_payload(vector_data):
            raise ValidationError(
                f"Vector {vector_name} received sparse payload but is not configured as sparse"
            )
        self._validate_dense_dimension(vector_name, vector_data)
        return cast(list[float] | list[list[float]], vector_data)

    def _validate_dense_dimension(self, vector_name: str, vector_data: Any) -> None:
        """Check a dense or multivector payload against the configured size.

        The payload must match its contract type, ``DenseVector`` or, for a
        name configured as a multivector, ``MultiVector``; each row must then
        have the configured dimension. qdrant-client rejects the rest with its
        own error, and Qdrant rejects NaN or an empty multivector only once the
        write is sent.

        Args:
            vector_name: Named vector field.
            vector_data: Dense list, or list of dense lists for a multivector.

        Raises:
            ValidationError: If the payload does not match its contract type,
                or a row has a size other than the configured dimension.
        """
        is_multivector = vector_name in self._multivector_vectors
        is_uint8 = self._vector_datatype == "uint8"
        if is_uint8:
            contract = _UINT8_MULTI_VECTOR if is_multivector else _UINT8_DENSE_VECTOR
        else:
            contract = _MULTI_VECTOR if is_multivector else _DENSE_VECTOR
        try:
            contract.validate_python(vector_data)
        except pydantic.ValidationError as error:
            element_desc = "integers in [0, 255]" if is_uint8 else "finite numbers"
            shape = (
                f"a non-empty list of rows of {element_desc}"
                if is_multivector
                else f"a list of {element_desc}"
            )
            raise ValidationError(f"Vector {vector_name} must be {shape}") from error
        expected = self._vector_names[vector_name]
        for row in vector_data if is_multivector else [vector_data]:
            if len(row) != expected:
                raise ValidationError(
                    f"Vector {vector_name} dimension mismatch: "
                    f"expected {expected}, got {len(row)}"
                )
