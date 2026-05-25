"""Unit tests for VectorNormalizer and module-level guard functions."""

import pytest
from qdrant_client.models import SparseVector
from qdrant_db.contracts import SparseVectorData
from qdrant_db.errors import ValidationError
from qdrant_db.normalizer import (
    VectorNormalizer,
    is_float_vector,
    is_sparse_payload,
)

# ---------------------------------------------------------------------------
# Module-level guards
# ---------------------------------------------------------------------------


def test_is_float_vector_returns_true_for_nonempty_list() -> None:
    assert is_float_vector([0.1, 0.2]) is True


def test_is_float_vector_returns_false_for_empty_list() -> None:
    assert is_float_vector([]) is False


def test_is_float_vector_returns_false_for_non_list() -> None:
    assert is_float_vector({"indices": [], "values": []}) is False


def test_is_sparse_payload_returns_true_for_valid_dict() -> None:
    assert is_sparse_payload({"indices": [0, 1], "values": [0.5, 0.8]}) is True


def test_is_sparse_payload_returns_false_for_non_dict() -> None:
    assert is_sparse_payload([0, 1]) is False


def test_is_sparse_payload_returns_false_when_indices_missing() -> None:
    assert is_sparse_payload({"values": [0.5]}) is False


def test_is_sparse_payload_returns_false_when_values_not_list() -> None:
    assert is_sparse_payload({"indices": [0], "values": "bad"}) is False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _normalizer(
    sparse_names: set[str] | None = None,
    multivector: set[str] | None = None,
    vector_names: dict[str, int] | None = None,
) -> VectorNormalizer:
    return VectorNormalizer(
        sparse_vector_names=sparse_names or set(),
        multivector_vectors=multivector or set(),
        vector_names=vector_names or {"text_vector": 4, "image_vector": 3},
    )


SPARSE_DICT = {"indices": [0, 2], "values": [0.9, 0.4]}


# ---------------------------------------------------------------------------
# to_sparse_vector_data
# ---------------------------------------------------------------------------


def test_to_sparse_vector_data_passthrough_already_model() -> None:
    norm = _normalizer()
    model = SparseVectorData(indices=[0], values=[1.0])
    assert norm.to_sparse_vector_data(model, "s") is model


def test_to_sparse_vector_data_coerces_dict() -> None:
    norm = _normalizer()
    result = norm.to_sparse_vector_data(SPARSE_DICT, "s")
    assert isinstance(result, SparseVectorData)
    assert result.indices == [0, 2]
    assert result.values == [0.9, 0.4]


def test_to_sparse_vector_data_raises_for_non_sparse_shape() -> None:
    norm = _normalizer()
    with pytest.raises(
        ValidationError, match="must be an object with indices and values"
    ):
        norm.to_sparse_vector_data([0.1, 0.2], "s")


def test_to_sparse_vector_data_raises_for_invalid_model() -> None:
    # indices is a list (passes is_sparse_payload) but contains non-int values
    # so SparseVectorData.model_validate fails, hitting the except branch.
    norm = _normalizer()
    with pytest.raises(ValidationError, match="Invalid sparse vector payload"):
        norm.to_sparse_vector_data(
            {"indices": ["not_int", "also_not"], "values": [0.1, 0.2]}, "s"
        )


# ---------------------------------------------------------------------------
# normalize_vector_payload
# ---------------------------------------------------------------------------


def test_normalize_sparse_vector_returns_qdrant_sparse() -> None:
    norm = _normalizer(sparse_names={"sparse_vector"})
    result = norm.normalize_vector_payload("sparse_vector", SPARSE_DICT)
    assert isinstance(result, SparseVector)
    assert result.indices == [0, 2]
    assert result.values == [0.9, 0.4]


def test_normalize_dense_vector_passthrough() -> None:
    norm = _normalizer()
    vec = [0.1, 0.2, 0.3, 0.4]
    result = norm.normalize_vector_payload("text_vector", vec)
    assert result is vec


def test_sparse_payload_to_dense_vector_raises() -> None:
    norm = _normalizer()
    with pytest.raises(ValidationError, match="not configured as sparse"):
        norm.normalize_vector_payload("text_vector", SPARSE_DICT)


# ---------------------------------------------------------------------------
# validate_payload_update
# ---------------------------------------------------------------------------


def test_validate_payload_update_raises_for_non_dict() -> None:
    norm = _normalizer()
    with pytest.raises(ValidationError, match="must be a dictionary"):
        norm.validate_payload_update("bad")  # type: ignore[arg-type]


def test_validate_payload_update_raises_for_empty_dict() -> None:
    norm = _normalizer()
    with pytest.raises(ValidationError, match="must not be empty"):
        norm.validate_payload_update({})


def test_validate_payload_update_accepts_nonempty_dict() -> None:
    norm = _normalizer()
    norm.validate_payload_update({"title": "Naruto"})  # no exception
