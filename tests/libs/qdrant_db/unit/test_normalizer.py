"""Unit tests for VectorNormalizer and module-level guard functions."""

import pytest
from qdrant_client.models import SparseVector
from qdrant_db.contracts import SparseVectorData
from qdrant_db.errors import ValidationError
from qdrant_db.normalizer import (
    VectorNormalizer,
    is_sparse_payload,
)

# ---------------------------------------------------------------------------
# Module-level guards
# ---------------------------------------------------------------------------


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
    with pytest.raises(ValidationError, match="Invalid sparse vector payload for s"):
        norm.to_sparse_vector_data([0.1, 0.2], "s")


def test_to_sparse_vector_data_raises_for_invalid_model() -> None:
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


@pytest.mark.parametrize(
    ("vector_name", "vec"),
    [
        ("text_vector", [0.1, 0.2, 0.3, 0.4]),
        ("image_vector", [[0.1, 0.2, 0.3], [1, 0, -1]]),
    ],
)
def test_normalize_dense_vector_passthrough(vector_name: str, vec: list) -> None:
    norm = _normalizer(multivector={"image_vector"})
    result = norm.normalize_vector_payload(vector_name, vec)
    assert result is vec


@pytest.mark.parametrize(
    ("vector_name", "vector_data", "error"),
    [
        ("text_vector", SPARSE_DICT, "not configured as sparse"),
        ("text_vector", (0.1, 0.2, 0.3, 0.4), "must be a list of finite numbers"),
        ("text_vector", [0.1, 0.2, 0.3], "dimension mismatch"),
        ("text_vector", [None, 0.2, 0.3, 0.4], "finite numbers"),
        ("text_vector", ["invalid", 0.2, 0.3, 0.4], "finite numbers"),
        ("text_vector", [True, 0.2, 0.3, 0.4], "finite numbers"),
        ("text_vector", [float("nan"), 0.2, 0.3, 0.4], "finite numbers"),
        ("text_vector", [float("-inf"), 0.2, 0.3, 0.4], "finite numbers"),
        ("text_vector", [10**400, 0.2, 0.3, 0.4], "finite numbers"),
        ("image_vector", [[0.1, 0.2, 0.3], "row"], "rows of finite numbers"),
        ("image_vector", [[0.1, 0.2, 0.3], [0.1, None, 0.3]], "rows of finite numbers"),
        ("image_vector", [], "non-empty list of rows"),
        ("image_vector", [0.1, 0.2, 0.3], "rows of finite numbers"),
    ],
)
def test_invalid_dense_payload_raises(
    vector_name: str, vector_data: object, error: str
) -> None:
    norm = _normalizer(multivector={"image_vector"})
    with pytest.raises(ValidationError, match=error):
        norm.normalize_vector_payload(vector_name, vector_data)
