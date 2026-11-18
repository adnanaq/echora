import httpx
import pytest
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import (
    OverwritePayloadOperation,
    SetPayloadOperation,
    SparseVector,
)
from qdrant_db.errors import DuplicateUpdateError
from qdrant_db.normalizer import VectorNormalizer
from qdrant_db.qdrant_models import (
    BatchPayloadUpdate,
    BatchVectorUpdate,
)
from qdrant_db.write_builder import (
    build_payload_operations,
    deduplicate_updates,
    group_vector_updates,
    is_transient_qdrant_error,
    normalize_document_vectors,
)


def _make_normalizer() -> VectorNormalizer:
    return VectorNormalizer(
        sparse_vector_names={"text_sparse_vector"},
        multivector_vectors={"image_vector"},
        vector_names={"text_vector": 3, "image_vector": 2},
    )


def test_is_transient_qdrant_error_response_handling_exception_returns_true() -> None:
    assert (
        is_transient_qdrant_error(ResponseHandlingException(RuntimeError("bad")))
        is True
    )


def test_is_transient_qdrant_error_connect_timeout_returns_true() -> None:
    assert is_transient_qdrant_error(httpx.ConnectTimeout("timeout")) is True


def test_is_transient_qdrant_error_value_error_returns_false() -> None:
    assert is_transient_qdrant_error(ValueError("invalid")) is False


def test_deduplicate_updates_fail_policy_with_duplicate_raises_duplicate_update_error() -> (
    None
):
    updates = [
        BatchPayloadUpdate(point_id="p1", payload={"year": 2020}),
        BatchPayloadUpdate(point_id="p1", payload={"year": 2021}),
    ]

    with pytest.raises(DuplicateUpdateError, match="p1"):
        deduplicate_updates(
            updates=updates,
            key_fn=lambda update: update.point_id,
            dedup_policy="fail",
            duplicate_message=lambda key: f"Duplicate update for {key}",
        )


def test_normalize_document_vectors_normalizes_dense_and_sparse_vectors() -> None:
    normalized = normalize_document_vectors(
        _make_normalizer(),
        {
            "text_vector": [0.1, 0.2, 0.3],
            "text_sparse_vector": {"indices": [1, 4], "values": [0.5, 0.9]},
        },
    )

    assert normalized["text_vector"] == [0.1, 0.2, 0.3]
    assert isinstance(normalized["text_sparse_vector"], SparseVector)


def test_group_vector_updates_merges_updates_by_point_id() -> None:
    grouped = group_vector_updates(
        _make_normalizer(),
        [
            BatchVectorUpdate(
                point_id="p1",
                vector_name="text_vector",
                vector_data=[0.1, 0.2, 0.3],
            ),
            BatchVectorUpdate(
                point_id="p1",
                vector_name="image_vector",
                vector_data=[[0.4, 0.5]],
            ),
        ],
    )

    assert len(grouped) == 1
    assert grouped[0].id == "p1"
    assert grouped[0].vector == {
        "text_vector": [0.1, 0.2, 0.3],
        "image_vector": [[0.4, 0.5]],
    }


def test_build_payload_operations_merge_and_overwrite_return_expected_models() -> None:
    updates = [BatchPayloadUpdate(point_id="p1", payload={"title": "A"}, key="meta")]

    merge_ops = build_payload_operations(updates, "merge")
    overwrite_ops = build_payload_operations(updates, "overwrite")

    assert isinstance(merge_ops[0], SetPayloadOperation)
    assert merge_ops[0].set_payload.key == "meta"
    assert isinstance(overwrite_ops[0], OverwritePayloadOperation)
    assert overwrite_ops[0].overwrite_payload.key is None
