import pytest
from pydantic import ValidationError
from qdrant_db.qdrant_models import (
    BatchPayloadUpdate,
    BatchVectorUpdate,
    SearchFilterCondition,
    SearchRange,
    SearchRequest,
    SparseVectorPayload,
)


def test_batch_payload_update_item_empty_payload_fails() -> None:
    with pytest.raises(ValidationError):
        BatchPayloadUpdate(point_id="p1", payload={})


def test_batch_payload_update_item_payload_is_kept() -> None:
    assert BatchPayloadUpdate(
        point_id="p1", payload={"title": "One Piece"}
    ).payload == {"title": "One Piece"}


@pytest.mark.parametrize(
    "vector_data",
    [
        [],
        [[]],
        [True, 0.2],
        [float("nan")],
        [float("inf")],
        ["0.1"],
        [[0.1], [float("-inf")]],
        [10**400],
    ],
)
def test_batch_vector_update_item_invalid_dense_vector_fails(
    vector_data: object,
) -> None:
    with pytest.raises(ValidationError):
        BatchVectorUpdate.model_validate(
            {"point_id": "p1", "vector_name": "text_vector", "vector_data": vector_data}
        )


@pytest.mark.parametrize("vector_data", [[0.1, 2], [[0.1, 0.2], [3, 0.4]]])
def test_batch_vector_update_item_finite_numbers_are_kept(vector_data: list) -> None:
    item = BatchVectorUpdate.model_validate(
        {"point_id": "p1", "vector_name": "text_vector", "vector_data": vector_data}
    )

    assert item.vector_data == vector_data


def test_batch_vector_update_item_sparse_vector_is_kept() -> None:
    item = BatchVectorUpdate.model_validate(
        {
            "point_id": "p1",
            "vector_name": "text_sparse_vector",
            "vector_data": {"indices": [1, 5], "values": [0.2, 0.4]},
        }
    )

    assert item.vector_data == SparseVectorPayload(indices=[1, 5], values=[0.2, 0.4])


@pytest.mark.parametrize("field", ["text_embedding", "image_embedding"])
@pytest.mark.parametrize("embedding", [[], [True], [float("nan")]])
def test_search_request_invalid_embedding_fails(field: str, embedding: list) -> None:
    with pytest.raises(ValidationError):
        SearchRequest.model_validate({field: embedding})


def test_search_request_empty_expanded_embedding_fails() -> None:
    with pytest.raises(ValidationError):
        SearchRequest.model_validate(
            {"text_embedding": [0.1], "expanded_text_embeddings": [[]]}
        )


def test_search_request_expanded_embeddings_without_text_embedding_fails() -> None:
    with pytest.raises(ValidationError, match="requires text_embedding"):
        SearchRequest.model_validate(
            {"image_embedding": [0.1], "expanded_text_embeddings": [[0.2]]}
        )


def test_search_request_without_any_embedding_fails() -> None:
    with pytest.raises(ValidationError, match="At least one"):
        SearchRequest.model_validate({"limit": 5})


def test_search_request_valid_embeddings_are_kept() -> None:
    request = SearchRequest.model_validate(
        {
            "text_embedding": [0.1, 0.2],
            "expanded_text_embeddings": [[0.3, 0.4]],
            "image_embedding": [1, 0.5],
        }
    )

    assert (
        request.text_embedding,
        request.expanded_text_embeddings,
        request.image_embedding,
    ) == (
        [0.1, 0.2],
        [[0.3, 0.4]],
        [1.0, 0.5],
    )


@pytest.mark.parametrize(
    ("indices", "values", "message"),
    [
        ([1, 2], [0.5], "indices and values must have the same length"),
        ([1, -5, 10], [0.5, 0.3, 0.2], "indices must be non-negative"),
        ([1, 5, 1], [0.5, 0.3, 0.2], "indices must be unique"),
        ([4294967296], [0.5], "indices must not exceed u32 maximum"),
    ],
)
def test_sparse_vector_data_invalid_indices_raises_validation_error(
    indices: list[int], values: list[float], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        SparseVectorPayload(indices=indices, values=values)


def test_search_filter_condition_invalid_values_raise_validation_error() -> None:
    with pytest.raises(ValidationError, match="must be scalar"):
        SearchFilterCondition(field="title", operator="eq", value=["x"])
    with pytest.raises(ValidationError, match="non-empty list"):
        SearchFilterCondition(field="genre", operator="in", value=[])
    with pytest.raises(ValidationError, match="must be scalar"):
        SearchFilterCondition(field="genre", operator="in", value=[["x"]])
    with pytest.raises(ValueError, match="Unsupported filter operator"):
        SearchFilterCondition.model_construct(
            field="title", operator="bad", value="x"
        ).validate_value()


def test_search_filter_condition_range_model_instance_is_kept() -> None:
    range_value = SearchRange(gte=2000)
    condition = SearchFilterCondition(field="year", operator="range", value=range_value)

    assert condition.value is range_value
