from unittest.mock import MagicMock, patch

import pytest
import torch
from vector_processing.embedding_models.text.flagembedding_model import (
    FlagEmbeddingModel,
)

SPECIAL_TOKEN_IDS = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}


class FakeTokenizer:
    special_tokens_map = {
        "cls_token": "<s>",
        "pad_token": "<pad>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
    }

    def __init__(self) -> None:
        self.word_ids: dict[str, int] = {}

    def convert_tokens_to_ids(self, token: str) -> int:
        return SPECIAL_TOKEN_IDS[token]

    def __call__(
        self, texts: list[str], **kwargs: object
    ) -> dict[str, list[list[int]]]:
        input_ids = [
            [
                0,
                *(
                    self.word_ids.setdefault(word, 100 + len(self.word_ids))
                    for word in text.split()
                ),
                2,
            ]
            for text in texts
        ]
        return {
            "input_ids": input_ids,
            "attention_mask": [[1] * len(ids) for ids in input_ids],
        }

    def pad(self, encoded: dict[str, list[list[int]]], **kwargs: object) -> PaddedBatch:
        width = max(len(ids) for ids in encoded["input_ids"])
        return PaddedBatch(
            {
                "input_ids": torch.tensor(
                    [ids + [1] * (width - len(ids)) for ids in encoded["input_ids"]]
                ),
                "attention_mask": torch.tensor(
                    [
                        mask + [0] * (width - len(mask))
                        for mask in encoded["attention_mask"]
                    ]
                ),
            }
        )


class PaddedBatch(dict):
    def to(self, device: str) -> PaddedBatch:
        return self


class FakeEncoder:
    def __init__(self, largest_batch: int = 1_000) -> None:
        self.largest_batch = largest_batch
        self.calls: list[dict[str, object]] = []

    def __call__(
        self, inputs: PaddedBatch, **options: object
    ) -> dict[str, torch.Tensor]:
        input_ids = inputs["input_ids"]
        self.calls.append({"batch": len(input_ids), **options})
        if len(input_ids) > self.largest_batch:
            raise torch.OutOfMemoryError("fake")
        token_counts = inputs["attention_mask"].sum(dim=1).float()
        positions = torch.arange(input_ids.shape[1]).float()
        return {
            "dense_vecs": torch.stack([token_counts, input_ids[:, 1].float()], dim=1),
            "sparse_vecs": (input_ids.float() / 100 + positions / 1000).unsqueeze(-1),
        }

    def to(self, device: str) -> None:
        pass

    def eval(self) -> None:
        pass

    def float(self) -> None:
        pass


def _model_with(
    encoder: FakeEncoder, chunk_size: int = 256, max_tokens_per_pass: int = 0
) -> FlagEmbeddingModel:
    bge_m3 = MagicMock()
    bge_m3.target_devices = ["cpu"]
    bge_m3.truncate_dim = None
    bge_m3.model = encoder
    bge_m3.tokenizer = FakeTokenizer()
    bge_m3._convert_to_numpy = lambda tensor, device=None: tensor.numpy()
    with patch("FlagEmbedding.BGEM3FlagModel", return_value=bge_m3):
        return FlagEmbeddingModel(
            "BAAI/bge-m3",
            chunk_size=chunk_size,
            max_tokens_per_pass=max_tokens_per_pass,
        )


def _loaded_bge_m3(device: str) -> MagicMock:
    bge_m3 = MagicMock()
    bge_m3.target_devices = [device]
    return bge_m3


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_moves_model_to_its_device_once_at_load(device):
    bge_m3 = _loaded_bge_m3(device)
    with patch("FlagEmbedding.BGEM3FlagModel", return_value=bge_m3):
        FlagEmbeddingModel("BAAI/bge-m3")

    bge_m3.model.to.assert_called_once_with(device)
    bge_m3.model.eval.assert_called_once_with()


def test_cpu_model_uses_full_precision():
    bge_m3 = _loaded_bge_m3("cpu")
    with patch("FlagEmbedding.BGEM3FlagModel", return_value=bge_m3):
        FlagEmbeddingModel("BAAI/bge-m3")

    bge_m3.model.float.assert_called_once_with()


def test_missing_model_fails_at_load():
    bge_m3 = _loaded_bge_m3("cuda:0")
    bge_m3.model = None
    with (
        patch("FlagEmbedding.BGEM3FlagModel", return_value=bge_m3),
        pytest.raises(RuntimeError, match="BAAI/bge-m3"),
    ):
        FlagEmbeddingModel("BAAI/bge-m3")


def test_gpu_model_keeps_its_precision():
    bge_m3 = _loaded_bge_m3("cuda:0")
    with patch("FlagEmbedding.BGEM3FlagModel", return_value=bge_m3):
        FlagEmbeddingModel("BAAI/bge-m3")

    bge_m3.model.float.assert_not_called()


def test_encodes_a_chunk_with_one_model_pass():
    encoder = FakeEncoder()
    _model_with(encoder).encode_with_sparse(
        ["pirate ship", "mecha", "school romance comedy"]
    )

    assert [call["batch"] for call in encoder.calls] == [3]


def test_returns_vectors_in_input_order():
    texts = ["mecha", "school romance comedy", "pirate ship"]
    dense, _ = _model_with(FakeEncoder()).encode_with_sparse(texts)

    assert [row[0] for row in dense] == [len(text.split()) + 2 for text in texts]


def test_sparse_weights_keep_words_and_drop_special_tokens():
    model = _model_with(FakeEncoder())
    _, sparse = model.encode_with_sparse(["idol idol music"])

    idol, music = 100, 101
    assert sparse[0] == {
        "indices": [idol, music],
        "values": pytest.approx([idol / 100 + 2 / 1000, music / 100 + 3 / 1000]),
    }


def test_out_of_memory_retries_with_smaller_chunks():
    encoder = FakeEncoder(largest_batch=2)
    texts = ["a", "b c", "d e f", "g h i j"]
    dense, sparse = _model_with(encoder, chunk_size=4).encode_with_sparse(texts)

    assert [call["batch"] for call in encoder.calls] == [4, 2, 2]
    assert [row[0] for row in dense] == [3, 4, 5, 6]
    assert len(sparse) == 4


def test_dense_only_encode_skips_sparse_output():
    encoder = FakeEncoder()
    dense = _model_with(encoder).encode(["pirate ship"])

    assert dense == [[4.0, 100.0]]
    assert encoder.calls[0]["return_sparse"] is False


def test_splits_a_batch_into_length_sorted_chunks():
    encoder = FakeEncoder()
    texts = ["a", "b c d e", "f g", "h i j", "k l m n o"]
    dense, sparse = _model_with(encoder, chunk_size=2).encode_with_sparse(texts)

    assert [call["batch"] for call in encoder.calls] == [2, 2, 1]
    assert [row[0] for row in dense] == [len(text.split()) + 2 for text in texts]
    assert len(sparse) == 5


def test_chunks_pad_only_to_their_own_longest_text():
    widths: list[int] = []
    encoder = FakeEncoder()
    original_call = encoder.__call__

    def record_width(inputs: PaddedBatch, **options: object) -> dict[str, torch.Tensor]:
        widths.append(inputs["input_ids"].shape[1])
        return original_call(inputs, **options)

    model = _model_with(encoder, chunk_size=2)
    model._encoder = record_width
    model.encode_with_sparse(["a", "b c d e f g", "h", "i j k l m n"])

    assert widths == [8, 3]


def test_default_chunk_size_keeps_a_query_batch_in_one_pass():
    encoder = FakeEncoder()
    bge_m3 = MagicMock()
    bge_m3.target_devices = ["cpu"]
    bge_m3.truncate_dim = None
    bge_m3.model = encoder
    bge_m3.tokenizer = FakeTokenizer()
    bge_m3._convert_to_numpy = lambda tensor, device=None: tensor.numpy()
    with patch("FlagEmbedding.BGEM3FlagModel", return_value=bge_m3):
        model = FlagEmbeddingModel("BAAI/bge-m3")
    model.encode_with_sparse([f"query {number}" for number in range(64)])

    assert [call["batch"] for call in encoder.calls] == [64]


def test_chunk_size_below_one_is_refused():
    with pytest.raises(ValueError, match="chunk_size"):
        _model_with(FakeEncoder(), chunk_size=0)


def _pass_widths(model: FlagEmbeddingModel, texts: list[str]) -> list[tuple[int, int]]:
    passes: list[tuple[int, int]] = []
    run_encoder = model._encoder

    def record_pass(inputs: PaddedBatch, **options: object) -> dict[str, torch.Tensor]:
        passes.append(tuple(inputs["input_ids"].shape))
        return run_encoder(inputs, **options)

    model._encoder = record_pass
    model.encode_with_sparse(texts)
    return passes


def test_token_budget_keeps_long_texts_apart_from_short_ones():
    model = _model_with(FakeEncoder(), max_tokens_per_pass=20)
    texts = ["a b c d e f g h", "i j k l m n o p", "q", "r", "s", "t"]

    assert _pass_widths(model, texts) == [(2, 10), (4, 3)]


def test_token_budget_and_text_count_both_end_a_pass():
    model = _model_with(FakeEncoder(), chunk_size=3, max_tokens_per_pass=100)
    texts = ["a", "b", "c", "d", "e"]

    assert _pass_widths(model, texts) == [(3, 3), (2, 3)]


def test_text_longer_than_the_budget_still_gets_its_own_pass():
    model = _model_with(FakeEncoder(), max_tokens_per_pass=4)
    texts = ["a b c d e f", "g"]

    assert _pass_widths(model, texts) == [(1, 8), (1, 3)]


def test_token_budget_keeps_vectors_in_input_order():
    texts = ["q", "a b c d e f g h", "r", "i j k l m n o p"]
    dense, sparse = _model_with(
        FakeEncoder(), max_tokens_per_pass=20
    ).encode_with_sparse(texts)

    assert [row[0] for row in dense] == [3, 10, 3, 10]
    assert len(sparse) == 4


def test_negative_token_budget_is_refused():
    with pytest.raises(ValueError, match="max_tokens_per_pass"):
        _model_with(FakeEncoder(), max_tokens_per_pass=-1)
