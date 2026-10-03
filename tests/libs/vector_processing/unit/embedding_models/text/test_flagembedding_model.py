from unittest.mock import create_autospec, patch

import pytest
import torch
from FlagEmbedding import BGEM3FlagModel
from transformers import BatchEncoding
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

    def pad(
        self, encoded: dict[str, list[list[int]]], **kwargs: object
    ) -> BatchEncoding:
        width = max(len(ids) for ids in encoded["input_ids"])
        return BatchEncoding(
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


class FakeEncoder(torch.nn.Module):
    def __init__(self, largest_batch: int = 1_000) -> None:
        super().__init__()
        self.largest_batch = largest_batch
        self.calls: list[dict[str, object]] = []

    def forward(
        self, inputs: BatchEncoding, **options: object
    ) -> dict[str, torch.Tensor]:
        input_ids = inputs["input_ids"]
        self.calls.append(
            {"batch": len(input_ids), "width": input_ids.shape[1], **options}
        )
        if len(input_ids) > self.largest_batch:
            raise torch.OutOfMemoryError("fake")
        token_counts = inputs["attention_mask"].sum(dim=1).float()
        positions = torch.arange(input_ids.shape[1]).float()
        return {
            "dense_vecs": torch.stack([token_counts, input_ids[:, 1].float()], dim=1),
            "sparse_vecs": (input_ids.float() / 100 + positions / 1000).unsqueeze(-1),
        }


def _model_with(
    encoder: FakeEncoder, chunk_size: int = 256, max_tokens_per_pass: int = 0
) -> FlagEmbeddingModel:
    bge_m3 = create_autospec(BGEM3FlagModel, instance=True)
    bge_m3.target_devices = ["cpu"]
    bge_m3.truncate_dim = None
    bge_m3.model = encoder
    bge_m3.tokenizer = FakeTokenizer()
    bge_m3._convert_to_numpy = lambda tensor, device=None: tensor.numpy()
    with patch("FlagEmbedding.BGEM3FlagModel", autospec=True, return_value=bge_m3):
        return FlagEmbeddingModel(
            "BAAI/bge-m3",
            chunk_size=chunk_size,
            max_tokens_per_pass=max_tokens_per_pass,
        )


def _load_with(device: str, module: torch.nn.Module | None) -> None:
    bge_m3 = create_autospec(BGEM3FlagModel, instance=True)
    bge_m3.target_devices = [device]
    bge_m3.model = module
    bge_m3.tokenizer = FakeTokenizer()
    with patch("FlagEmbedding.BGEM3FlagModel", autospec=True, return_value=bge_m3):
        FlagEmbeddingModel("BAAI/bge-m3")


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_flag_embedding_model_init_moves_model_to_its_device_once(device):
    module = create_autospec(torch.nn.Module, instance=True)
    _load_with(device, module)

    module.to.assert_called_once_with(device)
    module.eval.assert_called_once_with()


def test_flag_embedding_model_init_cpu_device_uses_full_precision():
    module = create_autospec(torch.nn.Module, instance=True)
    _load_with("cpu", module)

    module.float.assert_called_once_with()


def test_flag_embedding_model_init_missing_model_raises_runtime_error():
    with pytest.raises(RuntimeError, match="BAAI/bge-m3"):
        _load_with("cuda:0", None)


def test_flag_embedding_model_init_gpu_device_keeps_precision():
    module = create_autospec(torch.nn.Module, instance=True)
    _load_with("cuda:0", module)

    module.float.assert_not_called()


def test_encode_with_sparse_one_chunk_runs_one_model_pass():
    encoder = FakeEncoder()
    _model_with(encoder).encode_with_sparse(
        ["pirate ship", "mecha", "school romance comedy"]
    )

    assert [call["batch"] for call in encoder.calls] == [3]


def test_encode_with_sparse_returns_vectors_in_input_order():
    texts = ["mecha", "school romance comedy", "pirate ship"]
    dense, _ = _model_with(FakeEncoder()).encode_with_sparse(texts)

    assert [row[0] for row in dense] == [len(text.split()) + 2 for text in texts]


def test_encode_with_sparse_sparse_weights_keep_words_without_special_tokens():
    model = _model_with(FakeEncoder())
    _, sparse = model.encode_with_sparse(["idol idol music"])

    idol, music = 100, 101
    assert sparse[0] == {
        "indices": [idol, music],
        "values": pytest.approx([idol / 100 + 2 / 1000, music / 100 + 3 / 1000]),
    }


def test_encode_with_sparse_out_of_memory_retries_with_smaller_chunks():
    encoder = FakeEncoder(largest_batch=2)
    texts = ["a", "b c", "d e f", "g h i j"]
    dense, sparse = _model_with(encoder, chunk_size=4).encode_with_sparse(texts)

    assert [call["batch"] for call in encoder.calls] == [4, 2, 2]
    assert [row[0] for row in dense] == [3, 4, 5, 6]
    assert len(sparse) == 4


def test_encode_dense_only_skips_sparse_output():
    encoder = FakeEncoder()
    dense = _model_with(encoder).encode(["pirate ship"])

    assert dense == [[4.0, 100.0]]
    assert encoder.calls[0]["return_sparse"] is False


def test_encode_with_sparse_batch_splits_into_length_sorted_chunks():
    encoder = FakeEncoder()
    texts = ["a", "b c d e", "f g", "h i j", "k l m n o"]
    dense, sparse = _model_with(encoder, chunk_size=2).encode_with_sparse(texts)

    assert [call["batch"] for call in encoder.calls] == [2, 2, 1]
    assert [row[0] for row in dense] == [len(text.split()) + 2 for text in texts]
    assert len(sparse) == 5


def test_encode_with_sparse_chunks_pad_only_to_own_longest_text():
    encoder = FakeEncoder()
    _model_with(encoder, chunk_size=2).encode_with_sparse(
        ["a", "b c d e f g", "h", "i j k l m n"]
    )

    assert [call["width"] for call in encoder.calls] == [8, 3]


def test_encode_with_sparse_default_chunk_size_keeps_query_batch_in_one_pass():
    encoder = FakeEncoder()
    bge_m3 = create_autospec(BGEM3FlagModel, instance=True)
    bge_m3.target_devices = ["cpu"]
    bge_m3.truncate_dim = None
    bge_m3.model = encoder
    bge_m3.tokenizer = FakeTokenizer()
    bge_m3._convert_to_numpy = lambda tensor, device=None: tensor.numpy()
    with patch("FlagEmbedding.BGEM3FlagModel", autospec=True, return_value=bge_m3):
        model = FlagEmbeddingModel("BAAI/bge-m3")
    model.encode_with_sparse([f"query {number}" for number in range(64)])

    assert [call["batch"] for call in encoder.calls] == [64]


def test_flag_embedding_model_init_chunk_size_below_one_raises_value_error():
    with pytest.raises(ValueError, match="chunk_size"):
        _model_with(FakeEncoder(), chunk_size=0)


def _pass_shapes(texts: list[str], **limits: int) -> list[tuple[object, object]]:
    encoder = FakeEncoder()
    _model_with(encoder, **limits).encode_with_sparse(texts)
    return [(call["batch"], call["width"]) for call in encoder.calls]


def test_encode_with_sparse_token_budget_keeps_long_texts_apart_from_short():
    texts = ["a b c d e f g h", "i j k l m n o p", "q", "r", "s", "t"]

    assert _pass_shapes(texts, max_tokens_per_pass=20) == [(2, 10), (4, 3)]


def test_encode_with_sparse_token_budget_and_text_count_both_end_pass():
    texts = ["a", "b", "c", "d", "e"]

    assert _pass_shapes(texts, chunk_size=3, max_tokens_per_pass=100) == [
        (3, 3),
        (2, 3),
    ]


def test_encode_with_sparse_text_longer_than_budget_gets_own_pass():
    texts = ["a b c d e f", "g"]

    assert _pass_shapes(texts, max_tokens_per_pass=4) == [(1, 8), (1, 3)]


def test_encode_with_sparse_token_budget_keeps_vectors_in_input_order():
    texts = ["q", "a b c d e f g h", "r", "i j k l m n o p"]
    dense, sparse = _model_with(
        FakeEncoder(), max_tokens_per_pass=20
    ).encode_with_sparse(texts)

    assert [row[0] for row in dense] == [3, 10, 3, 10]
    assert len(sparse) == 4


def test_flag_embedding_model_init_negative_token_budget_raises_value_error():
    with pytest.raises(ValueError, match="max_tokens_per_pass"):
        _model_with(FakeEncoder(), max_tokens_per_pass=-1)
