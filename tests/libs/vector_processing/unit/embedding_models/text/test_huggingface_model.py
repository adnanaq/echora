import builtins
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vector_processing.embedding_models.text.huggingface_model import HuggingFaceModel


@pytest.fixture
def loaded_parts():
    model = MagicMock()
    model.config.hidden_size = 768
    tokenizer = MagicMock()
    tokenizer.model_max_length = 1024
    with (
        patch("transformers.AutoModel.from_pretrained", return_value=model),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
    ):
        yield model, tokenizer


def test_loads_model_and_tokenizer(loaded_parts):
    model = HuggingFaceModel("some/model")

    assert model.embedding_size == 768
    assert model.max_length == 512


def test_missing_tokenizer_fails_at_load(loaded_parts):
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=None):
        with pytest.raises(ValueError, match="some/model"):
            HuggingFaceModel("some/model")


def test_import_error_when_transformers_missing():
    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "transformers":
            raise ImportError("no transformers")
        return real_import(name, *args, **kwargs)

    with (
        patch.dict(sys.modules, {"transformers": None}),
        patch("builtins.__import__", side_effect=failing_import),
        pytest.raises(ImportError, match="HuggingFace dependencies missing"),
    ):
        HuggingFaceModel("some/model")


def test_properties_and_encode_success_and_error(loaded_parts):
    raw_model, tokenizer = loaded_parts
    batch_encoding = MagicMock()
    batch_encoding.to.return_value = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.tensor([[1, 0]]),
    }
    tokenizer.return_value = batch_encoding
    raw_model.return_value = SimpleNamespace(
        last_hidden_state=torch.tensor([[[3.0, 4.0], [100.0, 100.0]]])
    )

    model = HuggingFaceModel("BAAI/bge-m3")
    assert model.model_name == "BAAI/bge-m3"
    assert model.supports_multilingual is True
    assert model.encode(["hello"]) == [pytest.approx([0.6, 0.8])]

    raw_model.side_effect = RuntimeError("forward failed")
    with pytest.raises(RuntimeError, match="forward failed"):
        model.encode(["hello"])
