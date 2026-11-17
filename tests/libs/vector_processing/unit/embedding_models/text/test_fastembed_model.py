import builtins
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from vector_processing.embedding_models.text.fastembed_model import FastEmbedModel


def test_fastembed_model_init_properties_and_encode():
    fake_embedder = MagicMock()
    fake_embedder.embed.return_value = iter([np.array([0.1, 0.2])])
    fake_cls = MagicMock(return_value=fake_embedder)
    fake_module = SimpleNamespace(TextEmbedding=fake_cls)

    with patch.dict(sys.modules, {"fastembed": fake_module}):
        model = FastEmbedModel("BAAI/bge-m3", cache_dir="/tmp/models", threads=2)

    fake_cls.assert_called_once_with(
        model_name="BAAI/bge-m3", cache_dir="/tmp/models", threads=2
    )
    assert model.embedding_size == 1024
    assert model.model_name == "BAAI/bge-m3"
    assert model.max_length == 512
    assert model.supports_multilingual is True
    assert model.encode(["hello"]) == [[0.1, 0.2]]


def test_fastembed_model_default_size_and_encode_failure():
    fake_embedder = MagicMock()
    fake_embedder.embed.side_effect = RuntimeError("embed failed")
    fake_module = SimpleNamespace(TextEmbedding=MagicMock(return_value=fake_embedder))

    with patch.dict(sys.modules, {"fastembed": fake_module}):
        model = FastEmbedModel("custom/unknown-model")

    assert model.embedding_size == 384
    assert model.supports_multilingual is False
    with pytest.raises(RuntimeError, match="embed failed"):
        model.encode(["hello"])


def test_fastembed_model_import_error():
    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "fastembed":
            raise ImportError("no fastembed")
        return real_import(name, *args, **kwargs)

    with (
        patch.dict(sys.modules, {"fastembed": None}),
        patch("builtins.__import__", side_effect=failing_import),
        pytest.raises(ImportError, match="FastEmbed dependencies missing"),
    ):
        FastEmbedModel("BAAI/bge-m3")
