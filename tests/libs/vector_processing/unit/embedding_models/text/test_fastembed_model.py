import builtins
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from vector_processing.embedding_models.text.fastembed_model import FastEmbedModel


def test_fastembed_model_init_properties_and_encode():
    fake_embedder = MagicMock()
    fake_embedder.embedding_size = 1024
    fake_embedder.model.tokenizer.model_max_length = 8192
    fake_embedder.embed.return_value = iter([np.array([0.1, 0.2])])
    fake_cls = MagicMock(return_value=fake_embedder)
    fake_cls.list_supported_models = None
    fake_module = SimpleNamespace(TextEmbedding=fake_cls)

    with patch.dict(sys.modules, {"fastembed": fake_module}):
        model = FastEmbedModel("BAAI/bge-m3", cache_dir="/tmp/models", threads=2)

    fake_cls.assert_called_once_with(
        model_name="BAAI/bge-m3", cache_dir="/tmp/models", threads=2
    )
    assert model.embedding_size == 1024
    assert model.model_name == "BAAI/bge-m3"
    assert model.max_length == 8192
    assert model.supports_multilingual is True
    assert model.encode(["hello"]) == [[0.1, 0.2]]


def test_fastembed_model_registry_metadata_resolves_dim_length_and_multilingual():
    registry = [
        {
            "model": "jinaai/jina-embeddings-v3",
            "dim": 1024,
            "description": "Multi-task unimodal (text), multi-lingual (~100), 8192 input tokens truncation.",
        },
        {
            "model": "BAAI/bge-base-en",
            "dim": 768,
            "description": "Text embeddings, English, 512 input tokens truncation.",
        },
        {
            "model": "snowflake/snowflake-arctic-embed-m-long",
            "dim": 768,
            "description": "Text embeddings, English, 2048 input tokens truncation.",
        },
    ]
    fake_embedder = SimpleNamespace(embed=lambda _texts: iter([]))
    fake_cls = MagicMock(return_value=fake_embedder)
    fake_cls.list_supported_models = lambda: registry
    fake_module = SimpleNamespace(TextEmbedding=fake_cls)

    with patch.dict(sys.modules, {"fastembed": fake_module}):
        jina = FastEmbedModel("jinaai/jina-embeddings-v3")
        bge_base = FastEmbedModel("BAAI/bge-base-en")
        arctic_long = FastEmbedModel("snowflake/snowflake-arctic-embed-m-long")

    assert (jina.embedding_size, jina.max_length, jina.supports_multilingual) == (
        1024,
        8192,
        True,
    )
    assert (
        bge_base.embedding_size,
        bge_base.max_length,
        bge_base.supports_multilingual,
    ) == (768, 512, False)
    assert (
        arctic_long.embedding_size,
        arctic_long.max_length,
        arctic_long.supports_multilingual,
    ) == (768, 2048, False)


def test_fastembed_model_default_size_and_encode_failure():
    fake_embedder = MagicMock()
    fake_embedder.embedding_size = None
    fake_embedder.model.tokenizer.model_max_length = None
    fake_embedder.embed.side_effect = RuntimeError("embed failed")
    fake_cls = MagicMock(return_value=fake_embedder)
    fake_cls.list_supported_models = lambda: None
    fake_module = SimpleNamespace(TextEmbedding=fake_cls)

    with patch.dict(sys.modules, {"fastembed": fake_module}):
        model = FastEmbedModel("custom/unknown-model")
        fake_cls.list_supported_models = lambda: [{"model": "other/model", "dim": 512}]
        unlisted_model = FastEmbedModel("custom/another-unknown")

    assert (unlisted_model.embedding_size, unlisted_model.max_length) == (384, 512)
    assert model.embedding_size == 384
    assert model.max_length == 512
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
