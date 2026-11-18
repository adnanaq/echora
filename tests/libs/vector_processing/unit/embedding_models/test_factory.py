from unittest.mock import patch

import pytest
from common.config.embedding_config import EmbeddingConfig
from vector_processing.embedding_models import factory


def test_create_text_model_flagembedding_provider_passes_configured_chunk_size():
    config = EmbeddingConfig(
        text_embedding_provider="flagembedding", embed_model_chunk_size=16
    )
    with patch.object(
        factory, "FlagEmbeddingModel", autospec=True
    ) as flag_embedding_model:
        factory.EmbeddingModelFactory.create_text_model(config)

    assert flag_embedding_model.call_args.kwargs["chunk_size"] == 16


def test_create_text_model_flagembedding_provider_passes_configured_token_budget():
    config = EmbeddingConfig(
        text_embedding_provider="flagembedding", embed_model_max_tokens_per_pass=256
    )
    with patch.object(
        factory, "FlagEmbeddingModel", autospec=True
    ) as flag_embedding_model:
        factory.EmbeddingModelFactory.create_text_model(config)

    assert flag_embedding_model.call_args.kwargs["max_tokens_per_pass"] == 256


@pytest.mark.parametrize(
    ("provider", "attr"),
    [
        ("fastembed", "FastEmbedModel"),
        ("huggingface", "HuggingFaceModel"),
        ("sentence-transformers", "SentenceTransformerModel"),
    ],
)
def test_create_text_model_supported_providers(provider, attr):
    config = EmbeddingConfig(
        text_embedding_provider=provider,
        text_embedding_model="BAAI/bge-m3",
        model_cache_dir="/tmp/cache",
    )
    with patch.object(factory, attr, autospec=True) as model_cls:
        result = factory.EmbeddingModelFactory.create_text_model(config)

    model_cls.assert_called_once_with("BAAI/bge-m3", cache_dir="/tmp/cache")
    assert result is model_cls.return_value


def test_create_text_model_unsupported_provider_raises():
    config = EmbeddingConfig.model_construct(
        text_embedding_provider="unknown",
        text_embedding_model="m",
        model_cache_dir=None,
    )
    with pytest.raises(ValueError, match="Unsupported text embedding provider"):
        factory.EmbeddingModelFactory.create_text_model(config)


def test_create_vision_model_openclip_and_unsupported():
    config = EmbeddingConfig(
        image_embedding_provider="openclip",
        image_embedding_model="ViT-L-14/laion2b_s32b_b82k",
        model_cache_dir="/tmp/cache",
        image_batch_size=8,
    )
    with patch.object(factory, "OpenClipModel", autospec=True) as clip_cls:
        result = factory.EmbeddingModelFactory.create_vision_model(config)

    clip_cls.assert_called_once_with(
        "ViT-L-14/laion2b_s32b_b82k", cache_dir="/tmp/cache", batch_size=8
    )
    assert result is clip_cls.return_value

    bad_config = EmbeddingConfig.model_construct(
        image_embedding_provider="unknown",
        image_embedding_model="m",
        model_cache_dir=None,
        image_batch_size=8,
    )
    with pytest.raises(ValueError, match="Unsupported vision embedding provider"):
        factory.EmbeddingModelFactory.create_vision_model(bad_config)
