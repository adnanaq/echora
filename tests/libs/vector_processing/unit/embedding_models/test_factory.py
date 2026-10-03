from unittest.mock import patch

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
