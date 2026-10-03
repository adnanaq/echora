import os
from unittest.mock import patch

import pytest
from common.config.embedding_config import EmbeddingConfig
from common.config.settings import Settings
from pydantic import ValidationError


def test_embedding_config_default_chunk_size_is_256():
    assert EmbeddingConfig().embed_model_chunk_size == 256


def test_settings_chunk_size_environment_variable_sets_embedding_chunk_size():
    with patch.dict(
        os.environ, {"ENVIRONMENT": "development", "EMBED_MODEL_CHUNK_SIZE": "16"}
    ):
        assert Settings().embedding.embed_model_chunk_size == 16


@pytest.mark.parametrize("chunk_size", [0, 1025])
def test_embedding_config_chunk_size_out_of_range_raises_validation_error(chunk_size):
    with pytest.raises(ValidationError):
        EmbeddingConfig(embed_model_chunk_size=chunk_size)


def test_embedding_config_default_token_budget_is_zero():
    assert EmbeddingConfig().embed_model_max_tokens_per_pass == 0


def test_settings_token_budget_environment_variable_sets_embedding_token_budget():
    with patch.dict(
        os.environ,
        {"ENVIRONMENT": "development", "EMBED_MODEL_MAX_TOKENS_PER_PASS": "256"},
    ):
        assert Settings().embedding.embed_model_max_tokens_per_pass == 256


def test_embedding_config_negative_token_budget_raises_validation_error():
    with pytest.raises(ValidationError):
        EmbeddingConfig(embed_model_max_tokens_per_pass=-1)
