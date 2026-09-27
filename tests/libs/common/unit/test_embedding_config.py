import os
from unittest.mock import patch

import pytest
from common.config.embedding_config import EmbeddingConfig
from common.config.settings import Settings
from pydantic import ValidationError


def test_model_chunk_size_default_keeps_a_query_batch_in_one_pass():
    assert EmbeddingConfig().embed_model_chunk_size == 256


def test_model_chunk_size_is_read_from_the_environment():
    with patch.dict(
        os.environ, {"ENVIRONMENT": "development", "EMBED_MODEL_CHUNK_SIZE": "16"}
    ):
        assert Settings().embedding.embed_model_chunk_size == 16


@pytest.mark.parametrize("chunk_size", [0, 1025])
def test_model_chunk_size_outside_its_range_is_refused(chunk_size):
    with pytest.raises(ValidationError):
        EmbeddingConfig(embed_model_chunk_size=chunk_size)


def test_token_budget_per_pass_is_off_by_default():
    assert EmbeddingConfig().embed_model_max_tokens_per_pass == 0


def test_token_budget_per_pass_is_read_from_the_environment():
    with patch.dict(
        os.environ,
        {"ENVIRONMENT": "development", "EMBED_MODEL_MAX_TOKENS_PER_PASS": "256"},
    ):
        assert Settings().embedding.embed_model_max_tokens_per_pass == 256


def test_negative_token_budget_per_pass_is_refused():
    with pytest.raises(ValidationError):
        EmbeddingConfig(embed_model_max_tokens_per_pass=-1)
