from unittest.mock import MagicMock, patch

import pytest
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
