from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from vector_processing.embedding_models.text.sentence_transformer_model import (
    SentenceTransformerModel,
)


class TestSentenceTransformerModel:
    @pytest.fixture
    def mock_sentence_transformer(self):
        with patch("sentence_transformers.SentenceTransformer") as mock:
            # Setup mock instance
            mock_instance = MagicMock()
            mock.return_value = mock_instance

            # Setup default behavior
            mock_instance.get_sentence_embedding_dimension.return_value = 384
            mock_instance.max_seq_length = 512

            yield mock_instance

    def test_initialization(self, mock_sentence_transformer):
        """Test model initialization."""
        model = SentenceTransformerModel("test-model")

        assert model.model_name == "test-model"
        assert model.embedding_size == 384
        assert model.max_length == 512
        mock_sentence_transformer.get_sentence_embedding_dimension.assert_called_once()

    def test_encode_success(self, mock_sentence_transformer):
        """Test successful encoding."""
        model = SentenceTransformerModel("test-model")

        # Mock encode return value (numpy array)
        expected_embedding = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_sentence_transformer.encode.return_value = expected_embedding

        texts = ["hello", "world"]
        embeddings = model.encode(texts)

        assert len(embeddings) == 2
        assert embeddings == [[0.1, 0.2], [0.3, 0.4]]
        mock_sentence_transformer.encode.assert_called_with(texts)

    def test_encode_failure(self, mock_sentence_transformer):
        """Test encoding failure handling."""
        model = SentenceTransformerModel("test-model")

        mock_sentence_transformer.encode.side_effect = Exception("Encoding error")

        with pytest.raises(Exception, match="Encoding error"):
            model.encode(["test"])

    def test_multilingual_support(self, mock_sentence_transformer):
        """Test multilingual support detection."""
        # Test multilingual model
        model1 = SentenceTransformerModel(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        assert model1.supports_multilingual is True

        # Test monolingual model
        model2 = SentenceTransformerModel("sentence-transformers/all-MiniLM-L6-v2")
        assert model2.supports_multilingual is False

    def test_import_error(self):
        """Test handling of missing dependency."""
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with pytest.raises(
                ImportError, match="Sentence Transformers dependencies missing"
            ):
                SentenceTransformerModel("test-model")
