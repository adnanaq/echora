"""Unit tests for TextProcessor.

Tests cover all code paths including initialization, encoding,
batch processing, and edge cases.
"""

from unittest.mock import MagicMock, patch

import pytest
from vector_processing.processors.text_processor import TextProcessor

# Fixtures mock_text_model and mock_settings are provided by conftest.py


class TestTextProcessorInit:
    """Tests for TextProcessor initialization."""

    def test_init_with_settings(self, mock_text_model, mock_settings):
        """Test initialization with provided settings."""
        processor = TextProcessor(model=mock_text_model, config=mock_settings)

        assert processor.model == mock_text_model
        assert processor.config == mock_settings

    def test_init_without_settings_uses_defaults(self, mock_text_model):
        """Test initialization without config uses default EmbeddingConfig."""
        with patch(
            "vector_processing.processors.text_processor.EmbeddingConfig"
        ) as mock_config_class:
            mock_default_config = MagicMock()
            mock_default_config.embed_max_concurrency = 2
            mock_config_class.return_value = mock_default_config

            processor = TextProcessor(model=mock_text_model)

            mock_config_class.assert_called_once()
            assert processor.config == mock_default_config

    def test_init_logs_model_name(self, mock_text_model, mock_settings, caplog):
        """Test that initialization logs the model name."""
        with caplog.at_level("INFO"):
            TextProcessor(model=mock_text_model, config=mock_settings)

        assert "Initialized TextProcessor with model: test-text-model" in caplog.text


class TestEncodeText:
    """Tests for encode_text method."""

    @pytest.mark.asyncio
    async def test_encode_text_success(self, mock_text_model, mock_settings):
        """Test successful text encoding."""
        processor = TextProcessor(model=mock_text_model, config=mock_settings)

        result = await processor.encode_text("Hello world")

        assert result == [0.1] * 1024
        mock_text_model.encode.assert_called_once_with(["Hello world"])

    @pytest.mark.asyncio
    async def test_encode_text_empty_string_returns_zero_embedding(
        self, mock_text_model, mock_settings
    ):
        """Test empty string returns zero embedding."""
        processor = TextProcessor(model=mock_text_model, config=mock_settings)

        result = await processor.encode_text("")

        assert result == [0.0] * 1024
        mock_text_model.encode.assert_not_called()

    @pytest.mark.asyncio
    async def test_encode_text_whitespace_only_returns_zero_embedding(
        self, mock_text_model, mock_settings
    ):
        """Test whitespace-only string returns zero embedding."""
        processor = TextProcessor(model=mock_text_model, config=mock_settings)

        result = await processor.encode_text("   \t\n  ")

        assert result == [0.0] * 1024
        mock_text_model.encode.assert_not_called()

    @pytest.mark.asyncio
    async def test_encode_text_model_returns_empty_list(
        self, mock_text_model, mock_settings
    ):
        """Test when model returns empty list."""
        mock_text_model.encode.return_value = []
        processor = TextProcessor(model=mock_text_model, config=mock_settings)

        result = await processor.encode_text("Hello")

        assert result is None

    @pytest.mark.asyncio
    async def test_encode_text_model_raises_exception(
        self, mock_text_model, mock_settings, caplog
    ):
        """Test when model raises exception."""
        mock_text_model.encode.side_effect = RuntimeError("Model error")
        processor = TextProcessor(model=mock_text_model, config=mock_settings)

        with caplog.at_level("ERROR"):
            result = await processor.encode_text("Hello")

        assert result is None
        assert "Text encoding failed" in caplog.text


class TestEncodeTextsBatch:
    """Tests for encode_texts_batch method."""

    @pytest.mark.asyncio
    async def test_encode_texts_batch_success(self, mock_text_model, mock_settings):
        """Test successful batch encoding."""
        mock_text_model.encode.return_value = [
            [0.1] * 1024,
            [0.2] * 1024,
            [0.3] * 1024,
        ]
        processor = TextProcessor(model=mock_text_model, config=mock_settings)

        result = await processor.encode_texts_batch(["text1", "text2", "text3"])

        assert len(result) == 3
        assert result[0] == [0.1] * 1024
        assert result[1] == [0.2] * 1024
        assert result[2] == [0.3] * 1024
        mock_text_model.encode.assert_called_once_with(["text1", "text2", "text3"])

    @pytest.mark.asyncio
    async def test_encode_texts_batch_empty_list(self, mock_text_model, mock_settings):
        """Test batch encoding with empty list."""
        mock_text_model.encode.return_value = []
        processor = TextProcessor(model=mock_text_model, config=mock_settings)

        result = await processor.encode_texts_batch([])

        assert result == []

    @pytest.mark.asyncio
    async def test_encode_texts_batch_model_raises_exception(
        self, mock_text_model, mock_settings, caplog
    ):
        """Test when model raises exception during batch encoding."""
        mock_text_model.encode.side_effect = RuntimeError("Batch error")
        processor = TextProcessor(model=mock_text_model, config=mock_settings)

        with caplog.at_level("ERROR"):
            result = await processor.encode_texts_batch(["text1", "text2"])

        assert result == [None, None]
        assert "Batch text encoding failed" in caplog.text

    @pytest.mark.asyncio
    async def test_encode_texts_batch_with_empty_strings(
        self, mock_text_model, mock_settings
    ):
        """Test batch encoding filters out empty strings and returns zero vectors."""
        # Model should only receive non-empty texts
        mock_text_model.encode.return_value = [
            [0.1] * 1024,  # For "valid text"
        ]
        processor = TextProcessor(model=mock_text_model, config=mock_settings)

        result = await processor.encode_texts_batch(["", "valid text", ""])

        assert len(result) == 3
        assert result[0] == [0.0] * 1024  # Empty string -> zero vector
        assert result[1] == [0.1] * 1024  # Valid text -> encoded
        assert result[2] == [0.0] * 1024  # Empty string -> zero vector
        # Model should only be called with valid text
        mock_text_model.encode.assert_called_once_with(["valid text"])

    @pytest.mark.asyncio
    async def test_encode_texts_batch_with_whitespace_strings(
        self, mock_text_model, mock_settings
    ):
        """Test batch encoding filters out whitespace-only strings."""
        mock_text_model.encode.return_value = [
            [0.2] * 1024,  # For "real content"
        ]
        processor = TextProcessor(model=mock_text_model, config=mock_settings)

        result = await processor.encode_texts_batch(["   \t\n  ", "real content", "  "])

        assert len(result) == 3
        assert result[0] == [0.0] * 1024  # Whitespace -> zero vector
        assert result[1] == [0.2] * 1024  # Valid text -> encoded
        assert result[2] == [0.0] * 1024  # Whitespace -> zero vector
        mock_text_model.encode.assert_called_once_with(["real content"])

    @pytest.mark.asyncio
    async def test_encode_texts_batch_all_empty(self, mock_text_model, mock_settings):
        """Test batch encoding when all inputs are empty/whitespace."""
        processor = TextProcessor(model=mock_text_model, config=mock_settings)

        result = await processor.encode_texts_batch(["", "   ", "\t\n"])

        assert len(result) == 3
        assert all(vec == [0.0] * 1024 for vec in result)
        # Model should not be called at all
        mock_text_model.encode.assert_not_called()

    @pytest.mark.asyncio
    async def test_encode_texts_batch_mixed_content(
        self, mock_text_model, mock_settings
    ):
        """Test batch encoding with realistic mixed content."""
        mock_text_model.encode.return_value = [
            [0.1] * 1024,  # "Action anime"
            [0.2] * 1024,  # "Character development"
            [0.3] * 1024,  # "Epic finale"
        ]
        processor = TextProcessor(model=mock_text_model, config=mock_settings)

        result = await processor.encode_texts_batch(
            [
                "Action anime",
                "",
                "Character development",
                "  \t  ",
                "Epic finale",
            ]
        )

        assert len(result) == 5
        assert result[0] == [0.1] * 1024
        assert result[1] == [0.0] * 1024
        assert result[2] == [0.2] * 1024
        assert result[3] == [0.0] * 1024
        assert result[4] == [0.3] * 1024
        mock_text_model.encode.assert_called_once_with(
            [
                "Action anime",
                "Character development",
                "Epic finale",
            ]
        )

    @pytest.mark.asyncio
    async def test_encode_texts_batch_zero_vectors_are_independent(
        self, mock_text_model, mock_settings
    ):
        """Test that zero vectors are independent copies, not the same object.

        Regression test for issue where all empty inputs shared the same
        zero vector instance, causing mutations to affect all empty embeddings.
        """
        mock_text_model.encode.return_value = [[0.1] * 1024]
        processor = TextProcessor(model=mock_text_model, config=mock_settings)

        result = await processor.encode_texts_batch(["", "valid", "", ""])

        # All empty positions should have zero vectors
        assert result[0] == [0.0] * 1024
        assert result[2] == [0.0] * 1024
        assert result[3] == [0.0] * 1024

        # Critical: They must be DIFFERENT objects
        assert id(result[0]) != id(result[2])
        assert id(result[0]) != id(result[3])
        assert id(result[2]) != id(result[3])

        # Verify mutation isolation (narrow types — these are known zero vectors)
        assert result[0] is not None
        assert result[2] is not None
        assert result[3] is not None
        result[0][0] = 999.0
        assert result[2][0] == 0.0  # Should NOT be affected
        assert result[3][0] == 0.0  # Should NOT be affected

    @pytest.mark.asyncio
    async def test_encode_texts_batch_all_empty_produces_independent_vectors(
        self, mock_text_model, mock_settings
    ):
        """Test all-empty case produces independent zero vectors.

        Regression test for the same issue in the all-empty code path.
        """
        processor = TextProcessor(model=mock_text_model, config=mock_settings)

        result = await processor.encode_texts_batch(["", "", "", ""])

        # All should be zero vectors
        assert all(vec == [0.0] * 1024 for vec in result)

        # Critical: Each must be a different object
        assert id(result[0]) != id(result[1])
        assert id(result[0]) != id(result[2])
        assert id(result[0]) != id(result[3])

        # Verify mutation isolation (narrow types — these are known zero vectors)
        assert result[0] is not None
        assert result[1] is not None
        assert result[2] is not None
        assert result[3] is not None
        result[0][0] = 999.0
        assert result[1][0] == 0.0
        assert result[2][0] == 0.0
        assert result[3][0] == 0.0

    @pytest.mark.asyncio
    async def test_encode_texts_batch_large_batch_correctness(
        self, mock_text_model, mock_settings
    ):
        """Test correctness with large batch (regression test for O(n^2) complexity).

        While we can't easily test performance in a unit test, we verify that
        the set-based lookup produces correct results with large batches.
        """
        # Create large batch with alternating pattern
        size = 1000
        texts = ["valid" if i % 2 == 0 else "" for i in range(size)]

        # Mock should return embeddings for 500 valid texts
        mock_text_model.encode.return_value = [
            [float(i)] * 1024 for i in range(size // 2)
        ]
        processor = TextProcessor(model=mock_text_model, config=mock_settings)

        result = await processor.encode_texts_batch(texts)

        # Verify length
        assert len(result) == size

        # Verify alternating pattern: even indices have embeddings, odd have zeros
        for i in range(size):
            vec = result[i]
            assert vec is not None
            if i % 2 == 0:  # Valid text
                assert vec[0] == float(i // 2)
            else:  # Empty string
                assert vec == [0.0] * 1024


class TestGetZeroEmbedding:
    """Tests for get_zero_embedding method."""

    def test_get_zero_embedding_correct_size(self, mock_text_model, mock_settings):
        """Test zero embedding has correct dimensions."""
        mock_text_model.embedding_size = 512
        processor = TextProcessor(model=mock_text_model, config=mock_settings)

        result = processor.get_zero_embedding()

        assert len(result) == 512
        assert all(x == 0.0 for x in result)

    def test_get_zero_embedding_different_sizes(self, mock_text_model, mock_settings):
        """Test zero embedding works for different model sizes."""
        for size in [256, 768, 1024, 1536]:
            mock_text_model.embedding_size = size
            processor = TextProcessor(model=mock_text_model, config=mock_settings)

            result = processor.get_zero_embedding()

            assert len(result) == size


class TestGetModelInfo:
    """Tests for get_model_info method."""

    def test_get_model_info_returns_model_info(self, mock_text_model, mock_settings):
        """Test get_model_info delegates to model."""
        expected_info = {
            "model_name": "test-model",
            "embedding_size": 1024,
            "provider": "test",
        }
        mock_text_model.get_model_info.return_value = expected_info
        processor = TextProcessor(model=mock_text_model, config=mock_settings)

        result = processor.get_model_info()

        assert result == expected_info
        mock_text_model.get_model_info.assert_called_once()
