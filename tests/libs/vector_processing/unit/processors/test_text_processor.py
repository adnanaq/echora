"""Unit tests for TextProcessor.

Tests cover all code paths including initialization, encoding,
batch processing, and edge cases.
"""

import pytest
from unittest.mock import MagicMock, patch

from vector_processing.processors.text_processor import TextProcessor

# Fixtures mock_text_model and mock_settings are provided by conftest.py


class TestTextProcessorInit:
    """Tests for TextProcessor initialization."""

    def test_init_with_settings(self, mock_text_model, mock_settings):
        """Test initialization with provided settings."""
        processor = TextProcessor(model=mock_text_model, settings=mock_settings)

        assert processor.model == mock_text_model
        assert processor.settings == mock_settings

    def test_init_without_settings_uses_defaults(self, mock_text_model):
        """Test initialization without settings uses default Settings."""
        with patch(
            "vector_processing.processors.text_processor.Settings"
        ) as mock_settings_class:
            mock_default_settings = MagicMock()
            mock_settings_class.return_value = mock_default_settings

            processor = TextProcessor(model=mock_text_model)

            mock_settings_class.assert_called_once()
            assert processor.settings == mock_default_settings

    def test_init_logs_model_name(self, mock_text_model, mock_settings, caplog):
        """Test that initialization logs the model name."""
        with caplog.at_level("INFO"):
            TextProcessor(model=mock_text_model, settings=mock_settings)

        assert "Initialized TextProcessor with model: test-text-model" in caplog.text


class TestEncodeText:
    """Tests for encode_text method."""

    def test_encode_text_success(self, mock_text_model, mock_settings):
        """Test successful text encoding."""
        processor = TextProcessor(model=mock_text_model, settings=mock_settings)

        result = processor.encode_text("Hello world")

        assert result == [0.1] * 1024
        mock_text_model.encode.assert_called_once_with(["Hello world"])

    def test_encode_text_empty_string_returns_zero_embedding(
        self, mock_text_model, mock_settings
    ):
        """Test empty string returns zero embedding."""
        processor = TextProcessor(model=mock_text_model, settings=mock_settings)

        result = processor.encode_text("")

        assert result == [0.0] * 1024
        mock_text_model.encode.assert_not_called()

    def test_encode_text_whitespace_only_returns_zero_embedding(
        self, mock_text_model, mock_settings
    ):
        """Test whitespace-only string returns zero embedding."""
        processor = TextProcessor(model=mock_text_model, settings=mock_settings)

        result = processor.encode_text("   \t\n  ")

        assert result == [0.0] * 1024
        mock_text_model.encode.assert_not_called()

    def test_encode_text_none_like_empty_returns_zero_embedding(
        self, mock_text_model, mock_settings
    ):
        """Test that falsy empty string returns zero embedding."""
        processor = TextProcessor(model=mock_text_model, settings=mock_settings)

        # Empty string is falsy
        result = processor.encode_text("")

        assert result == [0.0] * 1024

    def test_encode_text_model_returns_empty_list(self, mock_text_model, mock_settings):
        """Test when model returns empty list."""
        mock_text_model.encode.return_value = []
        processor = TextProcessor(model=mock_text_model, settings=mock_settings)

        result = processor.encode_text("Hello")

        assert result is None

    def test_encode_text_model_raises_exception(
        self, mock_text_model, mock_settings, caplog
    ):
        """Test when model raises exception."""
        mock_text_model.encode.side_effect = RuntimeError("Model error")
        processor = TextProcessor(model=mock_text_model, settings=mock_settings)

        with caplog.at_level("ERROR"):
            result = processor.encode_text("Hello")

        assert result is None
        assert "Text encoding failed" in caplog.text


class TestEncodeTextsBatch:
    """Tests for encode_texts_batch method."""

    def test_encode_texts_batch_success(self, mock_text_model, mock_settings):
        """Test successful batch encoding."""
        mock_text_model.encode.return_value = [
            [0.1] * 1024,
            [0.2] * 1024,
            [0.3] * 1024,
        ]
        processor = TextProcessor(model=mock_text_model, settings=mock_settings)

        result = processor.encode_texts_batch(["text1", "text2", "text3"])

        assert len(result) == 3
        assert result[0] == [0.1] * 1024
        assert result[1] == [0.2] * 1024
        assert result[2] == [0.3] * 1024
        mock_text_model.encode.assert_called_once_with(["text1", "text2", "text3"])

    def test_encode_texts_batch_empty_list(self, mock_text_model, mock_settings):
        """Test batch encoding with empty list."""
        mock_text_model.encode.return_value = []
        processor = TextProcessor(model=mock_text_model, settings=mock_settings)

        result = processor.encode_texts_batch([])

        assert result == []

    def test_encode_texts_batch_model_raises_exception(
        self, mock_text_model, mock_settings, caplog
    ):
        """Test when model raises exception during batch encoding."""
        mock_text_model.encode.side_effect = RuntimeError("Batch error")
        processor = TextProcessor(model=mock_text_model, settings=mock_settings)

        with caplog.at_level("ERROR"):
            result = processor.encode_texts_batch(["text1", "text2"])

        assert result == [None, None]
        assert "Batch text encoding failed" in caplog.text


class TestGetZeroEmbedding:
    """Tests for _get_zero_embedding method."""

    def test_get_zero_embedding_correct_size(self, mock_text_model, mock_settings):
        """Test zero embedding has correct dimensions."""
        mock_text_model.embedding_size = 512
        processor = TextProcessor(model=mock_text_model, settings=mock_settings)

        result = processor._get_zero_embedding()

        assert len(result) == 512
        assert all(x == 0.0 for x in result)

    def test_get_zero_embedding_different_sizes(self, mock_text_model, mock_settings):
        """Test zero embedding works for different model sizes."""
        for size in [256, 768, 1024, 1536]:
            mock_text_model.embedding_size = size
            processor = TextProcessor(model=mock_text_model, settings=mock_settings)

            result = processor._get_zero_embedding()

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
        processor = TextProcessor(model=mock_text_model, settings=mock_settings)

        result = processor.get_model_info()

        assert result == expected_info
        mock_text_model.get_model_info.assert_called_once()
