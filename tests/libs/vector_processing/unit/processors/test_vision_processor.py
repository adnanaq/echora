"""Unit tests for VisionProcessor.

Tests cover all code paths including initialization, image encoding,
cache management, hashing, and edge cases.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from vector_processing.processors.vision_processor import VisionProcessor

# Fixtures mock_vision_model, mock_downloader, and mock_settings are provided by conftest.py


class TestVisionProcessorInit:
    """Tests for VisionProcessor initialization."""

    def test_init_with_settings(
        self, mock_vision_model, mock_downloader, mock_settings
    ):
        """Test initialization with provided settings."""
        processor = VisionProcessor(
            model=mock_vision_model,
            downloader=mock_downloader,
            settings=mock_settings,
        )

        assert processor.model == mock_vision_model
        assert processor.downloader == mock_downloader
        assert processor.settings == mock_settings

    def test_init_without_settings_uses_defaults(
        self, mock_vision_model, mock_downloader
    ):
        """Test initialization without settings uses default Settings."""
        with patch(
            "vector_processing.processors.vision_processor.Settings"
        ) as mock_settings_class:
            mock_default_settings = MagicMock()
            mock_settings_class.return_value = mock_default_settings

            processor = VisionProcessor(
                model=mock_vision_model, downloader=mock_downloader
            )

            mock_settings_class.assert_called_once()
            assert processor.settings == mock_default_settings

    def test_init_logs_model_name(
        self, mock_vision_model, mock_downloader, mock_settings, caplog
    ):
        """Test that initialization logs the model name."""
        with caplog.at_level("INFO"):
            VisionProcessor(
                model=mock_vision_model,
                downloader=mock_downloader,
                settings=mock_settings,
            )

        assert (
            "Initialized VisionProcessor with model: test-vision-model" in caplog.text
        )


class TestEncodeImage:
    """Tests for encode_image method."""

    def test_encode_image_success(
        self, mock_vision_model, mock_downloader, mock_settings
    ):
        """Test successful image encoding."""
        processor = VisionProcessor(
            model=mock_vision_model,
            downloader=mock_downloader,
            settings=mock_settings,
        )

        result = processor.encode_image("/path/to/image.jpg")

        assert result == [0.2] * 768
        mock_vision_model.encode_image.assert_called_once_with(["/path/to/image.jpg"])

    def test_encode_image_model_returns_empty_list(
        self, mock_vision_model, mock_downloader, mock_settings
    ):
        """Test when model returns empty list."""
        mock_vision_model.encode_image.return_value = []
        processor = VisionProcessor(
            model=mock_vision_model,
            downloader=mock_downloader,
            settings=mock_settings,
        )

        result = processor.encode_image("/path/to/image.jpg")

        assert result is None

    def test_encode_image_model_returns_none(
        self, mock_vision_model, mock_downloader, mock_settings
    ):
        """Test when model returns None."""
        mock_vision_model.encode_image.return_value = None
        processor = VisionProcessor(
            model=mock_vision_model,
            downloader=mock_downloader,
            settings=mock_settings,
        )

        result = processor.encode_image("/path/to/image.jpg")

        assert result is None

    def test_encode_image_model_raises_exception(
        self, mock_vision_model, mock_downloader, mock_settings, caplog
    ):
        """Test when model raises exception."""
        mock_vision_model.encode_image.side_effect = RuntimeError("Encoding error")
        processor = VisionProcessor(
            model=mock_vision_model,
            downloader=mock_downloader,
            settings=mock_settings,
        )

        with caplog.at_level("ERROR"):
            result = processor.encode_image("/path/to/image.jpg")

        assert result is None
        assert "Image encoding failed" in caplog.text


class TestHashEmbedding:
    """Tests for _hash_embedding method."""

    def test_hash_embedding_returns_md5_hash(
        self, mock_vision_model, mock_downloader, mock_settings
    ):
        """Test hash embedding returns consistent MD5 hash."""
        processor = VisionProcessor(
            model=mock_vision_model,
            downloader=mock_downloader,
            settings=mock_settings,
        )

        embedding = [0.1234, 0.5678, 0.9012]
        result = processor._hash_embedding(embedding)

        # Should return a 32-character hex string (MD5)
        assert len(result) == 32
        assert all(c in "0123456789abcdef" for c in result)

    def test_hash_embedding_consistent_for_same_input(
        self, mock_vision_model, mock_downloader, mock_settings
    ):
        """Test same embedding produces same hash."""
        processor = VisionProcessor(
            model=mock_vision_model,
            downloader=mock_downloader,
            settings=mock_settings,
        )

        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        result1 = processor._hash_embedding(embedding)
        result2 = processor._hash_embedding(embedding)

        assert result1 == result2

    def test_hash_embedding_different_for_different_input(
        self, mock_vision_model, mock_downloader, mock_settings
    ):
        """Test different embeddings produce different hashes."""
        processor = VisionProcessor(
            model=mock_vision_model,
            downloader=mock_downloader,
            settings=mock_settings,
        )

        embedding1 = [0.1, 0.2, 0.3]
        embedding2 = [0.4, 0.5, 0.6]

        result1 = processor._hash_embedding(embedding1)
        result2 = processor._hash_embedding(embedding2)

        assert result1 != result2

    def test_hash_embedding_respects_precision(
        self, mock_vision_model, mock_downloader, mock_settings
    ):
        """Test precision parameter affects rounding."""
        processor = VisionProcessor(
            model=mock_vision_model,
            downloader=mock_downloader,
            settings=mock_settings,
        )

        # These differ at 5th decimal place
        embedding1 = [0.12345678]
        embedding2 = [0.12346789]

        # With precision=4, they should hash the same (both round to 0.1235)
        result1 = processor._hash_embedding(embedding1, precision=4)
        result2 = processor._hash_embedding(embedding2, precision=4)

        assert result1 == result2

    def test_hash_embedding_exception_fallback(
        self, mock_vision_model, mock_downloader, mock_settings
    ):
        """Test fallback to hash() when exception occurs."""
        processor = VisionProcessor(
            model=mock_vision_model,
            downloader=mock_downloader,
            settings=mock_settings,
        )

        # Create an object that will cause round() to fail
        class BadFloat:
            def __round__(self, n):
                raise TypeError("Cannot round")  # noqa: TRY003

            def __iter__(self):
                return iter([self])

        # Use a mock to make round fail
        with patch(
            "vector_processing.processors.vision_processor.round",
            side_effect=TypeError("Cannot round"),
        ):
            embedding = [0.1, 0.2, 0.3]
            result = processor._hash_embedding(embedding)

            # Should return string representation of hash
            assert isinstance(result, str)


class TestGetCacheStats:
    """Tests for get_cache_stats method."""

    def test_get_cache_stats_delegates_to_downloader(
        self, mock_vision_model, mock_downloader, mock_settings
    ):
        """Test get_cache_stats delegates to downloader."""
        processor = VisionProcessor(
            model=mock_vision_model,
            downloader=mock_downloader,
            settings=mock_settings,
        )

        result = processor.get_cache_stats()

        assert result == {"cache_size": 100, "hit_rate": 0.85}
        mock_downloader.get_cache_stats.assert_called_once()


class TestClearCache:
    """Tests for clear_cache method."""

    def test_clear_cache_without_max_age(
        self, mock_vision_model, mock_downloader, mock_settings
    ):
        """Test clear_cache without max_age_days parameter."""
        processor = VisionProcessor(
            model=mock_vision_model,
            downloader=mock_downloader,
            settings=mock_settings,
        )

        result = processor.clear_cache()

        assert result == {"cleared": 50, "remaining": 50}
        mock_downloader.clear_cache.assert_called_once_with(None)

    def test_clear_cache_with_max_age(
        self, mock_vision_model, mock_downloader, mock_settings
    ):
        """Test clear_cache with max_age_days parameter."""
        processor = VisionProcessor(
            model=mock_vision_model,
            downloader=mock_downloader,
            settings=mock_settings,
        )

        result = processor.clear_cache(max_age_days=7)
        assert result == {"cleared": 50, "remaining": 50}

        mock_downloader.clear_cache.assert_called_once_with(7)


class TestGetSupportedFormats:
    """Tests for get_supported_formats method."""

    def test_get_supported_formats_returns_expected_list(
        self, mock_vision_model, mock_downloader, mock_settings
    ):
        """Test get_supported_formats returns expected formats."""
        processor = VisionProcessor(
            model=mock_vision_model,
            downloader=mock_downloader,
            settings=mock_settings,
        )

        result = processor.get_supported_formats()

        assert result == ["jpg", "jpeg", "png", "bmp", "tiff", "webp", "gif"]

    def test_get_supported_formats_includes_common_formats(
        self, mock_vision_model, mock_downloader, mock_settings
    ):
        """Test all common image formats are supported."""
        processor = VisionProcessor(
            model=mock_vision_model,
            downloader=mock_downloader,
            settings=mock_settings,
        )

        result = processor.get_supported_formats()

        # Check common formats are included
        assert "jpg" in result
        assert "png" in result
        assert "gif" in result
        assert "webp" in result


class TestGetModelInfo:
    """Tests for get_model_info method."""

    def test_get_model_info_returns_model_info(
        self, mock_vision_model, mock_downloader, mock_settings
    ):
        """Test get_model_info delegates to model."""
        expected_info = {
            "model_name": "test-vision-model",
            "embedding_size": 768,
            "provider": "openclip",
        }
        mock_vision_model.get_model_info.return_value = expected_info
        processor = VisionProcessor(
            model=mock_vision_model,
            downloader=mock_downloader,
            settings=mock_settings,
        )

        result = processor.get_model_info()

        assert result == expected_info
        mock_vision_model.get_model_info.assert_called_once()


class TestEncodeImagesBatch:
    """Tests for encode_images_batch async method."""

    @pytest.mark.asyncio
    async def test_encode_images_batch_returns_matrix(
        self, mock_vision_model, mock_settings
    ):
        """Test that encoding multiple images returns a matrix (list of lists)."""
        mock_downloader = AsyncMock()
        mock_downloader.download_and_cache_image.side_effect = [
            "/cache/image1.jpg",
            "/cache/image2.jpg",
        ]

        mock_vision_model.encode_image.side_effect = [
            [[0.1] * 768],
            [[0.2] * 768],
        ]

        processor = VisionProcessor(
            model=mock_vision_model,
            downloader=mock_downloader,
            settings=mock_settings,
        )

        urls = ["http://example.com/1.jpg", "http://example.com/2.jpg"]
        result = await processor.encode_images_batch(urls)

        # Should return list of lists (matrix)
        assert len(result) == 2
        assert len(result[0]) == 768
        assert len(result[1]) == 768

    @pytest.mark.asyncio
    async def test_encode_images_batch_empty_urls(
        self, mock_vision_model, mock_settings
    ):
        """Test that empty URL list returns empty matrix."""
        mock_downloader = AsyncMock()

        processor = VisionProcessor(
            model=mock_vision_model,
            downloader=mock_downloader,
            settings=mock_settings,
        )

        result = await processor.encode_images_batch([])

        assert result == []
        mock_vision_model.encode_image.assert_not_called()

    @pytest.mark.asyncio
    async def test_encode_images_batch_skips_failed_downloads(
        self, mock_vision_model, mock_settings
    ):
        """Test that failed downloads are skipped."""
        mock_downloader = AsyncMock()
        mock_downloader.download_and_cache_image.side_effect = [
            "/cache/image1.jpg",
            None,  # Failed download
            "/cache/image3.jpg",
        ]

        mock_vision_model.encode_image.side_effect = [
            [[0.1] * 768],
            [[0.3] * 768],
        ]

        processor = VisionProcessor(
            model=mock_vision_model,
            downloader=mock_downloader,
            settings=mock_settings,
        )

        urls = [
            "http://example.com/1.jpg",
            "http://example.com/2.jpg",
            "http://example.com/3.jpg",
        ]
        result = await processor.encode_images_batch(urls)

        # Should only have 2 results (skipping the failed download)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_encode_images_batch_skips_failed_encoding(
        self, mock_vision_model, mock_settings
    ):
        """Test that failed encodings are skipped."""
        mock_downloader = AsyncMock()
        mock_downloader.download_and_cache_image.side_effect = [
            "/cache/image1.jpg",
            "/cache/image2.jpg",
            "/cache/image3.jpg",
        ]

        mock_vision_model.encode_image.side_effect = [
            [[0.1] * 768],
            None,  # Failed encoding
            [[0.3] * 768],
        ]

        processor = VisionProcessor(
            model=mock_vision_model,
            downloader=mock_downloader,
            settings=mock_settings,
        )

        urls = [
            "http://example.com/1.jpg",
            "http://example.com/2.jpg",
            "http://example.com/3.jpg",
        ]
        result = await processor.encode_images_batch(urls)

        # Should only have 2 results (skipping the failed encoding)
        assert len(result) == 2
