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
            config=mock_settings,
        )

        assert processor.model == mock_vision_model
        assert processor.downloader == mock_downloader
        assert processor.config == mock_settings

    def test_init_without_settings_uses_defaults(
        self, mock_vision_model, mock_downloader
    ):
        """Test initialization without config uses default EmbeddingConfig."""
        with patch(
            "vector_processing.processors.vision_processor.EmbeddingConfig"
        ) as mock_config_class:
            mock_default_config = MagicMock()
            mock_default_config.embed_max_concurrency = 2
            mock_default_config.max_concurrent_image_downloads = 10
            mock_config_class.return_value = mock_default_config

            processor = VisionProcessor(
                model=mock_vision_model, downloader=mock_downloader
            )

            mock_config_class.assert_called_once()
            assert processor.config == mock_default_config

    def test_init_logs_model_name(
        self, mock_vision_model, mock_downloader, mock_settings, caplog
    ):
        """Test that initialization logs the model name."""
        with caplog.at_level("INFO"):
            VisionProcessor(
                model=mock_vision_model,
                downloader=mock_downloader,
                config=mock_settings,
            )

        assert (
            "Initialized VisionProcessor with model: test-vision-model" in caplog.text
        )


class TestEncodeImage:
    """Tests for encode_image method."""

    @pytest.mark.asyncio
    async def test_encode_image_success(
        self, mock_vision_model, mock_downloader, mock_settings
    ):
        """Test successful image encoding."""
        processor = VisionProcessor(
            model=mock_vision_model,
            downloader=mock_downloader,
            config=mock_settings,
        )

        result = await processor.encode_image("/path/to/image.jpg")

        assert result == [0.2] * 768
        mock_vision_model.encode_image.assert_called_once_with(["/path/to/image.jpg"])

    @pytest.mark.asyncio
    async def test_encode_image_model_returns_empty_list(
        self, mock_vision_model, mock_downloader, mock_settings
    ):
        """Test when model returns empty list."""
        mock_vision_model.encode_image.return_value = []
        processor = VisionProcessor(
            model=mock_vision_model,
            downloader=mock_downloader,
            config=mock_settings,
        )

        result = await processor.encode_image("/path/to/image.jpg")

        assert result is None

    @pytest.mark.asyncio
    async def test_encode_image_model_returns_none(
        self, mock_vision_model, mock_downloader, mock_settings
    ):
        """Test when model returns None."""
        mock_vision_model.encode_image.return_value = None
        processor = VisionProcessor(
            model=mock_vision_model,
            downloader=mock_downloader,
            config=mock_settings,
        )

        result = await processor.encode_image("/path/to/image.jpg")

        assert result is None

    @pytest.mark.asyncio
    async def test_encode_image_model_raises_exception(
        self, mock_vision_model, mock_downloader, mock_settings, caplog
    ):
        """Test when model raises exception."""
        mock_vision_model.encode_image.side_effect = RuntimeError("Encoding error")
        processor = VisionProcessor(
            model=mock_vision_model,
            downloader=mock_downloader,
            config=mock_settings,
        )

        with caplog.at_level("ERROR"):
            result = await processor.encode_image("/path/to/image.jpg")

        assert result is None
        assert "Image encoding failed" in caplog.text


class TestHashEmbedding:
    """Tests for _hash_embedding method."""

    def test_hash_embedding_returns_blake2b_hash(
        self, mock_vision_model, mock_downloader, mock_settings
    ):
        """Test hash embedding returns consistent blake2b hash."""
        processor = VisionProcessor(
            model=mock_vision_model,
            downloader=mock_downloader,
            config=mock_settings,
        )

        embedding = [0.1234, 0.5678, 0.9012]
        result = processor._hash_embedding(embedding)

        # Should return a 32-character hex string (blake2b with digest_size=16)
        assert len(result) == 32
        assert all(c in "0123456789abcdef" for c in result)

    def test_hash_embedding_consistent_for_same_input(
        self, mock_vision_model, mock_downloader, mock_settings
    ):
        """Test same embedding produces same hash."""
        processor = VisionProcessor(
            model=mock_vision_model,
            downloader=mock_downloader,
            config=mock_settings,
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
            config=mock_settings,
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
            config=mock_settings,
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
            config=mock_settings,
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
            config=mock_settings,
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
            config=mock_settings,
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
            config=mock_settings,
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
            config=mock_settings,
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
            config=mock_settings,
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
            config=mock_settings,
        )

        result = processor.get_model_info()

        assert result == expected_info
        mock_vision_model.get_model_info.assert_called_once()


class TestEncodeImagesBatch:
    """Tests for encode_images_batch async method with concurrent downloads."""

    @pytest.mark.asyncio
    async def test_downloads_all_images_concurrently(
        self, mock_vision_model, mock_settings
    ):
        """Test that all images are downloaded concurrently, not sequentially."""
        import asyncio

        mock_downloader = AsyncMock()

        # Track concurrent downloads using overlap counting (deterministic, not timing-based)
        active_downloads = 0
        max_concurrent = 0

        async def track_download(url, **_kwargs):
            nonlocal active_downloads, max_concurrent
            active_downloads += 1
            max_concurrent = max(max_concurrent, active_downloads)
            await asyncio.sleep(0.01)  # Simulate network delay
            active_downloads -= 1
            return f"/cache/{url.split('/')[-1]}"

        mock_downloader.download_and_cache_image.side_effect = track_download
        mock_vision_model.encode_image.return_value = [
            [0.1] * 768,
            [0.2] * 768,
            [0.3] * 768,
        ]

        processor = VisionProcessor(
            model=mock_vision_model,
            downloader=mock_downloader,
            config=mock_settings,
        )

        urls = [
            "http://example.com/1.jpg",
            "http://example.com/2.jpg",
            "http://example.com/3.jpg",
        ]
        await processor.encode_images_batch(urls)

        # All 3 downloads should overlap (semaphore allows 10 concurrent)
        # If sequential, max_concurrent would be 1
        assert max_concurrent == 3, (
            f"Expected all 3 downloads to run concurrently, but max_concurrent={max_concurrent}"
        )

    @pytest.mark.asyncio
    async def test_calls_encode_image_once_with_all_paths(
        self, mock_vision_model, mock_settings
    ):
        """Test that model.encode_image is called ONCE with ALL paths in batch."""
        mock_downloader = AsyncMock()
        mock_downloader.download_and_cache_image.side_effect = [
            "/cache/image1.jpg",
            "/cache/image2.jpg",
            "/cache/image3.jpg",
        ]

        mock_vision_model.encode_image.return_value = [
            [0.1] * 768,
            [0.2] * 768,
            [0.3] * 768,
        ]

        processor = VisionProcessor(
            model=mock_vision_model,
            downloader=mock_downloader,
            config=mock_settings,
        )

        urls = [
            "http://example.com/1.jpg",
            "http://example.com/2.jpg",
            "http://example.com/3.jpg",
        ]
        await processor.encode_images_batch(urls)

        # CRITICAL: Should call encode_image ONCE with all 3 paths
        mock_vision_model.encode_image.assert_called_once_with(
            [
                "/cache/image1.jpg",
                "/cache/image2.jpg",
                "/cache/image3.jpg",
            ]
        )

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

        # Single call returns all embeddings
        mock_vision_model.encode_image.return_value = [
            [0.1] * 768,
            [0.2] * 768,
        ]

        processor = VisionProcessor(
            model=mock_vision_model,
            downloader=mock_downloader,
            config=mock_settings,
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
            config=mock_settings,
        )

        result = await processor.encode_images_batch([])

        assert result == []
        mock_vision_model.encode_image.assert_not_called()

    @pytest.mark.asyncio
    async def test_encode_images_batch_skips_failed_downloads(
        self, mock_vision_model, mock_settings
    ):
        """Test that failed downloads are skipped, successful ones are batched."""
        mock_downloader = AsyncMock()
        mock_downloader.download_and_cache_image.side_effect = [
            "/cache/image1.jpg",
            None,  # Failed download
            "/cache/image3.jpg",
        ]

        # Single batch call with only successful downloads
        mock_vision_model.encode_image.return_value = [
            [0.1] * 768,
            [0.3] * 768,
        ]

        processor = VisionProcessor(
            model=mock_vision_model,
            downloader=mock_downloader,
            config=mock_settings,
        )

        urls = [
            "http://example.com/1.jpg",
            "http://example.com/2.jpg",
            "http://example.com/3.jpg",
        ]
        result = await processor.encode_images_batch(urls)

        # Should only have 2 results (skipping the failed download)
        assert len(result) == 2
        # Should batch encode only the 2 successful downloads
        mock_vision_model.encode_image.assert_called_once_with(
            [
                "/cache/image1.jpg",
                "/cache/image3.jpg",
            ]
        )

    @pytest.mark.asyncio
    async def test_encode_images_batch_handles_encoding_failure(
        self, mock_vision_model, mock_settings
    ):
        """Test that batch encoding failure returns empty list."""
        mock_downloader = AsyncMock()
        mock_downloader.download_and_cache_image.side_effect = [
            "/cache/image1.jpg",
            "/cache/image2.jpg",
            "/cache/image3.jpg",
        ]

        # Batch encoding fails
        mock_vision_model.encode_image.side_effect = RuntimeError("GPU error")

        processor = VisionProcessor(
            model=mock_vision_model,
            downloader=mock_downloader,
            config=mock_settings,
        )

        urls = [
            "http://example.com/1.jpg",
            "http://example.com/2.jpg",
            "http://example.com/3.jpg",
        ]
        result = await processor.encode_images_batch(urls)

        # Should return empty list on batch encoding failure
        assert result == []

    @pytest.mark.asyncio
    async def test_all_downloads_fail_returns_empty(
        self, mock_vision_model, mock_settings
    ):
        """Test that when all downloads fail, returns empty list without calling encode."""
        mock_downloader = AsyncMock()
        mock_downloader.download_and_cache_image.side_effect = [
            None,  # Failed
            None,  # Failed
            None,  # Failed
        ]

        processor = VisionProcessor(
            model=mock_vision_model,
            downloader=mock_downloader,
            config=mock_settings,
        )

        urls = [
            "http://example.com/1.jpg",
            "http://example.com/2.jpg",
            "http://example.com/3.jpg",
        ]
        result = await processor.encode_images_batch(urls)

        # Should return empty list
        assert result == []
        # Should NOT call encode_image when no downloads succeeded
        mock_vision_model.encode_image.assert_not_called()

    @pytest.mark.asyncio
    async def test_download_exception_does_not_stop_other_downloads(
        self, mock_vision_model, mock_settings
    ):
        """Test that exception in one download doesn't prevent others from completing."""
        mock_downloader = AsyncMock()

        async def download_with_exception(url, **_kwargs):
            if "2.jpg" in url:
                raise RuntimeError("Network error")  # noqa: TRY003
            return f"/cache/{url.split('/')[-1]}"

        mock_downloader.download_and_cache_image.side_effect = download_with_exception
        mock_vision_model.encode_image.return_value = [
            [0.1] * 768,
            [0.3] * 768,
        ]

        processor = VisionProcessor(
            model=mock_vision_model,
            downloader=mock_downloader,
            config=mock_settings,
        )

        urls = [
            "http://example.com/1.jpg",
            "http://example.com/2.jpg",  # Will raise exception
            "http://example.com/3.jpg",
        ]
        result = await processor.encode_images_batch(urls)

        # Should get 2 results (1 and 3 succeeded, 2 failed)
        assert len(result) == 2
        mock_vision_model.encode_image.assert_called_once_with(
            [
                "/cache/1.jpg",
                "/cache/3.jpg",
            ]
        )

    @pytest.mark.asyncio
    async def test_limits_concurrent_downloads_with_semaphore(
        self, mock_vision_model, mock_settings
    ):
        """Test that concurrent downloads are limited by semaphore to prevent resource exhaustion.

        This test verifies the fix for unbounded concurrency issue where all downloads
        ran simultaneously, causing potential resource exhaustion with large batches.
        """
        import asyncio

        mock_downloader = AsyncMock()

        # Track maximum concurrent downloads
        active_downloads = 0
        max_concurrent = 0

        async def track_concurrent_download(url, **_kwargs):
            nonlocal active_downloads, max_concurrent
            active_downloads += 1
            max_concurrent = max(max_concurrent, active_downloads)
            await asyncio.sleep(0.01)  # Simulate download time
            active_downloads -= 1
            return f"/cache/{url.split('/')[-1]}"

        mock_downloader.download_and_cache_image.side_effect = track_concurrent_download
        mock_vision_model.encode_image.return_value = [[0.1] * 768] * 20

        processor = VisionProcessor(
            model=mock_vision_model,
            downloader=mock_downloader,
            config=mock_settings,
            max_concurrent_downloads=5,  # NEW: Limit to 5 concurrent
        )

        # Test with 20 URLs
        urls = [f"http://example.com/{i}.jpg" for i in range(20)]
        await processor.encode_images_batch(urls)

        # CRITICAL: Max concurrent should respect the limit
        assert max_concurrent <= 5, (
            f"Exceeded concurrency limit: {max_concurrent} concurrent (limit: 5)"
        )

    @pytest.mark.asyncio
    async def test_uses_shared_session_for_downloads(
        self, mock_vision_model, mock_settings
    ):
        """Test that ImageDownloader uses a shared session instead of creating new ones.

        This test verifies the fix for session proliferation where each download
        created its own ClientSession (1:1 ratio).
        """
        mock_downloader = AsyncMock()

        # Track session usage
        sessions_created = []

        async def track_session_usage(url, session=None):
            sessions_created.append(session)
            return f"/cache/{url.split('/')[-1]}"

        mock_downloader.download_and_cache_image.side_effect = track_session_usage
        mock_vision_model.encode_image.return_value = [[0.1] * 768] * 10

        processor = VisionProcessor(
            model=mock_vision_model,
            downloader=mock_downloader,
            config=mock_settings,
        )

        urls = [f"http://example.com/{i}.jpg" for i in range(10)]
        await processor.encode_images_batch(urls)

        # CRITICAL: Downloader should be called with session parameter
        # All calls should have session parameter (not None)
        for call in mock_downloader.download_and_cache_image.call_args_list:
            kwargs = call.kwargs
            assert "session" in kwargs, "Session not passed to downloader"
            assert kwargs["session"] is not None, "Session is None"

    @pytest.mark.asyncio
    async def test_default_concurrency_limit_reasonable(
        self, mock_vision_model, mock_settings
    ):
        """Test that default concurrency limit is reasonable (not unbounded)."""
        import asyncio

        mock_downloader = AsyncMock()

        active_downloads = 0
        max_concurrent = 0

        async def track_concurrent(url, **_kwargs):
            nonlocal active_downloads, max_concurrent
            active_downloads += 1
            max_concurrent = max(max_concurrent, active_downloads)
            await asyncio.sleep(0.001)
            active_downloads -= 1
            return f"/cache/{url.split('/')[-1]}"

        mock_downloader.download_and_cache_image.side_effect = track_concurrent
        mock_vision_model.encode_image.return_value = [[0.1] * 768] * 50

        # Create processor WITHOUT specifying max_concurrent_downloads
        processor = VisionProcessor(
            model=mock_vision_model,
            downloader=mock_downloader,
            config=mock_settings,
        )

        urls = [f"http://example.com/{i}.jpg" for i in range(50)]
        await processor.encode_images_batch(urls)

        # Default should be reasonable (e.g., 10-20, not 50)
        assert max_concurrent <= 20, (
            f"Default concurrency too high: {max_concurrent} (should be ≤20)"
        )
        assert max_concurrent > 0, "No downloads were concurrent"


class TestRetryLogic:
    """Tests for download retry logic with exponential backoff."""

    @pytest.mark.asyncio
    async def test_retries_failed_downloads_with_backoff(
        self, mock_vision_model, mock_downloader, mock_settings
    ):
        """Test that failed downloads are retried with exponential backoff."""
        attempt_count = 0
        sleep_delays: list[float] = []

        async def failing_download(url, session=None):  # noqa: ARG001
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:  # Fail first 2 attempts, succeed on 3rd
                raise RuntimeError("Transient network error")  # noqa: TRY003
            return f"/cache/{url.split('/')[-1]}"

        async def tracking_sleep(delay):
            sleep_delays.append(delay)

        mock_downloader.download_and_cache_image = AsyncMock(
            side_effect=failing_download
        )
        mock_vision_model.encode_image.return_value = [[0.1] * 768]

        processor = VisionProcessor(
            model=mock_vision_model,
            downloader=mock_downloader,
            config=mock_settings,
        )

        urls = ["http://example.com/test.jpg"]
        with patch("asyncio.sleep", new=tracking_sleep):
            result = await processor.encode_images_batch(urls, max_retries=2)

        # Should succeed after retries
        assert len(result) == 1
        assert attempt_count == 3  # 3 total attempts (initial + 2 retries)

        # Verify exact exponential backoff delays passed to asyncio.sleep
        assert len(sleep_delays) == 2, "Should have 2 backoff sleeps"
        assert sleep_delays[0] == pytest.approx(0.5), "First retry should wait 0.5s"
        assert sleep_delays[1] == pytest.approx(1.0), "Second retry should wait 1.0s"

    @pytest.mark.asyncio
    async def test_gives_up_after_max_retries(
        self, mock_vision_model, mock_downloader, mock_settings
    ):
        """Test that downloads are abandoned after max_retries exhausted."""
        mock_downloader.download_and_cache_image = AsyncMock(
            side_effect=Exception("Persistent failure")
        )
        mock_vision_model.encode_image.return_value = []

        processor = VisionProcessor(
            model=mock_vision_model,
            downloader=mock_downloader,
            config=mock_settings,
        )

        urls = ["http://example.com/broken.jpg"]
        result = await processor.encode_images_batch(urls, max_retries=2)

        # Should fail after 3 total attempts (1 initial + 2 retries)
        assert len(result) == 0
        assert mock_downloader.download_and_cache_image.call_count == 3

    @pytest.mark.asyncio
    async def test_successful_downloads_dont_retry(
        self, mock_vision_model, mock_downloader, mock_settings
    ):
        """Test that successful downloads on first attempt don't trigger retries."""
        mock_downloader.download_and_cache_image = AsyncMock(
            return_value="/cache/test.jpg"
        )
        mock_vision_model.encode_image.return_value = [[0.1] * 768]

        processor = VisionProcessor(
            model=mock_vision_model,
            downloader=mock_downloader,
            config=mock_settings,
        )

        urls = ["http://example.com/test.jpg"]
        result = await processor.encode_images_batch(urls, max_retries=2)

        assert len(result) == 1
        # Should only call download once (no retries needed)
        assert mock_downloader.download_and_cache_image.call_count == 1


class TestMetricsLogging:
    """Tests for download/encoding metrics and logging."""

    @pytest.mark.asyncio
    async def test_logs_batch_download_metrics(
        self, mock_vision_model, mock_downloader, mock_settings, caplog
    ):
        """Test that batch download success rates are logged."""

        # 7 successful, 3 failed
        async def selective_download(url, session=None):  # noqa: ARG001
            if "fail" in url:
                raise RuntimeError("Download failed")  # noqa: TRY003
            return f"/cache/{url.split('/')[-1]}"

        mock_downloader.download_and_cache_image = AsyncMock(
            side_effect=selective_download
        )
        mock_vision_model.encode_image.return_value = [[0.1] * 768] * 7

        processor = VisionProcessor(
            model=mock_vision_model,
            downloader=mock_downloader,
            config=mock_settings,
        )

        urls = [f"http://example.com/{i}.jpg" for i in range(7)]
        urls += [f"http://example.com/fail{i}.jpg" for i in range(3)]

        with caplog.at_level("INFO"):
            result = await processor.encode_images_batch(urls, max_retries=0)

        # Verify metrics logged
        assert len(result) == 7
        log_messages = [record.message for record in caplog.records]

        # Should log download metrics
        download_log = [msg for msg in log_messages if "Batch download complete" in msg]
        assert len(download_log) > 0, "No download metrics logged"
        assert "7/10 successful" in download_log[0]
        assert "70.0% success rate" in download_log[0]

        # Should log encoding metrics
        encoding_log = [msg for msg in log_messages if "Batch encoding complete" in msg]
        assert len(encoding_log) > 0, "No encoding metrics logged"

    @pytest.mark.asyncio
    async def test_logs_warning_when_all_downloads_fail(
        self, mock_vision_model, mock_downloader, mock_settings, caplog
    ):
        """Test that warning is logged when no images download successfully."""
        mock_downloader.download_and_cache_image = AsyncMock(
            side_effect=Exception("All downloads failed")
        )

        processor = VisionProcessor(
            model=mock_vision_model,
            downloader=mock_downloader,
            config=mock_settings,
        )

        urls = ["http://example.com/1.jpg", "http://example.com/2.jpg"]

        with caplog.at_level("WARNING"):
            result = await processor.encode_images_batch(urls, max_retries=0)

        assert len(result) == 0
        log_messages = [record.message for record in caplog.records]
        warning_log = [
            msg for msg in log_messages if "No images downloaded successfully" in msg
        ]
        assert len(warning_log) > 0, "No warning logged for complete failure"


class TestConfigurationIntegration:
    """Tests for configuration-driven behavior."""

    def test_uses_settings_max_concurrent_downloads(
        self, mock_vision_model, mock_downloader, mock_settings
    ):
        """Test that processor uses settings.max_concurrent_image_downloads by default."""
        mock_settings.max_concurrent_image_downloads = 25

        processor = VisionProcessor(
            model=mock_vision_model,
            downloader=mock_downloader,
            config=mock_settings,
        )

        assert processor.max_concurrent_downloads == 25

    def test_allows_override_of_max_concurrent_downloads(
        self, mock_vision_model, mock_downloader, mock_settings
    ):
        """Test that max_concurrent_downloads can be overridden for testing."""
        mock_settings.max_concurrent_image_downloads = 25

        processor = VisionProcessor(
            model=mock_vision_model,
            downloader=mock_downloader,
            config=mock_settings,
            max_concurrent_downloads=5,  # Override
        )

        assert processor.max_concurrent_downloads == 5

    def test_logs_concurrency_limit_on_init(
        self, mock_vision_model, mock_downloader, mock_settings, caplog
    ):
        """Test that initialization logs the configured concurrency limit."""
        mock_settings.max_concurrent_image_downloads = 15

        with caplog.at_level("INFO"):
            VisionProcessor(
                model=mock_vision_model,
                downloader=mock_downloader,
                config=mock_settings,
            )

        log_messages = [record.message for record in caplog.records]
        init_log = [msg for msg in log_messages if "max_concurrent_downloads" in msg]
        assert len(init_log) > 0, "Concurrency limit not logged"
        assert "15" in init_log[0]
