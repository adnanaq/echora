"""Vision processor for converting images to vector embeddings.

This module provides the VisionProcessor class which serves as a pure compute
engine for image embedding. It is strictly responsible for converting image
files to embedding vectors using OpenCLIP models, with no domain-specific logic.
"""

import asyncio
import hashlib
import logging
from typing import Any

import aiohttp
from common.config import EmbeddingConfig

from ..embedding_models.vision.base import VisionEmbeddingModel
from ..utils.image_downloader import ImageDownloader

logger = logging.getLogger(__name__)


class VisionProcessor:
    """Pure vision embedding processor with no domain-specific logic.

    This class serves as the "Visual Compute Engine" in the vector processing
    pipeline. It knows HOW to turn images into numbers but has no knowledge
    of anime, characters, or any domain concepts.

    Responsibilities:
        - Interface with VisionEmbeddingModels (OpenCLIP).
        - Manage image downloading and caching via ImageDownloader.
        - Provide raw image encoding capabilities.
    """

    def __init__(
        self,
        model: VisionEmbeddingModel,
        downloader: ImageDownloader,
        config: EmbeddingConfig | None = None,
        max_concurrent_downloads: int | None = None,
    ):
        """Initialize the vision processor with model and downloader.

        Args:
            model: An initialized VisionEmbeddingModel instance.
            downloader: An initialized ImageDownloader for fetching images.
            config: Embedding configuration instance. Uses defaults if None.
            max_concurrent_downloads: Maximum number of concurrent image downloads.
                If None, uses config.max_concurrent_image_downloads (default: 10).
                Override for testing or special cases.
        """
        if config is None:
            config = EmbeddingConfig()

        self.config = config
        self.model = model
        self.downloader = downloader
        self._semaphore = asyncio.Semaphore(config.embed_max_concurrency)

        # Use config value as default, allow override for testing
        self.max_concurrent_downloads = (
            max_concurrent_downloads
            if max_concurrent_downloads is not None
            else config.max_concurrent_image_downloads
        )

        logger.info(
            f"Initialized VisionProcessor with model: {model.model_name}, "
            f"max_concurrent_downloads: {self.max_concurrent_downloads}"
        )

    async def encode_image(self, image_path: str) -> list[float] | None:
        """Encode a local image file to an embedding vector.

        Runs the model inference in a thread to avoid blocking the event loop,
        with semaphore-based concurrency control.

        Args:
            image_path: Absolute path to the local image file.

        Returns:
            A list of floats representing the image embedding vector,
            or None if encoding fails.
        """
        try:
            async with self._semaphore:
                embeddings = await asyncio.to_thread(
                    self.model.encode_image, [image_path]
                )
        except Exception:
            logger.exception("Image encoding failed")
            return None
        else:
            if embeddings:
                return embeddings[0]
            return None

    async def encode_images_batch(
        self, image_urls: list[str], max_retries: int = 2
    ) -> list[list[float]]:
        """Encode multiple images from URLs into a matrix for multivector storage.

        Downloads images concurrently with semaphore-based rate limiting and
        encodes them in a single batch for optimal performance. Uses shared
        aiohttp session to prevent connection proliferation. Failed downloads
        are automatically retried with exponential backoff.

        Args:
            image_urls: List of image URLs to download and encode.
            max_retries: Maximum retry attempts for failed downloads (default: 2).

        Returns:
            A list of embedding vectors (matrix). Failed downloads/encodings
            are skipped, so the result may have fewer vectors than input URLs.
        """
        if not image_urls:
            return []

        total_urls = len(image_urls)

        # Create shared session and semaphore for controlled concurrency
        semaphore = asyncio.Semaphore(self.max_concurrent_downloads)
        async with aiohttp.ClientSession() as session:
            # Concurrent downloads with semaphore limiting and retry logic (I/O-bound)
            async def download_single_with_retry(url: str) -> tuple[str, str | None]:
                async with semaphore:
                    for attempt in range(max_retries + 1):
                        try:
                            local_path = await self.downloader.download_and_cache_image(
                                url, session=session
                            )
                            if local_path is None:
                                raise RuntimeError(f"Download returned None for {url}")  # noqa: TRY003, TRY301
                        except Exception as e:
                            if attempt < max_retries:
                                # Exponential backoff: 0.5s, 1s, 2s, ...
                                wait_time = 0.5 * (2**attempt)
                                logger.debug(
                                    f"Download attempt {attempt + 1}/{max_retries + 1} "
                                    f"failed for {url}, retrying in {wait_time}s: {e}"
                                )
                                await asyncio.sleep(wait_time)
                            else:
                                logger.warning(
                                    f"Failed to download {url} after {max_retries + 1} attempts: {e}"
                                )
                                return (url, None)
                        else:
                            return (url, local_path)
                    return (
                        url,
                        None,
                    )  # pragma: no cover - unreachable defensive fallback

            download_results = await asyncio.gather(
                *[download_single_with_retry(url) for url in image_urls]
            )

        # Filter successful downloads and compute metrics
        local_paths = [path for _, path in download_results if path is not None]
        successful_downloads = len(local_paths)
        failed_downloads = total_urls - successful_downloads
        success_rate = (
            (successful_downloads / total_urls * 100) if total_urls > 0 else 0
        )

        # Log batch download metrics
        logger.info(
            f"Batch download complete: {successful_downloads}/{total_urls} successful "
            f"({success_rate:.1f}% success rate), {failed_downloads} failed"
        )

        if not local_paths:
            logger.warning("No images downloaded successfully in batch")
            return []

        # Single batch encoding (CPU/GPU-bound - leverage model's batch processing)
        try:
            async with self._semaphore:
                embeddings = await asyncio.to_thread(
                    self.model.encode_image, local_paths
                )
        except Exception:
            logger.exception("Batch encoding failed")
            return []
        else:
            logger.info(
                f"Batch encoding complete: {len(embeddings)}/{total_urls} images encoded successfully"
            )
            return embeddings

    def _hash_embedding(self, embedding: list[float], precision: int = 4) -> str:
        """Create a hash of an embedding vector for duplicate detection.

        Args:
            embedding: The embedding vector to hash.
            precision: Decimal places to round to before hashing.

        Returns:
            A blake2b hash string of the rounded embedding vector.
        """
        try:
            rounded_embedding = [round(x, precision) for x in embedding]
            embedding_str = ",".join(map(str, rounded_embedding))
            return hashlib.blake2b(embedding_str.encode(), digest_size=16).hexdigest()
        except Exception:
            return str(hash(tuple(embedding)))

    def get_cache_stats(self) -> dict[str, Any]:
        """Get image cache statistics from the downloader.

        Returns:
            A dictionary containing cache statistics such as size and hit rate.
        """
        return self.downloader.get_cache_stats()

    def clear_cache(self, max_age_days: int | None = None) -> dict[str, Any]:
        """Clear the image cache via the downloader.

        Args:
            max_age_days: If provided, only clear images older than this.
                If None, clears all cached images.

        Returns:
            A dictionary containing information about the cleared cache.
        """
        return self.downloader.clear_cache(max_age_days)

    def get_supported_formats(self) -> list[str]:
        """Get the list of supported image formats.

        Returns:
            A list of supported image file extensions.
        """
        return ["jpg", "jpeg", "png", "bmp", "tiff", "webp", "gif"]

    def get_model_info(self) -> dict[str, Any]:
        """Get metadata about the underlying vision model.

        Returns:
            A dictionary containing model information such as name,
            embedding size, and other model-specific details.
        """
        return self.model.get_model_info()
