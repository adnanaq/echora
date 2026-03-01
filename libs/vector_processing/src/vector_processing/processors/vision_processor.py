"""Vision processor for converting images to vector embeddings.

This module provides the VisionProcessor class which serves as a pure compute
engine for image embedding. It is strictly responsible for converting image
files to embedding vectors using OpenCLIP models, with no domain-specific logic.
"""

import asyncio
import hashlib
import logging
import time
from typing import Any

import aiohttp
from common.config import EmbeddingConfig
from opentelemetry import metrics as otel_metrics
from opentelemetry import trace as otel_trace

from ..cache import EmbeddingCache
from ..embedding_models.vision.base import VisionEmbeddingModel
from ..utils.image_downloader import ImageDownloader

logger = logging.getLogger(__name__)

_tracer = otel_trace.get_tracer("echora.vector_processing")
_meter = otel_metrics.get_meter("echora.vector_processing")
_embedding_duration = _meter.create_histogram(
    "echora_embedding_duration_seconds",
    unit="s",
    description="Embedding model inference duration in seconds",
)
_image_download_duration = _meter.create_histogram(
    "echora_image_download_duration_seconds",
    unit="s",
    description="Image download and cache duration per URL",
)
_image_download_failures = _meter.create_counter(
    "echora_image_download_failures_total",
    description="Total image download failures after all retries",
)
_embedding_cache_lookups = _meter.create_counter(
    "echora_embedding_cache_total",
    description="Embedding cache lookups by result (hit/miss) and modality",
)


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
        embedding_cache: EmbeddingCache | None = None,
    ):
        """Initialize the vision processor with model and downloader.

        Args:
            model: An initialized VisionEmbeddingModel instance.
            downloader: An initialized ImageDownloader for fetching images.
            config: Embedding configuration instance. Uses defaults if None.
            max_concurrent_downloads: Maximum number of concurrent image downloads.
                If None, uses config.max_concurrent_image_downloads (default: 10).
                Override for testing or special cases.
            embedding_cache: Optional Redis-backed embedding cache.
                When provided, image embeddings are cached by file-content hash
                to avoid redundant model inference.
        """
        if config is None:
            config = EmbeddingConfig()

        self.config = config
        self.model = model
        self.downloader = downloader
        self._semaphore = asyncio.Semaphore(config.embed_max_concurrency)
        self._cache = embedding_cache

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
        with semaphore-based concurrency control. When an embedding cache is
        configured, results are looked up by file-content hash before inference.

        Args:
            image_path: Absolute path to the local image file.

        Returns:
            A list of floats representing the image embedding vector,
            or None if encoding fails.
        """
        # Hash file contents for cache lookup (before semaphore — fast I/O)
        file_hash: str | None = None
        if self._cache is not None:
            try:
                file_hash = await asyncio.to_thread(self._hash_file, image_path)
                cached = await self._cache.get(self.model.model_name, file_hash)
                if cached is not None:
                    _embedding_cache_lookups.add(
                        1, {"result": "hit", "modality": "image"}
                    )
                    return cached
                _embedding_cache_lookups.add(1, {"result": "miss", "modality": "image"})
            except Exception:
                logger.debug("Failed to hash image for cache lookup", exc_info=True)

        with _tracer.start_as_current_span(
            "vector_processing.vision.encode",
            attributes={"embedding.model": self.model.model_name},
        ):
            try:
                async with self._semaphore:
                    _start = time.perf_counter()
                    embeddings = await asyncio.to_thread(
                        self.model.encode_image, [image_path]
                    )
                    _embedding_duration.record(
                        time.perf_counter() - _start, {"modality": "image"}
                    )
            except Exception:
                logger.exception("Image encoding failed")
                return None
            else:
                if embeddings:
                    result = embeddings[0]
                    if self._cache is not None and file_hash is not None:
                        await self._cache.set(self.model.model_name, file_hash, result)
                    return result
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
                            _dl_start = time.perf_counter()
                            local_path = await self.downloader.download_and_cache_image(
                                url, session=session
                            )
                            if local_path is None:
                                raise RuntimeError(f"Download returned None for {url}")  # noqa: TRY003, TRY301
                            _image_download_duration.record(
                                time.perf_counter() - _dl_start
                            )
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
                                _image_download_failures.add(1)
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

        # Hash downloaded files and check cache
        file_hashes: list[str | None] = []
        cached_results: list[list[float] | None] = [None] * len(local_paths)
        uncached_positions: list[int] = []

        if self._cache is not None:
            for i, path in enumerate(local_paths):
                try:
                    file_hashes.append(self._hash_file(path))
                except Exception:
                    file_hashes.append(None)

            valid_hashes = [h for h in file_hashes if h is not None]
            valid_hash_indices = [i for i, h in enumerate(file_hashes) if h is not None]

            if valid_hashes:
                batch_cached = await self._cache.get_batch(
                    self.model.model_name, valid_hashes
                )
                for j, idx in enumerate(valid_hash_indices):
                    cached_results[idx] = batch_cached[j]

            uncached_positions = [
                i for i in range(len(local_paths)) if cached_results[i] is None
            ]
            cache_hits = len(local_paths) - len(uncached_positions)
            if cache_hits:
                _embedding_cache_lookups.add(
                    cache_hits, {"result": "hit", "modality": "image"}
                )
            if uncached_positions:
                _embedding_cache_lookups.add(
                    len(uncached_positions), {"result": "miss", "modality": "image"}
                )
        else:
            uncached_positions = list(range(len(local_paths)))

        # Collect results from cache hits
        all_embeddings: list[list[float]] = []
        if not uncached_positions:
            # All cache hits
            all_embeddings = [e for e in cached_results if e is not None]
            logger.info(
                f"Batch encode: all {len(all_embeddings)} embeddings served from cache"
            )
            return all_embeddings

        # Encode only uncached images
        uncached_paths = [local_paths[i] for i in uncached_positions]

        with _tracer.start_as_current_span(
            "vector_processing.vision.encode_batch",
            attributes={
                "embedding.model": self.model.model_name,
                "embedding.batch_size": len(uncached_paths),
                "embedding.cache_hits": len(local_paths) - len(uncached_positions),
            },
        ):
            try:
                async with self._semaphore:
                    _start = time.perf_counter()
                    encoded_new = await asyncio.to_thread(
                        self.model.encode_image, uncached_paths
                    )
                    _embedding_duration.record(
                        time.perf_counter() - _start, {"modality": "image"}
                    )
            except Exception:
                logger.exception("Batch encoding failed")
                return [e for e in cached_results if e is not None]

            # Merge newly encoded into cached_results and write to cache
            to_cache: dict[str, list[float]] = {}
            for j, pos in enumerate(uncached_positions):
                cached_results[pos] = encoded_new[j]
                if (
                    self._cache is not None
                    and pos < len(file_hashes)
                    and file_hashes[pos] is not None
                ):
                    to_cache[file_hashes[pos]] = encoded_new[j]  # type: ignore[index]

            if self._cache is not None and to_cache:
                await self._cache.set_batch(self.model.model_name, to_cache)

        all_embeddings = [e for e in cached_results if e is not None]
        logger.info(
            f"Batch encoding complete: {len(all_embeddings)}/{total_urls} images encoded successfully"
        )
        return all_embeddings

    @staticmethod
    def _hash_file(path: str) -> str:
        """Compute SHA-256 hex digest of a file's contents."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

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
