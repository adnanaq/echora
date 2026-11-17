"""Vision processor for converting images to vector embeddings.

This module provides the VisionProcessor class, which serves as a pure compute
engine for image embedding. The processor converts image files into embedding
vectors using OpenCLIP models with no domain-specific logic.
"""

import asyncio
import hashlib
import logging
import time
from typing import Any

import aiohttp
from common.config import EmbeddingConfig
from common.utils.retry import retry_with_backoff
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

    This class serves as the visual compute engine in the vector processing
    pipeline. The processor turns images into numbers with no knowledge of anime,
    characters, or any domain concepts.

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

        local_paths = await self._download_images_batch(image_urls, max_retries)
        if not local_paths:
            logger.warning("No images downloaded successfully in batch")
            return []

        (
            file_hashes,
            cached_results,
            uncached_positions,
        ) = await self._lookup_cached_image_embeddings(local_paths)

        if not uncached_positions:
            all_embeddings = [e for e in cached_results if e is not None]
            logger.info(
                f"Batch encode: all {len(all_embeddings)} embeddings served from cache"
            )
            return all_embeddings

        await self._encode_uncached_images(
            local_paths, file_hashes, uncached_positions, cached_results
        )

        all_embeddings = [e for e in cached_results if e is not None]
        logger.info(
            f"Batch encoding complete: {len(all_embeddings)}/{len(image_urls)} images encoded successfully"
        )
        return all_embeddings

    async def _download_images_batch(
        self, image_urls: list[str], max_retries: int
    ) -> list[str]:
        """Download a batch of image URLs concurrently with retry backoff.

        Args:
            image_urls: Image URLs to download.
            max_retries: Maximum retry attempts per failed download.

        Returns:
            List of local file paths for successfully downloaded images.
        """
        total_urls = len(image_urls)
        semaphore = asyncio.Semaphore(self.max_concurrent_downloads)
        async with aiohttp.ClientSession() as session:
            download_results = await asyncio.gather(
                *[
                    self._download_single_image(session, semaphore, url, max_retries)
                    for url in image_urls
                ]
            )

        local_paths = [path for path in download_results if path is not None]
        successful_downloads = len(local_paths)
        failed_downloads = total_urls - successful_downloads
        success_rate = (successful_downloads / total_urls * 100) if total_urls else 0
        logger.info(
            f"Batch download complete: {successful_downloads}/{total_urls} successful "
            f"({success_rate:.1f}% success rate), {failed_downloads} failed"
        )
        return local_paths

    async def _download_single_image(
        self,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        url: str,
        max_retries: int,
    ) -> str | None:
        """Download a single image URL with semaphore control and exponential backoff.

        Args:
            session: Shared aiohttp client session.
            semaphore: Concurrency limiter for downloads.
            url: Source image URL.
            max_retries: Maximum retry attempts after the initial attempt.

        Returns:
            Local file path on success, or None if all attempts failed.
        """

        async def _attempt_download() -> str:
            _dl_start = time.perf_counter()
            local_path = await self.downloader.download_and_cache_image(
                url, session=session
            )
            if local_path is None:
                raise RuntimeError(f"Download returned None for {url}")  # noqa: TRY003
            _image_download_duration.record(time.perf_counter() - _dl_start)
            return local_path

        def _log_retry(
            attempt: int, max_retries: int, error: Exception, delay: float
        ) -> None:
            logger.debug(
                f"Download attempt {attempt}/{max_retries + 1} "
                f"failed for {url}, retrying in {delay}s: {error}"
            )

        async with semaphore:
            try:
                return await retry_with_backoff(
                    _attempt_download,
                    is_transient_error=lambda _exc: True,
                    max_retries=max_retries,
                    retry_delay=0.5,
                    on_retry=_log_retry,
                )
            except Exception as exc:
                logger.warning(
                    f"Failed to download {url} after {max_retries + 1} attempts: {exc}"
                )
                _image_download_failures.add(1)
                return None

    async def _lookup_cached_image_embeddings(
        self, local_paths: list[str]
    ) -> tuple[list[str | None], list[list[float] | None], list[int]]:
        """Hash downloaded image files and query the embedding cache.

        Args:
            local_paths: Local file paths for downloaded images.

        Returns:
            Tuple of ``(file_hashes, cached_results, uncached_positions)``.
        """
        cached_results: list[list[float] | None] = [None] * len(local_paths)
        if self._cache is None:
            return [], cached_results, list(range(len(local_paths)))

        file_hashes: list[str | None] = []
        for path in local_paths:
            try:
                file_hashes.append(await asyncio.to_thread(self._hash_file, path))
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
        return file_hashes, cached_results, uncached_positions

    async def _encode_uncached_images(
        self,
        local_paths: list[str],
        file_hashes: list[str | None],
        uncached_positions: list[int],
        cached_results: list[list[float] | None],
    ) -> None:
        """Encode uncached images, update ``cached_results`` in place, and populate cache.

        Args:
            local_paths: Local file paths for downloaded images.
            file_hashes: SHA-256 hex digests aligned with ``local_paths``.
            uncached_positions: Indices in ``local_paths`` missing from cache.
            cached_results: Mutable list of embeddings updated with new results.
        """
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
                return

            to_cache: dict[str, list[float]] = {}
            for j, pos in enumerate(uncached_positions):
                cached_results[pos] = encoded_new[j]
                file_hash = file_hashes[pos] if pos < len(file_hashes) else None
                if self._cache is not None and file_hash is not None:
                    to_cache[file_hash] = encoded_new[j]

            if self._cache is not None and to_cache:
                await self._cache.set_batch(self.model.model_name, to_cache)

    @staticmethod
    def _hash_file(path: str) -> str:
        """Compute SHA-256 hex digest of a file's contents.

        Args:
            path: Filesystem path to the file to hash.

        Returns:
            Hexadecimal SHA-256 digest string.
        """
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
