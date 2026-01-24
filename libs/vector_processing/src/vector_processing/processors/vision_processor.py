"""Vision processor for converting images to vector embeddings.

This module provides the VisionProcessor class which serves as a pure compute
engine for image embedding. It is strictly responsible for converting image
files to embedding vectors using OpenCLIP models, with no domain-specific logic.
"""

import asyncio
import hashlib
import logging
from typing import Any

from common.config import Settings

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
        settings: Settings | None = None,
    ):
        """Initialize the vision processor with model and downloader.

        Args:
            model: An initialized VisionEmbeddingModel instance.
            downloader: An initialized ImageDownloader for fetching images.
            settings: Configuration settings instance. Uses defaults if None.
        """
        if settings is None:
            settings = Settings()

        self.settings = settings
        self.model = model
        self.downloader = downloader

        logger.info(f"Initialized VisionProcessor with model: {model.model_name}")

    def encode_image(self, image_path: str) -> list[float] | None:
        """Encode a local image file to an embedding vector.

        Args:
            image_path: Absolute path to the local image file.

        Returns:
            A list of floats representing the image embedding vector,
            or None if encoding fails.
        """
        try:
            # Encode with model
            embeddings = self.model.encode_image([image_path])
            if embeddings:
                return embeddings[0]
            return None

        except Exception:
            logger.exception("Image encoding failed")
            return None

    async def encode_images_batch(self, image_urls: list[str]) -> list[list[float]]:
        """Encode multiple images from URLs into a matrix for multivector storage.

        Downloads images concurrently and encodes them in a single batch for
        optimal performance. Uses asyncio.gather for parallel I/O and leverages
        the model's native batch processing capabilities.

        Args:
            image_urls: List of image URLs to download and encode.

        Returns:
            A list of embedding vectors (matrix). Failed downloads/encodings
            are skipped, so the result may have fewer vectors than input URLs.
        """
        if not image_urls:
            return []

        # Concurrent downloads (I/O-bound - benefits from async)
        async def download_single(url: str) -> tuple[str, str | None]:
            try:
                local_path = await self.downloader.download_and_cache_image(url)
                return (url, local_path)
            except Exception as e:
                logger.warning(f"Error downloading image {url}: {e}")
                return (url, None)

        download_results = await asyncio.gather(
            *[download_single(url) for url in image_urls]
        )

        # Filter successful downloads
        local_paths = [path for _, path in download_results if path is not None]

        if not local_paths:
            logger.debug("No images downloaded successfully")
            return []

        # Single batch encoding (CPU/GPU-bound - leverage model's batch processing)
        try:
            embeddings = self.model.encode_image(local_paths)
            logger.debug(
                f"Encoded {len(embeddings)}/{len(image_urls)} images successfully"
            )
            return embeddings
        except Exception as e:
            logger.warning(f"Batch encoding failed: {e}")
            return []

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
