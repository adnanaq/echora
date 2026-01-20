"""Vision processor for converting images to vector embeddings.

This module provides the VisionProcessor class which serves as a pure compute
engine for image embedding. It is strictly responsible for converting image
files to embedding vectors using OpenCLIP models, with no domain-specific logic.
"""

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

        except Exception as e:
            logger.exception(f"Image encoding failed: {e}")
            return None

    async def encode_images_batch(
        self, image_urls: list[str]
    ) -> list[list[float]]:
        """Encode multiple images from URLs into a matrix for multivector storage.

        Downloads and encodes each image, returning a matrix (list of vectors)
        suitable for Qdrant's multivector storage with MAX_SIM comparator.

        Args:
            image_urls: List of image URLs to download and encode.

        Returns:
            A list of embedding vectors (matrix). Failed downloads/encodings
            are skipped, so the result may have fewer vectors than input URLs.
        """
        if not image_urls:
            return []

        embeddings_matrix: list[list[float]] = []

        for url in image_urls:
            try:
                # Download image
                local_path = await self.downloader.download_and_cache_image(url)
                if local_path is None:
                    logger.warning(f"Failed to download image: {url}")
                    continue

                # Encode image
                embeddings = self.model.encode_image([local_path])
                if embeddings and embeddings[0]:
                    embeddings_matrix.append(embeddings[0])
                else:
                    logger.warning(f"Failed to encode image: {url}")

            except Exception as e:
                logger.warning(f"Error processing image {url}: {e}")
                continue

        logger.debug(
            f"Encoded {len(embeddings_matrix)}/{len(image_urls)} images successfully"
        )
        return embeddings_matrix

    def _hash_embedding(self, embedding: list[float], precision: int = 4) -> str:
        """Create a hash of an embedding vector for duplicate detection.

        Args:
            embedding: The embedding vector to hash.
            precision: Decimal places to round to before hashing.

        Returns:
            An MD5 hash string of the rounded embedding vector.
        """
        try:
            rounded_embedding = [round(x, precision) for x in embedding]
            embedding_str = ",".join(map(str, rounded_embedding))
            return hashlib.md5(embedding_str.encode()).hexdigest()
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
