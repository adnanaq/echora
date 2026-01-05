"""Vision processor using OpenCLIP for anime image search.

Uses OpenCLIP ViT-L/14 model for high-quality image embeddings
with commercial-friendly licensing.
"""

import hashlib
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from common.config import Settings
from common.models.anime import AnimeEntry

from ..embedding_models.vision.base import VisionEmbeddingModel
from ..utils.image_downloader import ImageDownloader

if TYPE_CHECKING:
    from .anime_field_mapper import AnimeFieldMapper

logger = logging.getLogger(__name__)


class VisionProcessor:
    """Vision processor supporting multiple embedding models."""

    def __init__(
        self,
        model: VisionEmbeddingModel,
        downloader: ImageDownloader,
        field_mapper: "AnimeFieldMapper",
        settings: Settings | None = None,
    ):
        """Initialize modern vision processor with injected dependencies.

        Args:
            model: Initialized VisionEmbeddingModel instance
            downloader: Initialized ImageDownloader instance
            field_mapper: Initialized AnimeFieldMapper instance
            settings: Configuration settings instance
        """
        if settings is None:
            settings = Settings()

        self.settings = settings
        self.model = model
        self.downloader = downloader
        self.field_mapper = field_mapper

        logger.info(f"Initialized VisionProcessor with model: {model.model_name}")

    def encode_image(self, image_path: str) -> list[float] | None:
        """Encode image to embedding vector.

        Args:
            image_path: Path to image file

        Returns:
            Embedding vector or None if encoding fails
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

    def _hash_embedding(self, embedding: list[float], precision: int = 4) -> str:
        """Create hash of embedding vector to detect duplicates.

        Args:
            embedding: Embedding vector
            precision: Decimal precision for hash (default 4 for similarity detection)

        Returns:
            Hash string of the embedding
        """
        try:
            # Round to specified precision to catch near-identical embeddings
            rounded_embedding = [round(x, precision) for x in embedding]
            # Create hash from rounded values
            embedding_str = ",".join(map(str, rounded_embedding))
            return hashlib.md5(embedding_str.encode()).hexdigest()
        except Exception:
            # Fallback to string representation
            return str(hash(tuple(embedding)))

    async def process_anime_image_vector(self, anime: AnimeEntry) -> list[float] | None:
        """Process general anime images (covers, posters, banners, trailers) excluding character images.

        Args:
            anime: AnimeEntry instance with image data

        Returns:
            Combined general image embedding vector or None if processing fails
        """
        try:
            # Extract all image URLs from anime data
            image_urls = self.field_mapper._extract_image_content(anime)

            if not image_urls:
                logger.warning("No image URLs found for anime")
                return None

            logger.debug(
                f"Processing {len(image_urls)} general images for anime (excluding character images)"
            )

            # Process all images with duplicate vector detection
            successful_embeddings = []
            processed_vectors = set()  # Store vector hashes to detect duplicates

            for i, image_url in enumerate(image_urls):
                try:
                    logger.debug(
                        f"Processing image {i + 1}/{len(image_urls)}: {image_url}"
                    )

                    # Download and cache image using injected downloader
                    # Returns path to cached file
                    image_path = await self.downloader.download_and_cache_image(
                        image_url
                    )

                    logger.debug(f"Downloader returned path: {image_path}")

                    if image_path:
                        # Encode using path
                        logger.debug(f"Encoding image from path: {image_path}")
                        embedding = self.encode_image(image_path)
                        logger.debug(
                            f"Encoding result: {embedding is not None}, length: {len(embedding) if embedding else 0}"
                        )

                        if embedding:
                            # Create hash of embedding to check for duplicates
                            embedding_hash = self._hash_embedding(embedding)

                            if embedding_hash not in processed_vectors:
                                successful_embeddings.append(embedding)
                                processed_vectors.add(embedding_hash)
                                logger.debug(
                                    f"Successfully encoded unique image {i + 1}/{len(image_urls)}"
                                )
                            else:
                                logger.debug(
                                    f"Skipped duplicate image {i + 1}/{len(image_urls)}"
                                )
                        else:
                            logger.warning(f"Encoding returned None for image {i + 1}")
                    else:
                        logger.warning(
                            f"Downloader returned None for image {i + 1}: {image_url}"
                        )
                except Exception as e:
                    logger.error(f"Failed to process image {i + 1}: {e}", exc_info=True)
                    continue

            if successful_embeddings:
                # Combine multiple embeddings by averaging (preserves semantic information)
                if len(successful_embeddings) == 1:
                    logger.debug("Single unique image processed")
                    return successful_embeddings[0]
                else:
                    # Average multiple embeddings for comprehensive visual representation
                    combined_embedding: list[float] = np.mean(
                        successful_embeddings, axis=0
                    ).tolist()
                    logger.debug(
                        f"Combined {len(successful_embeddings)} unique image embeddings from {len(image_urls)} total images"
                    )
                    return combined_embedding

            # Fallback: return None to store URLs in payload instead
            logger.info(
                "All general image processing failed, URLs will be stored in payload"
            )
            return None

        except Exception as e:
            logger.exception(f"General image vector processing failed: {e}")
            return None

    async def process_anime_character_image_vector(
        self, anime: AnimeEntry
    ) -> list[float] | None:
        """Process character images from anime data for character identification and recommendations.

        Args:
            anime: AnimeEntry instance with character image data

        Returns:
            Combined character image embedding vector or None if processing fails
        """
        try:
            # Extract character image URLs from anime data (separate from general images)
            character_image_urls = self.field_mapper._extract_character_image_content(
                anime
            )

            if not character_image_urls:
                logger.debug("No character image URLs found for anime")
                return None

            logger.debug(
                f"Processing {len(character_image_urls)} character images for anime"
            )

            # Process character images with duplicate vector detection
            successful_embeddings = []
            processed_vectors = set()  # Store vector hashes to detect duplicates

            for i, image_url in enumerate(character_image_urls):
                try:
                    # Download and cache image using injected downloader
                    image_path = await self.downloader.download_and_cache_image(
                        image_url
                    )

                    if image_path:
                        embedding = self.encode_image(image_path)
                        if embedding:
                            # Create hash of embedding to check for duplicates
                            embedding_hash = self._hash_embedding(embedding)

                            if embedding_hash not in processed_vectors:
                                successful_embeddings.append(embedding)
                                processed_vectors.add(embedding_hash)
                                logger.debug(
                                    f"Successfully encoded unique character image {i + 1}/{len(character_image_urls)}"
                                )
                            else:
                                logger.debug(
                                    f"Skipped duplicate character image {i + 1}/{len(character_image_urls)}"
                                )
                except Exception as e:
                    logger.warning(f"Failed to process character image {i + 1}: {e}")
                    continue

            if successful_embeddings:
                # Combine multiple embeddings by averaging (preserves character identification features)
                if len(successful_embeddings) == 1:
                    logger.debug("Single unique character image processed")
                    return successful_embeddings[0]
                else:
                    # Average multiple embeddings for comprehensive character visual representation
                    combined_embedding: list[float] = np.mean(
                        successful_embeddings, axis=0
                    ).tolist()
                    logger.debug(
                        f"Combined {len(successful_embeddings)} unique character image embeddings from {len(character_image_urls)} total character images"
                    )
                    return combined_embedding

            # Fallback: return None to store character image URLs in payload instead
            logger.debug(
                "All character image processing failed, URLs will be stored in payload"
            )
            return None

        except Exception as e:
            logger.exception(f"Character image vector processing failed: {e}")
            return None

    def get_cache_stats(self) -> dict[str, Any]:
        """Get image cache statistics.

        Delegates to ImageDownloader for cache management.

        Returns:
            Dictionary with cache statistics
        """
        return self.downloader.get_cache_stats()

    def clear_cache(self, max_age_days: int | None = None) -> dict[str, Any]:
        """Clear image cache.

        Delegates to ImageDownloader for cache management.

        Args:
            max_age_days: Only clear files older than this many days (optional)

        Returns:
            Dictionary with cleanup statistics
        """
        return self.downloader.clear_cache(max_age_days)

    def get_supported_formats(self) -> list[str]:
        """Get list of supported image formats.

        Returns:
            List of supported image format extensions
        """
        return ["jpg", "jpeg", "png", "bmp", "tiff", "webp", "gif"]

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the current vision embedding model.

        Returns:
            Dictionary with model information
        """
        return self.model.get_model_info()
