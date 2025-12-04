import logging
from typing import Optional, TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from common.config import Settings
    from .embedding_models.vision.base import VisionEmbeddingModel

logger = logging.getLogger(__name__)


class LegacyCCIPS:
    """Legacy CCIPS logic extracted from VisionProcessor.

    This class preserves the character similarity calculation logic
    until it can be properly migrated to libs/enrichment.

    Includes OpenCLIP fallback for graceful degradation when CCIP unavailable.
    """

    def __init__(self, settings: Optional["Settings"] = None):
        """Initialize LegacyCCIPS with optional settings.

        Args:
            settings: Configuration settings for fallback model creation
        """
        self.settings = settings
        self._fallback_model: Optional["VisionEmbeddingModel"] = None

    def _get_fallback_model(self) -> Optional["VisionEmbeddingModel"]:
        """Lazy-load OpenCLIP model for fallback similarity calculation.

        Returns:
            Initialized VisionEmbeddingModel or None if unavailable
        """
        if self._fallback_model is None:
            try:
                from .embedding_models.factory import EmbeddingModelFactory

                if self.settings is None:
                    from common.config import Settings
                    self.settings = Settings()

                self._fallback_model = EmbeddingModelFactory.create_vision_model(
                    self.settings
                )
                logger.info("OpenCLIP fallback model initialized for CCIP")
            except Exception as e:
                logger.error(f"Failed to initialize OpenCLIP fallback model: {e}")
                return None
        return self._fallback_model

    def _load_image_for_ccip(self, url_or_path: str) -> Optional[Image.Image]:
        """Load image for CCIP, handling both URLs and local paths."""
        try:
            if url_or_path.startswith("http"):
                # Use requests for synchronous download
                from io import BytesIO

                import requests

                response = requests.get(url_or_path, timeout=10)
                response.raise_for_status()
                return Image.open(BytesIO(response.content))
            else:
                # Handle local file path
                return Image.open(url_or_path)
        except Exception as e:
            logger.error(f"Failed to load image from {url_or_path}: {e}")
            return None

    def _calculate_openclip_similarity(
        self, image_url_1: str, image_url_2: str
    ) -> float:
        """Fallback: Calculate similarity using OpenCLIP embeddings.

        Args:
            image_url_1: URL or path to first image
            image_url_2: URL or path to second image

        Returns:
            Cosine similarity score (0.0 - 1.0)
        """
        try:
            # Get fallback model
            model = self._get_fallback_model()
            if not model:
                logger.warning("OpenCLIP fallback model unavailable")
                return 0.0

            # Load images
            img1 = self._load_image_for_ccip(image_url_1)
            img2 = self._load_image_for_ccip(image_url_2)

            if not img1 or not img2:
                logger.warning("Could not load images for OpenCLIP fallback")
                return 0.0

            # Encode both images (model expects list of paths, but we have PIL Images)
            # Save images temporarily to encode them
            import tempfile
            from pathlib import Path

            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                img1_path = tmpdir_path / "img1.jpg"
                img2_path = tmpdir_path / "img2.jpg"

                img1.save(img1_path, "JPEG")
                img2.save(img2_path, "JPEG")

                # Encode images
                embeddings = model.encode_image([str(img1_path), str(img2_path)])

                if not embeddings or len(embeddings) < 2:
                    logger.warning("Failed to encode images with OpenCLIP")
                    return 0.0

                emb1 = np.array(embeddings[0])
                emb2 = np.array(embeddings[1])

            # Normalize and calculate cosine similarity
            emb1 = emb1 / np.linalg.norm(emb1)
            emb2 = emb2 / np.linalg.norm(emb2)

            similarity = float(np.dot(emb1, emb2))
            logger.debug(f"OpenCLIP fallback similarity: {similarity}")
            return similarity

        except Exception as e:
            logger.error(f"OpenCLIP fallback similarity calculation failed: {e}")
            return 0.0

    def calculate_character_similarity(
        self, image_url_1: str, image_url_2: str
    ) -> float:
        """Calculate character similarity using CCIP (anime-specialized model).

        Uses DeepGHS CCIP model for anime character similarity matching.
        Falls back to OpenCLIP embeddings if CCIP unavailable or fails.
        Returns similarity score (0.0 - 1.0, higher = more similar).

        Args:
            image_url_1: URL or path to first character image
            image_url_2: URL or path to second character image

        Returns:
            Similarity score (1.0 = identical, 0.0 = completely different)
        """
        try:
            from imgutils.metrics import ccip_difference

            # Load images, whether from URL or local path
            img1 = self._load_image_for_ccip(image_url_1)
            img2 = self._load_image_for_ccip(image_url_2)

            if not img1 or not img2:
                logger.warning(
                    "Could not load one or both images for CCIP, trying OpenCLIP fallback"
                )
                return self._calculate_openclip_similarity(image_url_1, image_url_2)

            # CCIP returns difference (0 = identical, 1 = different)
            difference = ccip_difference(img1, img2)

            # Convert to similarity
            similarity = 1.0 - difference

            return float(similarity)

        except Exception as e:
            logger.warning(
                f"CCIP character similarity calculation failed: {e}, trying OpenCLIP fallback"
            )
            return self._calculate_openclip_similarity(image_url_1, image_url_2)
