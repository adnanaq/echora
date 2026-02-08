import logging
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from common.config import EmbeddingConfig
    from vector_processing.embedding_models.vision.base import VisionEmbeddingModel

logger = logging.getLogger(__name__)


class CCIP:
    """Character Comparison Image Processing for anime character similarity.

    Provides character similarity calculation using dghs-imgutils CCIP algorithm
    with OpenCLIP fallback for graceful degradation when CCIP is unavailable.

    TODO: Convert to async/await pattern to avoid blocking event loop
    - Replace requests.get with httpx.AsyncClient for async HTTP calls
    - Make _load_image_for_ccip, _calculate_openclip_similarity, and
      calculate_character_similarity async methods
    - Add httpx as dependency
    - Reference: https://github.com/adnanaq/echora/pull/27#discussion_r2593563165
    """

    def __init__(self, config: "EmbeddingConfig | None" = None):
        """Initialize CCIP with optional config.

        Args:
            config: Embedding configuration for fallback model creation
        """
        self.config = config
        self._fallback_model: VisionEmbeddingModel | None = None

    def _get_fallback_model(self) -> "VisionEmbeddingModel | None":
        """Lazy-load OpenCLIP model for fallback similarity calculation.

        Note: If self.config is None, this method will initialize it with
        default EmbeddingConfig() as a side-effect (lazy initialization pattern).

        Returns:
            Initialized VisionEmbeddingModel or None if unavailable
        """
        if self._fallback_model is None:
            try:
                from vector_processing.embedding_models.factory import (
                    EmbeddingModelFactory,
                )

                if self.config is None:
                    from common.config import EmbeddingConfig

                    self.config = EmbeddingConfig()

                self._fallback_model = EmbeddingModelFactory.create_vision_model(
                    self.config
                )
                logger.info("OpenCLIP fallback model initialized for CCIP")
            except Exception:
                logger.exception("Failed to initialize OpenCLIP fallback model")
                return None
        return self._fallback_model

    def _load_image_for_ccip(self, url_or_path: str) -> Image.Image | None:
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
        except Exception:
            logger.exception(f"Failed to load image from {url_or_path}")
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
                logger.debug("OpenCLIP fallback model unavailable")
                return 0.0

            # Load images
            img1 = self._load_image_for_ccip(image_url_1)
            img2 = self._load_image_for_ccip(image_url_2)

            if not img1 or not img2:
                logger.debug("Could not load images for OpenCLIP fallback")
                return 0.0

            # Encode both images directly (OpenCLIP accepts PIL Images)
            embeddings = model.encode_image([img1, img2])

            if not embeddings or len(embeddings) < 2:
                logger.warning("Failed to encode images with OpenCLIP")
                return 0.0

            emb1 = np.array(embeddings[0])
            emb2 = np.array(embeddings[1])

            # Calculate cosine similarity
            # Note: OpenCLIP already returns normalized embeddings, so dot product = cosine similarity
            similarity = float(np.dot(emb1, emb2))
            logger.debug(f"OpenCLIP fallback similarity: {similarity}")
            return similarity

        except Exception:
            logger.exception("OpenCLIP fallback similarity calculation failed")
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
