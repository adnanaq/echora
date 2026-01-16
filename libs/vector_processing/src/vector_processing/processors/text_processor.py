"""Text embedding processor supporting multiple models for anime text search.

Supports FastEmbed, HuggingFace, and BGE models with dynamic model selection
for optimal performance.
"""

import logging
from typing import Any, cast

from common.config import Settings

from ..embedding_models.text.base import TextEmbeddingModel

logger = logging.getLogger(__name__)


class TextProcessor:
    """Text embedding processor supporting multiple models."""

    def __init__(
        self,
        model: TextEmbeddingModel,
        settings: Settings | None = None,
    ):
        """Initialize modern text processor with injected dependencies.

        Args:
            model: Initialized TextEmbeddingModel instance
            settings: Configuration settings instance
        """
        if settings is None:
            settings = Settings()

        self.settings = settings
        self.model = model

        logger.info(f"Initialized TextProcessor with model: {model.model_name}")

    def encode_text(self, text: str) -> list[float] | None:
        """Encode text to embedding vector.

        Args:
            text: Input text string

        Returns:
            Embedding vector or None if encoding fails
        """
        try:
            # Handle empty text
            if not text.strip():
                return self._get_zero_embedding()

            # Encode with model
            embeddings = self.model.encode([text])
            if embeddings:
                return embeddings[0]
            return None

        except Exception as e:
            logger.error(f"Text encoding failed: {e}")
            return None

    def encode_texts_batch(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[list[float] | None]:
        """Encode multiple texts in batches.

        Args:
            texts: List of text strings
            batch_size: Batch size (ignored, handled by model or caller)

        Returns:
            List of embedding vectors
        """
        try:
            # Simple delegation to model which handles batching or list processing
            # The base model.encode takes a list and returns a list
            return cast(list[list[float] | None], self.model.encode(texts))

        except Exception as e:
            logger.error(f"Batch text encoding failed: {e}")
            return [None] * len(texts)

    def validate_text(self, text: str) -> bool:
        """Validate if text can be processed.

        Args:
            text: Input text string

        Returns:
            True if text is valid, False otherwise
        """
        try:
            if not isinstance(text, str):
                return False

            # Check length
            if len(text) > self.model.max_length * 4:  # Rough token estimate
                return False

            return True

        except Exception:
            return False

    def _get_zero_embedding(self) -> list[float]:
        """Get zero embedding vector for empty/failed content."""
        return [0.0] * self.model.embedding_size

    def get_text_vector_names(self) -> list[str]:
        """Get list of text vector names supported by this processor."""
        return list(self.settings.vector_names.keys())

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the current text embedding model.

        Returns:
            Dictionary with model information
        """
        return self.model.get_model_info()
