"""Text embedding processor for converting text to vector embeddings.

This module provides the TextProcessor class which serves as a pure compute
engine for text embedding. It is strictly responsible for converting strings
to embedding vectors using configured ML models, with no domain-specific logic.
"""

import logging
from typing import Any, cast

from common.config import Settings

from ..embedding_models.text.base import TextEmbeddingModel

logger = logging.getLogger(__name__)


class TextProcessor:
    """Pure text embedding processor with no domain-specific logic.

    This class serves as the "Text Compute Engine" in the vector processing
    pipeline. It knows HOW to turn text into numbers but has no knowledge
    of anime, characters, or any domain concepts.

    Responsibilities:
        - Interface with TextEmbeddingModels (FastEmbed, BGE, etc.).
        - Handle batching and list processing for text.
        - Provide zero-vectors for empty content.
    """

    def __init__(
        self,
        model: TextEmbeddingModel,
        settings: Settings | None = None,
    ):
        """Initialize the text processor with an embedding model.

        Args:
            model: An initialized TextEmbeddingModel instance.
            settings: Configuration settings instance. Uses defaults if None.
        """
        if settings is None:
            settings = Settings()

        self.settings = settings
        self.model = model

        logger.info(f"Initialized TextProcessor with model: {model.model_name}")

    def encode_text(self, text: str) -> list[float] | None:
        """Encode a single string to an embedding vector.

        Args:
            text: The text string to encode.

        Returns:
            A list of floats representing the embedding vector, a zero vector
            if the input is empty/whitespace, or None if encoding fails.
        """
        try:
            if not text or not text.strip():
                return self.get_zero_embedding()

            embeddings = self.model.encode([text])
            if embeddings:
                return embeddings[0]
            return None

        except Exception as e:
            logger.error(f"Text encoding failed: {e}")
            return None

    def encode_texts_batch(self, texts: list[str]) -> list[list[float] | None]:
        """Encode multiple texts in a single batch call.

        More efficient than calling encode_text repeatedly as it leverages
        the model's batch processing capabilities.

        Args:
            texts: List of text strings to encode.

        Returns:
            A list of embedding vectors (or None for failed encodings),
            in the same order as the input texts.
        """
        try:
            return cast(list[list[float] | None], self.model.encode(texts))
        except Exception as e:
            logger.exception(f"Batch text encoding failed: {e}")
            return [None] * len(texts)

    def get_zero_embedding(self) -> list[float]:
        """Get a zero embedding vector matching model dimensions.

        Returns:
            A list of zeros with length equal to the model's embedding size.
        """
        return [0.0] * self.model.embedding_size

    def get_model_info(self) -> dict[str, Any]:
        """Get metadata about the underlying embedding model.

        Returns:
            A dictionary containing model information such as name,
            embedding size, and other model-specific details.
        """
        return self.model.get_model_info()
