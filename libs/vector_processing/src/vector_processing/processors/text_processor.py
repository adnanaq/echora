"""Text embedding processor for converting text to vector embeddings.

This module provides the TextProcessor class which serves as a pure compute
engine for text embedding. It is strictly responsible for converting strings
to embedding vectors using configured ML models, with no domain-specific logic.
"""

import asyncio
import logging
from typing import Any, cast

from common.config import EmbeddingConfig

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
        config: EmbeddingConfig | None = None,
    ):
        """Initialize the text processor with an embedding model.

        Args:
            model: An initialized TextEmbeddingModel instance.
            config: Embedding configuration instance. Uses defaults if None.
        """
        if config is None:
            config = EmbeddingConfig()

        self.config = config
        self.model = model
        self._semaphore = asyncio.Semaphore(config.embed_max_concurrency)

        logger.info(f"Initialized TextProcessor with model: {model.model_name}")

    async def encode_text(self, text: str) -> list[float] | None:
        """Encode a single string to an embedding vector.

        Runs the model inference in a thread to avoid blocking the event loop,
        with semaphore-based concurrency control.

        Args:
            text: The text string to encode.

        Returns:
            A list of floats representing the embedding vector, a zero vector
            if the input is empty/whitespace, or None if encoding fails.
        """
        if not text or not text.strip():
            return self.get_zero_embedding()

        try:
            async with self._semaphore:
                embeddings = await asyncio.to_thread(self.model.encode, [text])
        except Exception:
            logger.exception("Text encoding failed")
            return None
        else:
            if not embeddings:
                return None
            return embeddings[0]

    async def encode_texts_batch(self, texts: list[str]) -> list[list[float] | None]:
        """Encode multiple texts in a single batch call.

        More efficient than calling encode_text repeatedly as it leverages
        the model's batch processing capabilities. Pre-processing runs on
        the event loop; the model call runs in a thread with semaphore control.

        Args:
            texts: List of text strings to encode.

        Returns:
            A list of embedding vectors (or None for failed encodings),
            in the same order as the input texts. Empty or whitespace-only
            strings return zero vectors to maintain consistency with encode_text.
        """
        # Pre-process texts to identify empty/whitespace entries
        zero_embedding = self.get_zero_embedding()
        valid_indices = []
        valid_texts = []

        for i, text in enumerate(texts):
            if text and text.strip():
                valid_indices.append(i)
                valid_texts.append(text)

        # If all texts are empty, return independent zero vectors for all
        if not valid_texts:
            return [zero_embedding.copy() for _ in texts]

        # Encode only valid texts (CPU/GPU-bound — run in thread)
        try:
            async with self._semaphore:
                encoded_valid = await asyncio.to_thread(self.model.encode, valid_texts)
        except Exception:
            logger.exception("Batch text encoding failed")
            return [None] * len(texts)

        # Reconstruct result list with zero vectors for empty inputs
        result: list[list[float] | None] = []
        valid_idx = 0
        valid_index_set = set(valid_indices)

        for i in range(len(texts)):
            if i in valid_index_set:
                result.append(cast(list[float] | None, encoded_valid[valid_idx]))
                valid_idx += 1
            else:
                result.append(zero_embedding.copy())

        return result

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
