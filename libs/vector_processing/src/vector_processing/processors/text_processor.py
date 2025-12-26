"""Text embedding processor supporting multiple models for anime text search.

Supports FastEmbed, HuggingFace, and BGE models with dynamic model selection
for optimal performance.
"""

import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from common.config import Settings
from common.models.anime import AnimeEntry

from ..embedding_models.text.base import TextEmbeddingModel

if TYPE_CHECKING:
    from .anime_field_mapper import AnimeFieldMapper

logger = logging.getLogger(__name__)


class TextProcessor:
    """Text embedding processor supporting multiple models."""

    def __init__(self, model: TextEmbeddingModel, settings: Settings | None = None):
        """Initialize modern text processor with injected model.

        Args:
            model: Initialized TextEmbeddingModel instance
            settings: Configuration settings instance
        """
        if settings is None:
            settings = Settings()

        self.settings = settings
        self.model = model

        # Initialize field mapper for multi-vector processing
        self._field_mapper: AnimeFieldMapper | None = None

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

    # ============================================================================
    # MULTI-VECTOR PROCESSING METHODS
    # ============================================================================

    def _get_field_mapper(self) -> "AnimeFieldMapper":
        """Lazy initialization of field mapper."""
        if self._field_mapper is None:
            from .anime_field_mapper import AnimeFieldMapper

            self._field_mapper = AnimeFieldMapper()
        return self._field_mapper

    def process_anime_vectors(self, anime: AnimeEntry) -> dict[str, list[float]]:
        """
        Process anime data into multiple semantic text embeddings.

        Args:
            anime: AnimeEntry with comprehensive anime data

        Returns:
            Dict mapping vector names to their embeddings
        """
        try:
            field_mapper = self._get_field_mapper()

            # Extract field content for all vectors
            vector_data = field_mapper.map_anime_to_vectors(anime)

            # Generate embeddings for text vectors only
            text_embeddings = {}
            text_vectors = [
                name
                for name, vec_type in field_mapper.get_vector_types().items()
                if vec_type == "text"
            ]

            for vector_name in text_vectors:
                if vector_name in vector_data:
                    text_content = vector_data[vector_name]

                    # Convert to string if it's a list
                    if isinstance(text_content, list):
                        content_str = " ".join(str(item) for item in text_content)
                    else:
                        content_str = str(text_content)

                    # Apply field-specific preprocessing
                    processed_text = self._preprocess_field_content(
                        content_str, vector_name
                    )

                    # Generate embedding with hierarchical averaging for episode chunks
                    if processed_text.strip():
                        if (
                            vector_name == "episode_vector"
                            and "|| CHUNK_SEPARATOR ||" in processed_text
                        ):
                            # Handle hierarchical averaging for episode chunks
                            embedding = self._encode_with_hierarchical_averaging(
                                processed_text
                            )
                        else:
                            # Standard single embedding
                            embedding = self.encode_text(processed_text)

                        if embedding:
                            text_embeddings[vector_name] = embedding
                        else:
                            # Use zero vector for failed embedding
                            text_embeddings[vector_name] = self._get_zero_embedding()
                    else:
                        # Use zero vector for empty content
                        text_embeddings[vector_name] = self._get_zero_embedding()

            logger.debug(
                f"Generated embeddings for {len(text_embeddings)} text vectors"
            )
            return text_embeddings

        except Exception as e:
            logger.error(f"Failed to process anime vectors: {e}")
            raise

    def _encode_with_hierarchical_averaging(
        self, chunked_text: str
    ) -> list[float] | None:
        """
        Encode text with hierarchical averaging for episode chunks.

        Args:
            chunked_text: Text with "|| CHUNK_SEPARATOR ||" delimiters

        Returns:
            Single averaged embedding vector or None if encoding fails
        """
        try:
            # Split text into chunks
            chunks = [
                chunk.strip() for chunk in chunked_text.split("|| CHUNK_SEPARATOR ||")
            ]
            chunks = [chunk for chunk in chunks if chunk]  # Remove empty chunks

            if not chunks:
                return self._get_zero_embedding()

            # For single chunk, encode directly (no averaging needed)
            if len(chunks) == 1:
                return self.encode_text(chunks[0])

            logger.debug(
                f"Processing {len(chunks)} episode chunks with hierarchical averaging"
            )

            # Encode each chunk individually
            chunk_embeddings = []
            for i, chunk in enumerate(chunks):
                chunk_embedding = self.encode_text(chunk)
                if chunk_embedding:
                    chunk_embeddings.append(chunk_embedding)
                else:
                    logger.warning(
                        f"Failed to encode episode chunk {i + 1}/{len(chunks)}"
                    )

            if not chunk_embeddings:
                logger.error("All episode chunks failed to encode")
                return self._get_zero_embedding()

            # Hierarchical averaging: equal weight for all chunks
            # This preserves semantic coherence better than weighted averaging for BGE-M3

            # Convert to numpy for efficient averaging
            chunk_matrix = np.array(chunk_embeddings)
            averaged_embedding = np.mean(chunk_matrix, axis=0)

            # Convert back to list with proper typing
            result_embedding: list[float] = cast(
                list[float], averaged_embedding.tolist()
            )

            logger.debug(
                f"Successfully averaged {len(chunk_embeddings)} episode chunks"
            )
            return result_embedding

        except Exception as e:
            logger.error(f"Hierarchical averaging failed: {e}")
            return self._get_zero_embedding()

    def _preprocess_field_content(self, content: str, vector_name: str) -> str:
        """
        Apply field-specific preprocessing to improve embedding quality.

        Args:
            content: Raw text content from field mapper
            vector_name: Name of the vector being processed

        Returns:
            Preprocessed text optimized for embedding
        """
        if not content:
            return ""

        # Apply general preprocessing
        processed = content.strip()

        # Field-specific preprocessing rules
        if vector_name == "title_vector":
            processed = processed.replace("Synopsis:", "Story:")
            processed = processed.replace("Background:", "Production:")

        elif vector_name == "character_vector":
            processed = processed.replace("Role:", "Character Role:")
            processed = processed.replace("Description:", "Background:")

        elif vector_name == "genre_vector":
            processed = processed.replace("Shounen", "Shonen (young male)")
            processed = processed.replace("Shoujo", "Shojo (young female)")
            processed = processed.replace("Seinen", "Seinen (adult male)")
            processed = processed.replace("Josei", "Josei (adult female)")

        elif vector_name == "sources_vector":
            processed = processed.replace("Source:", "Platform:")
            processed = processed.replace("External:", "External Platform:")

        return processed

    def _get_zero_embedding(self) -> list[float]:
        """Get zero embedding vector for empty/failed content."""
        return [0.0] * self.model.embedding_size

    def get_text_vector_names(self) -> list[str]:
        """Get list of text vector names supported by this processor."""
        field_mapper = self._get_field_mapper()
        return [
            name
            for name, vec_type in field_mapper.get_vector_types().items()
            if vec_type == "text"
        ]

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the current text embedding model.

        Returns:
            Dictionary with model information
        """
        return self.model.get_model_info()
