import logging
from typing import Any, cast

from .base import TextEmbeddingModel

logger = logging.getLogger(__name__)


class FastEmbedModel(TextEmbeddingModel):
    """FastEmbed implementation of TextEmbeddingModel."""

    def __init__(self, model_name: str, cache_dir: str | None = None, **kwargs):
        """Initialize FastEmbed model.

        Args:
            model_name: FastEmbed model name
            cache_dir: Optional directory to cache model files
            **kwargs: Additional arguments passed to TextEmbedding
        """
        try:
            from fastembed import TextEmbedding

            self._model_name = model_name

            # Initialize FastEmbed model
            init_kwargs: dict[str, Any] = {"model_name": model_name}
            if cache_dir:
                init_kwargs["cache_dir"] = cache_dir
            init_kwargs.update(kwargs)

            self.model = TextEmbedding(**init_kwargs)

            # Determine embedding size
            self._embedding_size = self._get_fastembed_embedding_size(model_name)

            logger.info(f"Initialized FastEmbed model: {model_name}")

        except ImportError as e:
            logger.exception(
                "FastEmbed not installed. Install with: pip install fastembed"
            )
            raise ImportError("FastEmbed dependencies missing") from e

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode a list of texts into embeddings.

        Args:
            texts: List of text strings to encode

        Returns:
            List of embedding vectors
        """
        try:
            # Generate embeddings
            # FastEmbed returns a generator, convert to list
            embeddings = list(self.model.embed(texts))
            return [cast(list[float], e.tolist()) for e in embeddings]

        except Exception:
            logger.exception("FastEmbed encoding failed")
            # Re-raise to let caller handle encoding failures
            raise

    @property
    def embedding_size(self) -> int:
        return self._embedding_size

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def max_length(self) -> int:
        return 512  # FastEmbed default

    @property
    def supports_multilingual(self) -> bool:
        return (
            "multilingual" in self._model_name.lower()
            or "m3" in self._model_name.lower()
        )

    def _get_fastembed_embedding_size(self, model_name: str) -> int:
        """Get embedding size for FastEmbed model."""
        # Common FastEmbed model dimensions
        model_dimensions = {
            "BAAI/bge-small-en-v1.5": 384,
            "BAAI/bge-base-en-v1.5": 768,
            "BAAI/bge-large-en-v1.5": 1024,
            "BAAI/bge-m3": 1024,
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "sentence-transformers/all-mpnet-base-v2": 768,
            "intfloat/e5-small-v2": 384,
            "intfloat/e5-base-v2": 768,
            "intfloat/e5-large-v2": 1024,
        }
        return model_dimensions.get(model_name, 384)  # Default to 384
