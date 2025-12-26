"""Abstract base class for text embedding models.

Defines the contract that all text embedding model implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Any


class TextEmbeddingModel(ABC):
    """Abstract base class for text embedding models.

    All text embedding models (FastEmbed, HuggingFace, etc.) must implement this interface.
    """

    @abstractmethod
    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode a list of texts into embeddings.

        Args:
            texts: List of text strings to encode

        Returns:
            List of embedding vectors, one per input text
        """
        pass

    @property
    @abstractmethod
    def embedding_size(self) -> int:
        """Get the dimensionality of the embeddings produced by this model.

        Returns:
            Embedding dimension size
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the name/identifier of this model.

        Returns:
            Model name string
        """
        pass

    @property
    def max_length(self) -> int:
        """Get the maximum sequence length supported by this model.

        Returns:
            Maximum sequence length (default: 512)
        """
        return 512

    @property
    def supports_multilingual(self) -> bool:
        """Check if this model supports multilingual text.

        Returns:
            True if multilingual support, False otherwise
        """
        return False

    def get_model_info(self) -> dict[str, Any]:
        """Get comprehensive information about this text embedding model.

        Returns:
            Dictionary with model metadata
        """
        return {
            "model_name": self.model_name,
            "embedding_size": self.embedding_size,
            "max_length": self.max_length,
            "supports_multilingual": self.supports_multilingual,
        }
