"""Abstract base class for vision embedding models.

Defines the contract that all vision embedding model implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Any

from PIL import Image


class VisionEmbeddingModel(ABC):
    """Abstract base class for vision embedding models.

    All vision embedding models (OpenCLIP, etc.) must implement this interface.
    """

    @abstractmethod
    def encode_image(self, images: list[Image.Image | str]) -> list[list[float]]:
        """Encode a list of images into embeddings.

        Args:
            images: List of PIL Images or image file paths

        Returns:
            List of embedding vectors, one per input image
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
    def input_size(self) -> int:
        """Get the expected input image size for this model.

        Returns:
            Image size in pixels (default: 224)
        """
        return 224

    def get_model_info(self) -> dict[str, Any]:
        """Get comprehensive information about this vision embedding model.

        Returns:
            Dictionary with model metadata
        """
        return {
            "model_name": self.model_name,
            "embedding_size": self.embedding_size,
            "input_size": self.input_size,
        }
