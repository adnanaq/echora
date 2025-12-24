"""Embedding models for text and vision processing.

This module provides abstract base classes and concrete implementations for
embedding models, following the factory pattern for model creation.
"""

from vector_processing.embedding_models.factory import EmbeddingModelFactory
from vector_processing.embedding_models.text.base import TextEmbeddingModel
from vector_processing.embedding_models.vision.base import VisionEmbeddingModel

__all__ = [
    "TextEmbeddingModel",
    "VisionEmbeddingModel",
    "EmbeddingModelFactory",
]
