"""Vector processing library for embeddings and multi-vector generation."""

from vector_processing.processors.text_processor import TextProcessor
from vector_processing.processors.vision_processor import VisionProcessor
from vector_processing.processors.embedding_manager import MultiVectorEmbeddingManager
from vector_processing.processors.anime_field_mapper import AnimeFieldMapper

__all__ = [
    "TextProcessor",
    "VisionProcessor",
    "MultiVectorEmbeddingManager",
    "AnimeFieldMapper",
]
