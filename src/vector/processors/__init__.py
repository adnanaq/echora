"""Core Embedding Processors

Text, vision, and multi-vector embedding generation with field mapping.
"""

from .anime_field_mapper import AnimeFieldMapper
from .embedding_manager import MultiVectorEmbeddingManager
from .text_processor import TextProcessor
from .vision_processor import VisionProcessor

__all__ = [
    "TextProcessor",
    "VisionProcessor",
    "MultiVectorEmbeddingManager",
    "AnimeFieldMapper",
]
