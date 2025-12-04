"""Vector Processing Module for Anime Search

This module provides a comprehensive vector processing pipeline for anime content,
including text and image embedding generation, database operations, and AI enhancements.

Architecture:
- client/: Database operations and search infrastructure
- processors/: Core embedding generation (text, vision, multi-vector coordination)
- providers/: Model provider abstractions
"""

# Core database client
from .client.qdrant_client import QdrantClient
from .processors.anime_field_mapper import AnimeFieldMapper
from .processors.embedding_manager import MultiVectorEmbeddingManager

# Core processors
from .processors.text_processor import TextProcessor
from .processors.vision_processor import VisionProcessor


# Public API
__all__ = [
    # Core infrastructure
    "QdrantClient",
    # Processors
    "TextProcessor",
    "VisionProcessor",
    "MultiVectorEmbeddingManager",
    "AnimeFieldMapper",
]
