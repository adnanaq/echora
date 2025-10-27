"""Vector Processing Module for Anime Search

This module provides a comprehensive vector processing pipeline for anime content,
including text and image embedding generation, database operations, and AI enhancements.

Architecture:
- client/: Database operations and search infrastructure
- processors/: Core embedding generation (text, vision, multi-vector coordination)
- enhancement/: AI-powered improvements and fine-tuning capabilities
- providers/: Model provider abstractions
"""

from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from .enhancement.anime_dataset import AnimeDataset
    from .enhancement.anime_fine_tuning import AnimeFineTuner
    from .enhancement.art_style_classifier import ArtStyleClassifier
    from .enhancement.character_recognition import CharacterRecognitionFinetuner
    from .enhancement.genre_enhancement import GenreEnhancementFinetuner

# Core database client
from .client.qdrant_client import QdrantClient
from .processors.anime_field_mapper import AnimeFieldMapper
from .processors.embedding_manager import MultiVectorEmbeddingManager

# Core processors
from .processors.text_processor import TextProcessor
from .processors.vision_processor import VisionProcessor


# Enhancement modules (lazy imports to avoid heavy dependencies)
def get_art_style_classifier() -> Type["ArtStyleClassifier"]:
    """Lazy import for ArtStyleClassifier to avoid loading heavy ML dependencies unless needed."""
    from .enhancement.art_style_classifier import ArtStyleClassifier

    return ArtStyleClassifier


def get_character_recognition() -> Type["CharacterRecognitionFinetuner"]:
    """Lazy import for CharacterRecognitionFinetuner."""
    from .enhancement.character_recognition import CharacterRecognitionFinetuner

    return CharacterRecognitionFinetuner


def get_genre_enhancement() -> Type["GenreEnhancementFinetuner"]:
    """Lazy import for GenreEnhancementFinetuner."""
    from .enhancement.genre_enhancement import GenreEnhancementFinetuner

    return GenreEnhancementFinetuner


def get_anime_dataset() -> Type["AnimeDataset"]:
    """Lazy import for AnimeDataset."""
    from .enhancement.anime_dataset import AnimeDataset

    return AnimeDataset


def get_anime_fine_tuning() -> Type["AnimeFineTuner"]:
    """Lazy import for anime fine-tuning orchestrator."""
    from .enhancement.anime_fine_tuning import AnimeFineTuner

    return AnimeFineTuner


# Public API
__all__ = [
    # Core infrastructure
    "QdrantClient",
    # Processors
    "TextProcessor",
    "VisionProcessor",
    "MultiVectorEmbeddingManager",
    "AnimeFieldMapper",
    # Enhancement (via lazy loaders)
    "get_art_style_classifier",
    "get_character_recognition",
    "get_genre_enhancement",
    "get_anime_dataset",
    "get_anime_fine_tuning",
]
