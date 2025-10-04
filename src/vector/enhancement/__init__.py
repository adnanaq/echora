"""AI Enhancement and Fine-tuning

Advanced AI features for anime-specific improvements including character recognition,
art style classification, and domain-specific fine-tuning.
"""

from .anime_dataset import AnimeDataset
from .art_style_classifier import ArtStyleClassifier
from .character_recognition import CharacterRecognitionFinetuner
from .genre_enhancement import GenreEnhancementFinetuner


# Fine-tuning orchestrator - may have heavy dependencies
def get_anime_fine_tuning():
    """Lazy import for fine-tuning orchestrator."""
    from .anime_fine_tuning import AnimeFineTuning

    return AnimeFineTuning


__all__ = [
    "AnimeDataset",
    "ArtStyleClassifier",
    "CharacterRecognitionFinetuner",
    "GenreEnhancementFinetuner",
    "get_anime_fine_tuning",
]
