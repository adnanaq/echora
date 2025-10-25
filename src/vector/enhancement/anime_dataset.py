"""Anime dataset preparation pipeline for domain-specific fine-tuning.

This module creates training datasets from anime data for character recognition,
art style classification, and genre understanding tasks.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from ..processors.text_processor import TextProcessor
from ..processors.vision_processor import VisionProcessor

logger = logging.getLogger(__name__)


@dataclass
class DataSample:
    """Single training sample for anime fine-tuning."""

    anime_id: str
    title: str
    text: str
    image_data: Optional[str] = None

    # Labels for different tasks
    character_labels: Optional[List[str]] = None
    art_style_label: Optional[str] = None
    genre_labels: Optional[List[str]] = None

    # Additional metadata
    studio: Optional[str] = None
    year: Optional[int] = None
    type: Optional[str] = None
    tags: Optional[List[str]] = None


class AnimeDataset(Dataset):
    """PyTorch dataset for anime fine-tuning."""

    def __init__(
        self,
        anime_data: List[Dict[str, Any]],
        text_processor: TextProcessor,
        vision_processor: VisionProcessor,
        config: Any,
        augment_data: bool = True,
    ):
        """Initialize anime dataset.

        Args:
            anime_data: List of anime entries
            text_processor: Text embedding processor
            vision_processor: Vision embedding processor
            config: Fine-tuning configuration
            augment_data: Whether to augment training data
        """
        self.anime_data = anime_data
        self.text_processor = text_processor
        self.vision_processor = vision_processor
        self.config = config
        self.augment_data = augment_data

        # Process data into samples
        self.samples = self._prepare_samples()

        # Create label mappings
        self.character_vocab = self._build_character_vocab()
        self.art_style_vocab = self._build_art_style_vocab()
        self.genre_vocab = self._build_genre_vocab()

        logger.info(f"Dataset prepared with {len(self.samples)} samples")
        logger.info(f"Character vocabulary size: {len(self.character_vocab)}")
        logger.info(f"Art style vocabulary size: {len(self.art_style_vocab)}")
        logger.info(f"Genre vocabulary size: {len(self.genre_vocab)}")

    def _prepare_samples(self) -> List[DataSample]:
        """Prepare training samples from anime data.

        Returns:
            List of prepared data samples
        """
        samples = []

        for anime in self.anime_data:
            try:
                # Create base sample
                sample = self._create_base_sample(anime)
                if sample:
                    samples.append(sample)

                # Create augmented samples if enabled
                if self.augment_data:
                    augmented_samples = self._create_augmented_samples(anime)
                    samples.extend(augmented_samples)

            except Exception as e:
                logger.warning(
                    f"Error processing anime {anime.get('title', 'unknown')}: {e}"
                )
                continue

        return samples

    def _create_base_sample(self, anime: Dict[str, Any]) -> Optional[DataSample]:
        """Create base training sample from anime data.

        Args:
            anime: Anime data dictionary

        Returns:
            Data sample or None if invalid
        """
        try:
            # Extract basic information
            title = anime.get("title", "")
            if not title:
                return None

            # Create anime ID
            anime_id = self._generate_anime_id(anime)

            # Create text representation
            text = self._create_text_representation(anime)

            # Extract image data
            image_data = self._extract_image_data(anime)

            # Extract labels
            character_labels = self._extract_character_labels(anime)
            art_style_label = self._extract_art_style_label(anime)
            genre_labels = self._extract_genre_labels(anime)

            # Create sample
            sample = DataSample(
                anime_id=anime_id,
                title=title,
                text=text,
                image_data=image_data,
                character_labels=character_labels,
                art_style_label=art_style_label,
                genre_labels=genre_labels,
                studio=anime.get("studios", [None])[0],
                year=self._extract_year(anime),
                type=anime.get("type", "unknown"),
                tags=anime.get("tags", []),
            )

            return sample

        except Exception as e:
            logger.warning(f"Error creating base sample: {e}")
            return None

    def _create_augmented_samples(self, anime: Dict[str, Any]) -> List[DataSample]:
        """Create augmented training samples from anime data.

        Args:
            anime: Anime data dictionary

        Returns:
            List of augmented samples
        """
        augmented_samples: List[DataSample] = []
        base_sample = self._create_base_sample(anime)

        if not base_sample:
            return augmented_samples

        # Text augmentation variants
        text_variants = [
            self._create_synopsis_variant(anime),
            self._create_tag_variant(anime),
            self._create_studio_variant(anime),
            self._create_character_variant(anime),
        ]

        for variant_text in text_variants:
            if variant_text and variant_text != base_sample.text:
                augmented_sample = DataSample(
                    anime_id=base_sample.anime_id + "_aug",
                    title=base_sample.title,
                    text=variant_text,
                    image_data=base_sample.image_data,
                    character_labels=base_sample.character_labels,
                    art_style_label=base_sample.art_style_label,
                    genre_labels=base_sample.genre_labels,
                    studio=base_sample.studio,
                    year=base_sample.year,
                    type=base_sample.type,
                    tags=base_sample.tags,
                )
                augmented_samples.append(augmented_sample)

        return augmented_samples

    def _generate_anime_id(self, anime: Dict[str, Any]) -> str:
        """Generate unique anime ID.

        Args:
            anime: Anime data dictionary

        Returns:
            Unique anime ID
        """
        title = anime.get("title", "unknown")
        sources = anime.get("sources", [])

        if sources:
            # Use first source URL to create ID
            source_id = sources[0].split("/")[-1]
            return f"{title}_{source_id}"
        else:
            # Fallback to title-based ID
            return title.replace(" ", "_").lower()

    def _create_text_representation(self, anime: Dict[str, Any]) -> str:
        """Create comprehensive text representation.

        Args:
            anime: Anime data dictionary

        Returns:
            Text representation
        """
        parts = []

        # Title
        title = anime.get("title", "")
        if title:
            parts.append(f"Title: {title}")

        # Synopsis
        synopsis = anime.get("synopsis", "")
        if synopsis:
            parts.append(f"Synopsis: {synopsis}")

        # Genre/Tags
        tags = anime.get("tags", [])
        if tags:
            parts.append(f"Genres: {', '.join(tags)}")

        # Studio
        studios = anime.get("studios", [])
        if studios:
            parts.append(f"Studio: {', '.join(studios)}")

        # Type and year
        anime_type = anime.get("type", "")
        if anime_type:
            parts.append(f"Type: {anime_type}")

        year = self._extract_year(anime)
        if year:
            parts.append(f"Year: {year}")

        return " | ".join(parts)

    def _extract_image_data(self, anime: Dict[str, Any]) -> Optional[str]:
        """Extract image data from anime entry.

        Args:
            anime: Anime data dictionary

        Returns:
            Base64 encoded image data or None
        """
        # For now, return the picture URL as placeholder
        # In a full implementation, this would download and encode the image
        picture_url = anime.get("picture")
        if picture_url:
            # Placeholder: return URL as "image data"
            # In production, this would download and encode the image
            return picture_url

        return None

    def _extract_character_labels(self, anime: Dict[str, Any]) -> List[str]:
        """Extract character labels from anime data.

        Args:
            anime: Anime data dictionary

        Returns:
            List of character labels
        """
        # Extract character names from tags/synopsis
        character_tags = []

        # Look for character-related tags
        tags = anime.get("tags", [])
        for tag in tags:
            if any(
                keyword in tag.lower()
                for keyword in ["character", "protagonist", "main"]
            ):
                character_tags.append(tag)

        # Extract from synopsis (basic approach)
        synopsis = anime.get("synopsis", "")
        if synopsis:
            # Look for common character name patterns
            # This is a simplified approach - in production would use NER
            words = synopsis.split()
            potential_characters = [w for w in words if w[0].isupper() and len(w) > 3]
            character_tags.extend(potential_characters[:3])  # Limit to top 3

        return list(set(character_tags))

    def _extract_art_style_label(self, anime: Dict[str, Any]) -> str:
        """Extract art style label from anime data.

        Args:
            anime: Anime data dictionary

        Returns:
            Art style label
        """
        # Infer art style from studio and year
        studios = anime.get("studios", [])
        year = self._extract_year(anime)

        # Studio-based style mapping
        studio_style_mapping = {
            "Studio Ghibli": "ghibli",
            "Madhouse": "madhouse",
            "Toei Animation": "toei",
            "Sunrise": "sunrise",
            "Bones": "bones",
            "Pierrot": "pierrot",
            "Mappa": "mappa",
            "Wit Studio": "wit",
            "Ufotable": "ufotable",
            "Kyoto Animation": "kyoani",
        }

        for studio in studios:
            if studio in studio_style_mapping:
                return studio_style_mapping[studio]

        # Year-based style periods
        if year:
            if year >= 2010:
                return "modern"
            elif year >= 2000:
                return "digital"
            elif year >= 1990:
                return "classic"
            else:
                return "vintage"

        return "unknown"

    def _extract_genre_labels(self, anime: Dict[str, Any]) -> List[str]:
        """Extract genre labels from anime data.

        Args:
            anime: Anime data dictionary

        Returns:
            List of genre labels
        """
        # Extract from tags
        tags = anime.get("tags", [])

        # Common anime genre mappings
        genre_keywords = {
            "action": ["action", "fighting", "battle", "combat"],
            "adventure": ["adventure", "journey", "quest", "travel"],
            "comedy": ["comedy", "humor", "funny", "parody"],
            "drama": ["drama", "dramatic", "emotional", "tragic"],
            "fantasy": ["fantasy", "magic", "magical", "supernatural"],
            "horror": ["horror", "scary", "thriller", "suspense"],
            "mecha": ["mecha", "robot", "mechanical", "pilot"],
            "romance": ["romance", "love", "romantic", "relationship"],
            "sci-fi": ["sci-fi", "science", "future", "space", "technology"],
            "slice-of-life": ["slice of life", "daily", "school", "everyday"],
            "sports": ["sports", "competition", "tournament", "team"],
            "psychological": ["psychological", "mind", "mental", "psycho"],
        }

        genres = []
        for tag in tags:
            tag_lower = tag.lower()
            for genre, keywords in genre_keywords.items():
                if any(keyword in tag_lower for keyword in keywords):
                    genres.append(genre)

        return list(set(genres))

    def _extract_year(self, anime: Dict[str, Any]) -> Optional[int]:
        """Extract year from anime data.

        Args:
            anime: Anime data dictionary

        Returns:
            Year as integer or None
        """
        anime_season = anime.get("animeSeason")
        if anime_season and isinstance(anime_season, dict):
            return anime_season.get("year")

        return None

    def _create_synopsis_variant(self, anime: Dict[str, Any]) -> str:
        """Create synopsis-focused text variant.

        Args:
            anime: Anime data dictionary

        Returns:
            Synopsis variant text
        """
        title = anime.get("title", "")
        synopsis = anime.get("synopsis", "")

        if synopsis:
            return f"{title}: {synopsis}"

        return title

    def _create_tag_variant(self, anime: Dict[str, Any]) -> str:
        """Create tag-focused text variant.

        Args:
            anime: Anime data dictionary

        Returns:
            Tag variant text
        """
        title = anime.get("title", "")
        tags = anime.get("tags", [])

        if tags:
            return f"{title} - {', '.join(tags)}"

        return title

    def _create_studio_variant(self, anime: Dict[str, Any]) -> str:
        """Create studio-focused text variant.

        Args:
            anime: Anime data dictionary

        Returns:
            Studio variant text
        """
        title = anime.get("title", "")
        studios = anime.get("studios", [])

        if studios:
            return f"{title} by {', '.join(studios)}"

        return title

    def _create_character_variant(self, anime: Dict[str, Any]) -> str:
        """Create character-focused text variant.

        Args:
            anime: Anime data dictionary

        Returns:
            Character variant text
        """
        title = anime.get("title", "")
        character_labels = self._extract_character_labels(anime)

        if character_labels:
            return f"{title} featuring {', '.join(character_labels)}"

        return title

    def _build_character_vocab(self) -> Dict[str, int]:
        """Build character vocabulary from samples.

        Returns:
            Character to index mapping
        """
        vocab = {}
        index = 0

        for sample in self.samples:
            if sample.character_labels:
                for character in sample.character_labels:
                    if character not in vocab:
                        vocab[character] = index
                        index += 1

        return vocab

    def _build_art_style_vocab(self) -> Dict[str, int]:
        """Build art style vocabulary from samples.

        Returns:
            Art style to index mapping
        """
        vocab = {}
        index = 0

        for sample in self.samples:
            if sample.art_style_label and sample.art_style_label not in vocab:
                vocab[sample.art_style_label] = index
                index += 1

        return vocab

    def _build_genre_vocab(self) -> Dict[str, int]:
        """Build genre vocabulary from samples.

        Returns:
            Genre to index mapping
        """
        vocab = {}
        index = 0

        for sample in self.samples:
            if sample.genre_labels:
                for genre in sample.genre_labels:
                    if genre not in vocab:
                        vocab[genre] = index
                        index += 1

        return vocab

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item.

        Args:
            idx: Sample index

        Returns:
            Sample data dictionary
        """
        sample = self.samples[idx]

        # Encode text
        text_embedding = self.text_processor.encode_text(sample.text)

        # Encode image (if available)
        image_embedding = None
        if sample.image_data:
            try:
                image_embedding = self.vision_processor.encode_image(sample.image_data)
            except Exception as e:
                logger.warning(f"Error encoding image for {sample.anime_id}: {e}")

        # Create label tensors
        character_labels = torch.zeros(len(self.character_vocab))
        if sample.character_labels:
            for char in sample.character_labels:
                if char in self.character_vocab:
                    character_labels[self.character_vocab[char]] = 1.0

        art_style_label = torch.zeros(len(self.art_style_vocab))
        if sample.art_style_label and sample.art_style_label in self.art_style_vocab:
            art_style_label[self.art_style_vocab[sample.art_style_label]] = 1.0

        genre_labels = torch.zeros(len(self.genre_vocab))
        if sample.genre_labels:
            for genre in sample.genre_labels:
                if genre in self.genre_vocab:
                    genre_labels[self.genre_vocab[genre]] = 1.0

        return {
            "anime_id": sample.anime_id,
            "title": sample.title,
            "text": sample.text,
            "text_embedding": (
                torch.tensor(text_embedding) if text_embedding is not None else None
            ),
            "image_embedding": (
                torch.tensor(image_embedding) if image_embedding is not None else None
            ),
            "character_labels": character_labels,
            "art_style_label": art_style_label,
            "genre_labels": genre_labels,
            "metadata": {
                "studio": sample.studio,
                "year": sample.year,
                "type": sample.type,
                "tags": sample.tags,
            },
        }

    def get_vocab_sizes(self) -> Dict[str, int]:
        """Get vocabulary sizes for each task.

        Returns:
            Dictionary with vocabulary sizes
        """
        return {
            "character": len(self.character_vocab),
            "art_style": len(self.art_style_vocab),
            "genre": len(self.genre_vocab),
        }

    def get_class_weights(self) -> Dict[str, torch.Tensor]:
        """Calculate class weights for balanced training.

        Returns:
            Class weights for each task
        """
        # Calculate character weights
        character_counts: Dict[str, int] = defaultdict(int)
        art_style_counts: Dict[str, int] = defaultdict(int)
        genre_counts: Dict[str, int] = defaultdict(int)

        for sample in self.samples:
            if sample.character_labels:
                for char in sample.character_labels:
                    character_counts[char] += 1

            if sample.art_style_label:
                art_style_counts[sample.art_style_label] += 1

            if sample.genre_labels:
                for genre in sample.genre_labels:
                    genre_counts[genre] += 1

        # Create weight tensors
        character_weights = torch.ones(len(self.character_vocab))
        for char, idx in self.character_vocab.items():
            if char in character_counts:
                character_weights[idx] = 1.0 / character_counts[char]

        art_style_weights = torch.ones(len(self.art_style_vocab))
        for style, idx in self.art_style_vocab.items():
            if style in art_style_counts:
                art_style_weights[idx] = 1.0 / art_style_counts[style]

        genre_weights = torch.ones(len(self.genre_vocab))
        for genre, idx in self.genre_vocab.items():
            if genre in genre_counts:
                genre_weights[idx] = 1.0 / genre_counts[genre]

        return {
            "character": character_weights,
            "art_style": art_style_weights,
            "genre": genre_weights,
        }
