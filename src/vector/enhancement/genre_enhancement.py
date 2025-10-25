"""Genre enhancement fine-tuner for anime-specific genre understanding.

This module implements genre understanding enhancement through fine-tuning
text models for better anime genre classification and semantic understanding.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ...config import Settings

logger = logging.getLogger(__name__)


class GenreClassificationHead(nn.Module):
    """Multi-label classification head for genre classification."""

    def __init__(self, input_dim: int, num_genres: int, dropout: float = 0.1):
        """Initialize genre classification head.

        Args:
            input_dim: Input embedding dimension
            num_genres: Number of genre classes
            dropout: Dropout probability
        """
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 4, num_genres),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass through classification head.

        Args:
            embeddings: Input embeddings

        Returns:
            Genre classification logits
        """
        return self.classifier(embeddings)


class GenreSemanticEncoder(nn.Module):
    """Semantic encoder for genre-aware text understanding."""

    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout: float = 0.1):
        """Initialize genre semantic encoder.

        Args:
            input_dim: Input embedding dimension
            hidden_dim: Hidden dimension
            dropout: Dropout probability
        """
        super().__init__()

        # Semantic understanding layers
        self.semantic_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Genre-specific attention mechanism
        self.genre_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, dropout=dropout, batch_first=True
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass through semantic encoder.

        Args:
            embeddings: Input embeddings

        Returns:
            Genre-aware embeddings
        """
        # Semantic encoding
        encoded = self.semantic_encoder(embeddings)

        # Apply genre-specific attention
        encoded_expanded = encoded.unsqueeze(1)  # Add sequence dimension
        attended, _ = self.genre_attention(
            encoded_expanded, encoded_expanded, encoded_expanded
        )
        attended = attended.squeeze(1)  # Remove sequence dimension

        # Layer normalization with residual
        encoded = self.layer_norm(encoded + attended)

        # Project back to original dimension
        output = self.output_proj(encoded)

        return output


class GenreEnhancementModel(nn.Module):
    """Genre enhancement model for better anime genre understanding."""

    def __init__(
        self,
        input_dim: int,
        num_genres: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        """Initialize genre enhancement model.

        Args:
            input_dim: Input embedding dimension
            num_genres: Number of genre classes
            hidden_dim: Hidden dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_genres = num_genres
        self.hidden_dim = hidden_dim

        # Semantic encoder
        self.semantic_encoder = GenreSemanticEncoder(input_dim, hidden_dim, dropout)

        # Genre classifier
        self.genre_classifier = GenreClassificationHead(input_dim, num_genres, dropout)

        # Auxiliary tasks for better understanding
        self.theme_classifier = nn.Linear(input_dim, 30)  # Theme classification
        self.target_classifier = nn.Linear(input_dim, 10)  # Target audience
        self.mood_classifier = nn.Linear(input_dim, 15)  # Mood classification

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through genre enhancement model.

        Args:
            text_embeddings: Text embeddings

        Returns:
            Genre enhancement outputs
        """
        # Enhance embeddings with genre understanding
        enhanced_embeddings = self.semantic_encoder(text_embeddings)
        enhanced_embeddings = self.dropout(enhanced_embeddings)

        # Genre classification
        genre_logits = self.genre_classifier(enhanced_embeddings)

        # Auxiliary predictions
        theme_logits = self.theme_classifier(enhanced_embeddings)
        target_logits = self.target_classifier(enhanced_embeddings)
        mood_logits = self.mood_classifier(enhanced_embeddings)

        return {
            "genre_logits": genre_logits,
            "theme_logits": theme_logits,
            "target_logits": target_logits,
            "mood_logits": mood_logits,
            "enhanced_embeddings": enhanced_embeddings,
        }


class GenreEnhancementFinetuner:
    """Genre enhancement fine-tuner for anime genre understanding."""

    def __init__(self, settings: Settings, text_processor: Any):
        """Initialize genre enhancement fine-tuner.

        Args:
            settings: Configuration settings instance
            text_processor: Text processing utility
        """
        self.settings = settings
        self.text_processor = text_processor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model components
        self.enhancement_model: Optional[GenreEnhancementModel] = None
        self.optimizer: Optional[AdamW] = None
        self.loss_fn: Optional[Dict[str, nn.Module]] = None

        # Training state
        self.num_genres = 0
        self.genre_vocab: Dict[str, int] = {}
        self.theme_vocab: Dict[str, int] = {}
        self.target_vocab: Dict[str, int] = {}
        self.mood_vocab: Dict[str, int] = {}
        self.is_trained = False

        logger.info(f"Genre enhancement fine-tuner initialized on {self.device}")

    def setup_lora_model(
        self, lora_config: LoraConfig, fine_tuning_config: Any
    ) -> None:
        """Setup LoRA model for parameter-efficient fine-tuning.

        Args:
            lora_config: LoRA configuration
            fine_tuning_config: Fine-tuning configuration
        """
        self.fine_tuning_config = fine_tuning_config

        try:
            # Get text model info
            model_info = self.text_processor.get_model_info()
            input_dim = model_info.get("embedding_size", 384)

            # Create genre enhancement model
            self.enhancement_model = GenreEnhancementModel(
                input_dim=input_dim,
                num_genres=self.num_genres or 20,  # Default placeholder
                hidden_dim=512,
                dropout=0.1,
            )

            # Move to device
            self.enhancement_model = self.enhancement_model.to(self.device)

            # Setup optimizer
            self.optimizer = AdamW(
                self.enhancement_model.parameters(), lr=1e-4, weight_decay=1e-5
            )

            # Setup loss function (multi-task loss)
            self.loss_fn = {
                "genre": nn.BCEWithLogitsLoss(),
                "theme": nn.BCEWithLogitsLoss(),
                "target": nn.CrossEntropyLoss(),
                "mood": nn.CrossEntropyLoss(),
            }

            logger.info("LoRA model setup completed")

        except Exception as e:
            logger.error(f"Error setting up LoRA model: {e}")
            raise

    def prepare_for_training(self, dataset):
        """Prepare model for training with dataset vocabulary.

        Args:
            dataset: Training dataset
        """
        vocab_sizes = dataset.get_vocab_sizes()
        self.num_genres = vocab_sizes["genre"]
        self.genre_vocab = dataset.genre_vocab

        # Create auxiliary vocabularies
        self._create_auxiliary_vocabularies(dataset)

        # Recreate model with correct vocabulary size
        if self.enhancement_model is not None:
            input_dim = self.enhancement_model.input_dim

            # Create new model with correct vocabulary sizes
            self.enhancement_model = GenreEnhancementModel(
                input_dim=input_dim,
                num_genres=self.num_genres,
                hidden_dim=512,
                dropout=0.1,
            )

            # Update auxiliary classifiers
            self.enhancement_model.theme_classifier = nn.Linear(
                input_dim, len(self.theme_vocab)
            )
            self.enhancement_model.target_classifier = nn.Linear(
                input_dim, len(self.target_vocab)
            )
            self.enhancement_model.mood_classifier = nn.Linear(
                input_dim, len(self.mood_vocab)
            )

            # Move to device
            self.enhancement_model = self.enhancement_model.to(self.device)

            # Setup optimizer
            self.optimizer = torch.optim.AdamW(
                self.enhancement_model.parameters(),
                lr=self.fine_tuning_config.learning_rate,
                weight_decay=self.fine_tuning_config.weight_decay,
            )

        logger.info(f"Model prepared for training with {self.num_genres} genres")

    def _create_auxiliary_vocabularies(self, dataset):
        """Create auxiliary vocabularies for multi-task learning.

        Args:
            dataset: Training dataset
        """
        # Theme vocabulary from tags
        themes = set()
        for sample in dataset.samples:
            if sample.tags:
                for tag in sample.tags:
                    if any(
                        keyword in tag.lower()
                        for keyword in ["school", "military", "magic", "space"]
                    ):
                        themes.add(tag.lower())

        self.theme_vocab = {theme: i for i, theme in enumerate(sorted(themes))}

        # Target audience vocabulary
        targets = [
            "kids",
            "shounen",
            "shoujo",
            "seinen",
            "josei",
            "adult",
            "family",
            "general",
        ]
        self.target_vocab = {target: i for i, target in enumerate(targets)}

        # Mood vocabulary
        moods = [
            "happy",
            "sad",
            "exciting",
            "calm",
            "dark",
            "bright",
            "serious",
            "funny",
            "mysterious",
            "romantic",
        ]
        self.mood_vocab = {mood: i for i, mood in enumerate(moods)}

        logger.info(
            f"Created auxiliary vocabularies: {len(self.theme_vocab)} themes, {len(self.target_vocab)} targets, {len(self.mood_vocab)} moods"
        )

    def _infer_auxiliary_labels(self, sample) -> Tuple[List[str], str, str]:
        """Infer auxiliary labels from sample data.

        Args:
            sample: Data sample

        Returns:
            Tuple of (themes, target, mood)
        """
        themes = []
        target = "general"
        mood = "general"

        if sample.tags:
            for tag in sample.tags:
                tag_lower = tag.lower()

                # Theme inference
                if tag_lower in self.theme_vocab:
                    themes.append(tag_lower)

                # Target inference
                if any(
                    keyword in tag_lower
                    for keyword in ["shounen", "action", "adventure"]
                ):
                    target = "shounen"
                elif any(
                    keyword in tag_lower for keyword in ["shoujo", "romance", "school"]
                ):
                    target = "shoujo"
                elif any(
                    keyword in tag_lower
                    for keyword in ["seinen", "mature", "psychological"]
                ):
                    target = "seinen"
                elif any(
                    keyword in tag_lower for keyword in ["josei", "adult", "drama"]
                ):
                    target = "josei"
                elif any(
                    keyword in tag_lower for keyword in ["kids", "child", "family"]
                ):
                    target = "kids"

                # Mood inference
                if any(
                    keyword in tag_lower for keyword in ["comedy", "funny", "humor"]
                ):
                    mood = "funny"
                elif any(
                    keyword in tag_lower for keyword in ["dark", "horror", "thriller"]
                ):
                    mood = "dark"
                elif any(keyword in tag_lower for keyword in ["romance", "love"]):
                    mood = "romantic"
                elif any(keyword in tag_lower for keyword in ["mystery", "suspense"]):
                    mood = "mysterious"
                elif any(keyword in tag_lower for keyword in ["action", "adventure"]):
                    mood = "exciting"

        return themes, target, mood

    def train_step(self, batch: Dict[str, Any]) -> float:
        """Perform one training step.

        Args:
            batch: Training batch

        Returns:
            Training loss
        """
        if self.enhancement_model is None:
            raise RuntimeError("Model not initialized. Call setup_lora_model first.")

        self.enhancement_model.train()
        if self.optimizer is not None:
            self.optimizer.zero_grad()

        # Get inputs
        text_embeddings = batch.get("text_embedding")
        if text_embeddings is None:
            # Skip batch if no text embeddings
            return 0.0

        text_embeddings = text_embeddings.to(self.device)
        genre_labels = batch["genre_labels"].to(self.device)
        metadata = batch["metadata"]

        # Create auxiliary labels
        theme_labels = torch.zeros(
            len(metadata), len(self.theme_vocab), device=self.device
        )
        target_labels = torch.zeros(len(metadata), dtype=torch.long, device=self.device)
        mood_labels = torch.zeros(len(metadata), dtype=torch.long, device=self.device)

        for i, meta in enumerate(metadata):
            # Create a simple sample object for auxiliary label inference
            class SimpleSample:
                def __init__(self, tags):
                    self.tags = tags

            sample = SimpleSample(meta.get("tags", []))
            themes, target, mood = self._infer_auxiliary_labels(sample)

            # Theme labels (multi-label)
            for theme in themes:
                if theme in self.theme_vocab:
                    theme_labels[i, self.theme_vocab[theme]] = 1.0

            # Target label
            if target in self.target_vocab:
                target_labels[i] = self.target_vocab[target]

            # Mood label
            if mood in self.mood_vocab:
                mood_labels[i] = self.mood_vocab[mood]

        # Forward pass
        outputs = self.enhancement_model(text_embeddings)

        # Calculate losses
        if self.loss_fn is not None:
            genre_loss = self.loss_fn["genre"](outputs["genre_logits"], genre_labels)
            theme_loss = self.loss_fn["theme"](outputs["theme_logits"], theme_labels)
            target_loss = self.loss_fn["target"](
                outputs["target_logits"], target_labels
            )
            mood_loss = self.loss_fn["mood"](outputs["mood_logits"], mood_labels)
        else:
            return 0.0

        # Combined loss
        total_loss = genre_loss + 0.3 * theme_loss + 0.2 * target_loss + 0.2 * mood_loss

        # Backward pass
        total_loss.backward()
        if self.optimizer is not None:
            self.optimizer.step()

        return total_loss.item()

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation set.

        Args:
            dataloader: Validation data loader

        Returns:
            Evaluation metrics
        """
        if self.enhancement_model is None:
            raise RuntimeError("Model not initialized")

        self.enhancement_model.eval()

        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in dataloader:
                # Get inputs
                text_embeddings = batch.get("text_embedding")
                if text_embeddings is None:
                    continue

                text_embeddings = text_embeddings.to(self.device)
                genre_labels = batch["genre_labels"].to(self.device)

                # Forward pass
                outputs = self.enhancement_model(text_embeddings)

                # Calculate loss
                if self.loss_fn is not None:
                    genre_loss = self.loss_fn["genre"](
                        outputs["genre_logits"], genre_labels
                    )
                else:
                    continue
                total_loss += genre_loss.item()

                # Calculate accuracy
                predictions = torch.sigmoid(outputs["genre_logits"]) > 0.5
                correct_predictions += (predictions == genre_labels).sum().item()
                total_predictions += genre_labels.numel()

        avg_loss = total_loss / len(dataloader)
        accuracy = (
            correct_predictions / total_predictions if total_predictions > 0 else 0.0
        )

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "total_predictions": total_predictions,
        }

    def get_enhanced_embedding(self, text_embedding: np.ndarray) -> np.ndarray:
        """Get enhanced genre-aware embedding.

        Args:
            text_embedding: Text embedding

        Returns:
            Enhanced embedding
        """
        if self.enhancement_model is None:
            logger.warning("Model not initialized, returning original embedding")
            return text_embedding

        self.enhancement_model.eval()

        with torch.no_grad():
            # Convert to tensor
            text_tensor = (
                torch.tensor(text_embedding, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
            )

            # Get enhanced features
            outputs = self.enhancement_model(text_tensor)

            # Return enhanced embeddings
            enhanced_embeddings = outputs["enhanced_embeddings"]
            return enhanced_embeddings.cpu().numpy().flatten()

    def predict_genres(
        self, text_embedding: np.ndarray, threshold: float = 0.5
    ) -> List[Tuple[str, float]]:
        """Predict genres from text embedding.

        Args:
            text_embedding: Text embedding
            threshold: Prediction threshold

        Returns:
            List of (genre, confidence) tuples
        """
        if self.enhancement_model is None:
            logger.warning("Model not initialized, returning empty predictions")
            return []

        self.enhancement_model.eval()

        with torch.no_grad():
            # Convert to tensor
            text_tensor = (
                torch.tensor(text_embedding, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
            )

            # Get predictions
            outputs = self.enhancement_model(text_tensor)

            # Get genre probabilities
            genre_logits = outputs["genre_logits"]
            genre_probs = torch.sigmoid(genre_logits)

            # Get predictions above threshold
            predictions = []
            for i, prob in enumerate(genre_probs[0]):
                if prob > threshold:
                    # Find genre name
                    genre_name = None
                    for genre, idx in self.genre_vocab.items():
                        if idx == i:
                            genre_name = genre
                            break

                    if genre_name:
                        predictions.append((genre_name, prob.item()))

            # Sort by confidence
            predictions.sort(key=lambda x: x[1], reverse=True)

            return predictions

    def save_model(self, save_path: Path) -> None:
        """Save fine-tuned model.

        Args:
            save_path: Path to save model
        """
        if self.enhancement_model is None:
            raise RuntimeError("Model not initialized")

        save_path.mkdir(parents=True, exist_ok=True)

        # Save model state
        model_state = {
            "model_state_dict": self.enhancement_model.state_dict(),
            "optimizer_state_dict": (
                self.optimizer.state_dict() if self.optimizer is not None else {}
            ),
            "num_genres": self.num_genres,
            "genre_vocab": self.genre_vocab,
            "theme_vocab": self.theme_vocab,
            "target_vocab": self.target_vocab,
            "mood_vocab": self.mood_vocab,
            "is_trained": self.is_trained,
        }

        torch.save(model_state, save_path / "genre_enhancement_model.pth")
        logger.info(f"Genre enhancement model saved to {save_path}")

    def load_model(self, load_path: Path) -> None:
        """Load fine-tuned model.

        Args:
            load_path: Path to load model from
        """
        model_path = load_path / "genre_enhancement_model.pth"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load model state
        model_state = torch.load(model_path, map_location=self.device)

        # Restore configuration
        self.num_genres = model_state["num_genres"]
        self.genre_vocab = model_state["genre_vocab"]
        self.theme_vocab = model_state["theme_vocab"]
        self.target_vocab = model_state["target_vocab"]
        self.mood_vocab = model_state["mood_vocab"]
        self.is_trained = model_state["is_trained"]

        # Recreate model with correct configuration
        self.enhancement_model = GenreEnhancementModel(
            input_dim=384,  # Default dimension
            num_genres=self.num_genres,
            hidden_dim=512,
            dropout=0.1,
        )

        # Update auxiliary classifiers
        if self.enhancement_model is not None:
            self.enhancement_model.theme_classifier = nn.Linear(
                384, len(self.theme_vocab)
            )
            self.enhancement_model.target_classifier = nn.Linear(
                384, len(self.target_vocab)
            )
            self.enhancement_model.mood_classifier = nn.Linear(
                384, len(self.mood_vocab)
            )

            # Load state dict
            self.enhancement_model.load_state_dict(model_state["model_state_dict"])
            self.enhancement_model = self.enhancement_model.to(self.device)

            # Setup optimizer
            self.optimizer = AdamW(
                self.enhancement_model.parameters(), lr=1e-4, weight_decay=1e-5
            )
            if self.optimizer is not None:
                self.optimizer.load_state_dict(model_state["optimizer_state_dict"])

        logger.info(f"Genre enhancement model loaded from {load_path}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.

        Returns:
            Model information dictionary
        """
        return {
            "num_genres": self.num_genres,
            "genre_vocab_size": len(self.genre_vocab),
            "theme_vocab_size": len(self.theme_vocab),
            "target_vocab_size": len(self.target_vocab),
            "mood_vocab_size": len(self.mood_vocab),
            "is_trained": self.is_trained,
            "device": str(self.device),
            "model_type": "genre_enhancement",
        }
