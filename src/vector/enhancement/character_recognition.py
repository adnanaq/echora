"""Character recognition fine-tuner for anime-specific character identification.

This module implements character recognition capabilities through fine-tuning
multimodal models for anime character identification tasks.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig
from torch.utils.data import DataLoader

from ...config import Settings

logger = logging.getLogger(__name__)


class CharacterRecognitionHead(nn.Module):
    """Classification head for character recognition."""

    def __init__(self, input_dim: int, num_characters: int, dropout: float = 0.1):
        """Initialize character recognition head.

        Args:
            input_dim: Input embedding dimension
            num_characters: Number of character classes
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
            nn.Linear(input_dim // 4, num_characters),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass through classification head.

        Args:
            embeddings: Input embeddings

        Returns:
            Character classification logits
        """
        return self.classifier(embeddings)


class MultimodalCharacterRecognizer(nn.Module):
    """Multimodal character recognition model."""

    def __init__(
        self,
        text_dim: int,
        image_dim: int,
        num_characters: int,
        fusion_dim: int = 512,
        dropout: float = 0.1,
    ):
        """Initialize multimodal character recognizer.

        Args:
            text_dim: Text embedding dimension
            image_dim: Image embedding dimension
            num_characters: Number of character classes
            fusion_dim: Fusion layer dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.text_dim = text_dim
        self.image_dim = image_dim
        self.fusion_dim = fusion_dim

        # Projection layers
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.image_proj = nn.Linear(image_dim, fusion_dim)

        # Attention mechanism for fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=8, dropout=dropout, batch_first=True
        )

        # Character recognition head
        self.character_head = CharacterRecognitionHead(
            fusion_dim, num_characters, dropout
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        text_embeddings: Optional[torch.Tensor] = None,
        image_embeddings: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through multimodal character recognizer.

        Args:
            text_embeddings: Text embeddings
            image_embeddings: Image embeddings

        Returns:
            Character recognition outputs
        """
        features = []

        # Process text embeddings
        if text_embeddings is not None:
            text_features = self.text_proj(text_embeddings)
            text_features = F.relu(text_features)
            text_features = self.dropout(text_features)
            features.append(text_features)

        # Process image embeddings
        if image_embeddings is not None:
            image_features = self.image_proj(image_embeddings)
            image_features = F.relu(image_features)
            image_features = self.dropout(image_features)
            features.append(image_features)

        # Fusion
        if len(features) == 2:
            # Multimodal fusion with attention
            stacked_features = torch.stack(features, dim=1)
            fused_features, attention_weights = self.attention(
                stacked_features, stacked_features, stacked_features
            )
            fused_features = fused_features.mean(dim=1)
        elif len(features) == 1:
            # Single modality
            fused_features = features[0]
            attention_weights = None
        else:
            raise ValueError("At least one modality must be provided")

        # Character recognition
        character_logits = self.character_head(fused_features)

        return {
            "character_logits": character_logits,
            "fused_features": fused_features,
            "attention_weights": attention_weights,
        }


class CharacterRecognitionFinetuner:
    """Character recognition fine-tuner for anime characters."""

    def __init__(self, settings: Settings, text_processor: Any, vision_processor: Any):
        """Initialize character recognition fine-tuner.

        Args:
            settings: Configuration settings instance
            text_processor: Text processing utility
            vision_processor: Vision processing utility
        """
        self.settings = settings
        self.text_processor = text_processor
        self.vision_processor = vision_processor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model components
        self.base_model: Optional[Any] = None
        self.recognition_model: Optional[MultimodalCharacterRecognizer] = None
        self.optimizer: Optional[torch.optim.AdamW] = None
        self.loss_fn: Optional[nn.BCEWithLogitsLoss] = None

        # Training state
        self.num_characters = 0
        self.character_vocab: Dict[str, int] = {}
        self.is_trained = False

        logger.info(f"Character recognition fine-tuner initialized on {self.device}")

    def setup_lora_model(
        self, lora_config: LoraConfig, fine_tuning_config: Any
    ) -> None:
        """Setup LoRA model for parameter-efficient fine-tuning.

        Args:
            lora_config: LoRA configuration
            fine_tuning_config: Fine-tuning configuration
        """
        self.fine_tuning_config = fine_tuning_config

        # Load base model (using text processor's model as base)
        try:
            from .text_processor import TextProcessor

            TextProcessor(self.settings)

            # Get model info
            model_info = self.text_processor.get_model_info()
            text_dim = model_info.get("embedding_size", 384)

            # Load vision model info
            vision_info = self.vision_processor.get_model_info()
            image_dim = vision_info.get("embedding_size", 512)

            # Create multimodal model
            self.recognition_model = MultimodalCharacterRecognizer(
                text_dim=text_dim,
                image_dim=image_dim,
                num_characters=self.num_characters or 100,  # Default placeholder
                fusion_dim=512,
                dropout=0.1,
            )

            # Move to device
            self.recognition_model = self.recognition_model.to(self.device)

            # Setup optimizer
            self.optimizer = torch.optim.AdamW(
                self.recognition_model.parameters(),
                lr=fine_tuning_config.learning_rate,
                weight_decay=fine_tuning_config.weight_decay,
            )

            # Setup loss function
            self.loss_fn = nn.BCEWithLogitsLoss()

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
        self.num_characters = vocab_sizes["character"]
        self.character_vocab = dataset.character_vocab

        # Recreate model with correct vocabulary size
        if self.recognition_model is not None:
            # Get model dimensions
            text_dim = self.recognition_model.text_dim
            image_dim = self.recognition_model.image_dim
            fusion_dim = self.recognition_model.fusion_dim

            # Create new model with correct vocabulary size
            self.recognition_model = MultimodalCharacterRecognizer(
                text_dim=text_dim,
                image_dim=image_dim,
                num_characters=self.num_characters,
                fusion_dim=fusion_dim,
                dropout=0.1,
            )

            # Move to device
            self.recognition_model = self.recognition_model.to(self.device)

            # Setup optimizer
            self.optimizer = torch.optim.AdamW(
                self.recognition_model.parameters(),
                lr=self.fine_tuning_config.learning_rate,
                weight_decay=self.fine_tuning_config.weight_decay,
            )

        logger.info(
            f"Model prepared for training with {self.num_characters} characters"
        )

    def train_step(self, batch: Dict[str, Any]) -> float:
        """Perform one training step.

        Args:
            batch: Training batch

        Returns:
            Training loss
        """
        if (
            self.recognition_model is None
            or self.optimizer is None
            or self.loss_fn is None
        ):
            raise RuntimeError("Model not initialized. Call setup_lora_model first.")

        self.recognition_model.train()
        self.optimizer.zero_grad()

        # Get inputs
        text_embeddings = batch.get("text_embedding")
        image_embeddings = batch.get("image_embedding")
        character_labels = batch["character_labels"].to(self.device)

        # Move embeddings to device
        if text_embeddings is not None:
            text_embeddings = text_embeddings.to(self.device)
        if image_embeddings is not None:
            image_embeddings = image_embeddings.to(self.device)

        # Forward pass
        outputs = self.recognition_model(
            text_embeddings=text_embeddings, image_embeddings=image_embeddings
        )

        # Calculate loss
        character_logits = outputs["character_logits"]
        loss = self.loss_fn(character_logits, character_labels)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation set.

        Args:
            dataloader: Validation data loader

        Returns:
            Evaluation metrics
        """
        if self.recognition_model is None or self.loss_fn is None:
            raise RuntimeError("Model not initialized")

        self.recognition_model.eval()

        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in dataloader:
                # Get inputs
                text_embeddings = batch.get("text_embedding")
                image_embeddings = batch.get("image_embedding")
                character_labels = batch["character_labels"].to(self.device)

                # Move embeddings to device
                if text_embeddings is not None:
                    text_embeddings = text_embeddings.to(self.device)
                if image_embeddings is not None:
                    image_embeddings = image_embeddings.to(self.device)

                # Forward pass
                outputs = self.recognition_model(
                    text_embeddings=text_embeddings, image_embeddings=image_embeddings
                )

                # Calculate loss
                character_logits = outputs["character_logits"]
                loss = self.loss_fn(character_logits, character_labels)
                total_loss += loss.item()

                # Calculate accuracy
                predictions = torch.sigmoid(character_logits) > 0.5
                correct_predictions += (predictions == character_labels).sum().item()
                total_predictions += character_labels.numel()

        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "total_predictions": total_predictions,
        }

    def get_enhanced_embedding(
        self,
        text_embedding: Optional[np.ndarray] = None,
        image_embedding: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Get enhanced character-aware embedding.

        Args:
            text_embedding: Text embedding
            image_embedding: Image embedding

        Returns:
            Enhanced embedding
        """
        if self.recognition_model is None:
            logger.warning("Model not initialized, returning original embeddings")
            if text_embedding is not None:
                return text_embedding
            elif image_embedding is not None:
                return image_embedding
            else:
                return np.zeros(512)

        self.recognition_model.eval()

        with torch.no_grad():
            # Convert to tensors
            text_tensor = None
            if text_embedding is not None:
                text_tensor = (
                    torch.tensor(text_embedding, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(self.device)
                )

            image_tensor = None
            if image_embedding is not None:
                image_tensor = (
                    torch.tensor(image_embedding, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(self.device)
                )

            # Get enhanced features
            outputs = self.recognition_model(
                text_embeddings=text_tensor, image_embeddings=image_tensor
            )

            # Return fused features
            fused_features = outputs["fused_features"]
            return fused_features.cpu().numpy().flatten()

    def predict_characters(
        self,
        text_embedding: Optional[np.ndarray] = None,
        image_embedding: Optional[np.ndarray] = None,
        threshold: float = 0.5,
    ) -> List[Tuple[str, float]]:
        """Predict characters from embeddings.

        Args:
            text_embedding: Text embedding
            image_embedding: Image embedding
            threshold: Prediction threshold

        Returns:
            List of (character, confidence) tuples
        """
        if self.recognition_model is None:
            logger.warning("Model not initialized, returning empty predictions")
            return []

        self.recognition_model.eval()

        with torch.no_grad():
            # Convert to tensors
            text_tensor = None
            if text_embedding is not None:
                text_tensor = (
                    torch.tensor(text_embedding, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(self.device)
                )

            image_tensor = None
            if image_embedding is not None:
                image_tensor = (
                    torch.tensor(image_embedding, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(self.device)
                )

            # Get predictions
            outputs = self.recognition_model(
                text_embeddings=text_tensor, image_embeddings=image_tensor
            )

            # Get character probabilities
            character_logits = outputs["character_logits"]
            character_probs = torch.sigmoid(character_logits)

            # Get predictions above threshold
            predictions = []
            for i, prob in enumerate(character_probs[0]):
                if prob > threshold:
                    # Find character name
                    character_name = None
                    for char, idx in self.character_vocab.items():
                        if idx == i:
                            character_name = char
                            break

                    if character_name:
                        predictions.append((character_name, prob.item()))

            # Sort by confidence
            predictions.sort(key=lambda x: x[1], reverse=True)

            return predictions

    def save_model(self, save_path: Path) -> None:
        """Save fine-tuned model.

        Args:
            save_path: Path to save model
        """
        if self.recognition_model is None or self.optimizer is None:
            raise RuntimeError("Model not initialized")

        save_path.mkdir(parents=True, exist_ok=True)

        # Save model state
        model_state = {
            "model_state_dict": self.recognition_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "num_characters": self.num_characters,
            "character_vocab": self.character_vocab,
            "is_trained": self.is_trained,
        }

        torch.save(model_state, save_path / "character_recognition_model.pth")
        logger.info(f"Character recognition model saved to {save_path}")

    def load_model(self, load_path: Path) -> None:
        """Load fine-tuned model.

        Args:
            load_path: Path to load model from
        """
        model_path = load_path / "character_recognition_model.pth"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load model state
        model_state = torch.load(model_path, map_location=self.device)

        # Restore configuration
        self.num_characters = model_state["num_characters"]
        self.character_vocab = model_state["character_vocab"]
        self.is_trained = model_state["is_trained"]

        # Recreate model with correct configuration
        self.recognition_model = MultimodalCharacterRecognizer(
            text_dim=384,  # Default dimension
            image_dim=512,  # Default dimension
            num_characters=self.num_characters,
            fusion_dim=512,
            dropout=0.1,
        )

        # Load state dict
        self.recognition_model.load_state_dict(model_state["model_state_dict"])
        self.recognition_model = self.recognition_model.to(self.device)

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.recognition_model.parameters(), lr=1e-4, weight_decay=1e-5
        )
        self.optimizer.load_state_dict(model_state["optimizer_state_dict"])

        logger.info(f"Character recognition model loaded from {load_path}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.

        Returns:
            Model information dictionary
        """
        return {
            "num_characters": self.num_characters,
            "character_vocab_size": len(self.character_vocab),
            "is_trained": self.is_trained,
            "device": str(self.device),
            "model_type": "multimodal_character_recognizer",
        }
