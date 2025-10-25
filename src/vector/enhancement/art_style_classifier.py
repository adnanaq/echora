"""Art style classifier for anime-specific visual style classification.

This module implements art style classification capabilities through fine-tuning
vision models for anime art style recognition tasks.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import PreTrainedModel

try:
    import open_clip  # type: ignore[import-untyped]
except ImportError:
    open_clip = None

from ...config import Settings

logger = logging.getLogger(__name__)


class LoRAEnhancedVisionModel(nn.Module):
    """LoRA-enhanced vision model for anime-specific fine-tuning."""

    def __init__(
        self,
        base_vision_model: nn.Module,
        lora_config: LoraConfig,
        freeze_base: bool = True,
    ):
        """Initialize LoRA-enhanced vision model.

        Args:
            base_vision_model: Base vision model (e.g., OpenCLIP)
            lora_config: LoRA configuration for fine-tuning
            freeze_base: Whether to freeze base model parameters
        """
        super().__init__()

        self.base_model = base_vision_model
        self.lora_config = lora_config

        # Freeze base model parameters if specified
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Apply LoRA to the vision model
        self.lora_model = self._apply_lora_to_vision_model()

        logger.info(f"LoRA-enhanced vision model initialized with r={lora_config.r}")

    def _apply_lora_to_vision_model(self) -> nn.Module | PeftModel | Any:
        """Apply LoRA to vision transformer layers.

        Returns:
            LoRA-enhanced model or base model if LoRA fails
        """
        try:
            # For OpenCLIP models, we need to handle the visual component
            if hasattr(self.base_model, "visual"):
                # OpenCLIP visual encoder
                visual_model = self.base_model.visual

                # Ensure we have a proper Module, not Tensor
                if not isinstance(visual_model, nn.Module):
                    logger.error(
                        f"Visual model is not a proper Module: {type(visual_model)}"
                    )
                    return self.base_model

                # Apply LoRA to the visual transformer
                if hasattr(visual_model, "transformer"):
                    # Create a wrapper model for PEFT compatibility
                    vision_wrapper = VisionTransformerWrapper(visual_model)
                    lora_model = get_peft_model(vision_wrapper, self.lora_config)
                    logger.info("Applied LoRA to vision transformer layers")
                    return lora_model
                else:
                    logger.warning(
                        "Vision model doesn't have transformer layers, applying LoRA to entire visual component"
                    )
                    # Fallback: apply to entire visual model
                    fallback_wrapper = VisionModelWrapper(visual_model)
                    lora_model = get_peft_model(fallback_wrapper, self.lora_config)
                    logger.info("Applied LoRA to entire visual model")
                    return lora_model
            else:
                # Direct model (e.g., standalone ViT)
                if isinstance(self.base_model, nn.Module):
                    model_wrapper = VisionModelWrapper(self.base_model)
                    lora_model = get_peft_model(model_wrapper, self.lora_config)
                    logger.info("Applied LoRA to standalone vision model")
                    return lora_model
                else:
                    logger.error(
                        f"Base model is not a proper Module: {type(self.base_model)}"
                    )
                    return self.base_model

        except Exception as e:
            logger.error(f"Failed to apply LoRA to vision model: {e}")
            # Fallback: return base model without LoRA
            return self.base_model

    def encode_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Encode image using LoRA-enhanced model.

        Args:
            image_tensor: Input image tensor

        Returns:
            Enhanced image embeddings
        """
        try:
            # Check if we have a PeftModel (LoRA-enhanced)
            if isinstance(self.lora_model, PeftModel):
                # Use the LoRA-enhanced model
                return self.lora_model(image_tensor)
            elif hasattr(self.lora_model, "base_model") and hasattr(
                self.lora_model.base_model, "visual_model"
            ):
                # Wrapper-based LoRA model
                visual_model = self.lora_model.base_model.visual_model
                if callable(visual_model):
                    return visual_model(image_tensor)
                else:
                    raise RuntimeError("Visual model is not callable")
            else:
                # Fallback to base model
                if hasattr(self.base_model, "encode_image") and callable(
                    self.base_model.encode_image
                ):
                    return self.base_model.encode_image(image_tensor)
                elif hasattr(self.base_model, "visual") and callable(
                    self.base_model.visual
                ):
                    return self.base_model.visual(image_tensor)
                elif callable(self.base_model):
                    return self.base_model(image_tensor)
                else:
                    raise RuntimeError("Base model is not callable")
        except Exception as e:
            logger.error(f"Error in LoRA vision encoding: {e}")
            # Final fallback to base model
            if hasattr(self.base_model, "encode_image") and callable(
                self.base_model.encode_image
            ):
                return self.base_model.encode_image(image_tensor)
            elif hasattr(self.base_model, "visual") and callable(
                self.base_model.visual
            ):
                return self.base_model.visual(image_tensor)
            elif callable(self.base_model):
                return self.base_model(image_tensor)
            else:
                raise RuntimeError("Base model is not callable")

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass through LoRA-enhanced vision model."""
        return self.encode_image(image_tensor)


class VisionTransformerWrapper(PreTrainedModel):
    """Wrapper for vision transformer to make it compatible with PEFT."""

    def __init__(self, visual_model: nn.Module, config=None):
        # Create a minimal config if none provided
        if config is None:
            from transformers import PretrainedConfig

            config = PretrainedConfig()

        super().__init__(config)
        self.visual_model = visual_model

        # Identify and store transformer layers for LoRA targeting
        self.transformer_layers: list[nn.Module] = []
        if hasattr(visual_model, "transformer"):
            transformer = visual_model.transformer

            # Handle OpenCLIP-style resblocks (ModuleList)
            if hasattr(transformer, "resblocks"):
                resblocks = transformer.resblocks
                if isinstance(resblocks, (nn.ModuleList, nn.Sequential)):
                    self.transformer_layers = list(resblocks)
                else:
                    # Handle unknown iterable types (cast to Any to bypass mypy)
                    try:
                        # Test if it's iterable
                        iterator = iter(resblocks)  # type: ignore[arg-type]
                        self.transformer_layers = [
                            block for block in resblocks if isinstance(block, nn.Module)
                        ]  # type: ignore[union-attr]
                    except (TypeError, AttributeError):
                        logger.warning("Could not iterate over transformer resblocks")
                        self.transformer_layers = []

            # Handle other transformer implementations with 'layers'
            elif hasattr(transformer, "layers"):
                layers = transformer.layers
                if isinstance(layers, (nn.ModuleList, nn.Sequential)):
                    self.transformer_layers = list(layers)
                else:
                    # Handle unknown iterable types (cast to Any to bypass mypy)
                    try:
                        # Test if it's iterable
                        iterator = iter(layers)  # type: ignore[arg-type]
                        self.transformer_layers = [
                            layer for layer in layers if isinstance(layer, nn.Module)
                        ]  # type: ignore[union-attr]
                    except (TypeError, AttributeError):
                        logger.warning("Could not iterate over transformer layers")
                        self.transformer_layers = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through wrapped vision model."""
        return self.visual_model(x)

    def get_input_embeddings(self) -> None:
        """Required method for PreTrainedModel."""
        return None

    def set_input_embeddings(self, value):
        """Required method for PreTrainedModel."""


class VisionModelWrapper(PreTrainedModel):
    """Generic wrapper for vision models to make them compatible with PEFT."""

    def __init__(self, vision_model: nn.Module, config=None):
        # Create a minimal config if none provided
        if config is None:
            from transformers import PretrainedConfig

            config = PretrainedConfig()

        super().__init__(config)
        self.vision_model = vision_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through wrapped vision model."""
        return self.vision_model(x)

    def get_input_embeddings(self) -> None:
        """Required method for PreTrainedModel."""
        return None

    def set_input_embeddings(self, value):
        """Required method for PreTrainedModel."""


class ArtStyleClassificationHead(nn.Module):
    """Classification head for art style classification."""

    def __init__(self, input_dim: int, num_styles: int, dropout: float = 0.1):
        """Initialize art style classification head.

        Args:
            input_dim: Input embedding dimension
            num_styles: Number of art style classes
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
            nn.Linear(input_dim // 4, num_styles),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass through classification head.

        Args:
            embeddings: Input embeddings

        Returns:
            Art style classification logits
        """
        return self.classifier(embeddings)


class ArtStyleFeatureExtractor(nn.Module):
    """Feature extractor for art style-specific features."""

    def __init__(self, input_dim: int, feature_dim: int = 256, dropout: float = 0.1):
        """Initialize art style feature extractor.

        Args:
            input_dim: Input embedding dimension
            feature_dim: Feature dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Style-specific attention
        self.style_attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=8, dropout=dropout, batch_first=True
        )

        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass through feature extractor.

        Args:
            embeddings: Input embeddings

        Returns:
            Style-specific features
        """
        # Extract features
        features = self.feature_extractor(embeddings)

        # Apply self-attention for style-specific features
        features_expanded = features.unsqueeze(1)  # Add sequence dimension
        attended_features, _ = self.style_attention(
            features_expanded, features_expanded, features_expanded
        )
        attended_features = attended_features.squeeze(1)  # Remove sequence dimension

        # Layer normalization
        features = self.layer_norm(features + attended_features)

        return features


class ArtStyleClassifierModel(nn.Module):
    """Art style classifier model with enhanced visual features."""

    def __init__(
        self,
        input_dim: int,
        num_styles: int,
        feature_dim: int = 256,
        dropout: float = 0.1,
    ):
        """Initialize art style classifier model.

        Args:
            input_dim: Input embedding dimension
            num_styles: Number of art style classes
            feature_dim: Feature dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_styles = num_styles
        self.feature_dim = feature_dim

        # Feature extractor
        self.feature_extractor = ArtStyleFeatureExtractor(
            input_dim, feature_dim, dropout
        )

        # Classification head
        self.classifier = ArtStyleClassificationHead(feature_dim, num_styles, dropout)

        # Auxiliary tasks for better feature learning
        self.studio_classifier = nn.Linear(feature_dim, 50)  # Predict studio
        self.era_classifier = nn.Linear(
            feature_dim, 5
        )  # Predict era (vintage, classic, digital, modern, contemporary)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, image_embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass through art style classifier.

        Args:
            image_embeddings: Image embeddings

        Returns:
            Art style classification outputs
        """
        # Extract style-specific features
        style_features = self.feature_extractor(image_embeddings)
        style_features = self.dropout(style_features)

        # Art style classification
        style_logits = self.classifier(style_features)

        # Auxiliary predictions
        studio_logits = self.studio_classifier(style_features)
        era_logits = self.era_classifier(style_features)

        return {
            "style_logits": style_logits,
            "studio_logits": studio_logits,
            "era_logits": era_logits,
            "style_features": style_features,
        }


class ArtStyleClassifier:
    """Art style classifier for anime visual styles."""

    def __init__(self, settings: Settings, vision_processor: Any):
        """Initialize art style classifier.

        Args:
            settings: Configuration settings instance
            vision_processor: Vision processing utility
        """
        self.settings = settings
        self.vision_processor = vision_processor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model components
        self.classifier_model: ArtStyleClassifierModel | None = None
        self.lora_vision_model: LoRAEnhancedVisionModel | None = None
        self.optimizer: AdamW | None = None
        self.loss_fn: dict[str, nn.CrossEntropyLoss] | None = None

        # Training state
        self.num_styles = 0
        self.style_vocab: dict[str, int] = {}
        self.studio_vocab: dict[str, int] = {}
        self.era_vocab: dict[str, int] = {}
        self.is_trained = False

        logger.info(f"Art style classifier initialized on {self.device}")

    def setup_lora_model(
        self, lora_config: LoraConfig, fine_tuning_config: Any
    ) -> None:
        """Setup LoRA model for parameter-efficient fine-tuning on vision backbone.

        Args:
            lora_config: LoRA configuration for vision model
            fine_tuning_config: Fine-tuning configuration
        """
        self.fine_tuning_config = fine_tuning_config

        try:
            # Get vision model info
            vision_info = self.vision_processor.get_model_info()
            input_dim = vision_info.get("embedding_size", 512)

            # Setup LoRA on vision backbone
            self._setup_lora_vision_model(lora_config)

            # Create art style classifier (smaller now, as LoRA handles feature adaptation)
            self.classifier_model = ArtStyleClassifierModel(
                input_dim=input_dim,
                num_styles=self.num_styles or 20,  # Default placeholder
                feature_dim=256,
                dropout=0.1,
            )

            # Move to device
            self.classifier_model = self.classifier_model.to(self.device)

            # Setup optimizer for both LoRA vision model and classifier
            trainable_params = list(self.classifier_model.parameters())
            if self.lora_vision_model is not None:
                trainable_params.extend(self.lora_vision_model.parameters())

            self.optimizer = AdamW(trainable_params, lr=1e-4, weight_decay=1e-5)

            # Setup loss function (multi-task loss)
            self.loss_fn = {
                "style": nn.CrossEntropyLoss(),
                "studio": nn.CrossEntropyLoss(),
                "era": nn.CrossEntropyLoss(),
            }

            logger.info("LoRA model setup completed with vision backbone enhancement")

        except Exception as e:
            logger.error(f"Error setting up LoRA model: {e}")
            raise

    def _setup_lora_vision_model(self, lora_config: LoraConfig) -> None:
        """Setup LoRA-enhanced vision model using configuration settings.

        Args:
            lora_config: LoRA configuration for vision model
        """
        try:
            # Check if LoRA is enabled in settings
            if not self.settings.lora_enabled:
                logger.info("LoRA is disabled in settings, skipping LoRA vision setup")
                return

            # Get the base vision model from the processor
            if self.vision_processor.model is None:
                logger.warning(
                    "Vision processor model not initialized, skipping LoRA vision setup"
                )
                return

            base_vision_model = self.vision_processor.model.get("model")
            if base_vision_model is None:
                logger.warning(
                    "Base vision model not found, skipping LoRA vision setup"
                )
                return

            # Create LoRA configuration from settings
            vision_lora_config = LoraConfig(
                task_type=getattr(TaskType, self.settings.lora_task_type),
                inference_mode=False,
                r=self.settings.lora_rank,
                lora_alpha=self.settings.lora_alpha,
                lora_dropout=self.settings.lora_dropout,
                target_modules=self.settings.lora_target_modules,
                bias=self.settings.lora_bias,
            )

            # Create LoRA-enhanced vision model
            self.lora_vision_model = LoRAEnhancedVisionModel(
                base_vision_model=base_vision_model,
                lora_config=vision_lora_config,
                freeze_base=True,
            )

            # Move to device
            self.lora_vision_model = self.lora_vision_model.to(self.device)

            logger.info(
                f"LoRA vision model setup completed with r={self.settings.lora_rank}, "
                f"alpha={self.settings.lora_alpha}, target_modules={self.settings.lora_target_modules}"
            )

        except Exception as e:
            logger.error(f"Failed to setup LoRA vision model: {e}")
            # Continue without LoRA vision model
            self.lora_vision_model = None

    def prepare_for_training(self, dataset):
        """Prepare model for training with dataset vocabulary.

        Args:
            dataset: Training dataset
        """
        vocab_sizes = dataset.get_vocab_sizes()
        self.num_styles = vocab_sizes["art_style"]
        self.style_vocab = dataset.art_style_vocab

        # Create auxiliary vocabularies
        self._create_auxiliary_vocabularies(dataset)

        # Recreate model with correct vocabulary size
        if self.classifier_model is not None:
            input_dim = self.classifier_model.input_dim

            # Create new model with correct vocabulary sizes
            self.classifier_model = ArtStyleClassifierModel(
                input_dim=input_dim,
                num_styles=self.num_styles,
                feature_dim=256,
                dropout=0.1,
            )

            # Update auxiliary classifiers
            self.classifier_model.studio_classifier = nn.Linear(
                256, len(self.studio_vocab)
            )
            self.classifier_model.era_classifier = nn.Linear(256, len(self.era_vocab))

            # Move to device
            self.classifier_model = self.classifier_model.to(self.device)

            # Setup optimizer
            self.optimizer = AdamW(
                self.classifier_model.parameters(),
                lr=self.fine_tuning_config.learning_rate,
                weight_decay=self.fine_tuning_config.weight_decay,
            )

        logger.info(f"Model prepared for training with {self.num_styles} art styles")

    def _create_auxiliary_vocabularies(self, dataset):
        """Create auxiliary vocabularies for multi-task learning.

        Args:
            dataset: Training dataset
        """
        # Studio vocabulary
        studios = set()
        for sample in dataset.samples:
            if sample.studio:
                studios.add(sample.studio)

        self.studio_vocab = {studio: i for i, studio in enumerate(sorted(studios))}

        # Era vocabulary based on years
        eras = ["vintage", "classic", "digital", "modern", "contemporary"]
        self.era_vocab = {era: i for i, era in enumerate(eras)}

        logger.info(
            f"Created auxiliary vocabularies: {len(self.studio_vocab)} studios, {len(self.era_vocab)} eras"
        )

    def _get_era_from_year(self, year: int | None) -> str:
        """Get era from year.

        Args:
            year: Year

        Returns:
            Era label
        """
        if year is None:
            return "unknown"

        if year < 1980:
            return "vintage"
        elif year < 1995:
            return "classic"
        elif year < 2005:
            return "digital"
        elif year < 2015:
            return "modern"
        else:
            return "contemporary"

    def train_step(self, batch: dict[str, Any]) -> float:
        """Perform one training step.

        Args:
            batch: Training batch

        Returns:
            Training loss
        """
        if (
            self.classifier_model is None
            or self.optimizer is None
            or self.loss_fn is None
        ):
            raise RuntimeError("Model not initialized. Call setup_lora_model first.")

        self.classifier_model.train()
        self.optimizer.zero_grad()

        # Get inputs
        image_embeddings = batch.get("image_embedding")
        if image_embeddings is None:
            # Skip batch if no image embeddings
            return 0.0

        image_embeddings = image_embeddings.to(self.device)
        art_style_labels = batch["art_style_label"].to(self.device)
        metadata = batch["metadata"]

        # Create auxiliary labels
        studio_labels = torch.zeros(len(metadata), dtype=torch.long, device=self.device)
        era_labels = torch.zeros(len(metadata), dtype=torch.long, device=self.device)

        for i, meta in enumerate(metadata):
            # Studio label
            if meta.get("studio") and meta["studio"] in self.studio_vocab:
                studio_labels[i] = self.studio_vocab[meta["studio"]]

            # Era label
            era = self._get_era_from_year(meta.get("year"))
            if era in self.era_vocab:
                era_labels[i] = self.era_vocab[era]

        # Forward pass
        outputs = self.classifier_model(image_embeddings)

        # Calculate losses
        style_loss = self.loss_fn["style"](
            outputs["style_logits"], art_style_labels.argmax(dim=1)
        )
        studio_loss = self.loss_fn["studio"](outputs["studio_logits"], studio_labels)
        era_loss = self.loss_fn["era"](outputs["era_logits"], era_labels)

        # Combined loss
        total_loss = style_loss + 0.3 * studio_loss + 0.2 * era_loss

        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """Evaluate model on validation set.

        Args:
            dataloader: Validation data loader

        Returns:
            Evaluation metrics
        """
        if self.classifier_model is None or self.loss_fn is None:
            raise RuntimeError("Model not initialized")

        self.classifier_model.eval()

        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in dataloader:
                # Get inputs
                image_embeddings = batch.get("image_embedding")
                if image_embeddings is None:
                    continue

                image_embeddings = image_embeddings.to(self.device)
                art_style_labels = batch["art_style_label"].to(self.device)

                # Forward pass
                outputs = self.classifier_model(image_embeddings)

                # Calculate loss
                style_loss = self.loss_fn["style"](
                    outputs["style_logits"], art_style_labels.argmax(dim=1)
                )
                total_loss += style_loss.item()

                # Calculate accuracy
                predictions = outputs["style_logits"].argmax(dim=1)
                correct_predictions += (
                    (predictions == art_style_labels.argmax(dim=1)).sum().item()
                )
                total_predictions += art_style_labels.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = (
            correct_predictions / total_predictions if total_predictions > 0 else 0.0
        )

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "total_predictions": total_predictions,
        }

    def get_enhanced_embedding(self, image_data: str) -> np.ndarray:
        """Get enhanced art style-aware embedding using LoRA vision model.

        Args:
            image_data: Base64 encoded image data

        Returns:
            Enhanced embedding from LoRA-adapted vision model
        """
        if self.lora_vision_model is None:
            logger.warning(
                "LoRA vision model not initialized, falling back to original embedding"
            )
            # Fallback to original processor
            original_embedding = self.vision_processor.encode_image(image_data)
            if original_embedding is None:
                return np.array([])
            return np.array(original_embedding)

        try:
            self.lora_vision_model.eval()

            with torch.no_grad():
                # Decode and preprocess image
                image = self.vision_processor._decode_base64_image(image_data)
                if image is None:
                    logger.error("Failed to decode image")
                    return np.array([])

                # Get preprocessing from vision processor
                preprocess = self.vision_processor.model.get("preprocess")
                if preprocess is None:
                    logger.error("Preprocessor not available")
                    return np.array([])

                # Preprocess image
                image_tensor = preprocess(image).unsqueeze(0).to(self.device)

                # Get LoRA-enhanced features
                enhanced_features = self.lora_vision_model.encode_image(image_tensor)

                # Normalize features
                enhanced_features = enhanced_features / enhanced_features.norm(
                    dim=-1, keepdim=True
                )

                return enhanced_features.cpu().numpy().flatten()

        except Exception as e:
            logger.error(f"Enhanced embedding generation failed: {e}")
            # Fallback to original processor
            original_embedding = self.vision_processor.encode_image(image_data)
            if original_embedding is None:
                return np.array([])
            return np.array(original_embedding)

    def predict_style(self, image_embedding: np.ndarray) -> list[tuple[str, float]]:
        """Predict art style from image embedding.

        Args:
            image_embedding: Image embedding

        Returns:
            List of (style, confidence) tuples
        """
        if self.classifier_model is None:
            logger.warning("Model not initialized, returning empty predictions")
            return []

        self.classifier_model.eval()

        with torch.no_grad():
            # Convert to tensor
            image_tensor = (
                torch.tensor(image_embedding, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
            )

            # Get predictions
            outputs = self.classifier_model(image_tensor)

            # Get style probabilities
            style_logits = outputs["style_logits"]
            style_probs = F.softmax(style_logits, dim=1)

            # Get top predictions
            top_probs, top_indices = torch.topk(
                style_probs, k=min(5, len(self.style_vocab))
            )

            predictions = []
            for prob, idx in zip(top_probs[0], top_indices[0]):
                # Find style name
                style_name = None
                for style, style_idx in self.style_vocab.items():
                    if style_idx == idx.item():
                        style_name = style
                        break

                if style_name:
                    predictions.append((style_name, prob.item()))

            return predictions

    def save_model(self, save_path: Path) -> None:
        """Save fine-tuned model.

        Args:
            save_path: Path to save model
        """
        if self.classifier_model is None or self.optimizer is None:
            raise RuntimeError("Model not initialized")

        save_path.mkdir(parents=True, exist_ok=True)

        # Save model state
        model_state = {
            "model_state_dict": self.classifier_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "num_styles": self.num_styles,
            "style_vocab": self.style_vocab,
            "studio_vocab": self.studio_vocab,
            "era_vocab": self.era_vocab,
            "is_trained": self.is_trained,
            "has_lora_vision": self.lora_vision_model is not None,
        }

        # Save LoRA vision model if available
        if self.lora_vision_model is not None:
            try:
                # Save LoRA adapter weights
                if (
                    hasattr(self.lora_vision_model, "lora_model")
                    and hasattr(self.lora_vision_model.lora_model, "save_pretrained")
                    and callable(self.lora_vision_model.lora_model.save_pretrained)
                ):
                    lora_save_path = save_path / "lora_vision_adapter"
                    self.lora_vision_model.lora_model.save_pretrained(lora_save_path)
                    logger.info(f"LoRA vision adapter saved to {lora_save_path}")
                else:
                    # Fallback: save state dict
                    model_state["lora_vision_state_dict"] = (
                        self.lora_vision_model.state_dict()
                    )
                    logger.info("LoRA vision model state saved to main checkpoint")
            except Exception as e:
                logger.warning(f"Failed to save LoRA vision model: {e}")

        torch.save(model_state, save_path / "art_style_classifier.pth")
        logger.info(f"Art style classifier saved to {save_path}")

    def load_model(self, load_path: Path) -> None:
        """Load fine-tuned model.

        Args:
            load_path: Path to load model from
        """
        model_path = load_path / "art_style_classifier.pth"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load model state
        model_state = torch.load(model_path, map_location=self.device)

        # Restore configuration
        self.num_styles = model_state["num_styles"]
        self.style_vocab = model_state["style_vocab"]
        self.studio_vocab = model_state["studio_vocab"]
        self.era_vocab = model_state["era_vocab"]
        self.is_trained = model_state["is_trained"]

        # Load LoRA vision model if available
        has_lora_vision = model_state.get("has_lora_vision", False)
        if has_lora_vision:
            try:
                # Try to load LoRA adapter
                lora_adapter_path = load_path / "lora_vision_adapter"
                if lora_adapter_path.exists():
                    # Recreate LoRA vision model
                    base_vision_model = (
                        self.vision_processor.model.get("model")
                        if self.vision_processor.model
                        else None
                    )
                    if base_vision_model:
                        # Create minimal LoRA config for loading
                        lora_config = LoraConfig(
                            task_type=TaskType.FEATURE_EXTRACTION,
                            inference_mode=True,  # For inference
                            r=16,  # Default values
                            lora_alpha=32,
                            lora_dropout=0.1,
                        )
                        self.lora_vision_model = LoRAEnhancedVisionModel(
                            base_vision_model=base_vision_model,
                            lora_config=lora_config,
                            freeze_base=True,
                        )
                        # Load LoRA weights
                        if hasattr(self.lora_vision_model, "lora_model"):
                            try:
                                if isinstance(
                                    self.lora_vision_model.lora_model, PeftModel
                                ):
                                    self.lora_vision_model.lora_model = PeftModel.from_pretrained(
                                        self.lora_vision_model.lora_model.base_model,
                                        str(lora_adapter_path),
                                    )
                                    logger.info(
                                        "LoRA vision adapter loaded successfully"
                                    )
                                else:
                                    logger.warning(
                                        "LoRA model is not a PeftModel, skipping adapter loading"
                                    )
                            except Exception as e:
                                logger.warning(f"Failed to load LoRA adapter: {e}")
                elif "lora_vision_state_dict" in model_state:
                    # Load from state dict
                    base_vision_model = (
                        self.vision_processor.model.get("model")
                        if self.vision_processor.model
                        else None
                    )
                    if base_vision_model:
                        lora_config = LoraConfig(
                            task_type=TaskType.FEATURE_EXTRACTION,
                            inference_mode=True,
                            r=16,
                            lora_alpha=32,
                            lora_dropout=0.1,
                        )
                        self.lora_vision_model = LoRAEnhancedVisionModel(
                            base_vision_model=base_vision_model,
                            lora_config=lora_config,
                            freeze_base=True,
                        )
                        self.lora_vision_model.load_state_dict(
                            model_state["lora_vision_state_dict"]
                        )
                        logger.info("LoRA vision model loaded from state dict")
            except Exception as e:
                logger.warning(f"Failed to load LoRA vision model: {e}")
                self.lora_vision_model = None

        # Recreate classifier model with correct configuration
        self.classifier_model = ArtStyleClassifierModel(
            input_dim=512,  # Default dimension
            num_styles=self.num_styles,
            feature_dim=256,
            dropout=0.1,
        )

        # Update auxiliary classifiers
        self.classifier_model.studio_classifier = nn.Linear(256, len(self.studio_vocab))
        self.classifier_model.era_classifier = nn.Linear(256, len(self.era_vocab))

        # Load state dict
        self.classifier_model.load_state_dict(model_state["model_state_dict"])
        self.classifier_model = self.classifier_model.to(self.device)

        # Setup optimizer with both models
        trainable_params = list(self.classifier_model.parameters())
        if self.lora_vision_model is not None:
            trainable_params.extend(self.lora_vision_model.parameters())

        self.optimizer = AdamW(trainable_params, lr=1e-4, weight_decay=1e-5)
        self.optimizer.load_state_dict(model_state["optimizer_state_dict"])

        logger.info(f"Art style classifier loaded from {load_path}")

    def get_model_info(self) -> dict[str, Any]:
        """Get model information including LoRA enhancement status.

        Returns:
            Model information dictionary
        """
        info = {
            "num_styles": self.num_styles,
            "style_vocab_size": len(self.style_vocab),
            "studio_vocab_size": len(self.studio_vocab),
            "era_vocab_size": len(self.era_vocab),
            "is_trained": self.is_trained,
            "device": str(self.device),
            "model_type": "art_style_classifier",
            "has_lora_vision": self.lora_vision_model is not None,
        }

        # Add LoRA-specific information
        if self.lora_vision_model is not None:
            try:
                lora_config = self.lora_vision_model.lora_config
                info.update(
                    {
                        "lora_rank": lora_config.r,
                        "lora_alpha": lora_config.lora_alpha,
                        "lora_dropout": lora_config.lora_dropout,
                        "lora_target_modules": lora_config.target_modules,
                        "vision_enhancement": "LoRA-enhanced",
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to get LoRA info: {e}")
                info["vision_enhancement"] = "LoRA-enhanced (info unavailable)"
        else:
            info["vision_enhancement"] = "Standard"

        return info
