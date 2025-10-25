"""Anime-specific fine-tuning orchestrator for domain-specific model optimization.

This module implements parameter-efficient fine-tuning techniques for anime content,
including character recognition, art style classification, and genre understanding.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from peft import LoraConfig, TaskType
from torch.utils.data import DataLoader

from ...config import Settings
from ..processors.text_processor import TextProcessor
from ..processors.vision_processor import VisionProcessor
from .anime_dataset import AnimeDataset
from .art_style_classifier import ArtStyleClassifier
from .character_recognition import CharacterRecognitionFinetuner
from .genre_enhancement import GenreEnhancementFinetuner

logger = logging.getLogger(__name__)


@dataclass
class FineTuningConfig:
    """Configuration for anime fine-tuning."""

    # Model configuration
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "out_proj"]
    )

    # Training configuration
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 3
    warmup_steps: int = 100

    # Data configuration
    max_text_length: int = 512
    image_size: int = 224
    train_split: float = 0.8
    validation_split: float = 0.1

    # Task weights
    character_weight: float = 0.4
    art_style_weight: float = 0.3
    genre_weight: float = 0.3

    # Output configuration
    save_model: bool = True
    model_output_dir: str = "models/anime_finetuned"
    checkpoint_steps: int = 500


class AnimeFineTuner:
    """Main orchestrator for anime-specific fine-tuning."""

    def __init__(self, settings: Settings):
        """Initialize anime fine-tuner.

        Args:
            settings: Configuration settings instance
        """
        self.settings = settings
        self.config = FineTuningConfig()

        # Initialize processors
        self.text_processor = TextProcessor(settings)
        self.vision_processor = VisionProcessor(settings)

        # Initialize specialized fine-tuners
        self.character_finetuner = CharacterRecognitionFinetuner(
            settings, self.text_processor, self.vision_processor
        )
        self.art_style_classifier = ArtStyleClassifier(settings, self.vision_processor)
        self.genre_enhancer = GenreEnhancementFinetuner(settings, self.text_processor)

        # Training state
        self.training_stats: dict[str, Any] = {}
        self.best_model_path: Path | None = None

    def prepare_dataset(self, data_path: str) -> AnimeDataset | None:
        """Prepare anime dataset for fine-tuning.

        Args:
            data_path: Path to anime data file

        Returns:
            Prepared anime dataset or None if an error occurs
        """
        logger.info(f"Preparing anime dataset from {data_path}")
        try:
            with open(data_path, encoding="utf-8") as f:
                anime_data = json.load(f)
        except FileNotFoundError:
            logger.error(f"Data file not found at path: {data_path}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {data_path}: {e}")
            return None
        except Exception as e:
            logger.error(
                f"An unexpected error occurred while preparing the dataset: {e}"
            )
            return None

        # Create dataset
        dataset = AnimeDataset(
            anime_data=anime_data,
            text_processor=self.text_processor,
            vision_processor=self.vision_processor,
            config=self.config,
        )

        logger.info(f"Prepared dataset with {len(dataset)} samples")
        return dataset

    def create_lora_config(self, task_type: TaskType) -> LoraConfig:
        """Create LoRA configuration for parameter-efficient fine-tuning.

        Args:
            task_type: Type of task (feature extraction, classification, etc.)

        Returns:
            LoRA configuration
        """
        return LoraConfig(
            task_type=task_type,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
        )

    def setup_models_for_finetuning(self) -> None:
        """Setup models for fine-tuning with LoRA."""
        logger.info("Setting up models for fine-tuning")

        # Setup character recognition model
        self.character_finetuner.setup_lora_model(
            self.create_lora_config(TaskType.FEATURE_EXTRACTION), self.config
        )

        # Setup art style classification model
        self.art_style_classifier.setup_lora_model(
            self.create_lora_config(TaskType.IMAGE_CLASSIFICATION), self.config
        )

        # Setup genre enhancement model
        self.genre_enhancer.setup_lora_model(
            self.create_lora_config(TaskType.FEATURE_EXTRACTION), self.config
        )

    def train_multi_task(self, dataset: AnimeDataset) -> dict[str, Any]:
        """Train all fine-tuning tasks simultaneously.

        Args:
            dataset: Prepared anime dataset

        Returns:
            Training statistics
        """
        logger.info("Starting multi-task fine-tuning")

        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        # Setup models
        self.setup_models_for_finetuning()

        # Training loop
        training_stats: dict[str, Any] = {
            "character_loss": [],
            "art_style_loss": [],
            "genre_loss": [],
            "total_loss": [],
            "best_epoch": 0,
            "best_loss": float("inf"),
        }

        for epoch in range(self.config.num_epochs):
            epoch_stats = self._train_epoch(dataloader, epoch)

            # Update training stats
            for key, value in epoch_stats.items():
                if key in training_stats and isinstance(training_stats[key], list):
                    training_stats[key].append(value)

            # Save best model
            if epoch_stats["total_loss"] < training_stats["best_loss"]:
                training_stats["best_loss"] = epoch_stats["total_loss"]
                training_stats["best_epoch"] = epoch
                if self.config.save_model:
                    self._save_best_model(epoch)

        self.training_stats = training_stats
        logger.info(
            f"Training completed. Best loss: {training_stats['best_loss']:.4f} at epoch {training_stats['best_epoch']}"
        )

        return training_stats

    def _train_epoch(self, dataloader: DataLoader, epoch: int) -> dict[str, float]:
        """Train one epoch with multi-task learning.

        Args:
            dataloader: Data loader for training
            epoch: Current epoch number

        Returns:
            Epoch training statistics
        """
        total_loss = 0.0
        character_loss = 0.0
        art_style_loss = 0.0
        genre_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            # Character recognition task
            char_loss = self.character_finetuner.train_step(batch)
            character_loss += char_loss

            # Art style classification task
            style_loss = self.art_style_classifier.train_step(batch)
            art_style_loss += style_loss

            # Genre enhancement task
            genre_loss_val = self.genre_enhancer.train_step(batch)
            genre_loss += genre_loss_val

            # Combined loss
            batch_loss = (
                char_loss * self.config.character_weight
                + style_loss * self.config.art_style_weight
                + genre_loss_val * self.config.genre_weight
            )
            total_loss += batch_loss
            num_batches += 1

            # Logging
            if batch_idx % 50 == 0:
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}: "
                    f"Total Loss: {batch_loss:.4f}, "
                    f"Character: {char_loss:.4f}, "
                    f"Style: {style_loss:.4f}, "
                    f"Genre: {genre_loss_val:.4f}"
                )

        # Average losses
        return {
            "total_loss": total_loss / num_batches,
            "character_loss": character_loss / num_batches,
            "art_style_loss": art_style_loss / num_batches,
            "genre_loss": genre_loss / num_batches,
        }

    def _save_best_model(self, epoch: int) -> None:
        """Save the best performing model.

        Args:
            epoch: Current epoch number
        """
        try:
            output_dir = Path(self.config.model_output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save individual models
            self.character_finetuner.save_model(output_dir / "character_model")
            self.art_style_classifier.save_model(output_dir / "art_style_model")
            self.genre_enhancer.save_model(output_dir / "genre_model")

            # Save configuration
            config_path = output_dir / "fine_tuning_config.json"
            with open(config_path, "w") as f:
                json.dump(self.config.__dict__, f, indent=2)

            self.best_model_path = Path(output_dir)
            logger.info(f"Saved best model at epoch {epoch} to {output_dir}")
        except PermissionError:
            logger.error(
                f"Permission denied to write to {self.config.model_output_dir}"
            )
        except Exception as e:
            logger.error(f"An unexpected error occurred while saving the model: {e}")

    def evaluate_models(self, validation_dataset: AnimeDataset) -> dict[str, float]:
        """Evaluate fine-tuned models on validation set.

        Args:
            validation_dataset: Validation dataset

        Returns:
            Evaluation metrics
        """
        logger.info("Evaluating fine-tuned models")

        validation_loader = DataLoader(
            validation_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
        )

        # Evaluate each component
        character_metrics = self.character_finetuner.evaluate(validation_loader)
        art_style_metrics = self.art_style_classifier.evaluate(validation_loader)
        genre_metrics = self.genre_enhancer.evaluate(validation_loader)

        # Combined metrics
        combined_metrics = {
            "character_accuracy": character_metrics.get("accuracy", 0.0),
            "art_style_accuracy": art_style_metrics.get("accuracy", 0.0),
            "genre_accuracy": genre_metrics.get("accuracy", 0.0),
            "overall_accuracy": (
                character_metrics.get("accuracy", 0.0) * self.config.character_weight
                + art_style_metrics.get("accuracy", 0.0) * self.config.art_style_weight
                + genre_metrics.get("accuracy", 0.0) * self.config.genre_weight
            ),
        }

        logger.info(f"Evaluation results: {combined_metrics}")
        return combined_metrics

    def load_finetuned_models(self, model_path: str) -> None:
        """Load fine-tuned models from saved path.

        Args:
            model_path: Path to saved models
        """
        logger.info(f"Loading fine-tuned models from {model_path}")
        try:
            model_dir = Path(model_path)

            # Load individual models
            self.character_finetuner.load_model(model_dir / "character_model")
            self.art_style_classifier.load_model(model_dir / "art_style_model")
            self.genre_enhancer.load_model(model_dir / "genre_model")

            # Load configuration
            config_path = model_dir / "fine_tuning_config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config_data = json.load(f)
                    # Update configuration
                    for key, value in config_data.items():
                        setattr(self.config, key, value)

            logger.info("Fine-tuned models loaded successfully")
        except FileNotFoundError as e:
            logger.error(f"Could not load model: {e.filename} not found.")
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading the models: {e}")

    def get_enhanced_embeddings(
        self, text: str, image_data: str | None = None
    ) -> dict[str, np.ndarray]:
        """Get enhanced embeddings using fine-tuned models.

        Args:
            text: Text to encode
            image_data: Optional image data (base64 encoded)

        Returns:
            Enhanced embeddings for different tasks
        """
        embeddings = {}

        # Get base embeddings
        text_embedding = self.text_processor.encode_text(text)
        if image_data:
            image_embedding = self.vision_processor.encode_image(image_data)
        else:
            image_embedding = None

        # Get enhanced embeddings
        if text_embedding is not None:
            text_array = (
                np.array(text_embedding)
                if isinstance(text_embedding, list)
                else text_embedding
            )
            image_array = (
                np.array(image_embedding)
                if isinstance(image_embedding, list) and image_embedding is not None
                else image_embedding
            )
            embeddings["character"] = self.character_finetuner.get_enhanced_embedding(
                text_array, image_array
            )
            embeddings["genre"] = self.genre_enhancer.get_enhanced_embedding(text_array)

        if image_embedding is not None and image_data is not None:
            image_array = (
                np.array(image_embedding)
                if isinstance(image_embedding, list)
                else image_embedding
            )
            embeddings["art_style"] = self.art_style_classifier.get_enhanced_embedding(
                image_data
            )

        return embeddings

    def get_training_summary(self) -> dict[str, Any]:
        """Get comprehensive training summary.

        Returns:
            Training summary with statistics and configuration
        """
        return {
            "config": self.config.__dict__,
            "training_stats": self.training_stats,
            "best_model_path": (
                str(self.best_model_path) if self.best_model_path else None
            ),
            "timestamp": datetime.now().isoformat(),
        }
