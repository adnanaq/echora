import logging

from common.config import Settings

from .text.base import TextEmbeddingModel
from .text.fastembed_model import FastEmbedModel
from .text.huggingface_model import HuggingFaceModel
from .text.sentence_transformer_model import SentenceTransformerModel
from .vision.base import VisionEmbeddingModel
from .vision.openclip_model import OpenClipModel

logger = logging.getLogger(__name__)


class EmbeddingModelFactory:
    """Factory for creating embedding models based on configuration."""

    @staticmethod
    def create_text_model(settings: Settings) -> TextEmbeddingModel:
        """Create text embedding model based on settings.

        Args:
            settings: Configuration settings

        Returns:
            Initialized TextEmbeddingModel
        """
        provider = settings.text_embedding_provider
        model_name = settings.text_embedding_model
        cache_dir = settings.model_cache_dir

        logger.info(f"Creating text embedding model: {provider} - {model_name}")

        if provider == "fastembed":
            return FastEmbedModel(model_name, cache_dir=cache_dir)
        elif provider == "huggingface":
            return HuggingFaceModel(model_name, cache_dir=cache_dir)
        elif provider == "sentence-transformers":
            return SentenceTransformerModel(model_name, cache_dir=cache_dir)
        else:
            raise ValueError(f"Unsupported text embedding provider: {provider}")

    @staticmethod
    def create_vision_model(settings: Settings) -> VisionEmbeddingModel:
        """Create vision embedding model based on settings.

        Args:
            settings: Configuration settings

        Returns:
            Initialized VisionEmbeddingModel
        """
        provider = settings.image_embedding_provider
        model_name = settings.image_embedding_model
        cache_dir = settings.model_cache_dir
        batch_size = getattr(settings, "image_batch_size", 16)

        logger.info(f"Creating vision embedding model: {provider} - {model_name}")

        if provider == "openclip":
            return OpenClipModel(model_name, cache_dir=cache_dir, batch_size=batch_size)
        else:
            raise ValueError(f"Unsupported vision embedding provider: {provider}")
