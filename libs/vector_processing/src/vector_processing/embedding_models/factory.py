import logging

from common.config import EmbeddingConfig
from vector_processing.reranking import RerankerModel, SentenceTransformerReranker

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
    def create_text_model(config: EmbeddingConfig) -> TextEmbeddingModel:
        """Create text embedding model based on config.

        Args:
            config: Embedding configuration

        Returns:
            Initialized TextEmbeddingModel
        """
        provider = config.text_embedding_provider
        model_name = config.text_embedding_model
        cache_dir = config.model_cache_dir

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
    def create_vision_model(config: EmbeddingConfig) -> VisionEmbeddingModel:
        """Create vision embedding model based on config.

        Args:
            config: Embedding configuration

        Returns:
            Initialized VisionEmbeddingModel
        """
        provider = config.image_embedding_provider
        model_name = config.image_embedding_model
        cache_dir = config.model_cache_dir
        batch_size = config.image_batch_size

        logger.info(f"Creating vision embedding model: {provider} - {model_name}")

        if provider == "openclip":
            return OpenClipModel(model_name, cache_dir=cache_dir, batch_size=batch_size)
        else:
            raise ValueError(f"Unsupported vision embedding provider: {provider}")

    @staticmethod
    def create_reranker_model(config: EmbeddingConfig) -> RerankerModel:
        """Create reranker model from configuration.

        Args:
            config: Embedding configuration.

        Returns:
            Initialized reranker model.

        Raises:
            ValueError: If reranking provider is unsupported.
        """
        provider = config.reranking_provider.lower()

        if provider == "sentence-transformers":
            return SentenceTransformerReranker(
                model_name=config.reranking_model,
                cache_dir=config.model_cache_dir,
                max_length=512,  # Standard for most rerankers
            )

        raise ValueError(f"Unsupported reranking provider: {provider}")
