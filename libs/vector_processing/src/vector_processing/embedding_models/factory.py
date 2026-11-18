"""Factory for creating text and vision embedding model instances."""

import logging
from collections.abc import Callable

from common.config import EmbeddingConfig

from .text.base import TextEmbeddingModel
from .text.fastembed_model import FastEmbedModel
from .text.flagembedding_model import FlagEmbeddingModel
from .text.huggingface_model import HuggingFaceModel
from .text.sentence_transformer_model import SentenceTransformerModel
from .vision.base import VisionEmbeddingModel
from .vision.openclip_model import OpenClipModel

logger = logging.getLogger(__name__)


def _build_flagembedding_model(config: EmbeddingConfig) -> TextEmbeddingModel:
    return FlagEmbeddingModel(
        config.text_embedding_model,
        cache_dir=config.model_cache_dir,
        max_length=config.bge_max_length,
        chunk_size=config.embed_model_chunk_size,
        max_tokens_per_pass=config.embed_model_max_tokens_per_pass,
    )


def _build_fastembed_model(config: EmbeddingConfig) -> TextEmbeddingModel:
    return FastEmbedModel(config.text_embedding_model, cache_dir=config.model_cache_dir)


def _build_huggingface_model(config: EmbeddingConfig) -> TextEmbeddingModel:
    return HuggingFaceModel(
        config.text_embedding_model, cache_dir=config.model_cache_dir
    )


def _build_sentence_transformer_model(config: EmbeddingConfig) -> TextEmbeddingModel:
    return SentenceTransformerModel(
        config.text_embedding_model, cache_dir=config.model_cache_dir
    )


def _build_openclip_model(config: EmbeddingConfig) -> VisionEmbeddingModel:
    return OpenClipModel(
        config.image_embedding_model,
        cache_dir=config.model_cache_dir,
        batch_size=config.image_batch_size,
    )


_TEXT_MODEL_BUILDERS: dict[str, Callable[[EmbeddingConfig], TextEmbeddingModel]] = {
    "flagembedding": _build_flagembedding_model,
    "fastembed": _build_fastembed_model,
    "huggingface": _build_huggingface_model,
    "sentence-transformers": _build_sentence_transformer_model,
}

_VISION_MODEL_BUILDERS: dict[str, Callable[[EmbeddingConfig], VisionEmbeddingModel]] = {
    "openclip": _build_openclip_model,
}


class EmbeddingModelFactory:
    """Factory for creating embedding models based on configuration."""

    @staticmethod
    def create_text_model(config: EmbeddingConfig) -> TextEmbeddingModel:
        """Create text embedding model based on config.

        Args:
            config: Embedding configuration.

        Returns:
            Initialized TextEmbeddingModel.

        Raises:
            ValueError: If the configured text embedding provider is unsupported.
        """
        provider = config.text_embedding_provider
        model_name = config.text_embedding_model
        logger.info(f"Creating text embedding model: {provider} - {model_name}")

        model_builder = _TEXT_MODEL_BUILDERS.get(provider)
        if model_builder is None:
            raise ValueError(f"Unsupported text embedding provider: {provider}")
        return model_builder(config)

    @staticmethod
    def create_vision_model(config: EmbeddingConfig) -> VisionEmbeddingModel:
        """Create vision embedding model based on config.

        Args:
            config: Embedding configuration.

        Returns:
            Initialized VisionEmbeddingModel.

        Raises:
            ValueError: If the configured vision embedding provider is unsupported.
        """
        provider = config.image_embedding_provider
        model_name = config.image_embedding_model
        logger.info(f"Creating vision embedding model: {provider} - {model_name}")

        model_builder = _VISION_MODEL_BUILDERS.get(provider)
        if model_builder is None:
            raise ValueError(f"Unsupported vision embedding provider: {provider}")
        return model_builder(config)
