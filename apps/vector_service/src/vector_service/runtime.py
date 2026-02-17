"""Build runtime dependencies for vector_service.

This module defines the runtime container and startup factory used by
vector_service. It initializes model processors and Qdrant clients required
by route handlers.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from common.config import Settings
from qdrant_client import AsyncQdrantClient
from qdrant_db import QdrantClient
from vector_processing import (
    AnimeFieldMapper,
    MultiVectorEmbeddingManager,
    TextProcessor,
    VisionProcessor,
)
from vector_processing.embedding_models.factory import EmbeddingModelFactory
from vector_processing.utils.image_downloader import ImageDownloader

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class VectorRuntime:
    """Runtime dependencies owned by vector_service."""

    qdrant_client: QdrantClient
    async_qdrant_client: AsyncQdrantClient
    text_processor: TextProcessor
    vision_processor: VisionProcessor
    embedding_manager: MultiVectorEmbeddingManager


async def build_runtime(settings: Settings) -> VectorRuntime:
    """Initialize runtime state for vector_service.

    Args:
        settings: Resolved application settings with model and Qdrant config.

    Returns:
        Fully initialized runtime dependencies.

    Raises:
        Exception: If any dependency initialization fails.
    """
    logger.info("Initializing vector_service runtime dependencies")

    if not settings.service.enable_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    async_qdrant_client: AsyncQdrantClient | None = None
    try:
        if settings.qdrant.qdrant_api_key:
            async_qdrant_client = AsyncQdrantClient(
                url=settings.qdrant.qdrant_url,
                api_key=settings.qdrant.qdrant_api_key,
            )
        else:
            async_qdrant_client = AsyncQdrantClient(url=settings.qdrant.qdrant_url)

        text_model = EmbeddingModelFactory.create_text_model(settings.embedding)
        vision_model = EmbeddingModelFactory.create_vision_model(settings.embedding)
        image_downloader = ImageDownloader(cache_dir=settings.embedding.model_cache_dir)
        field_mapper = AnimeFieldMapper()
        text_processor = TextProcessor(text_model, settings.embedding)
        vision_processor = VisionProcessor(
            vision_model, image_downloader, settings.embedding
        )
        embedding_manager = MultiVectorEmbeddingManager(
            text_processor=text_processor,
            vision_processor=vision_processor,
            field_mapper=field_mapper,
        )
        qdrant_client = await QdrantClient.create(
            config=settings.qdrant,
            async_qdrant_client=async_qdrant_client,
            url=settings.qdrant.qdrant_url,
            collection_name=settings.qdrant.qdrant_collection_name,
        )
        return VectorRuntime(
            qdrant_client=qdrant_client,
            async_qdrant_client=async_qdrant_client,
            text_processor=text_processor,
            vision_processor=vision_processor,
            embedding_manager=embedding_manager,
        )
    except Exception:
        if async_qdrant_client is not None:
            await async_qdrant_client.close()
        raise
