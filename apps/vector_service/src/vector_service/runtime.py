"""Build runtime dependencies for vector_service.

This module defines the runtime container and startup factory used by
vector_service. It initializes model processors and Qdrant clients required
by route handlers.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass

from common.config import QdrantConfig, Settings
from qdrant_client import AsyncQdrantClient
from qdrant_db import QdrantClient
from qdrant_db.errors import ConfigurationError
from vector_processing import (
    AnimeFieldMapper,
    EmbeddingCache,
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
    embedding_cache: EmbeddingCache | None
    record_query_text: bool = False


class GpuUnavailableError(RuntimeError):
    """Raised when ENABLE_GPU is true but PyTorch cannot use a GPU."""

    def __init__(self, cuda_build: str | None) -> None:
        if cuda_build is None:
            reason = "PyTorch is a CPU-only build"
        else:
            reason = (
                f"PyTorch (CUDA {cuda_build}) finds no usable CUDA device; check "
                "the NVIDIA driver, and in a container the NVIDIA Container "
                "Toolkit and the GPU reservation"
            )
        super().__init__(
            f"ENABLE_GPU is true but {reason}; the models would run on the CPU, "
            "far slower. Fix the GPU setup, or set ENABLE_GPU=false to run on the "
            "CPU on purpose"
        )


def _require_usable_gpu(cuda_available: bool, cuda_build: str | None) -> None:
    """Stop start-up when a GPU is requested but PyTorch cannot use one.

    Args:
        cuda_available: ``torch.cuda.is_available()``.
        cuda_build: ``torch.version.cuda``; ``None`` for a CPU-only build.

    Raises:
        GpuUnavailableError: If no CUDA device is usable.
    """
    if not cuda_available:
        raise GpuUnavailableError(cuda_build)


def _validate_model_dimensions(
    settings: Settings,
    text_processor: TextProcessor,
    vision_processor: VisionProcessor,
) -> None:
    """Assert model output dimensions match vector_names in config.

    Raises:
        ConfigurationError: If any model's embedding size differs from the
            configured dimension for its primary vector name.
    """
    checks = (
        (
            settings.qdrant.primary_text_vector_name,
            text_processor.model.embedding_size,
        ),
        (
            settings.qdrant.primary_image_vector_name,
            vision_processor.model.embedding_size,
        ),
    )
    for vector_name, actual_dim in checks:
        expected_dim = settings.qdrant.vector_names[vector_name]
        if actual_dim != expected_dim:
            raise ConfigurationError(
                f"Embedding dimension mismatch for '{vector_name}': "
                f"model produces {actual_dim}-dim vectors but config expects {expected_dim}. "
                f"Update vector_names['{vector_name}'] or change the embedding model."
            )


async def _warm_up_models(
    text_processor: TextProcessor, vision_processor: VisionProcessor
) -> None:
    """Run each model once so the first searches do not pay for it.

    A model's first pass is several times slower than later ones (~230 ms
    against ~15 ms for a whole search on the laptop GPU, ECHO-54 finding 37).
    Runs before the service reports healthy.
    """
    from PIL import Image

    started = time.perf_counter()
    await text_processor.encode_text_with_sparse("warm up")
    await asyncio.to_thread(
        vision_processor.model.encode_image, [Image.new("RGB", (224, 224))]
    )
    logger.info(f"Models warmed up in {time.perf_counter() - started:.1f} s")


def _create_qdrant_client(qdrant_settings: QdrantConfig) -> AsyncQdrantClient:
    """Create the async Qdrant client over HTTP, or gRPC when preferred.

    ``cloud_inference=True`` stops the client from searching every request for
    ``Document`` or ``Image`` objects to embed locally with FastEmbed. The
    service always sends finished vectors, and that search walks every number
    of every query vector: ~1.2 ms of CPU per hybrid query in qdrant-client
    1.19.1. Nothing is sent to Qdrant for inference unless a request holds
    such an object.
    """
    return AsyncQdrantClient(
        url=qdrant_settings.qdrant_url,
        api_key=qdrant_settings.qdrant_api_key,
        prefer_grpc=qdrant_settings.qdrant_prefer_grpc,
        grpc_port=qdrant_settings.qdrant_grpc_port,
        cloud_inference=True,
    )


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

    if settings.service.enable_gpu:
        import torch

        _require_usable_gpu(torch.cuda.is_available(), torch.version.cuda)
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    async_qdrant_client: AsyncQdrantClient | None = None
    embedding_cache: EmbeddingCache | None = None
    try:
        async_qdrant_client = _create_qdrant_client(settings.qdrant)

        # Build optional embedding cache from Redis config
        if settings.redis.redis_url:
            from redis.asyncio import Redis

            redis_client = Redis.from_url(
                settings.redis.redis_url,
                max_connections=settings.redis.redis_max_connections,
                socket_connect_timeout=settings.redis.redis_socket_connect_timeout,
                socket_timeout=settings.redis.redis_socket_timeout,
            )
            embedding_cache = EmbeddingCache(redis_client)
            logger.info(f"Embedding cache enabled (Redis: {settings.redis.redis_url})")
        else:
            logger.info("Embedding cache disabled (no REDIS_URL configured)")

        text_model = EmbeddingModelFactory.create_text_model(settings.embedding)
        vision_model = EmbeddingModelFactory.create_vision_model(settings.embedding)
        image_downloader = ImageDownloader(cache_dir=settings.embedding.model_cache_dir)
        field_mapper = AnimeFieldMapper()
        text_processor = TextProcessor(
            text_model, settings.embedding, embedding_cache=embedding_cache
        )
        vision_processor = VisionProcessor(
            vision_model,
            image_downloader,
            settings.embedding,
            embedding_cache=embedding_cache,
        )
        embedding_manager = MultiVectorEmbeddingManager(
            text_processor=text_processor,
            vision_processor=vision_processor,
            field_mapper=field_mapper,
        )

        # Verify model output dimensions match configured vector_names before
        # any Qdrant I/O. Catches model/config drift (e.g. wrong IMAGE_EMBEDDING_MODEL
        # env var) at the earliest possible moment — before collection init or writes.
        _validate_model_dimensions(settings, text_processor, vision_processor)
        if settings.embedding.model_warm_up:
            await _warm_up_models(text_processor, vision_processor)

        telemetry_registry = None
        if settings.observability.otel_enabled:
            from observability import registry

            telemetry_registry = registry

        qdrant_client = await QdrantClient.create(
            config=settings.qdrant,
            async_qdrant_client=async_qdrant_client,
            url=settings.qdrant.qdrant_url,
            collection_name=settings.qdrant.qdrant_collection_name,
            telemetry=telemetry_registry,
        )
        return VectorRuntime(
            qdrant_client=qdrant_client,
            async_qdrant_client=async_qdrant_client,
            text_processor=text_processor,
            vision_processor=vision_processor,
            embedding_manager=embedding_manager,
            embedding_cache=embedding_cache,
            record_query_text=settings.observability.otel_record_query_text,
        )
    except Exception:
        if async_qdrant_client is not None:
            await async_qdrant_client.close()
        if embedding_cache is not None:
            await embedding_cache.close()
        raise
