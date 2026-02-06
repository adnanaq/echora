"""Anime Vector Service - FastAPI application for vector database operations.

This microservice provides semantic search capabilities using Qdrant vector
database with multi-modal embeddings (text + image) for anime content.
"""

import logging
import os
from contextlib import asynccontextmanager

# Disable CUDA to force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any

from common.config import get_settings
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from http_cache.instance import http_cache_manager
from http_cache.result_cache import close_result_cache_redis_client
from qdrant_client import AsyncQdrantClient
from qdrant_db import QdrantClient
from vector_db_interface import VectorDBClient
from vector_processing import (
    AnimeFieldMapper,
    MultiVectorEmbeddingManager,
    TextProcessor,
    VisionProcessor,
)
from vector_processing.embedding_models.factory import EmbeddingModelFactory
from vector_processing.utils.image_downloader import ImageDownloader

from .dependencies import get_vector_db_client
from .routes import admin

# Get application settings
settings = get_settings()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.service.log_level),
    format=settings.service.log_format,
)
logger = logging.getLogger(__name__)


# No more global qdrant_client


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle.

    Loads embedding models (BGE-M3 and OpenCLIP ViT-L/14), initializes the
    Qdrant client, and stores all processors on ``app.state`` for dependency
    injection. Ensures proper cleanup of the AsyncQdrantClient connection on
    shutdown.

    Args:
        app: FastAPI application instance.

    Yields:
        Control back to the framework after successful initialization.

    Raises:
        RuntimeError: If the vector database health check fails on startup.
    """
    logger.info("Initializing Qdrant client, embedding models, and dependencies...")

    async_qdrant_client = None
    try:
        # Initialize AsyncQdrantClient from qdrant-client library
        if settings.qdrant.qdrant_api_key:
            async_qdrant_client = AsyncQdrantClient(
                url=settings.qdrant.qdrant_url,
                api_key=settings.qdrant.qdrant_api_key,
            )
        else:
            async_qdrant_client = AsyncQdrantClient(url=settings.qdrant.qdrant_url)

        # Initialize embedding processors
        logger.info("Loading embedding models...")

        # Create models via factory
        text_model = EmbeddingModelFactory.create_text_model(settings.embedding)
        vision_model = EmbeddingModelFactory.create_vision_model(settings.embedding)

        # Create utilities
        image_downloader = ImageDownloader(cache_dir=settings.embedding.model_cache_dir)
        field_mapper = AnimeFieldMapper()

        # Initialize processors with injected dependencies
        text_processor = TextProcessor(text_model, settings.embedding)
        vision_processor = VisionProcessor(
            vision_model, image_downloader, settings.embedding
        )

        embedding_manager = MultiVectorEmbeddingManager(
            text_processor=text_processor,
            vision_processor=vision_processor,
            field_mapper=field_mapper,
        )
        logger.info("Embedding models loaded successfully")

        # Initialize QdrantClient
        qdrant_client_instance = await QdrantClient.create(
            config=settings.qdrant,
            async_qdrant_client=async_qdrant_client,
            url=settings.qdrant.qdrant_url,
            collection_name=settings.qdrant.qdrant_collection_name,
        )

        # Store all clients and processors on app state
        app.state.qdrant_client = qdrant_client_instance
        app.state.async_qdrant_client = async_qdrant_client
        app.state.embedding_manager = embedding_manager
        app.state.text_processor = text_processor
        app.state.vision_processor = vision_processor
        app.state.field_mapper = field_mapper

        # Health check
        healthy = await app.state.qdrant_client.health_check()
        if not healthy:
            logger.error("Qdrant health check failed!")
            raise RuntimeError("Vector database is not available")

        logger.info(
            "Vector service initialized successfully with embedding models ready"
        )
        yield

    finally:
        # Cleanup on shutdown - guaranteed to run
        logger.info("Shutting down vector service and closing clients...")
        if async_qdrant_client:
            try:
                await async_qdrant_client.close()
                logger.info("AsyncQdrantClient closed successfully")
            except Exception:
                logger.exception("Error closing AsyncQdrantClient")

        try:
            await http_cache_manager.close_async()
            logger.info("HTTP cache client closed successfully")
        except Exception:
            logger.exception("Error closing HTTP cache manager")

        try:
            await close_result_cache_redis_client()
            logger.info("Result cache Redis client closed successfully")
        except Exception:
            logger.exception("Error closing result cache Redis client")


# Create FastAPI app
app = FastAPI(
    title=settings.service.api_title,
    description=settings.service.api_description,
    version=settings.service.api_version,
    lifespan=lifespan,
)

# Add CORS middleware
# Known FastAPI/Starlette typing issue with _MiddlewareFactory
# See: https://github.com/fastapi/fastapi/discussions/10968
app.add_middleware(
    CORSMiddleware,  # ty: ignore[invalid-argument-type]
    allow_origins=settings.service.allowed_origins,
    allow_credentials=True,
    allow_methods=settings.service.allowed_methods,
    allow_headers=settings.service.allowed_headers,
)


@app.get("/health")
async def health_check(
    db_client: VectorDBClient = Depends(get_vector_db_client),
) -> dict[str, Any]:
    """Return service health status with database diagnostics.

    Checks the vector database connection and optionally retrieves collection
    statistics. Used by load balancers, Kubernetes probes, and admin dashboards.

    Args:
        db_client: Injected vector database client.

    Returns:
        Health status dict including service metadata, database connectivity,
        and collection-level statistics when available.

    Raises:
        HTTPException: 503 if the database client is unreachable or an
            unexpected error occurs during the check.
    """
    try:
        db_healthy = await db_client.health_check()

        stats: dict[str, Any] = {}
        try:
            stats = await db_client.get_stats()
        except Exception:
            logger.warning("Could not retrieve stats during health check")

        return {
            "status": "healthy" if db_healthy else "unhealthy",
            "timestamp": datetime.now(UTC).isoformat(),
            "service": "anime-vector-service",
            "version": settings.service.api_version,
            "database": {
                "healthy": db_healthy,
                "collection_name": stats.get("collection_name", "unknown"),
                "document_count": stats.get("total_documents", 0),
            },
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("Health check failed")
        raise HTTPException(status_code=503, detail="Service unhealthy")


# Include API routers
app.include_router(admin.router, prefix="/api/v1/admin", tags=["admin"])


@app.get("/")
async def root() -> dict[str, Any]:
    """Return service metadata and a directory of available endpoints.

    Returns:
        Dict with service name, version, and endpoint paths.
    """
    return {
        "service": "Anime Vector Service",
        "version": settings.service.api_version,
        "description": "Microservice for anime vector database operations",
        "endpoints": {
            "health": "/health",
            "stats": "/api/v1/admin/stats",
            "collection": "/api/v1/admin/collection",
            "docs": "/docs",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "service.main:app",
        host=settings.service.vector_service_host,
        port=settings.service.vector_service_port,
        reload=settings.debug,
        log_level=settings.service.log_level.lower(),
    )
