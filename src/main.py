"""
Anime Vector Service - FastAPI application for vector database operations.

This microservice provides semantic search capabilities using Qdrant vector database
with multi-modal embeddings (text + image) for anime content.
"""

import logging
import os
from contextlib import asynccontextmanager

# Disable CUDA to force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from datetime import datetime
from typing import Any, AsyncGenerator, Dict

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware

from .api import admin
from .config import get_settings
from .vector.client.qdrant_client import QdrantClient
from .vector.processors.embedding_manager import MultiVectorEmbeddingManager
from .vector.processors.text_processor import TextProcessor
from .vector.processors.vision_processor import VisionProcessor
from qdrant_client import AsyncQdrantClient as QdrantSDK
# from src.cache_manager.instance import http_cache_manager
# from src.cache_manager.result_cache import close_result_cache_redis_client
from .dependencies import get_qdrant_client # New import

# Get application settings
settings = get_settings()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level), format=settings.log_format
)
logger = logging.getLogger(__name__)



# No more global qdrant_client

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize services on startup and cleanup on shutdown."""
    logger.info("Initializing Qdrant client and its dependencies...")

    # Initialize Qdrant SDK client
    if settings.qdrant_api_key:
        qdrant_sdk_client = QdrantSDK(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    else:
        qdrant_sdk_client = QdrantSDK(url=settings.qdrant_url)

    # Initialize embedding manager and processors
    text_processor = TextProcessor(settings)
    vision_processor = VisionProcessor(settings)
    embedding_manager = MultiVectorEmbeddingManager(
        text_processor=text_processor,
        vision_processor=vision_processor,
        settings=settings
    )

    # Initialize Qdrant client
    qdrant_client_instance = await QdrantClient.create(
        settings=settings,
        qdrant_sdk_client=qdrant_sdk_client,
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection_name,
    )
    
    # Store the client on the app state
    app.state.qdrant_client = qdrant_client_instance

    # Health check
    healthy = await app.state.qdrant_client.health_check()
    if not healthy:
        logger.error("Qdrant health check failed!")
        raise RuntimeError("Vector database is not available")

    logger.info("Vector service initialized successfully")
    yield

    # Cleanup on shutdown
    logger.info("Shutting down vector service...")


# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=settings.allowed_methods,
    allow_headers=settings.allowed_headers,
)


# Health check endpoint
@app.get("/health")
async def health_check(qdrant_client: QdrantClient = Depends(get_qdrant_client)) -> Dict[str, Any]:
    """Health check endpoint."""
    try:
        qdrant_status = await qdrant_client.health_check()
        return {
            "status": "healthy" if qdrant_status else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "anime-vector-service",
            "version": settings.api_version,
            "qdrant_status": qdrant_status,
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


# Include API routers
app.include_router(admin.router, prefix="/api/v1/admin", tags=["admin"])


# Root endpoint
@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint with service information."""
    return {
        "service": "Anime Vector Service",
        "version": settings.api_version,
        "description": "Microservice for anime vector database operations",
        "endpoints": {
            "health": "/health",
            "search": "/api/v1/search",
            "image_search": "/api/v1/search/image",
            "multimodal_search": "/api/v1/search/multimodal",
            "similar": "/api/v1/similarity/anime/{anime_id}",
            "visual_similar": "/api/v1/similarity/visual/{anime_id}",
            "stats": "/api/v1/admin/stats",
            "docs": "/docs",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.vector_service_host,
        port=settings.vector_service_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
