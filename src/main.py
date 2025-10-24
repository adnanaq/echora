"""
Anime Vector Service - FastAPI application for vector database operations.

This microservice provides semantic search capabilities using Qdrant vector database
with multi-modal embeddings (text + image) for anime content.
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .api import admin
from .config import get_settings
from .vector.client.vector_db_client import VectorDBClient
from .vector.client.vector_db_factory import VectorDBClientFactory
from .vector.processors.embedding_manager import MultiVectorEmbeddingManager

# Get application settings
settings = get_settings()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level), format=settings.log_format
)
logger = logging.getLogger(__name__)

# Global instances
vector_db_client: VectorDBClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize services on startup"""
    global vector_db_client

    # Initialize embedding manager
    logger.info("Initializing embedding manager...")
    embedding_manager = MultiVectorEmbeddingManager(settings)

    # Initialize VectorDB client using the factory
    logger.info(f"Initializing {settings.vector_db_provider} client...")
    settings.vector_db_provider = "milvus" # TEMPORARY: For testing Milvus initialization
    vector_db_client = VectorDBClientFactory.get_client(
        embedding_manager=embedding_manager,
        settings=settings,
        url=settings.qdrant_url if settings.vector_db_provider == "qdrant" else None, # Pass Qdrant URL only if Qdrant
        collection_name=settings.qdrant_collection_name if settings.vector_db_provider == "qdrant" else None, # Pass Qdrant collection name only if Qdrant
    )

    # Health check
    healthy = await vector_db_client.health_check()
    if not healthy:
        logger.error(f"{settings.vector_db_provider} health check failed!")
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
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    try:
        if vector_db_client is None:
            raise HTTPException(status_code=503, detail="VectorDB client not initialized")

        db_status = await vector_db_client.health_check()
        return {
            "status": "healthy" if db_status else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "anime-vector-service",
            "version": settings.api_version,
            f"{settings.vector_db_provider}_status": db_status,
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
