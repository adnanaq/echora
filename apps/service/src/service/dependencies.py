import logging

from fastapi import Request
from vector_db_interface import VectorDBClient
from vector_processing import (
    MultiVectorEmbeddingManager,
    TextProcessor,
    VisionProcessor,
)

logger = logging.getLogger(__name__)


async def get_vector_db_client(request: Request) -> VectorDBClient:
    """
    Dependency that provides a VectorDBClient instance.

    The vector DB client is initialized in the FastAPI lifespan event and stored
    in the app's state. This dependency retrieves it. No cleanup is needed
    here since the client lifecycle is managed by the lifespan context manager.

    Args:
        request: FastAPI request object containing app state

    Returns:
        Initialized VectorDBClient instance

    Raises:
        RuntimeError: If vector DB client not available in app state
    """
    if (
        not hasattr(request.app.state, "qdrant_client")
        or request.app.state.qdrant_client is None
    ):
        logger.error("Vector DB client not initialized in app state.")
        raise RuntimeError("Vector DB client not available.")
    return request.app.state.qdrant_client


async def get_embedding_manager(request: Request) -> MultiVectorEmbeddingManager:
    """
    Dependency that provides a MultiVectorEmbeddingManager instance.

    The embedding manager is initialized in the FastAPI lifespan event and stored
    in the app's state. This dependency retrieves it. No cleanup is needed here
    since the lifecycle is managed by lifespan.

    Args:
        request: FastAPI request object containing app state

    Returns:
        Initialized MultiVectorEmbeddingManager instance

    Raises:
        RuntimeError: If EmbeddingManager not available in app state
    """
    if (
        not hasattr(request.app.state, "embedding_manager")
        or request.app.state.embedding_manager is None
    ):
        logger.error("EmbeddingManager not initialized in app state.")
        raise RuntimeError("EmbeddingManager not available.")
    return request.app.state.embedding_manager


async def get_text_processor(request: Request) -> TextProcessor:
    """
    Dependency that provides a TextProcessor instance.

    The text processor is initialized in the FastAPI lifespan event and stored
    in the app's state. This dependency retrieves it. No cleanup is needed here
    since the lifecycle is managed by lifespan.

    Args:
        request: FastAPI request object containing app state

    Returns:
        Initialized TextProcessor instance

    Raises:
        RuntimeError: If TextProcessor not available in app state
    """
    if (
        not hasattr(request.app.state, "text_processor")
        or request.app.state.text_processor is None
    ):
        logger.error("TextProcessor not initialized in app state.")
        raise RuntimeError("TextProcessor not available.")
    return request.app.state.text_processor


async def get_vision_processor(request: Request) -> VisionProcessor:
    """
    Dependency that provides a VisionProcessor instance.

    The vision processor is initialized in the FastAPI lifespan event and stored
    in the app's state. This dependency retrieves it. No cleanup is needed here
    since the lifecycle is managed by lifespan.

    Args:
        request: FastAPI request object containing app state

    Returns:
        Initialized VisionProcessor instance

    Raises:
        RuntimeError: If VisionProcessor not available in app state
    """
    if (
        not hasattr(request.app.state, "vision_processor")
        or request.app.state.vision_processor is None
    ):
        logger.error("VisionProcessor not initialized in app state.")
        raise RuntimeError("VisionProcessor not available.")
    return request.app.state.vision_processor
