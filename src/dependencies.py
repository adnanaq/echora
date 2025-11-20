import logging
from typing import AsyncGenerator

from fastapi import Request

from src.vector.client.qdrant_client import QdrantClient

logger = logging.getLogger(__name__)


async def get_qdrant_client(request: Request) -> AsyncGenerator[QdrantClient, None]:
    """
    Dependency that provides a QdrantClient instance.

    The QdrantClient is initialized in the FastAPI lifespan event
    and stored in the app's state. This dependency retrieves it.
    """
    if not hasattr(request.app.state, "qdrant_client") or request.app.state.qdrant_client is None:
        logger.error("QdrantClient not initialized in app state.")
        raise RuntimeError("QdrantClient not available.")
    yield request.app.state.qdrant_client
