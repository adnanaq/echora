"""
Anime Vector Service - gRPC Agent Service entrypoint.

This service initializes the required clients and agents and starts the gRPC server
to handle internal search and AI tasks.
"""

import logging
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from src.poc.atomic_agents_poc import AnimeQueryAgent
from .config import get_settings
from .vector.client.qdrant_client import QdrantClient
from .server import serve_async
from . import globals as app_globals

# Get application settings
settings = get_settings()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level), format=settings.log_format
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan() -> AsyncGenerator[None, None]:
    """
    Initialize services on startup and handles graceful shutdown.
    """
    logger.info("Initializing services...")

    # Initialize Qdrant client
    app_globals.qdrant_client = QdrantClient(
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection_name,
        settings=settings,
    )

    # Initialize AnimeQueryAgent
    app_globals.query_parser_agent = AnimeQueryAgent(
        qdrant_client=app_globals.qdrant_client,
        settings=settings,
    )

    # Health check
    if not await app_globals.qdrant_client.health_check():
        raise RuntimeError("Vector database is not available")

    logger.info("Services initialized successfully.")
    yield
    logger.info("Shutting down services...")


async def main():
    """
    Runs the service initializers and then starts the async gRPC server.
    """
    async with lifespan():
        logger.info("Initialization complete. Starting gRPC server.")
        await serve_async()


if __name__ == "__main__":
    asyncio.run(main())
