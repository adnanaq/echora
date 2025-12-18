"""Integration test configuration for qdrant_db library.

These tests require real Qdrant database and ML models.
Inherits fixtures from parent conftest files.
"""

import pytest_asyncio
from vector_processing import MultiVectorEmbeddingManager
from vector_processing.embedding_models.factory import EmbeddingModelFactory
from vector_processing.utils.image_downloader import ImageDownloader


@pytest_asyncio.fixture(scope="session")
async def text_processor(settings):
    """Create TextProcessor for integration tests."""
    from vector_processing import TextProcessor
    
    # Create text model using factory
    text_model = EmbeddingModelFactory.create_text_model(settings)
    return TextProcessor(model=text_model, settings=settings)


@pytest_asyncio.fixture(scope="session")
async def vision_processor(settings):
    """Create VisionProcessor for integration tests."""
    from vector_processing import VisionProcessor
    
    # Create vision model and downloader using factory
    vision_model = EmbeddingModelFactory.create_vision_model(settings)
    downloader = ImageDownloader(settings.model_cache_dir)
    return VisionProcessor(model=vision_model, downloader=downloader, settings=settings)


@pytest_asyncio.fixture(scope="session")
async def embedding_manager(text_processor, vision_processor, settings):
    """Create MultiVectorEmbeddingManager for integration tests."""
    return MultiVectorEmbeddingManager(
        text_processor=text_processor,
        vision_processor=vision_processor,
        settings=settings,
    )
