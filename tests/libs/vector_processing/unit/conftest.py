"""Common fixtures for vector_processing tests."""

import pytest
from unittest.mock import MagicMock

from common.config import Settings


@pytest.fixture
def mock_settings():
    """Create mock Settings instance for unit tests."""
    settings = MagicMock(spec=Settings)
    # Add new configuration field
    settings.max_concurrent_image_downloads = 10
    return settings


@pytest.fixture
def mock_text_model():
    """Create a mock TextEmbeddingModel for unit tests."""
    model = MagicMock()
    model.model_name = "test-text-model"
    model.embedding_size = 1024
    model.encode.return_value = [[0.1] * 1024]
    model.get_model_info.return_value = {
        "model_name": "test-text-model",
        "embedding_size": 1024,
    }
    return model


@pytest.fixture
def mock_vision_model():
    """Create a mock VisionEmbeddingModel for unit tests."""
    model = MagicMock()
    model.model_name = "test-vision-model"
    model.embedding_size = 768
    model.encode_image.return_value = [[0.2] * 768]
    model.get_model_info.return_value = {
        "model_name": "test-vision-model",
        "embedding_size": 768,
    }
    return model


@pytest.fixture
def mock_downloader():
    """Create a mock ImageDownloader for unit tests."""
    downloader = MagicMock()
    downloader.get_cache_stats.return_value = {
        "cache_size": 100,
        "hit_rate": 0.85,
    }
    downloader.clear_cache.return_value = {
        "cleared": 50,
        "remaining": 50,
    }
    return downloader
