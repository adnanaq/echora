from unittest.mock import create_autospec

import pytest
from common.config import EmbeddingConfig
from vector_processing.embedding_models.text.base import TextEmbeddingModel
from vector_processing.embedding_models.vision.base import VisionEmbeddingModel
from vector_processing.utils.image_downloader import ImageDownloader


@pytest.fixture
def embedding_config() -> EmbeddingConfig:
    return EmbeddingConfig(
        max_concurrent_image_downloads=10,
        embed_max_concurrency=2,
        embed_batch_max_size=1,
        embed_batch_max_wait_ms=0.0,
    )


@pytest.fixture
def text_model() -> TextEmbeddingModel:
    model = create_autospec(TextEmbeddingModel, instance=True)
    model.model_name = "test-text-model"
    model.embedding_size = 1024
    model.encode.return_value = [[0.1] * 1024]
    model.get_model_info.return_value = {
        "model_name": "test-text-model",
        "embedding_size": 1024,
    }
    return model


@pytest.fixture
def vision_model() -> VisionEmbeddingModel:
    model = create_autospec(VisionEmbeddingModel, instance=True)
    model.model_name = "test-vision-model"
    model.embedding_size = 768
    model.encode_image.return_value = [[0.2] * 768]
    model.get_model_info.return_value = {
        "model_name": "test-vision-model",
        "embedding_size": 768,
    }
    return model


@pytest.fixture
def image_downloader() -> ImageDownloader:
    downloader = create_autospec(ImageDownloader, instance=True)
    downloader.get_cache_stats.return_value = {
        "cache_size": 100,
        "hit_rate": 0.85,
    }
    downloader.clear_cache.return_value = {
        "cleared": 50,
        "remaining": 50,
    }
    return downloader
