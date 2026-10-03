from unittest.mock import create_autospec, patch

import pytest
from common.config import Settings, get_settings
from qdrant_client.http.models import QueryRequest
from qdrant_db.errors import ConfigurationError
from vector_processing import TextProcessor, VisionProcessor
from vector_processing.embedding_models.text.base import TextEmbeddingModel
from vector_processing.embedding_models.vision.base import VisionEmbeddingModel
from vector_service.runtime import (
    GpuUnavailableError,
    _create_qdrant_client,
    _require_usable_gpu,
    _validate_model_dimensions,
    _warm_up_models,
    build_runtime,
)


def _settings_with_dimensions(text_dim: int, image_dim: int) -> Settings:
    settings = get_settings()
    qdrant = settings.qdrant.model_copy(
        update={"vector_names": {"text_vector": text_dim, "image_vector": image_dim}}
    )
    return settings.model_copy(update={"qdrant": qdrant})


def _processors(text_dim: int, image_dim: int) -> tuple[TextProcessor, VisionProcessor]:
    text_processor = create_autospec(TextProcessor, instance=True)
    text_processor.model = create_autospec(TextEmbeddingModel, instance=True)
    text_processor.model.embedding_size = text_dim
    vision_processor = create_autospec(VisionProcessor, instance=True)
    vision_processor.model = create_autospec(VisionEmbeddingModel, instance=True)
    vision_processor.model.embedding_size = image_dim
    return text_processor, vision_processor


def test_validate_model_dimensions_matching_dimensions_passes() -> None:
    text_processor, vision_processor = _processors(1024, 768)

    _validate_model_dimensions(
        _settings_with_dimensions(1024, 768), text_processor, vision_processor
    )


@pytest.mark.parametrize(
    (
        "vector_name",
        "text_model_dim",
        "image_model_dim",
        "config_text_dim",
        "config_image_dim",
    ),
    [
        ("text_vector", 512, 768, 1024, 768),
        ("image_vector", 1024, 512, 1024, 768),
    ],
    ids=["text_mismatch", "image_mismatch"],
)
def test_validate_model_dimensions_mismatch_raises_configuration_error_with_both_sizes(
    vector_name: str,
    text_model_dim: int,
    image_model_dim: int,
    config_text_dim: int,
    config_image_dim: int,
) -> None:
    text_processor, vision_processor = _processors(text_model_dim, image_model_dim)
    settings = _settings_with_dimensions(config_text_dim, config_image_dim)

    with pytest.raises(ConfigurationError, match=vector_name) as raised:
        _validate_model_dimensions(settings, text_processor, vision_processor)

    actual = text_model_dim if vector_name == "text_vector" else image_model_dim
    expected = config_text_dim if vector_name == "text_vector" else config_image_dim
    assert str(actual) in str(raised.value)
    assert str(expected) in str(raised.value)


def test_validate_model_dimensions_both_mismatched_reports_text_vector_first() -> None:
    text_processor, vision_processor = _processors(text_dim=512, image_dim=512)
    settings = _settings_with_dimensions(text_dim=1024, image_dim=768)

    with pytest.raises(ConfigurationError, match="text_vector"):
        _validate_model_dimensions(settings, text_processor, vision_processor)


def test_create_qdrant_client_default_settings_uses_http() -> None:
    qdrant_settings = get_settings().qdrant.model_copy(update={"qdrant_api_key": None})
    with patch(
        "vector_service.runtime.AsyncQdrantClient", autospec=True
    ) as client_class:
        _create_qdrant_client(qdrant_settings)

    client_class.assert_called_once_with(
        url=qdrant_settings.qdrant_url,
        api_key=None,
        prefer_grpc=False,
        grpc_port=6334,
        cloud_inference=True,
    )


def test_create_qdrant_client_prefer_grpc_setting_uses_grpc_port_and_key() -> None:
    qdrant_settings = get_settings().qdrant.model_copy(
        update={
            "qdrant_prefer_grpc": True,
            "qdrant_grpc_port": 7334,
            "qdrant_api_key": "key",
        }
    )
    with patch(
        "vector_service.runtime.AsyncQdrantClient", autospec=True
    ) as client_class:
        _create_qdrant_client(qdrant_settings)

    client_class.assert_called_once_with(
        url=qdrant_settings.qdrant_url,
        api_key="key",
        prefer_grpc=True,
        grpc_port=7334,
        cloud_inference=True,
    )


async def test_create_qdrant_client_query_batch_points_skips_local_inference_inspection() -> (
    None
):
    qdrant_settings = get_settings().qdrant.model_copy(update={"qdrant_api_key": None})
    client = _create_qdrant_client(qdrant_settings)
    request = QueryRequest(query=[0.1] * 1024, using="text_vector", limit=10)

    with (
        patch.object(client._inference_inspector, "inspect", autospec=True) as inspect,
        patch.object(
            client._client, "query_batch_points", autospec=True, return_value=[]
        ) as query_batch_points,
    ):
        await client.query_batch_points("anime", [request])

    inspect.assert_not_called()
    query_batch_points.assert_awaited_once()


def test_require_usable_gpu_cuda_device_available_passes() -> None:
    _require_usable_gpu(cuda_available=True, cuda_build="13.0")


def test_require_usable_gpu_cpu_only_build_names_cpu_only_build() -> None:
    with pytest.raises(GpuUnavailableError, match="CPU-only"):
        _require_usable_gpu(cuda_available=False, cuda_build=None)


def test_require_usable_gpu_cuda_build_without_device_names_missing_device() -> None:
    with pytest.raises(GpuUnavailableError, match="no usable CUDA device"):
        _require_usable_gpu(cuda_available=False, cuda_build="13.0")


def test_require_usable_gpu_no_gpu_error_says_how_to_run_on_cpu() -> None:
    with pytest.raises(GpuUnavailableError, match="ENABLE_GPU=false"):
        _require_usable_gpu(cuda_available=False, cuda_build=None)


@pytest.mark.asyncio
async def test_build_runtime_gpu_requested_but_unusable_stops_before_qdrant_client() -> (
    None
):
    settings = get_settings()
    settings = settings.model_copy(
        update={"service": settings.service.model_copy(update={"enable_gpu": True})}
    )
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.version.cuda", None),
        patch(
            "vector_service.runtime._create_qdrant_client", autospec=True
        ) as create_client,
        pytest.raises(GpuUnavailableError),
    ):
        await build_runtime(settings)

    create_client.assert_not_called()


@pytest.mark.asyncio
async def test_warm_up_models_runs_each_model_once() -> None:
    text_processor = create_autospec(TextProcessor, instance=True)
    vision_processor = create_autospec(VisionProcessor, instance=True)
    vision_processor.model = create_autospec(VisionEmbeddingModel, instance=True)
    vision_processor.model.encode_image.return_value = [[1.0]]

    await _warm_up_models(text_processor, vision_processor)

    text_processor.encode_text_with_sparse.assert_awaited_once()
    vision_processor.model.encode_image.assert_called_once()
    assert len(vision_processor.model.encode_image.call_args.args[0]) == 1
