"""Unit tests for vector_service runtime helpers."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from common.config import get_settings
from qdrant_client.http.models import QueryRequest
from qdrant_db.errors import ConfigurationError
from vector_service.runtime import (
    GpuUnavailableError,
    _create_qdrant_client,
    _require_usable_gpu,
    _validate_model_dimensions,
    _warm_up_models,
    build_runtime,
)


def _make_settings(text_dim: int = 1024, image_dim: int = 768) -> SimpleNamespace:
    return SimpleNamespace(
        qdrant=SimpleNamespace(
            primary_text_vector_name="text_vector",
            primary_image_vector_name="image_vector",
            vector_names={"text_vector": text_dim, "image_vector": image_dim},
        )
    )


def _make_processors(
    text_dim: int, image_dim: int
) -> tuple[SimpleNamespace, SimpleNamespace]:
    return (
        SimpleNamespace(model=SimpleNamespace(embedding_size=text_dim)),
        SimpleNamespace(model=SimpleNamespace(embedding_size=image_dim)),
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_validate_model_dimensions_passes_when_all_match() -> None:
    text_proc, vision_proc = _make_processors(1024, 768)
    # Must not raise
    _validate_model_dimensions(_make_settings(1024, 768), text_proc, vision_proc)


# ---------------------------------------------------------------------------
# Dimension mismatch — parameterized over text and image
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "vector_name, text_model_dim, image_model_dim, config_text_dim, config_image_dim",
    [
        # Text model produces 512 but config expects 1024
        ("text_vector", 512, 768, 1024, 768),
        # Image model produces 512 but config expects 768
        ("image_vector", 1024, 512, 1024, 768),
    ],
    ids=["text_mismatch", "image_mismatch"],
)
def test_validate_model_dimensions_raises_on_mismatch(
    vector_name: str,
    text_model_dim: int,
    image_model_dim: int,
    config_text_dim: int,
    config_image_dim: int,
) -> None:
    text_proc, vision_proc = _make_processors(text_model_dim, image_model_dim)
    settings = _make_settings(config_text_dim, config_image_dim)

    with pytest.raises(ConfigurationError, match=vector_name) as exc_info:
        _validate_model_dimensions(settings, text_proc, vision_proc)

    # Error message must include both actual and expected dims for actionability
    error_msg = str(exc_info.value)
    actual = text_model_dim if vector_name == "text_vector" else image_model_dim
    expected = config_text_dim if vector_name == "text_vector" else config_image_dim
    assert str(actual) in error_msg
    assert str(expected) in error_msg


# ---------------------------------------------------------------------------
# Fail-fast: text is checked before image
# ---------------------------------------------------------------------------


def test_validate_model_dimensions_fails_on_text_first_when_both_wrong() -> None:
    text_proc, vision_proc = _make_processors(text_dim=512, image_dim=512)
    settings = _make_settings(text_dim=1024, image_dim=768)

    with pytest.raises(ConfigurationError, match="text_vector"):
        _validate_model_dimensions(settings, text_proc, vision_proc)


def test_qdrant_client_uses_http_by_default() -> None:
    qdrant_settings = get_settings().qdrant.model_copy(update={"qdrant_api_key": None})
    with patch("vector_service.runtime.AsyncQdrantClient") as client_class:
        _create_qdrant_client(qdrant_settings)

    client_class.assert_called_once_with(
        url=qdrant_settings.qdrant_url,
        api_key=None,
        prefer_grpc=False,
        grpc_port=6334,
        cloud_inference=True,
    )


def test_qdrant_client_can_prefer_grpc() -> None:
    qdrant_settings = get_settings().qdrant.model_copy(
        update={
            "qdrant_prefer_grpc": True,
            "qdrant_grpc_port": 7334,
            "qdrant_api_key": "key",
        }
    )
    with patch("vector_service.runtime.AsyncQdrantClient") as client_class:
        _create_qdrant_client(qdrant_settings)

    client_class.assert_called_once_with(
        url=qdrant_settings.qdrant_url,
        api_key="key",
        prefer_grpc=True,
        grpc_port=7334,
        cloud_inference=True,
    )


async def test_qdrant_client_skips_local_inference_inspection() -> None:
    qdrant_settings = get_settings().qdrant.model_copy(update={"qdrant_api_key": None})
    client = _create_qdrant_client(qdrant_settings)
    request = QueryRequest(query=[0.1] * 1024, using="text_vector", limit=10)

    with (
        patch.object(client._inference_inspector, "inspect") as inspect,
        patch.object(
            client._client, "query_batch_points", AsyncMock(return_value=[])
        ) as query_batch_points,
    ):
        await client.query_batch_points("anime", [request])

    inspect.assert_not_called()
    query_batch_points.assert_awaited_once()


def test_gpu_check_passes_when_a_cuda_device_is_usable() -> None:
    _require_usable_gpu(cuda_available=True, cuda_build="13.0")


def test_gpu_check_names_a_cpu_only_torch_build() -> None:
    with pytest.raises(GpuUnavailableError, match="CPU-only"):
        _require_usable_gpu(cuda_available=False, cuda_build=None)


def test_gpu_check_names_a_missing_device_on_a_cuda_build() -> None:
    with pytest.raises(GpuUnavailableError, match="no usable CUDA device"):
        _require_usable_gpu(cuda_available=False, cuda_build="13.0")


def test_gpu_check_error_says_how_to_run_on_cpu() -> None:
    with pytest.raises(GpuUnavailableError, match="ENABLE_GPU=false"):
        _require_usable_gpu(cuda_available=False, cuda_build=None)


@pytest.mark.asyncio
async def test_runtime_stops_when_gpu_is_requested_but_not_usable() -> None:
    settings = SimpleNamespace(service=SimpleNamespace(enable_gpu=True))
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.version.cuda", None),
        patch("vector_service.runtime._create_qdrant_client") as create_client,
        pytest.raises(GpuUnavailableError),
    ):
        await build_runtime(settings)

    create_client.assert_not_called()


@pytest.mark.asyncio
async def test_warm_up_runs_each_model_once() -> None:
    text_processor = SimpleNamespace(encode_text_with_sparse=AsyncMock())
    image_model = SimpleNamespace(encode_image=MagicMock(return_value=[[1.0]]))
    vision_processor = SimpleNamespace(model=image_model)

    await _warm_up_models(text_processor, vision_processor)

    text_processor.encode_text_with_sparse.assert_awaited_once()
    image_model.encode_image.assert_called_once()
    assert len(image_model.encode_image.call_args.args[0]) == 1
