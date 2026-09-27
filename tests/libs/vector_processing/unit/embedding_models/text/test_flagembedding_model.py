from unittest.mock import MagicMock, patch

import pytest
from vector_processing.embedding_models.text.flagembedding_model import (
    FlagEmbeddingModel,
)


def _loaded_bge_m3(device: str) -> MagicMock:
    bge_m3 = MagicMock()
    bge_m3.target_devices = [device]
    return bge_m3


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_moves_model_to_its_device_once_at_load(device):
    bge_m3 = _loaded_bge_m3(device)
    with patch("FlagEmbedding.BGEM3FlagModel", return_value=bge_m3):
        FlagEmbeddingModel("BAAI/bge-m3")

    bge_m3.model.to.assert_called_once_with(device)
    bge_m3.model.eval.assert_called_once_with()


def test_cpu_model_uses_full_precision():
    bge_m3 = _loaded_bge_m3("cpu")
    with patch("FlagEmbedding.BGEM3FlagModel", return_value=bge_m3):
        FlagEmbeddingModel("BAAI/bge-m3")

    bge_m3.model.float.assert_called_once_with()


def test_missing_model_fails_at_load():
    bge_m3 = _loaded_bge_m3("cuda:0")
    bge_m3.model = None
    with (
        patch("FlagEmbedding.BGEM3FlagModel", return_value=bge_m3),
        pytest.raises(RuntimeError, match="BAAI/bge-m3"),
    ):
        FlagEmbeddingModel("BAAI/bge-m3")


def test_gpu_model_keeps_its_precision():
    bge_m3 = _loaded_bge_m3("cuda:0")
    with patch("FlagEmbedding.BGEM3FlagModel", return_value=bge_m3):
        FlagEmbeddingModel("BAAI/bge-m3")

    bge_m3.model.float.assert_not_called()
