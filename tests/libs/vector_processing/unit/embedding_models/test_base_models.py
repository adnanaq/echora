from collections.abc import Sequence

from PIL import Image
from vector_processing.embedding_models.text.base import TextEmbeddingModel
from vector_processing.embedding_models.vision.base import VisionEmbeddingModel


class _DummyTextModel(TextEmbeddingModel):
    def encode(self, texts: list[str]) -> list[list[float]]:
        super().encode(texts)
        return [[1.0, 2.0] for _ in texts]

    @property
    def embedding_size(self) -> int:
        _ = super().embedding_size
        return 2

    @property
    def model_name(self) -> str:
        _ = super().model_name
        return "dummy-text"


class _DummyVisionModel(VisionEmbeddingModel):
    def encode_image(self, images: Sequence[Image.Image | str]) -> list[list[float]]:
        super().encode_image(images)
        return [[0.5, 0.5] for _ in images]

    @property
    def embedding_size(self) -> int:
        _ = super().embedding_size
        return 2

    @property
    def model_name(self) -> str:
        _ = super().model_name
        return "dummy-vision"


def test_text_embedding_model_base_defaults():
    model = _DummyTextModel()
    assert model.max_length == 512
    assert model.supports_multilingual is False
    assert model.supports_sparse is False
    assert model.encode_with_sparse(["a", "b"]) == (
        [[1.0, 2.0], [1.0, 2.0]],
        [None, None],
    )
    assert model.get_model_info() == {
        "model_name": "dummy-text",
        "embedding_size": 2,
        "max_length": 512,
        "supports_multilingual": False,
    }


def test_vision_embedding_model_base_defaults():
    model = _DummyVisionModel()
    assert model.input_size == 224
    assert model.encode_image(["a.png"]) == [[0.5, 0.5]]
    assert model.get_model_info() == {
        "model_name": "dummy-vision",
        "embedding_size": 2,
        "input_size": 224,
    }
