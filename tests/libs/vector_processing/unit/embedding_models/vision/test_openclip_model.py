from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image
from vector_processing.embedding_models.vision.openclip_model import OpenClipModel


class FakeClip:
    def __init__(self) -> None:
        self.visual = MagicMock(output_dim=2)
        self.text_projection = torch.zeros(4, 2)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[3.0, 4.0]] * len(images))

    def eval(self) -> None:
        pass


def _model_on(device_has_cuda: bool) -> OpenClipModel:
    preprocess = MagicMock(side_effect=lambda image: torch.zeros(3, 2, 2))
    preprocess.transforms = [MagicMock(size=224)]
    with (
        patch("torch.cuda.is_available", return_value=device_has_cuda),
        patch(
            "open_clip.list_pretrained",
            return_value=[("ViT-L-14", "laion2b_s32b_b82k")],
        ),
        patch(
            "open_clip.create_model_and_transforms",
            return_value=(FakeClip(), None, preprocess),
        ),
        patch("open_clip.get_tokenizer"),
    ):
        return OpenClipModel("ViT-L-14/laion2b_s32b_b82k")


@pytest.mark.parametrize(("has_cuda", "half"), [(True, True), (False, False)])
def test_half_precision_only_on_a_gpu(has_cuda, half):
    assert _model_on(has_cuda).uses_half_precision is half


def test_embeddings_are_unit_length_floats():
    model = _model_on(False)
    embeddings = model.encode_image(
        [Image.new("RGB", (4, 4)), Image.new("RGB", (4, 4))]
    )

    assert embeddings == [pytest.approx([0.6, 0.8])] * 2
