from unittest.mock import patch

import pytest
import torch
from PIL import Image
from torchvision import transforms
from vector_processing.embedding_models.vision.openclip_model import OpenClipModel


class FakeVisualTower(torch.nn.Module):
    output_dim = 2


class FakeClip(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = FakeVisualTower()
        self.text_projection = torch.zeros(4, 2)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[3.0, 4.0]] * len(images))


def _model_on(device_has_cuda: bool) -> OpenClipModel:
    preprocess = transforms.Compose(
        [transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()]
    )
    with (
        patch("torch.cuda.is_available", autospec=True, return_value=device_has_cuda),
        patch(
            "open_clip.list_pretrained",
            autospec=True,
            return_value=[("ViT-L-14", "laion2b_s32b_b82k")],
        ),
        patch(
            "open_clip.create_model_and_transforms",
            autospec=True,
            return_value=(FakeClip(), None, preprocess),
        ),
        patch("open_clip.get_tokenizer", autospec=True),
    ):
        return OpenClipModel("ViT-L-14/laion2b_s32b_b82k")


@pytest.mark.parametrize(("has_cuda", "half"), [(True, True), (False, False)])
def test_open_clip_model_uses_half_precision_only_with_cuda(has_cuda, half):
    assert _model_on(has_cuda).uses_half_precision is half


def test_encode_image_returns_unit_length_float_embeddings():
    model = _model_on(False)
    embeddings = model.encode_image(
        [Image.new("RGB", (4, 4)), Image.new("RGB", (4, 4))]
    )

    assert embeddings == [pytest.approx([0.6, 0.8])] * 2
