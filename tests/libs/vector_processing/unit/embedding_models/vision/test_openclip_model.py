import builtins
import sys
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


def _model_on(device_has_cuda: bool, batch_size: int = 16) -> OpenClipModel:
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
        return OpenClipModel("ViT-L-14/laion2b_s32b_b82k", batch_size=batch_size)


@pytest.mark.parametrize(("has_cuda", "half"), [(True, True), (False, False)])
def test_open_clip_model_uses_half_precision_only_with_cuda(has_cuda, half):
    assert _model_on(has_cuda).uses_half_precision is half


def test_encode_image_returns_unit_length_float_embeddings():
    model = _model_on(False)
    embeddings = model.encode_image(
        [Image.new("RGB", (4, 4)), Image.new("RGB", (4, 4))]
    )

    assert embeddings == [pytest.approx([0.6, 0.8])] * 2


def test_encode_image_chunks_by_batch_size():
    model = _model_on(False, batch_size=2)
    batch_sizes: list[int] = []
    original_encode = model.model.encode_image

    def recording_encode(images: torch.Tensor) -> torch.Tensor:
        batch_sizes.append(len(images))
        return original_encode(images)

    with patch.object(model.model, "encode_image", side_effect=recording_encode):
        embeddings = model.encode_image([Image.new("RGB", (4, 4)) for _ in range(5)])

    assert batch_sizes == [2, 2, 1]
    assert len(embeddings) == 5


def test_encode_image_accepts_file_paths(tmp_path):
    img_path = tmp_path / "sample.png"
    Image.new("RGB", (4, 4)).save(img_path)
    model = _model_on(False)

    embeddings = model.encode_image([str(img_path)])

    assert embeddings == [pytest.approx([0.6, 0.8])]


def test_openclip_properties_empty_encode_and_visual_output_dim_fallback():
    model = _model_on(False, batch_size=4)
    assert model.embedding_size == 2
    assert model.model_name == "ViT-L-14/laion2b_s32b_b82k"
    assert model.input_size == 224
    assert model.batch_size == 4
    assert model.encode_image([]) == []

    clip_without_text_proj = FakeClip()
    del clip_without_text_proj.text_projection
    preprocess = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
    with (
        patch("torch.cuda.is_available", autospec=True, return_value=False),
        patch(
            "open_clip.list_pretrained",
            autospec=True,
            return_value=[("ViT-L-14", "laion2b_s32b_b82k")],
        ),
        patch(
            "open_clip.create_model_and_transforms",
            autospec=True,
            return_value=(clip_without_text_proj, None, preprocess),
        ),
        patch("open_clip.get_tokenizer", autospec=True),
    ):
        fallback_model = OpenClipModel("ViT-L-14")
    assert fallback_model.embedding_size == 2


def test_openclip_init_and_encode_errors():
    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "open_clip":
            raise ImportError("no open_clip")
        return real_import(name, *args, **kwargs)

    with (
        patch.dict(sys.modules, {"open_clip": None}),
        patch("builtins.__import__", side_effect=failing_import),
        pytest.raises(ImportError, match="OpenCLIP dependencies missing"),
    ):
        OpenClipModel("ViT-L-14/laion2b_s32b_b82k")

    with (
        patch("torch.cuda.is_available", autospec=True, return_value=False),
        patch(
            "open_clip.list_pretrained",
            autospec=True,
            return_value=[("ViT-L-14", "laion2b_s32b_b82k")],
        ),
    ):
        with pytest.raises(ValueError, match="Unknown OpenCLIP model"):
            OpenClipModel("MissingModel/ckpt")
        with pytest.raises(ValueError, match="Invalid checkpoint"):
            OpenClipModel("ViT-L-14/wrong_ckpt")

    model = _model_on(False)
    with (
        patch.object(model.model, "encode_image", side_effect=RuntimeError("boom")),
        pytest.raises(RuntimeError, match="boom"),
    ):
        model.encode_image([Image.new("RGB", (4, 4))])
