"""Image models to compare for image search, behind one embedding function.

A model is named ``<kind>:<name>``:

- ``service:<architecture>/<pretrained>``: the service's own ``OpenClipModel``
  (``vector_processing``), exactly as the service embeds images; the default,
  ``service:ViT-L-14/laion2b_s32b_b82k``, is what production uses today
- ``openclip:<architecture>/<pretrained>``: any other OpenCLIP model, e.g.
  SigLIP 2 (``ViT-SO400M-14-SigLIP2-378/webli``)
- ``hf-clip:<repo>``: a Hugging Face ``transformers`` CLIP model, e.g.
  ``OysterQAQ/DanbooruCLIP``

Images are embedded in fp16 on a GPU and normalized to unit length, and saved
per model in the results folder so a rerun does not embed them again.
"""

import hashlib
import re
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from benchmarks.vector_service.toolkit.settings import REPOSITORY_ROOT

SERVICE_MODEL = "service:ViT-L-14/laion2b_s32b_b82k"
IMAGE_CACHE = REPOSITORY_ROOT / "cache" / "images"
Embed = Callable[[list[Image.Image]], np.ndarray]


class UnknownImageModelError(ValueError):
    def __init__(self, model: str) -> None:
        super().__init__(
            f"unknown image model {model!r}: use service:, openclip:<architecture>/<pretrained> "
            "or hf-clip:<repo>"
        )


def cached_path(url: str) -> Path:
    return (
        IMAGE_CACHE / f"{hashlib.blake2b(url.encode(), digest_size=16).hexdigest()}.jpg"
    )


def embeddings_store(results_dir: Path, model: str) -> Path:
    """Where a model's embeddings are saved; the service's model keeps the original name."""
    if model == SERVICE_MODEL:
        return results_dir / "image_embeddings.npz"
    return (
        results_dir
        / f"image_embeddings_{re.sub(r'[^a-z0-9]+', '_', model.lower()).strip('_')}.npz"
    )


def unit_rows(features: torch.Tensor) -> np.ndarray:
    features = features.float()
    return (features / features.norm(dim=-1, keepdim=True)).cpu().numpy()


def service_embedder(name: str) -> Embed:
    """The service's own image model class, so results match what it stores."""
    from vector_processing.embedding_models.vision.openclip_model import OpenClipModel

    model = OpenClipModel(name)

    def embed(images: list[Image.Image]) -> np.ndarray:
        return np.asarray(model.encode_image(images), dtype=np.float32)

    return embed


def openclip_embedder(name: str, device: str) -> Embed:
    import open_clip

    architecture, pretrained = name.split("/", 1)
    model, _, preprocess = open_clip.create_model_and_transforms(
        architecture, pretrained=pretrained, device=device
    )
    model.eval()

    def embed(images: list[Image.Image]) -> np.ndarray:
        batch = torch.stack([preprocess(image) for image in images]).to(device)
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=device, dtype=torch.float16, enabled=device == "cuda"
            ),
        ):
            return unit_rows(model.encode_image(batch))

    return embed


def hf_clip_embedder(repo: str, device: str) -> Embed:
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(repo)
    torch.nn.Module.to(model, torch.device(device))
    model.eval()
    processor = CLIPProcessor.from_pretrained(repo)

    def embed(images: list[Image.Image]) -> np.ndarray:
        inputs = processor(images=images, return_tensors="pt").to(device)
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=device, dtype=torch.float16, enabled=device == "cuda"
            ),
        ):
            features = model.get_image_features(**inputs)
        # transformers 5 returns an output object; pooler_output is the
        # projected image vector (projection_dim), not the vision hidden size
        if not isinstance(features, torch.Tensor):
            features = features.pooler_output
        return unit_rows(features)

    return embed


def load_embedder(model: str) -> Embed:
    kind, name = model.split(":", 1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if kind == "service":
        return service_embedder(name)
    if kind == "openclip":
        return openclip_embedder(name, device)
    if kind == "hf-clip":
        return hf_clip_embedder(name, device)
    raise UnknownImageModelError(model)


def embed_images(
    urls: Sequence[str], model: str, store: Path, batch_size: int = 32
) -> dict[str, np.ndarray]:
    """Embeddings per URL for ``model``, reusing the ones saved in ``store``."""
    saved = dict(np.load(store)) if store.exists() else {}
    missing = [url for url in urls if url not in saved]
    if missing:
        embed = load_embedder(model)
        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            vectors = embed(
                [Image.open(cached_path(url)).convert("RGB") for url in batch]
            )
            saved.update(zip(batch, vectors.astype(np.float32), strict=True))
        np.savez(store, **saved)
    return {url: saved[url] for url in urls}
