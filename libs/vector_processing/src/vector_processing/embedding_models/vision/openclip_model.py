"""OpenCLIP vision embedding model implementation."""

import logging
from collections.abc import Sequence
from itertools import batched
from typing import Any, cast

from PIL import Image

from .base import VisionEmbeddingModel

logger = logging.getLogger(__name__)


class OpenClipModel(VisionEmbeddingModel):
    """OpenCLIP implementation of VisionEmbeddingModel."""

    def __init__(
        self, model_name: str, cache_dir: str | None = None, batch_size: int = 16
    ):
        """Initialize OpenCLIP model.

        Args:
            model_name: OpenCLIP model name (e.g., 'ViT-L-14/laion2b_s32b_b82k').
            cache_dir: Optional directory to cache model files.
            batch_size: Default batch size for encoding.
        """
        try:
            import open_clip
            import torch

            self._model_name = model_name
            self._batch_size = batch_size
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            clip_model_name, pretrained = self._resolve_model_and_checkpoint(
                model_name, open_clip.list_pretrained()
            )
            self._clip_model_name = clip_model_name
            self._pretrained = pretrained
            logger.info(
                f"Using OpenCLIP model: {clip_model_name} with checkpoint: {pretrained}"
            )

            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                clip_model_name,
                pretrained=pretrained,
                device=self.device,
                cache_dir=cache_dir,
            )
            self.tokenizer = open_clip.get_tokenizer(clip_model_name)
            self.model.eval()

            try:
                self._embedding_size = self.model.text_projection.shape[1]
            except AttributeError:
                self._embedding_size = self.model.visual.output_dim

            self._input_resolution = getattr(self.preprocess.transforms[0], "size", 224)
            logger.info(f"Initialized OpenCLIP model: {model_name} on {self.device}")
        except ImportError as e:
            logger.exception(
                "OpenCLIP dependencies not installed. Install with: pip install open-clip-torch"
            )
            raise ImportError("OpenCLIP dependencies missing") from e
        except Exception:
            logger.exception(f"Failed to load OpenCLIP model {model_name}")
            raise

    @property
    def uses_half_precision(self) -> bool:
        """Whether images are encoded in fp16, which only pays off on a GPU."""
        return self.device == "cuda"

    @property
    def embedding_size(self) -> int:
        return self._embedding_size

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def input_size(self) -> int:
        return self._input_resolution

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def encode_image(self, images: Sequence[Image.Image | str]) -> list[list[float]]:
        """Encode a list of images into normalized embeddings.

        Args:
            images: List of PIL Images or image file paths.

        Returns:
            List of normalized embedding vectors.
        """
        if not images:
            return []

        try:
            import torch

            step = max(1, self._batch_size)
            embeddings: list[list[float]] = []

            for chunk in batched(images, step, strict=False):
                processed = [self._preprocess_image(img) for img in chunk]
                image_tensor = torch.stack(processed).to(self.device)

                # Half precision on a GPU: ~2.8x more images/s, same embeddings
                # within rounding (cosine >= 0.9998 on real images, ECHO-54)
                with (
                    torch.inference_mode(),
                    torch.autocast(
                        device_type=self.device,
                        dtype=torch.float16,
                        enabled=self.uses_half_precision,
                    ),
                ):
                    image_features = self.model.encode_image(image_tensor).float()
                    image_features = image_features / image_features.norm(
                        dim=-1, keepdim=True
                    )
                    embeddings.extend(
                        cast(list[list[float]], image_features.cpu().numpy().tolist())
                    )

            return embeddings
        except Exception:
            logger.exception("OpenCLIP encoding failed")
            raise

    @staticmethod
    def _resolve_model_and_checkpoint(
        model_name: str, available_models: list[tuple[str, str]]
    ) -> tuple[str, str]:
        """Parse and validate an OpenCLIP model name and checkpoint pair.

        Args:
            model_name: Model specification as ``model/checkpoint`` or ``model``.
            available_models: Pairs of ``(model_name, pretrained)`` from OpenCLIP.

        Returns:
            Tuple of validated ``(clip_model_name, pretrained_checkpoint)``.

        Raises:
            ValueError: If the model name or checkpoint is not available.
        """
        if "/" in model_name:
            clip_model_name, requested_checkpoint = model_name.split("/", 1)
        else:
            clip_model_name, requested_checkpoint = model_name, None

        valid_checkpoints = [
            cp for model, cp in available_models if model == clip_model_name
        ]
        if not valid_checkpoints:
            available_names = sorted({model for model, _ in available_models})
            raise ValueError(
                f"Unknown OpenCLIP model '{clip_model_name}'. "
                f"Available models: {', '.join(available_names[:10])}... "
                f"(use open_clip.list_pretrained() for full list)"
            )

        if requested_checkpoint is None:
            pretrained = valid_checkpoints[0]
            logger.warning(
                f"No checkpoint specified for {clip_model_name}, using default: {pretrained}"
            )
            return clip_model_name, pretrained

        if requested_checkpoint not in valid_checkpoints:
            raise ValueError(
                f"Invalid checkpoint '{requested_checkpoint}' for model '{clip_model_name}'. "
                f"Available checkpoints: {', '.join(valid_checkpoints)}"
            )
        return clip_model_name, requested_checkpoint

    def _preprocess_image(self, image: Image.Image | str) -> Any:
        """Load if needed and apply OpenCLIP preprocessing to a single image.

        Args:
            image: PIL Image instance or path to an image file.

        Returns:
            Preprocessed image tensor ready for model input.
        """
        if isinstance(image, str):
            with Image.open(image) as loaded_img:
                rgb_img = (
                    loaded_img.convert("RGB")
                    if loaded_img.mode != "RGB"
                    else loaded_img
                )
                return self.preprocess(rgb_img)

        rgb_img = image.convert("RGB") if image.mode != "RGB" else image
        return self.preprocess(rgb_img)
