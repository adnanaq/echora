import logging
from typing import cast

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
            model_name: OpenCLIP model name (e.g., 'laion/CLIP-ViT-L-14-laion2B-s32B-b82K')
            cache_dir: Optional directory to cache model files
            batch_size: Default batch size for encoding
        """
        try:
            import open_clip  # type: ignore[import-untyped]
            import torch

            self._model_name = model_name
            self._batch_size = batch_size

            # Set device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            # Parse model name to extract OpenCLIP model and pretrained weights
            if "/" in model_name:
                # Format: model_name/checkpoint (e.g., ViT-L-14/laion2b_s32b_b82k)
                clip_model_name, pretrained = model_name.split("/", 1)
            else:
                # Direct OpenCLIP model name without checkpoint
                clip_model_name = model_name
                # Get first available checkpoint for this model
                available_models = open_clip.list_pretrained()
                matching_checkpoints = [
                    cp for m, cp in available_models if m == clip_model_name
                ]
                if not matching_checkpoints:
                    available_model_names = sorted(set(m for m, _ in available_models))
                    raise ValueError(
                        f"Unknown OpenCLIP model '{clip_model_name}'. "
                        f"Available models: {', '.join(available_model_names[:10])}... "
                        f"(use open_clip.list_pretrained() for full list)"
                    )
                pretrained = matching_checkpoints[0]
                logger.warning(
                    f"No checkpoint specified for {clip_model_name}, using default: {pretrained}"
                )

            # Validate model/checkpoint combination
            available_models = open_clip.list_pretrained()
            valid_checkpoints = [
                cp for m, cp in available_models if m == clip_model_name
            ]
            if not valid_checkpoints:
                available_model_names = sorted(set(m for m, _ in available_models))
                raise ValueError(
                    f"Unknown OpenCLIP model '{clip_model_name}'. "
                    f"Available models: {', '.join(available_model_names[:10])}... "
                    f"(use open_clip.list_pretrained() for full list)"
                )

            if pretrained not in valid_checkpoints:
                raise ValueError(
                    f"Invalid checkpoint '{pretrained}' for model '{clip_model_name}'. "
                    f"Available checkpoints: {', '.join(valid_checkpoints)}"
                )

            self._clip_model_name = clip_model_name
            self._pretrained = pretrained
            logger.info(
                f"Using OpenCLIP model: {clip_model_name} with checkpoint: {pretrained}"
            )

            # Load OpenCLIP model
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                clip_model_name,
                pretrained=pretrained,
                device=self.device,
                cache_dir=cache_dir,
            )

            # Load tokenizer
            self.tokenizer = open_clip.get_tokenizer(clip_model_name)

            self.model.eval()

            # Dynamically determine embedding size
            try:
                self._embedding_size = self.model.text_projection.shape[1]
            except AttributeError:
                self._embedding_size = self.model.visual.output_dim

            # Dynamically determine input resolution
            self._input_resolution = getattr(self.preprocess.transforms[0], "size", 224)

            logger.info(f"Initialized OpenCLIP model: {model_name} on {self.device}")

        except ImportError as e:
            logger.exception(
                "OpenCLIP dependencies not installed. Install with: pip install open-clip-torch"
            )
            raise ImportError("OpenCLIP dependencies missing") from e
        except Exception as e:
            logger.exception(f"Failed to load OpenCLIP model {model_name}: {e}")
            raise

    def encode_image(self, images: list[Image.Image | str]) -> list[list[float]]:
        """Encode a list of images into embeddings.

        Args:
            images: List of PIL Images or image file paths

        Returns:
            List of embedding vectors
        """
        try:
            import torch

            # Preprocess images
            processed_images = []
            for img in images:
                pil_img: Image.Image
                if isinstance(img, str):
                    # Load from path if string
                    try:
                        loaded_img = Image.open(img)
                        if loaded_img.mode != "RGB":
                            pil_img = loaded_img.convert("RGB")
                        else:
                            pil_img = loaded_img  # type: ignore[assignment]  # ImageFile is compatible with Image
                    except Exception as e:
                        logger.error(f"Failed to load image from path {img}: {e}")
                        # Skip failed images or raise?
                        # To keep consistent list length, we should probably raise or handle gracefully.
                        # For now, let's raise as the input is expected to be valid.
                        raise
                else:
                    pil_img = img

                processed_images.append(self.preprocess(pil_img))

            if not processed_images:
                return []

            # Stack images
            image_tensor = torch.stack(processed_images).to(self.device)

            # Generate embeddings
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                # Normalize features
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )

                return cast(list[list[float]], image_features.cpu().numpy().tolist())

        except Exception as e:
            logger.exception(f"OpenCLIP encoding failed: {e}")
            raise

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
