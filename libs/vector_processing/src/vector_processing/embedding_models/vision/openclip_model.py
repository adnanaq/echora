import logging
from typing import Any, Dict, List, Optional, Union, cast

from PIL import Image
from .base import VisionEmbeddingModel

logger = logging.getLogger(__name__)


class OpenClipModel(VisionEmbeddingModel):
    """OpenCLIP implementation of VisionEmbeddingModel."""

    def __init__(self, model_name: str, cache_dir: Optional[str] = None, batch_size: int = 16):
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
                # HuggingFace style name
                _, model_part = model_name.split("/", 1)
                if "ViT-L-14" in model_part:
                    clip_model_name = "ViT-L-14"
                    pretrained = model_part
                elif "ViT-B-32" in model_part:
                    clip_model_name = "ViT-B-32"
                    pretrained = model_part
                else:
                    # Default fallback
                    clip_model_name = "ViT-L-14"
                    pretrained = "laion2b_s32b_b82k"
            else:
                # Direct OpenCLIP model name
                clip_model_name = model_name
                pretrained = "laion2b_s32b_b82k"  # Default for ViT-L-14

            self._clip_model_name = clip_model_name
            self._pretrained = pretrained

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
            logger.error(
                "OpenCLIP dependencies not installed. Install with: pip install open-clip-torch"
            )
            raise ImportError("OpenCLIP dependencies missing") from e
        except Exception as e:
            logger.error(f"Failed to load OpenCLIP model {model_name}: {e}")
            raise

    def encode_image(self, images: List[Union[Image.Image, str]]) -> List[List[float]]:
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
                if isinstance(img, str):
                    # Load from path if string
                    try:
                        pil_img = Image.open(img)
                        if pil_img.mode != "RGB":
                            pil_img = pil_img.convert("RGB")
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
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                return cast(List[List[float]], image_features.cpu().numpy().tolist())

        except Exception as e:
            logger.error(f"OpenCLIP encoding failed: {e}")
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
