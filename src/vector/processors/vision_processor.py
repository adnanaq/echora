"""Vision processor using OpenCLIP for anime image search.

Uses OpenCLIP ViT-L/14 model for high-quality image embeddings
with commercial-friendly licensing.
"""

import base64
import hashlib
import io
import logging
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
from PIL import Image

from ...config import Settings
from ...models.anime import AnimeEntry
from .anime_field_mapper import AnimeFieldMapper

logger = logging.getLogger(__name__)


class VisionProcessor:
    """Vision processor supporting multiple embedding models."""

    def __init__(self, settings: Settings | None = None):
        """Initialize modern vision processor with configuration.

        Args:
            settings: Configuration settings instance
        """
        if settings is None:
            from ...config import Settings

            settings = Settings()

        self.settings = settings
        self.provider = settings.image_embedding_provider
        self.model_name = settings.image_embedding_model
        self.cache_dir = settings.model_cache_dir

        # Model instance
        self.model: dict[str, Any] | None = None

        # Device configuration
        self.device: str | None = None

        # Model metadata
        self.model_info: dict[str, Any] = {}

        # Image caching configuration
        self.image_cache_dir = (
            Path("cache/images")
            if not settings.model_cache_dir
            else Path(settings.model_cache_dir) / "images"
        )
        self.image_cache_dir.mkdir(parents=True, exist_ok=True)

        # Field mapper for anime data extraction
        self._field_mapper: AnimeFieldMapper | None = None

        # Initialize models
        self._init_models()

    def _init_models(self) -> None:
        """Initialize vision embedding model."""
        try:
            # Initialize model
            self.model = self._create_model(self.provider, self.model_name)

            # Warm up model if enabled
            if self.settings.model_warm_up:
                self._warm_up_model()

            logger.info(
                f"Initialized modern vision processor with {self.provider} model: {self.model_name}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize modern vision processor: {e}")
            raise

    def _create_model(self, provider: str, model_name: str) -> dict[str, Any]:
        """Create OpenCLIP model instance.

        Args:
            provider: Model provider (must be "openclip")
            model_name: OpenCLIP model name

        Returns:
            Dictionary containing model instance and metadata
        """
        if provider != "openclip":
            raise ValueError(f"Only 'openclip' provider is supported, got: {provider}")

        return self._create_openclip_model(model_name)

    def _create_openclip_model(self, model_name: str) -> dict[str, Any]:
        """Create OpenCLIP model instance.

        Args:
            model_name: OpenCLIP model name (e.g., 'laion/CLIP-ViT-L-14-laion2B-s32B-b82K')

        Returns:
            Dictionary with OpenCLIP model and metadata
        """
        try:
            import open_clip  # type: ignore[import-untyped]
            import torch

            # Set device
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Parse model name to extract OpenCLIP model and pretrained weights
            # Format: 'laion/CLIP-ViT-L-14-laion2B-s32B-b82K'
            if "/" in model_name:
                # HuggingFace style name
                _, model_part = model_name.split("/", 1)
                if "ViT-L-14" in model_part:
                    clip_model_name = "ViT-L-14"
                    pretrained = (
                        model_part  # Use full model part as pretrained identifier
                    )
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

            # Load OpenCLIP model
            model, _, preprocess = open_clip.create_model_and_transforms(
                clip_model_name,
                pretrained=pretrained,
                device=device,
                cache_dir=self.cache_dir,
            )

            # Load tokenizer using model architecture name only
            # Per OpenCLIP documentation: get_tokenizer() expects model name, not model/pretrained
            tokenizer = open_clip.get_tokenizer(clip_model_name)
            logger.debug(f"Loaded tokenizer for model architecture: {clip_model_name}")

            model.eval()

            # Dynamically determine embedding size
            try:
                embedding_size = model.text_projection.shape[1]
            except AttributeError:
                embedding_size = (
                    model.visual.output_dim
                )  # fallback for some OpenCLIP builds

            # Dynamically determine input resolution from preprocess
            input_resolution = getattr(preprocess.transforms[0], "size", 224)

            return {
                "model": model,
                "preprocess": preprocess,
                "tokenizer": tokenizer,
                "device": device,
                "provider": "openclip",
                "model_name": model_name,
                "clip_model_name": clip_model_name,
                "pretrained": pretrained,
                "embedding_size": embedding_size,
                "input_resolution": input_resolution,
                "supports_text": True,
                "supports_image": True,
                "batch_size": getattr(self.settings, "image_batch_size", 16),
            }

        except ImportError as e:
            logger.error(
                "OpenCLIP dependencies not installed. Install with: pip install open-clip-torch"
            )
            raise ImportError("OpenCLIP dependencies missing") from e
        except Exception as e:
            logger.error(f"Failed to load OpenCLIP model {model_name}: {e}")
            raise

    def _detect_model_provider(self, model_name: str) -> str:
        """Detect model provider from model name.

        Args:
            model_name: Model name or path

        Returns:
            Provider name (always "openclip")
        """
        # Only OpenCLIP is supported now
        return "openclip"

    def _warm_up_model(self) -> None:
        """Warm up model with dummy data."""
        try:
            # Create dummy image
            dummy_image = Image.new("RGB", (224, 224), color="red")
            dummy_image_b64 = self._pil_to_base64(dummy_image)

            # Warm up model
            if self.model is not None:
                self._encode_image_with_model(dummy_image_b64, self.model)
            logger.info("Model warmed up successfully")

        except Exception as e:
            logger.warning(f"Model warm-up failed: {e}")

    def _pil_to_base64(self, image: Image.Image) -> str:
        """Convert PIL image to base64 string.

        Args:
            image: PIL Image

        Returns:
            Base64 encoded image string
        """
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str

    def encode_image(self, image_data: str) -> list[float] | None:
        """Encode image to embedding vector.

        Args:
            image_data: Base64 encoded image data

        Returns:
            Embedding vector or None if encoding fails
        """
        try:
            # Encode with model
            if self.model is not None:
                embedding = self._encode_image_with_model(image_data, self.model)
                return embedding
            else:
                logger.error("Model not initialized")
                return None

        except Exception as e:
            logger.error(f"Image encoding failed: {e}")
            return None

    def _encode_image_with_model(
        self, image_data: str, model_dict: dict[str, Any]
    ) -> list[float] | None:
        """Encode image with specific model.

        Args:
            image_data: Base64 encoded image data
            model_dict: Model dictionary with instance and metadata

        Returns:
            Embedding vector or None if encoding fails
        """
        try:
            # Decode image
            image = self._decode_base64_image(image_data)
            if image is None:
                return None

            # Encode based on provider
            provider = model_dict["provider"]

            if provider == "openclip":
                return self._encode_with_openclip(image, model_dict)
            else:
                logger.error(f"Unsupported provider: {provider}")
                return None

        except Exception as e:
            logger.error(f"Model encoding failed: {e}")
            return None

    def _encode_with_openclip(
        self, image: Image.Image, model_dict: dict[str, Any]
    ) -> list[float] | None:
        """Encode image using OpenCLIP model.

        Args:
            image: PIL Image to encode
            model_dict: Dictionary containing OpenCLIP model and preprocessing

        Returns:
            Embedding vector as list of floats or None if encoding fails
        """
        try:
            import torch

            model = model_dict["model"]
            preprocess = model_dict["preprocess"]

            # Preprocess image
            image_tensor = preprocess(image).unsqueeze(0)

            # Move to device if available
            device = model_dict.get("device", "cpu")
            if device != "cpu":
                image_tensor = image_tensor.to(device)

            # Generate embedding
            with torch.no_grad():
                image_features = model.encode_image(image_tensor)
                # Normalize features
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )
                embedding: list[float] = image_features.cpu().numpy().flatten().tolist()

            return embedding

        except Exception as e:
            logger.error(f"OpenCLIP encoding failed: {e}")
            return None

    def _decode_base64_image(self, image_data: str) -> Image.Image | None:
        """Decode base64 image data to PIL Image.

        Args:
            image_data: Base64 encoded image string

        Returns:
            PIL Image object or None if decoding fails
        """
        try:
            # Handle data URL format
            if image_data.startswith("data:"):
                base64_part = image_data.split(",", 1)[1]
            else:
                base64_part = image_data

            # Decode base64
            image_bytes = base64.b64decode(base64_part)

            # Create PIL Image
            image: Image.Image = Image.open(io.BytesIO(image_bytes))

            # Convert to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")

            return image

        except Exception as e:
            logger.error(f"Failed to decode base64 image: {e}")
            return None

    def encode_images_batch(
        self, image_data_list: list[str], batch_size: int | None = None
    ) -> list[list[float] | None]:
        """Encode multiple images in batches.

        Args:
            image_data_list: List of base64 encoded images
            batch_size: Batch size (uses model-specific default if None)

        Returns:
            List of embedding vectors
        """
        try:
            if batch_size is None:
                batch_size = self.model.get("batch_size", 8) if self.model else 8

            embeddings = []
            total_images = len(image_data_list)

            logger.info(f"Processing {total_images} images in batches of {batch_size}")

            for i in range(0, total_images, batch_size):
                batch = image_data_list[i : i + batch_size]
                batch_embeddings = []

                for image_data in batch:
                    embedding = self.encode_image(image_data)
                    batch_embeddings.append(embedding)

                embeddings.extend(batch_embeddings)

                # Log progress
                processed = min(i + batch_size, total_images)
                logger.info(f"Processed {processed}/{total_images} images")

            return embeddings

        except Exception as e:
            logger.error(f"Batch image encoding failed: {e}")
            return [None] * len(image_data_list)

    def _load_image_for_ccip(self, url_or_path: str) -> Image.Image | None:
        """Load image for CCIP, handling both URLs and local paths."""
        try:
            if url_or_path.startswith("http"):
                # Use requests for synchronous download
                from io import BytesIO

                import requests

                response = requests.get(url_or_path, timeout=10)
                response.raise_for_status()
                return Image.open(BytesIO(response.content))
            else:
                # Handle local file path
                return Image.open(url_or_path)
        except Exception as e:
            logger.error(f"Failed to load image from {url_or_path}: {e}")
            return None

    def calculate_character_similarity(
        self, image_url_1: str, image_url_2: str
    ) -> float:
        """Calculate character similarity using CCIP (anime-specialized model).

        Uses DeepGHS CCIP model for anime character similarity matching.
        Returns similarity score (0.0 - 1.0, higher = more similar).

        Args:
            image_url_1: URL or path to first character image
            image_url_2: URL or path to second character image

        Returns:
            Similarity score (1.0 = identical, 0.0 = completely different)
        """
        try:
            from imgutils.metrics import ccip_difference

            # Load images, whether from URL or local path
            img1 = self._load_image_for_ccip(image_url_1)
            img2 = self._load_image_for_ccip(image_url_2)

            if not img1 or not img2:
                logger.warning(
                    "Could not load one or both images for CCIP, falling back to OpenCLIP."
                )
                return self._calculate_openclip_similarity(image_url_1, image_url_2)

            # CCIP returns difference (0 = identical, 1 = different)
            difference = ccip_difference(img1, img2)

            # Convert to similarity
            similarity = 1.0 - difference

            return float(similarity)

        except Exception as e:
            logger.error(f"CCIP character similarity calculation failed: {e}")
            # Fallback to OpenCLIP if CCIP fails
            return self._calculate_openclip_similarity(image_url_1, image_url_2)

    def _calculate_openclip_similarity(
        self, image_url_1: str, image_url_2: str
    ) -> float:
        """Fallback: Calculate similarity using OpenCLIP embeddings.

        Args:
            image_url_1: URL or path to first image
            image_url_2: URL or path to second image

        Returns:
            Cosine similarity score (0.0 - 1.0)
        """
        try:
            from io import BytesIO

            import requests
            import torch
            from PIL import Image

            if not self.model:
                return 0.0

            # Load images
            def load_image(url_or_path: str) -> Image.Image | None:
                if url_or_path.startswith("http"):
                    response = requests.get(url_or_path, timeout=10)
                    return Image.open(BytesIO(response.content))
                else:
                    return Image.open(url_or_path)

            img1 = load_image(image_url_1)
            img2 = load_image(image_url_2)

            if not img1 or not img2:
                return 0.0

            # Encode images
            preprocess = self.model["preprocess"]
            model = self.model["model"]
            device = self.model["device"]

            img1_input = preprocess(img1).unsqueeze(0).to(device)
            img2_input = preprocess(img2).unsqueeze(0).to(device)

            with torch.no_grad():
                emb1 = model.encode_image(img1_input).cpu().numpy().flatten()
                emb2 = model.encode_image(img2_input).cpu().numpy().flatten()

            # Normalize and calculate cosine similarity
            emb1 = emb1 / np.linalg.norm(emb1)
            emb2 = emb2 / np.linalg.norm(emb2)

            similarity = float(np.dot(emb1, emb2))
            return similarity

        except Exception as e:
            logger.error(f"OpenCLIP fallback similarity calculation failed: {e}")
            return 0.0

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the current model.

        Returns:
            Dictionary with model information
        """
        if self.model:
            return {
                "provider": self.model["provider"],
                "model_name": self.model["model_name"],
                "embedding_size": self.model["embedding_size"],
                "input_resolution": self.model["input_resolution"],
                "device": self.model["device"],
                "supports_text": self.model["supports_text"],
                "supports_image": self.model["supports_image"],
                "batch_size": self.model["batch_size"],
            }
        else:
            return {"error": "No model loaded"}

    def switch_model(self, provider: str, model_name: str) -> bool:
        """Switch to a different model.

        Args:
            provider: New model provider
            model_name: New model name

        Returns:
            True if switch successful, False otherwise
        """
        try:
            # Create new model
            new_model = self._create_model(provider, model_name)

            # Switch to new model
            self.model = new_model
            self.provider = provider
            self.model_name = model_name

            logger.info(f"Switched to {provider} model: {model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to switch model: {e}")
            return False

    def validate_image_data(self, image_data: str) -> bool:
        """Validate if image data can be processed.

        Args:
            image_data: Base64 encoded image data

        Returns:
            True if image data is valid, False otherwise
        """
        try:
            image = self._decode_base64_image(image_data)
            return image is not None
        except Exception:
            return False

    def _get_field_mapper(self) -> AnimeFieldMapper:
        """Get field mapper instance for anime data extraction."""
        if self._field_mapper is None:
            self._field_mapper = AnimeFieldMapper()
        return self._field_mapper

    async def process_anime_image_vector(self, anime: AnimeEntry) -> list[float] | None:
        """Process general anime images (covers, posters, banners, trailers) excluding character images.

        Args:
            anime: AnimeEntry instance with image data

        Returns:
            Combined general image embedding vector or None if processing fails
        """
        try:
            field_mapper = self._get_field_mapper()

            # Extract all image URLs from anime data
            image_urls = field_mapper._extract_image_content(anime)

            if not image_urls:
                logger.warning("No image URLs found for anime")
                return None

            logger.debug(
                f"Processing {len(image_urls)} general images for anime (excluding character images)"
            )

            # Process all images with duplicate vector detection
            successful_embeddings = []
            processed_vectors = set()  # Store vector hashes to detect duplicates

            for i, image_url in enumerate(image_urls):
                try:
                    image_data = await self._download_and_cache_image(image_url)
                    if image_data:
                        embedding = self.encode_image(image_data)
                        if embedding:
                            # Create hash of embedding to check for duplicates
                            embedding_hash = self._hash_embedding(embedding)

                            if embedding_hash not in processed_vectors:
                                successful_embeddings.append(embedding)
                                processed_vectors.add(embedding_hash)
                                logger.debug(
                                    f"Successfully encoded unique image {i+1}/{len(image_urls)}"
                                )
                            else:
                                logger.debug(
                                    f"Skipped duplicate image {i+1}/{len(image_urls)}"
                                )
                except Exception as e:
                    logger.warning(f"Failed to process image {i+1}: {e}")
                    continue

            if successful_embeddings:
                # Combine multiple embeddings by averaging (preserves semantic information)
                if len(successful_embeddings) == 1:
                    logger.debug("Single unique image processed")
                    return successful_embeddings[0]
                else:
                    # Average multiple embeddings for comprehensive visual representation
                    combined_embedding: list[float] = np.mean(
                        successful_embeddings, axis=0
                    ).tolist()
                    logger.debug(
                        f"Combined {len(successful_embeddings)} unique image embeddings from {len(image_urls)} total images"
                    )
                    return combined_embedding

            # Fallback: return None to store URLs in payload instead
            logger.info(
                "All general image processing failed, URLs will be stored in payload"
            )
            return None

        except Exception as e:
            logger.error(f"General image vector processing failed: {e}")
            return None

    async def process_anime_character_image_vector(
        self, anime: AnimeEntry
    ) -> list[float] | None:
        """Process character images from anime data for character identification and recommendations.

        Args:
            anime: AnimeEntry instance with character image data

        Returns:
            Combined character image embedding vector or None if processing fails
        """
        try:
            field_mapper = self._get_field_mapper()

            # Extract character image URLs from anime data (separate from general images)
            character_image_urls = field_mapper._extract_character_image_content(anime)

            if not character_image_urls:
                logger.debug("No character image URLs found for anime")
                return None

            logger.debug(
                f"Processing {len(character_image_urls)} character images for anime"
            )

            # Process character images with duplicate vector detection
            successful_embeddings = []
            processed_vectors = set()  # Store vector hashes to detect duplicates

            for i, image_url in enumerate(character_image_urls):
                try:
                    image_data = await self._download_and_cache_image(image_url)
                    if image_data:
                        embedding = self.encode_image(image_data)
                        if embedding:
                            # Create hash of embedding to check for duplicates
                            embedding_hash = self._hash_embedding(embedding)

                            if embedding_hash not in processed_vectors:
                                successful_embeddings.append(embedding)
                                processed_vectors.add(embedding_hash)
                                logger.debug(
                                    f"Successfully encoded unique character image {i+1}/{len(character_image_urls)}"
                                )
                            else:
                                logger.debug(
                                    f"Skipped duplicate character image {i+1}/{len(character_image_urls)}"
                                )
                except Exception as e:
                    logger.warning(f"Failed to process character image {i+1}: {e}")
                    continue

            if successful_embeddings:
                # Combine multiple embeddings by averaging (preserves character identification features)
                if len(successful_embeddings) == 1:
                    logger.debug("Single unique character image processed")
                    return successful_embeddings[0]
                else:
                    # Average multiple embeddings for comprehensive character visual representation
                    combined_embedding: list[float] = np.mean(
                        successful_embeddings, axis=0
                    ).tolist()
                    logger.debug(
                        f"Combined {len(successful_embeddings)} unique character image embeddings from {len(character_image_urls)} total character images"
                    )
                    return combined_embedding

            # Fallback: return None to store character image URLs in payload instead
            logger.debug(
                "All character image processing failed, URLs will be stored in payload"
            )
            return None

        except Exception as e:
            logger.error(f"Character image vector processing failed: {e}")
            return None

    def _hash_embedding(self, embedding: list[float], precision: int = 4) -> str:
        """Create hash of embedding vector to detect duplicates.

        Args:
            embedding: Embedding vector
            precision: Decimal precision for hash (default 4 for similarity detection)

        Returns:
            Hash string of the embedding
        """
        try:
            # Round to specified precision to catch near-identical embeddings
            rounded_embedding = [round(x, precision) for x in embedding]
            # Create hash from rounded values
            embedding_str = ",".join(map(str, rounded_embedding))
            return hashlib.md5(embedding_str.encode()).hexdigest()
        except Exception:
            # Fallback to string representation
            return str(hash(tuple(embedding)))

    async def _download_and_cache_image(self, image_url: str) -> str | None:
        """Download image from URL and cache locally.

        Args:
            image_url: URL of the image to download

        Returns:
            Base64 encoded image data or None if download fails
        """
        try:
            # Generate cache key from URL
            cache_key = hashlib.md5(image_url.encode()).hexdigest()
            cache_file = self.image_cache_dir / f"{cache_key}.jpg"

            # Check if already cached
            if cache_file.exists():
                logger.debug(f"Loading cached image: {cache_key}")
                with open(cache_file, "rb") as f:
                    image_bytes = f.read()
                    return base64.b64encode(image_bytes).decode()

            # Download image
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    image_url,
                    timeout=aiohttp.ClientTimeout(total=10),
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    },
                ) as response:
                    if response.status == 200:
                        image_bytes = await response.read()

                        # Validate image
                        try:
                            image: Image.Image = Image.open(io.BytesIO(image_bytes))
                            if image.mode != "RGB":
                                image = image.convert("RGB")

                            # Cache image
                            image.save(cache_file, "JPEG", quality=85)
                            logger.debug(f"Cached image: {cache_key}")

                            # Return base64 encoded
                            return base64.b64encode(image_bytes).decode()

                        except Exception as e:
                            logger.error(f"Invalid image format from {image_url}: {e}")
                            return None
                    else:
                        logger.warning(
                            f"Failed to download image from {image_url}: {response.status}"
                        )
                        return None

        except TimeoutError:
            logger.warning(f"Timeout downloading image from {image_url}")
            return None
        except Exception as e:
            logger.error(f"Error downloading image from {image_url}: {e}")
            return None

    def get_cache_stats(self) -> dict[str, Any]:
        """Get image cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        try:
            cache_files = list(self.image_cache_dir.glob("*.jpg"))
            total_size = sum(f.stat().st_size for f in cache_files)

            return {
                "cache_dir": str(self.image_cache_dir),
                "cached_images": len(cache_files),
                "total_cache_size_mb": round(total_size / (1024 * 1024), 2),
                "cache_enabled": True,
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"cache_enabled": False, "error": str(e)}

    def clear_cache(self, max_age_days: int | None = None) -> dict[str, Any]:
        """Clear image cache.

        Args:
            max_age_days: Only clear files older than this many days (optional)

        Returns:
            Dictionary with cleanup statistics
        """
        try:
            import time

            cache_files = list(self.image_cache_dir.glob("*.jpg"))
            removed_count = 0
            total_removed_size = 0

            cutoff_time = None
            if max_age_days:
                cutoff_time = time.time() - (max_age_days * 24 * 3600)

            for cache_file in cache_files:
                if cutoff_time and cache_file.stat().st_mtime > cutoff_time:
                    continue

                file_size = cache_file.stat().st_size
                cache_file.unlink()
                removed_count += 1
                total_removed_size += file_size

            return {
                "removed_files": removed_count,
                "removed_size_mb": round(total_removed_size / (1024 * 1024), 2),
                "remaining_files": len(list(self.image_cache_dir.glob("*.jpg"))),
            }

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return {"error": str(e)}

    def get_supported_formats(self) -> list[str]:
        """Get list of supported image formats.

        Returns:
            List of supported image format extensions
        """
        return ["jpg", "jpeg", "png", "bmp", "tiff", "webp", "gif"]
