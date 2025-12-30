import hashlib
import io
import logging
from pathlib import Path
from typing import Any

import aiohttp
from PIL import Image

logger = logging.getLogger(__name__)


class ImageDownloader:
    """Utility for downloading and caching images."""

    def __init__(self, cache_dir: str | None = None):
        """Initialize image downloader.

        Args:
            cache_dir: Directory to store cached images
        """
        self.image_cache_dir = (
            Path("cache/images") if not cache_dir else Path(cache_dir) / "images"
        )
        self.image_cache_dir.mkdir(parents=True, exist_ok=True)

    async def download_and_cache_image(self, image_url: str) -> str | None:
        """Download image from URL and cache locally.

        Args:
            image_url: URL of the image to download

        Returns:
            Path to cached image file or None if download fails
        """
        try:
            # Generate cache key from URL
            cache_key = hashlib.md5(image_url.encode()).hexdigest()
            cache_file = self.image_cache_dir / f"{cache_key}.jpg"

            # Check if already cached
            if cache_file.exists():
                return str(cache_file.absolute())

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

                            # Return path
                            return str(cache_file.absolute())

                        except Exception as e:
                            logger.error(f"Invalid image data from {image_url}: {e}")
                            return None
                    else:
                        logger.warning(
                            f"Failed to download image {image_url}: status {response.status}"
                        )
                        return None

        except TimeoutError:
            logger.error(f"Timeout downloading image from {image_url}")
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
