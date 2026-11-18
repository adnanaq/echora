import io
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from PIL import Image
from vector_processing.utils.image_downloader import ImageDownloader


def _png_bytes(mode: str = "RGB") -> bytes:
    buf = io.BytesIO()
    Image.new(mode, (4, 4)).save(buf, format="PNG")
    return buf.getvalue()


def test_init_uses_default_and_custom_cache_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    default_downloader = ImageDownloader()
    assert default_downloader.image_cache_dir == Path("cache/images")
    assert default_downloader.image_cache_dir.is_dir()

    custom_dir = tmp_path / "custom"
    custom_downloader = ImageDownloader(cache_dir=str(custom_dir))
    assert custom_downloader.image_cache_dir == custom_dir / "images"
    assert custom_downloader.image_cache_dir.is_dir()


@pytest.mark.asyncio
async def test_download_returns_cached_file_without_http(tmp_path):
    downloader = ImageDownloader(cache_dir=str(tmp_path))
    session = MagicMock(spec=aiohttp.ClientSession)

    response = MagicMock()
    response.status = 200
    response.read = AsyncMock(return_value=_png_bytes("RGBA"))
    session.get = AsyncMock(return_value=response)

    first_path = await downloader.download_and_cache_image(
        "https://example.com/a.png", session
    )
    assert first_path is not None
    assert Path(first_path).exists()
    response.close.assert_called_once()

    session.get.reset_mock()
    second_path = await downloader.download_and_cache_image(
        "https://example.com/a.png", session
    )
    assert second_path == first_path
    session.get.assert_not_called()


@pytest.mark.asyncio
async def test_download_rgb_image_saves_jpeg(tmp_path):
    downloader = ImageDownloader(cache_dir=str(tmp_path))
    session = MagicMock(spec=aiohttp.ClientSession)
    response = MagicMock()
    response.status = 200
    response.read = AsyncMock(return_value=_png_bytes("RGB"))
    session.get = AsyncMock(return_value=response)

    saved_path = await downloader.download_and_cache_image(
        "https://example.com/rgb.png", session
    )
    assert saved_path is not None
    with Image.open(saved_path) as img:
        assert img.format == "JPEG"
        assert img.mode == "RGB"


@pytest.mark.asyncio
async def test_download_handles_non_200_invalid_bytes_timeout_and_error(tmp_path):
    downloader = ImageDownloader(cache_dir=str(tmp_path))
    session = MagicMock(spec=aiohttp.ClientSession)

    bad_status_response = MagicMock()
    bad_status_response.status = 404
    session.get = AsyncMock(return_value=bad_status_response)
    assert (
        await downloader.download_and_cache_image(
            "https://example.com/missing.jpg", session
        )
        is None
    )
    bad_status_response.close.assert_called_once()

    invalid_response = MagicMock()
    invalid_response.status = 200
    invalid_response.read = AsyncMock(return_value=b"not-an-image")
    session.get = AsyncMock(return_value=invalid_response)
    assert (
        await downloader.download_and_cache_image(
            "https://example.com/corrupt.jpg", session
        )
        is None
    )
    invalid_response.close.assert_called_once()

    session.get = AsyncMock(side_effect=TimeoutError("timed out"))
    assert (
        await downloader.download_and_cache_image(
            "https://example.com/slow.jpg", session
        )
        is None
    )

    session.get = AsyncMock(side_effect=RuntimeError("boom"))
    assert (
        await downloader.download_and_cache_image(
            "https://example.com/err.jpg", session
        )
        is None
    )


def test_get_cache_stats_and_error(tmp_path):
    downloader = ImageDownloader(cache_dir=str(tmp_path))
    sample = downloader.image_cache_dir / "one.jpg"
    sample.write_bytes(b"12345")

    stats = downloader.get_cache_stats()
    assert stats["cached_images"] == 1
    assert stats["cache_enabled"] is True

    with patch.object(Path, "glob", side_effect=OSError("disk error")):
        error_stats = downloader.get_cache_stats()
    assert error_stats["cache_enabled"] is False
    assert "disk error" in error_stats["error"]


def test_clear_cache_all_by_age_and_error(tmp_path):
    downloader = ImageDownloader(cache_dir=str(tmp_path))
    old_file = downloader.image_cache_dir / "old.jpg"
    fresh_file = downloader.image_cache_dir / "fresh.jpg"
    old_file.write_bytes(b"old")
    fresh_file.write_bytes(b"fresh")

    old_mtime = time.time() - (5 * 24 * 3600)
    os.utime(old_file, (old_mtime, old_mtime))

    age_result = downloader.clear_cache(max_age_days=2)
    assert age_result["removed_files"] == 1
    assert age_result["remaining_files"] == 1
    assert not old_file.exists()
    assert fresh_file.exists()

    all_result = downloader.clear_cache()
    assert all_result["removed_files"] == 1
    assert all_result["remaining_files"] == 0
    assert not fresh_file.exists()

    with patch.object(Path, "glob", side_effect=OSError("clear error")):
        error_result = downloader.clear_cache()
    assert "clear error" in error_result["error"]
