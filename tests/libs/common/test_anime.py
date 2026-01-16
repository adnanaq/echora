"""Unit tests for Image model validation."""

import pytest
from common.models.anime import Image
from common.utils.id_generation import generate_ulid
from pydantic import ValidationError


class TestImageModel:
    """Test suite for Image model validation."""

    def test_image_with_anime_id_valid(self):
        """Test creating image with anime_id is valid."""
        image = Image(
            id=generate_ulid("image"),
            image_url="https://example.com/cover.jpg",
            anime_id="anime_01ARZ3NDEKTSV4RRFFQ69G5FAV",
        )
        assert image.anime_id is not None
        assert image.character_id is None
        assert image.id.startswith("img_")

    def test_image_with_character_id_valid(self):
        """Test creating image with character_id is valid."""
        image = Image(
            id=generate_ulid("image"),
            image_url="https://example.com/portrait.jpg",
            character_id="char_01ARZ3NDEKTSV4RRFFQ69G5FAV",
        )
        assert image.anime_id is None
        assert image.character_id is not None
        assert image.id.startswith("img_")

    def test_image_with_both_ids_invalid(self):
        """Test creating image with both anime_id and character_id raises error."""
        with pytest.raises(ValidationError) as exc_info:
            Image(
                id=generate_ulid("image"),
                image_url="https://example.com/image.jpg",
                anime_id="anime_01ARZ3NDEKTSV4RRFFQ69G5FAV",
                character_id="char_01ARZ3NDEKTSV4RRFFQ69G5FAV",
            )
        assert "cannot belong to both" in str(exc_info.value).lower()

    def test_image_with_no_parent_invalid(self):
        """Test creating image without any parent ID raises error."""
        with pytest.raises(ValidationError) as exc_info:
            Image(
                id=generate_ulid("image"),
                image_url="https://example.com/orphan.jpg",
            )
        assert "must belong to either" in str(exc_info.value).lower()

    def test_image_model_dump_excludes_none(self):
        """Test model_dump excludes None values correctly."""
        image = Image(
            id="img_test123",
            image_url="https://example.com/cover.jpg",
            anime_id="anime_test456",
        )
        dumped = image.model_dump(exclude_none=True)

        assert "id" in dumped
        assert "image_url" in dumped
        assert "anime_id" in dumped
        assert "character_id" not in dumped  # Should be excluded

    def test_image_required_fields(self):
        """Test that id and image_url are required."""
        with pytest.raises(ValidationError):
            Image(
                image_url="https://example.com/image.jpg",
                anime_id="anime_test",
            )  # Missing id

        with pytest.raises(ValidationError):
            Image(
                id="img_test",
                anime_id="anime_test",
            )  # Missing image_url

    def test_image_url_formats(self):
        """Test various image URL formats are accepted."""
        valid_urls = [
            "https://example.com/image.jpg",
            "http://example.com/image.png",
            "https://cdn.example.com/path/to/image.webp",
            "https://example.com/image?size=large",
        ]

        for url in valid_urls:
            image = Image(
                id=generate_ulid("image"),
                image_url=url,
                anime_id="anime_test",
            )
            assert image.image_url == url


class TestImageIdGeneration:
    """Test suite for Image ID generation."""

    def test_generate_ulid_with_image_type(self):
        """Test generate_ulid creates img_ prefixed IDs."""
        image_id = generate_ulid("image")
        assert image_id.startswith("img_")
        assert len(image_id) > 4  # prefix + ULID

    def test_image_ids_are_unique(self):
        """Test that generated image IDs are unique."""
        ids = [generate_ulid("image") for _ in range(100)]
        assert len(ids) == len(set(ids))  # All unique
