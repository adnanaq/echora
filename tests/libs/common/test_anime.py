"""Unit tests for Image model validation."""

import uuid

import pytest
from common.models.anime import Image
from common.utils.id_generation import generate_entity_id
from pydantic import ValidationError


def _is_valid_uuid(val: str) -> bool:
    """Check if a string is a valid UUID."""
    try:
        uuid.UUID(val)
        return True
    except ValueError:
        return False


class TestImageModel:
    """Test suite for Image model validation."""

    def test_image_with_anime_id_valid(self):
        """Test creating image with anime_id is valid."""
        image = Image(
            id=generate_entity_id(),
            image_url="https://example.com/cover.jpg",
            anime_id="01934f6c-8a2b-4c3d-9e5f-1a2b3c4d5e6f",
        )
        assert image.anime_id is not None
        assert image.character_id is None
        assert _is_valid_uuid(image.id)

    def test_image_with_character_id_valid(self):
        """Test creating image with character_id is valid."""
        image = Image(
            id=generate_entity_id(),
            image_url="https://example.com/portrait.jpg",
            character_id="01934f6c-1111-2222-3333-444455556666",
        )
        assert image.anime_id is None
        assert image.character_id is not None
        assert _is_valid_uuid(image.id)

    def test_image_with_both_ids_invalid(self):
        """Test creating image with both anime_id and character_id raises error."""
        with pytest.raises(ValidationError) as exc_info:
            Image(
                id=generate_entity_id(),
                image_url="https://example.com/image.jpg",
                anime_id="01934f6c-8a2b-4c3d-9e5f-1a2b3c4d5e6f",
                character_id="01934f6c-1111-2222-3333-444455556666",
            )
        assert "cannot belong to both" in str(exc_info.value).lower()

    def test_image_with_no_parent_invalid(self):
        """Test creating image without any parent ID raises error."""
        with pytest.raises(ValidationError) as exc_info:
            Image(
                id=generate_entity_id(),
                image_url="https://example.com/orphan.jpg",
            )
        assert "must belong to either" in str(exc_info.value).lower()

    def test_image_model_dump_excludes_none(self):
        """Test model_dump excludes None values correctly."""
        image = Image(
            id="01934f6c-0000-0000-0000-000000000001",
            image_url="https://example.com/cover.jpg",
            anime_id="01934f6c-0000-0000-0000-000000000002",
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
                anime_id="01934f6c-0000-0000-0000-000000000001",
            )  # Missing id

        with pytest.raises(ValidationError):
            Image(
                id="01934f6c-0000-0000-0000-000000000001",
                anime_id="01934f6c-0000-0000-0000-000000000002",
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
                id=generate_entity_id(),
                image_url=url,
                anime_id="01934f6c-0000-0000-0000-000000000001",
            )
            assert image.image_url == url


class TestImageIdGeneration:
    def test_generate_entity_id_creates_valid_uuid(self):
        """Test generate_entity_id creates valid UUID strings."""
        image_id = generate_entity_id()
        assert isinstance(image_id, str)
        assert len(image_id) == 36
        # Basic UUID structure check
        assert len(image_id.split("-")) == 5

    def test_image_ids_are_unique(self):
        """Test that generated IDs are unique."""
        ids = [generate_entity_id() for _ in range(100)]
        assert len(set(ids)) == 100
