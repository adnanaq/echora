"""Tests for explicit primary vector config fields."""

import pytest
from common.config.qdrant_config import QdrantConfig


def test_primary_vector_names_must_exist_in_vector_names() -> None:
    with pytest.raises(ValueError, match="primary_text_vector_name"):
        QdrantConfig(
            vector_names={"custom_text": 16, "custom_image": 8},
            primary_text_vector_name="text_vector",
            primary_image_vector_name="custom_image",
        )


def test_primary_vector_names_accept_valid_custom_names() -> None:
    cfg = QdrantConfig(
        vector_names={"custom_text": 16, "custom_image": 8},
        primary_text_vector_name="custom_text",
        primary_image_vector_name="custom_image",
    )
    assert cfg.primary_text_vector_name == "custom_text"
    assert cfg.primary_image_vector_name == "custom_image"
