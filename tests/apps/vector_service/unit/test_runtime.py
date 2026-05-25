"""Unit tests for vector_service runtime helpers."""

from types import SimpleNamespace

import pytest
from qdrant_db.errors import ConfigurationError
from vector_service.runtime import _validate_model_dimensions


def _make_settings(text_dim: int = 1024, image_dim: int = 768) -> SimpleNamespace:
    return SimpleNamespace(
        qdrant=SimpleNamespace(
            primary_text_vector_name="text_vector",
            primary_image_vector_name="image_vector",
            vector_names={"text_vector": text_dim, "image_vector": image_dim},
        )
    )


def _make_processors(text_dim: int, image_dim: int) -> tuple[SimpleNamespace, SimpleNamespace]:
    return (
        SimpleNamespace(model=SimpleNamespace(embedding_size=text_dim)),
        SimpleNamespace(model=SimpleNamespace(embedding_size=image_dim)),
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_validate_model_dimensions_passes_when_all_match() -> None:
    text_proc, vision_proc = _make_processors(1024, 768)
    # Must not raise
    _validate_model_dimensions(_make_settings(1024, 768), text_proc, vision_proc)


# ---------------------------------------------------------------------------
# Dimension mismatch — parameterized over text and image
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "vector_name, text_model_dim, image_model_dim, config_text_dim, config_image_dim",
    [
        # Text model produces 512 but config expects 1024
        ("text_vector", 512, 768, 1024, 768),
        # Image model produces 512 but config expects 768
        ("image_vector", 1024, 512, 1024, 768),
    ],
    ids=["text_mismatch", "image_mismatch"],
)
def test_validate_model_dimensions_raises_on_mismatch(
    vector_name: str,
    text_model_dim: int,
    image_model_dim: int,
    config_text_dim: int,
    config_image_dim: int,
) -> None:
    text_proc, vision_proc = _make_processors(text_model_dim, image_model_dim)
    settings = _make_settings(config_text_dim, config_image_dim)

    with pytest.raises(ConfigurationError, match=vector_name) as exc_info:
        _validate_model_dimensions(settings, text_proc, vision_proc)

    # Error message must include both actual and expected dims for actionability
    error_msg = str(exc_info.value)
    actual = text_model_dim if vector_name == "text_vector" else image_model_dim
    expected = config_text_dim if vector_name == "text_vector" else config_image_dim
    assert str(actual) in error_msg
    assert str(expected) in error_msg


# ---------------------------------------------------------------------------
# Fail-fast: text is checked before image
# ---------------------------------------------------------------------------


def test_validate_model_dimensions_fails_on_text_first_when_both_wrong() -> None:
    text_proc, vision_proc = _make_processors(text_dim=512, image_dim=512)
    settings = _make_settings(text_dim=1024, image_dim=768)

    with pytest.raises(ConfigurationError, match="text_vector"):
        _validate_model_dimensions(settings, text_proc, vision_proc)
