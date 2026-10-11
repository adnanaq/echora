from unittest.mock import patch

import pytest
from common.config.qdrant_config import HnswSettings, QdrantConfig, QuantizationSettings
from pydantic import ValidationError

_CUSTOM = {
    "vector_names": {"custom_text": 16, "custom_image": 8},
    "multivector_vectors": ["custom_image"],
    "vector_priorities": {
        "high": ["custom_text", "custom_image"],
        "medium": [],
        "low": [],
    },
}


def test_qdrant_config_primary_text_vector_missing_from_vector_names_fails() -> None:
    with pytest.raises(ValueError, match="primary_text_vector_name"):
        QdrantConfig(
            **_CUSTOM,
            primary_text_vector_name="text_vector",
            primary_image_vector_name="custom_image",
        )


def test_qdrant_config_custom_primary_vector_names_are_kept() -> None:
    config = QdrantConfig(
        **_CUSTOM,
        primary_text_vector_name="custom_text",
        primary_image_vector_name="custom_image",
    )

    assert (config.primary_text_vector_name, config.primary_image_vector_name) == (
        "custom_text",
        "custom_image",
    )


def test_qdrant_config_defaults_load() -> None:
    config = QdrantConfig()

    assert config.hnsw_config["high"] == HnswSettings(ef_construct=256, m=64)
    assert config.quantization_config["low"] == QuantizationSettings(
        type="binary", always_ram=False
    )


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("qdrant_distance_metric", "COSINE", "cosine"),
        ("qdrant_distance_metric", "Dot", "dot"),
        ("qdrant_quantization_type", "Scalar", "scalar"),
        ("sparse_vector_modifier", "IDF", "idf"),
    ],
)
def test_qdrant_config_choice_any_case_is_lowered(
    field: str, value: str, expected: str
) -> None:
    assert getattr(QdrantConfig(**{field: value}), field) == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("qdrant_distance_metric", "manhattan"),
        ("qdrant_quantization_type", "int4"),
        ("sparse_vector_modifier", "bm25"),
    ],
)
def test_qdrant_config_unknown_choice_fails(field: str, value: str) -> None:
    with pytest.raises(ValidationError):
        QdrantConfig(**{field: value})


def test_qdrant_config_choice_from_environment_is_lowered() -> None:
    with patch.dict(
        "os.environ",
        {"QDRANT_DISTANCE_METRIC": "EUCLID", "QDRANT_QUANTIZATION_TYPE": "BINARY"},
    ):
        config = QdrantConfig()

    assert (config.qdrant_distance_metric, config.qdrant_quantization_type) == (
        "euclid",
        "binary",
    )


@pytest.mark.parametrize("vector_names", [{}, {"text_vector": 0}, {"text_vector": -1}])
def test_qdrant_config_invalid_vector_names_fail(vector_names: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        QdrantConfig(vector_names=vector_names)


def test_qdrant_config_unknown_priority_fails() -> None:
    with pytest.raises(ValidationError):
        QdrantConfig(vector_priorities={"urgent": ["text_vector"]})  # ty: ignore[invalid-argument-type]


def test_qdrant_config_priority_vector_missing_from_vector_names_fails() -> None:
    with pytest.raises(ValidationError, match="unknown_vector"):
        QdrantConfig(
            vector_priorities={
                "high": ["text_vector", "unknown_vector"],
                "medium": [],
                "low": [],
            }
        )


@pytest.mark.parametrize("field", ["hnsw_config", "quantization_config"])
def test_qdrant_config_priority_without_settings_fails(field: str) -> None:
    settings = QdrantConfig().model_dump()[field]
    del settings["medium"]

    with pytest.raises(ValidationError, match="medium"):
        QdrantConfig(**{field: settings})


@pytest.mark.parametrize(
    "settings",
    [
        {"ef_construct": 0, "m": 16},
        {"ef_construct": 100, "m": -1},
        {"ef_construct": 100},
    ],
)
def test_hnsw_settings_invalid_values_fail(settings: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        HnswSettings.model_validate(settings)


def test_quantization_settings_unknown_type_fails() -> None:
    with pytest.raises(ValidationError):
        QuantizationSettings.model_validate({"type": "product"})


def test_qdrant_config_unknown_multivector_vector_fails() -> None:
    with pytest.raises(ValidationError, match="Unknown multivector vectors"):
        QdrantConfig(multivector_vectors=["missing_vector"])
