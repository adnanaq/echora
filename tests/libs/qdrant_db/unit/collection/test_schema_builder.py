"""Unit tests for schema_builder pure functions."""

from unittest.mock import patch

import pytest
from common.config import QdrantConfig
from qdrant_client.models import (
    BinaryQuantization,
    Distance,
    HnswConfigDiff,
    Modifier,
    MultiVectorConfig,
    OptimizersConfigDiff,
    ScalarQuantization,
    SparseIndexParams,
    WalConfigDiff,
)
from qdrant_db.collection.schema_builder import (
    build_optimizers_config,
    build_quantization_config,
    build_sparse_vector_config,
    build_vector_config,
    build_wal_config,
    get_hnsw_config,
    get_per_vector_quantization_config,
    get_vector_priority,
    validate_vector_config,
)
from qdrant_db.errors import ConfigurationError


def _base_config(**overrides: object) -> QdrantConfig:
    """Minimal valid QdrantConfig for schema builder tests."""
    defaults: dict[str, object] = {
        "vector_names": {"text_vector": 1024, "image_vector": 768},
        "multivector_vectors": ["image_vector"],
        "sparse_vector_names": ["text_sparse_vector"],
        "primary_sparse_vector_name": "text_sparse_vector",
        "primary_text_vector_name": "text_vector",
        "primary_image_vector_name": "image_vector",
        "vector_priorities": {
            "high": ["text_vector", "image_vector"],
            "medium": [],
            "low": [],
        },
        "hnsw_config": {
            "high": {"ef_construct": 256, "m": 64},
            "medium": {"ef_construct": 200, "m": 48},
            "low": {"ef_construct": 128, "m": 32},
        },
        "quantization_config": {
            "high": {"type": "scalar", "always_ram": True},
            "medium": {"type": "scalar", "always_ram": False},
            "low": {"type": "binary", "always_ram": False},
        },
    }
    defaults.update(overrides)
    return QdrantConfig.model_validate(defaults)


# ---------------------------------------------------------------------------
# get_vector_priority
# ---------------------------------------------------------------------------


def test_get_vector_priority_returns_correct_bucket() -> None:
    config = _base_config()
    assert get_vector_priority(config, "text_vector") == "high"
    assert get_vector_priority(config, "image_vector") == "high"


def test_get_vector_priority_defaults_to_medium_for_unknown() -> None:
    config = _base_config()
    assert get_vector_priority(config, "nonexistent_vector") == "medium"


def test_get_vector_priority_low_bucket() -> None:
    config = _base_config(
        vector_priorities={
            "high": [],
            "medium": [],
            "low": ["text_vector", "image_vector"],
        }
    )
    assert get_vector_priority(config, "text_vector") == "low"


# ---------------------------------------------------------------------------
# get_hnsw_config
# ---------------------------------------------------------------------------


def test_get_hnsw_config_high_priority() -> None:
    config = _base_config()
    result = get_hnsw_config(config, "high")
    assert isinstance(result, HnswConfigDiff)
    assert result.ef_construct == 256
    assert result.m == 64


def test_get_hnsw_config_medium_priority() -> None:
    config = _base_config()
    result = get_hnsw_config(config, "medium")
    assert result.ef_construct == 200
    assert result.m == 48


def test_get_hnsw_config_unknown_priority_uses_defaults() -> None:
    config = _base_config()
    result = get_hnsw_config(config, "nonexistent")
    assert result.ef_construct == 200
    assert result.m == 48


# ---------------------------------------------------------------------------
# get_per_vector_quantization_config
# ---------------------------------------------------------------------------


def test_get_per_vector_quantization_config_scalar() -> None:
    config = _base_config()
    result = get_per_vector_quantization_config(config, "high")
    assert isinstance(result, ScalarQuantization)
    assert result.scalar.always_ram is True


def test_get_per_vector_quantization_config_binary() -> None:
    config = _base_config()
    result = get_per_vector_quantization_config(config, "low")
    assert isinstance(result, BinaryQuantization)
    assert result.binary.always_ram is False


def test_get_per_vector_quantization_config_unknown_returns_none() -> None:
    config = _base_config()
    result = get_per_vector_quantization_config(config, "nonexistent")
    assert result is None


# ---------------------------------------------------------------------------
# build_vector_config
# ---------------------------------------------------------------------------


def test_build_vector_config_returns_all_named_vectors() -> None:
    config = _base_config()
    result = build_vector_config(config)
    assert set(result.keys()) == {"text_vector", "image_vector"}


def test_build_vector_config_correct_dimensions() -> None:
    config = _base_config()
    result = build_vector_config(config)
    assert result["text_vector"].size == 1024
    assert result["image_vector"].size == 768


def test_build_vector_config_cosine_distance() -> None:
    config = _base_config(qdrant_distance_metric="cosine")
    result = build_vector_config(config)
    assert result["text_vector"].distance == Distance.COSINE


def test_build_vector_config_dot_distance() -> None:
    config = _base_config(qdrant_distance_metric="dot")
    result = build_vector_config(config)
    assert result["text_vector"].distance == Distance.DOT


def test_build_vector_config_multivector_only_on_configured_vectors() -> None:
    config = _base_config()
    result = build_vector_config(config)
    assert result["image_vector"].multivector_config is not None
    assert isinstance(result["image_vector"].multivector_config, MultiVectorConfig)
    assert result["text_vector"].multivector_config is None


def test_build_vector_config_hnsw_config_applied() -> None:
    config = _base_config()
    result = build_vector_config(config)
    assert result["text_vector"].hnsw_config is not None
    assert result["text_vector"].hnsw_config.ef_construct == 256


# ---------------------------------------------------------------------------
# build_sparse_vector_config
# ---------------------------------------------------------------------------


def test_build_sparse_vector_config_returns_configured_names() -> None:
    config = _base_config()
    result = build_sparse_vector_config(config)
    assert result is not None
    assert "text_sparse_vector" in result


def test_build_sparse_vector_config_on_disk_false_by_default() -> None:
    config = _base_config(sparse_index_on_disk=False)
    result = build_sparse_vector_config(config)
    assert result is not None
    assert isinstance(result["text_sparse_vector"].index, SparseIndexParams)
    assert result["text_sparse_vector"].index.on_disk is False


def test_build_sparse_vector_config_on_disk_true() -> None:
    config = _base_config(sparse_index_on_disk=True)
    result = build_sparse_vector_config(config)
    assert result is not None
    assert result["text_sparse_vector"].index.on_disk is True


def test_build_sparse_vector_config_idf_modifier() -> None:
    config = _base_config(sparse_vector_modifier="idf")
    result = build_sparse_vector_config(config)
    assert result is not None
    assert result["text_sparse_vector"].modifier == Modifier.IDF


def test_build_sparse_vector_config_no_modifier_when_none() -> None:
    config = _base_config(sparse_vector_modifier="none")
    result = build_sparse_vector_config(config)
    assert result is not None
    assert result["text_sparse_vector"].modifier is None


# ---------------------------------------------------------------------------
# build_quantization_config
# ---------------------------------------------------------------------------


def test_build_quantization_config_disabled_returns_none() -> None:
    config = _base_config(qdrant_enable_quantization=False)
    assert build_quantization_config(config) is None


def test_build_quantization_config_scalar() -> None:
    config = _base_config(
        qdrant_enable_quantization=True,
        qdrant_quantization_type="scalar",
        qdrant_quantization_always_ram=True,
    )
    result = build_quantization_config(config)
    assert isinstance(result, ScalarQuantization)
    assert result.scalar.always_ram is True


def test_build_quantization_config_binary() -> None:
    from qdrant_client.models import ProductQuantization

    config = _base_config(
        qdrant_enable_quantization=True,
        qdrant_quantization_type="binary",
        qdrant_quantization_always_ram=False,
    )
    result = build_quantization_config(config)
    assert isinstance(result, BinaryQuantization)
    assert result.binary.always_ram is False

    product_config = _base_config(
        qdrant_enable_quantization=True,
        qdrant_quantization_type="product",
    )
    assert isinstance(build_quantization_config(product_config), ProductQuantization)


def test_build_quantization_config_scalar_exception_returns_none() -> None:
    config = _base_config(
        qdrant_enable_quantization=True,
        qdrant_quantization_type="scalar",
    )
    with patch(
        "qdrant_db.collection.schema_builder.ScalarQuantization",
        side_effect=RuntimeError("boom"),
    ):
        assert build_quantization_config(config) is None


def test_build_quantization_config_product() -> None:
    from qdrant_client.models import ProductQuantization

    config = _base_config(
        qdrant_enable_quantization=True,
        qdrant_quantization_type="product",
    )
    result = build_quantization_config(config)
    assert isinstance(result, ProductQuantization)


def test_build_quantization_config_exception_returns_none() -> None:
    config = _base_config(
        qdrant_enable_quantization=True,
        qdrant_quantization_type="scalar",
    )
    with patch(
        "qdrant_db.collection.schema_builder.ScalarQuantization",
        side_effect=RuntimeError("boom"),
    ):
        result = build_quantization_config(config)
    assert result is None


# ---------------------------------------------------------------------------
# build_optimizers_config
# ---------------------------------------------------------------------------


def test_build_optimizers_config_returns_diff() -> None:
    config = _base_config(memory_mapping_threshold_mb=50)
    result = build_optimizers_config(config)
    assert isinstance(result, OptimizersConfigDiff)
    assert result.default_segment_number == 4
    assert result.indexing_threshold == 20000
    assert result.memmap_threshold == 50 * 1024

    with patch(
        "qdrant_db.collection.schema_builder.OptimizersConfigDiff",
        side_effect=RuntimeError("boom"),
    ):
        assert build_optimizers_config(config) is None


def test_build_optimizers_config_memmap_scales_with_mb() -> None:
    config = _base_config(memory_mapping_threshold_mb=100)
    result = build_optimizers_config(config)
    assert result is not None
    assert result.memmap_threshold == 100 * 1024


def test_build_optimizers_config_exception_returns_none() -> None:
    config = _base_config()
    with patch(
        "qdrant_db.collection.schema_builder.OptimizersConfigDiff",
        side_effect=RuntimeError("boom"),
    ):
        result = build_optimizers_config(config)
    assert result is None


# ---------------------------------------------------------------------------
# build_wal_config
# ---------------------------------------------------------------------------


def test_build_wal_config_disabled_returns_none() -> None:
    config = _base_config(qdrant_enable_wal=False)
    assert build_wal_config(config) is None


def test_build_wal_config_none_returns_none() -> None:
    config = _base_config(qdrant_enable_wal=None)
    assert build_wal_config(config) is None


def test_build_wal_config_enabled_returns_diff() -> None:
    config = _base_config(qdrant_enable_wal=True)
    result = build_wal_config(config)
    assert isinstance(result, WalConfigDiff)
    assert result.wal_capacity_mb == 32
    assert result.wal_segments_ahead == 0

    with patch(
        "qdrant_db.collection.schema_builder.WalConfigDiff",
        side_effect=RuntimeError("boom"),
    ):
        assert build_wal_config(config) is None


# ---------------------------------------------------------------------------
# validate_vector_config
# ---------------------------------------------------------------------------


def test_validate_vector_config_raises_on_empty() -> None:
    config = _base_config()
    with pytest.raises(ConfigurationError, match="empty"):
        validate_vector_config(config, {})


def test_validate_vector_config_raises_on_count_mismatch() -> None:
    config = _base_config()
    vectors_config = build_vector_config(config)
    # Remove one vector to create mismatch
    del vectors_config["image_vector"]
    with pytest.raises(ConfigurationError, match="mismatch"):
        validate_vector_config(config, vectors_config)


def test_validate_vector_config_raises_on_wrong_vector_name() -> None:
    config = _base_config()
    vectors_config = build_vector_config(config)
    # Rename a key to simulate a vector name not in config.vector_names
    vectors_config["ghost_vector"] = vectors_config.pop("image_vector")
    with pytest.raises(ConfigurationError):
        validate_vector_config(config, vectors_config)

    # Size mismatch: build from a config with different dims, validate against original
    small_config = _base_config(
        vector_names={"text_vector": 512, "image_vector": 768},
        multivector_vectors=["image_vector"],
    )
    mismatched = build_vector_config(small_config)
    with pytest.raises(ConfigurationError, match="size mismatch"):
        validate_vector_config(config, mismatched)


def test_validate_vector_config_passes_on_valid_config() -> None:
    config = _base_config()
    vectors_config = build_vector_config(config)
    # Should not raise
    validate_vector_config(config, vectors_config)
