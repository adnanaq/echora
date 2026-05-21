"""Pure config → Qdrant model translation for collection schema building.

All functions are stateless and side-effect free — they take a QdrantConfig
and return Qdrant client model objects. No I/O, no async.
"""

import logging
from typing import Any

from common.config import QdrantConfig
from qdrant_client.models import (
    BinaryQuantization,
    BinaryQuantizationConfig,
    Distance,
    HnswConfigDiff,
    Modifier,
    MultiVectorComparator,
    MultiVectorConfig,
    OptimizersConfigDiff,
    ProductQuantization,
    QuantizationConfig,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
    WalConfigDiff,
)

from qdrant_db.errors import ConfigurationError

logger = logging.getLogger(__name__)

_DISTANCE_MAPPING = {
    "cosine": Distance.COSINE,
    "euclid": Distance.EUCLID,
    "dot": Distance.DOT,
}


def get_vector_priority(config: QdrantConfig, vector_name: str) -> str:
    """Resolve priority bucket for a vector name.

    Args:
        config: Qdrant runtime settings.
        vector_name: Vector name configured for the collection.

    Returns:
        Priority label used by HNSW/quantization lookup.
    """
    for priority, vectors in config.vector_priorities.items():
        if vector_name in vectors:
            return str(priority)
    return "medium"


def get_hnsw_config(config: QdrantConfig, priority: str) -> HnswConfigDiff:
    """Build HNSW configuration for a priority class.

    Args:
        config: Qdrant runtime settings.
        priority: Priority bucket key.

    Returns:
        Configured HNSW diff model.
    """
    hnsw_cfg = config.hnsw_config.get(priority, {})
    return HnswConfigDiff(
        ef_construct=hnsw_cfg.get("ef_construct", 200),
        m=hnsw_cfg.get("m", 48),
    )


def get_per_vector_quantization_config(
    config: QdrantConfig, priority: str
) -> QuantizationConfig | None:
    """Build per-vector quantization config for a priority class.

    Args:
        config: Qdrant runtime settings.
        priority: Priority bucket key.

    Returns:
        Quantization config model when configured; otherwise ``None``.
    """
    quant_cfg = config.quantization_config.get(priority, {})
    if quant_cfg.get("type") == "scalar":
        return ScalarQuantization(
            scalar=ScalarQuantizationConfig(
                type=ScalarType.INT8,
                always_ram=bool(quant_cfg.get("always_ram", False)),
            )
        )
    if quant_cfg.get("type") == "binary":
        return BinaryQuantization(
            binary=BinaryQuantizationConfig(
                always_ram=bool(quant_cfg.get("always_ram", False))
            )
        )
    return None


def build_vector_config(config: QdrantConfig) -> dict[str, VectorParams]:
    """Create named-vector configuration for collection creation.

    Args:
        config: Qdrant runtime settings.

    Returns:
        Mapping of vector name to :class:`VectorParams`.
    """
    distance = _DISTANCE_MAPPING.get(config.qdrant_distance_metric, Distance.COSINE)
    multivector_names = set(config.multivector_vectors)

    vector_params: dict[str, VectorParams] = {}
    for vector_name, dimension in config.vector_names.items():
        priority = get_vector_priority(config, vector_name)
        is_multivector = vector_name in multivector_names
        params_kwargs: dict[str, Any] = {
            "size": dimension,
            "distance": distance,
            # MaxSim is asymmetric — HNSW assumes symmetric distances and cannot
            # pre-compute valid neighbors for multivector. Disable it (m=0).
            "hnsw_config": HnswConfigDiff(m=0)
            if is_multivector
            else get_hnsw_config(config, priority),
            "quantization_config": get_per_vector_quantization_config(config, priority),
        }
        if is_multivector:
            params_kwargs["multivector_config"] = MultiVectorConfig(
                comparator=MultiVectorComparator.MAX_SIM
            )
        vector_params[vector_name] = VectorParams(**params_kwargs)

    return vector_params


def build_sparse_vector_config(
    config: QdrantConfig,
) -> dict[str, SparseVectorParams]:
    """Create sparse vector configuration for collection creation.

    Sparse vectors are always required alongside dense vectors. The config
    validator enforces at least one entry in ``sparse_vector_names``.

    Args:
        config: Qdrant runtime settings.

    Returns:
        Mapping of sparse vector name to :class:`SparseVectorParams`.
    """
    modifier = Modifier.IDF if config.sparse_vector_modifier == "idf" else None
    sparse_vectors_config: dict[str, SparseVectorParams] = {}
    for vector_name in config.sparse_vector_names:
        params_kwargs: dict[str, Any] = {
            "index": SparseIndexParams(on_disk=config.sparse_index_on_disk),
        }
        if modifier is not None:
            params_kwargs["modifier"] = modifier
        sparse_vectors_config[vector_name] = SparseVectorParams(**params_kwargs)

    return sparse_vectors_config


def build_quantization_config(
    config: QdrantConfig,
) -> BinaryQuantization | ScalarQuantization | ProductQuantization | None:
    """Create global collection quantization config from settings.

    Args:
        config: Qdrant runtime settings.

    Returns:
        One quantization config model or ``None`` when disabled/invalid.
    """
    if not config.qdrant_enable_quantization:
        return None

    quantization_type = config.qdrant_quantization_type
    always_ram = config.qdrant_quantization_always_ram

    try:
        if quantization_type == "binary":
            return BinaryQuantization(
                binary=BinaryQuantizationConfig(always_ram=always_ram)
            )
        if quantization_type == "scalar":
            return ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    always_ram=always_ram,
                )
            )
        if quantization_type == "product":
            from qdrant_client.models import CompressionRatio, ProductQuantizationConfig

            return ProductQuantization(
                product=ProductQuantizationConfig(compression=CompressionRatio.X16)
            )
        else:
            return None  # pragma: no cover
    except Exception:
        logger.exception("Failed to create quantization config")
        return None


def build_optimizers_config(config: QdrantConfig) -> OptimizersConfigDiff | None:
    """Create optimizer tuning parameters for collection creation.

    Args:
        config: Qdrant runtime settings.

    Returns:
        Optimizer config when construction succeeds, otherwise ``None``.
    """
    try:
        return OptimizersConfigDiff(
            default_segment_number=4,
            indexing_threshold=20000,
            memmap_threshold=config.memory_mapping_threshold_mb * 1024,
        )
    except Exception:
        logger.exception("Failed to create optimizers config")
        return None


def build_wal_config(config: QdrantConfig) -> WalConfigDiff | None:
    """Create WAL config from settings.

    Args:
        config: Qdrant runtime settings.

    Returns:
        WAL config when enabled and valid; otherwise ``None``.
    """
    if not config.qdrant_enable_wal:
        return None
    try:
        return WalConfigDiff(wal_capacity_mb=32, wal_segments_ahead=0)
    except Exception:
        logger.exception("Failed to create WAL config")
        return None


def validate_vector_config(
    config: QdrantConfig, vectors_config: dict[str, VectorParams]
) -> None:
    """Validate generated vector configuration before collection creation.

    Args:
        config: Qdrant runtime settings.
        vectors_config: Generated mapping of vector names to params.

    Raises:
        ConfigurationError: If config is empty, has mismatched count, or
            contains unexpected dimensions.
    """
    if not vectors_config:
        raise ConfigurationError("Vector configuration is empty")

    if len(vectors_config) != len(config.vector_names):
        raise ConfigurationError(
            "Vector count mismatch between generated config and settings"
        )

    for vector_name, vector_params in vectors_config.items():
        expected_dim = config.vector_names.get(vector_name)
        if expected_dim is None:
            raise ConfigurationError(
                f"Vector {vector_name} is not present in configured vector_names"
            )
        if vector_params.size != expected_dim:
            raise ConfigurationError(
                f"Vector {vector_name} size mismatch: expected {expected_dim}, got {vector_params.size}"
            )
