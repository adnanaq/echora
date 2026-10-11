"""Pure config → Qdrant model translation for collection schema building.

All functions are stateless and side-effect free — they take a QdrantConfig
and return Qdrant client model objects. No I/O, no async.
"""

import logging
from typing import Any

from common.config import QdrantConfig
from common.config.qdrant_config import Priority
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

logger = logging.getLogger(__name__)

_DISTANCE_MAPPING = {
    "cosine": Distance.COSINE,
    "euclid": Distance.EUCLID,
    "dot": Distance.DOT,
}


def get_vector_priority(config: QdrantConfig, vector_name: str) -> Priority:
    """Resolve priority bucket for a vector name.

    Args:
        config: Qdrant runtime settings.
        vector_name: Vector name configured for the collection.

    Returns:
        The priority listing the vector; "medium" when none does.
    """
    for priority, vectors in config.vector_priorities.items():
        if vector_name in vectors:
            return priority
    return "medium"


def get_hnsw_config(config: QdrantConfig, priority: Priority) -> HnswConfigDiff:
    """Build HNSW configuration for a priority class.

    Args:
        config: Qdrant runtime settings.
        priority: Priority bucket key.

    Returns:
        Configured HNSW diff model.
    """
    settings = config.hnsw_config[priority]
    return HnswConfigDiff(ef_construct=settings.ef_construct, m=settings.m)


def get_per_vector_quantization_config(
    config: QdrantConfig, priority: Priority
) -> QuantizationConfig:
    """Build per-vector quantization config for a priority class.

    Args:
        config: Qdrant runtime settings.
        priority: Priority bucket key.

    Returns:
        The scalar or binary quantization the priority is configured with.
    """
    settings = config.quantization_config[priority]
    if settings.type == "scalar":
        return ScalarQuantization(
            scalar=ScalarQuantizationConfig(
                type=ScalarType.INT8, always_ram=settings.always_ram
            )
        )
    return BinaryQuantization(
        binary=BinaryQuantizationConfig(always_ram=settings.always_ram)
    )


def build_vector_config(config: QdrantConfig) -> dict[str, VectorParams]:
    """Create named-vector configuration for collection creation.

    Args:
        config: Qdrant runtime settings.

    Returns:
        Mapping of vector name to :class:`VectorParams`.
    """
    distance = _DISTANCE_MAPPING[config.qdrant_distance_metric]
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
        from qdrant_client.models import CompressionRatio, ProductQuantizationConfig

        return ProductQuantization(
            product=ProductQuantizationConfig(compression=CompressionRatio.X16)
        )
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
            default_segment_number=config.default_segment_number,
            indexing_threshold=config.indexing_threshold,
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
        return WalConfigDiff(
            wal_capacity_mb=config.wal_capacity_mb,
            wal_segments_ahead=config.wal_segments_ahead,
        )
    except Exception:
        logger.exception("Failed to create WAL config")
        return None
