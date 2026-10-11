"""Qdrant database configuration model."""

from typing import Annotated, Literal, get_args

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    ValidationInfo,
    field_validator,
    model_validator,
)

from .env_settings import EnvSettings


def _lower(value: object) -> object:
    """Lower-case text, so a choice set as "COSINE" reads as "cosine"."""
    return value.lower() if isinstance(value, str) else value


DistanceMetric = Annotated[Literal["cosine", "euclid", "dot"], BeforeValidator(_lower)]
QuantizationType = Annotated[
    Literal["binary", "scalar", "product"], BeforeValidator(_lower)
]
SparseModifier = Annotated[Literal["none", "idf"], BeforeValidator(_lower)]
Priority = Literal["high", "medium", "low"]


class HnswSettings(BaseModel):
    """HNSW index settings for the vectors of one priority."""

    model_config = ConfigDict(extra="forbid")

    ef_construct: PositiveInt = Field(
        description="Neighbours considered while building"
    )
    m: NonNegativeInt = Field(description="Edges per node; 0 disables the graph")


class QuantizationSettings(BaseModel):
    """Quantization settings for the vectors of one priority."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["scalar", "binary"] = Field(description="Quantization kind")
    scalar_type: Literal["int8"] = Field(
        default="int8", description="Scalar quantization element type"
    )
    always_ram: bool = Field(default=False, description="Keep quantized vectors in RAM")


class QdrantConfig(EnvSettings):
    """Configuration for Qdrant vector database connection and optimization."""

    # Connection
    qdrant_url: str = Field(
        default="http://localhost:6333", description="Qdrant server URL"
    )
    qdrant_api_key: str | None = Field(
        default=None, description="Qdrant API key for cloud authentication"
    )
    qdrant_collection_name: str = Field(
        default="anime_database", description="Qdrant collection name"
    )
    qdrant_distance_metric: DistanceMetric = Field(
        default="cosine", description="Distance metric for similarity"
    )

    # Vector Architecture & Dimensions
    vector_names: dict[str, PositiveInt] = Field(
        default={
            "text_vector": 1024,
            "image_vector": 768,
        },
        min_length=1,
        description="Unified semantic architecture (BGE-M3 text: 1024-dim, OpenCLIP images: 768-dim)",
    )
    primary_text_vector_name: str = Field(
        default="text_vector",
        description="Explicit primary text vector name used for text search routing",
    )
    primary_image_vector_name: str = Field(
        default="image_vector",
        description="Explicit primary image vector name used for image search routing",
    )
    multivector_vectors: list[str] = Field(
        default=["image_vector"],
        description="Vector names that use multivector storage (list of vectors per point)",
    )
    sparse_vector_names: list[str] = Field(
        default=["text_sparse_vector"],
        description="Named sparse vectors configured for the collection",
    )
    primary_sparse_vector_name: str = Field(
        default="text_sparse_vector",
        description="Primary sparse vector used for sparse and hybrid text search",
    )
    sparse_vector_modifier: SparseModifier = Field(
        default="idf",
        description="Sparse vector modifier: none or idf",
    )
    sparse_index_on_disk: bool = Field(
        default=False,
        description="Store sparse index on disk instead of RAM",
    )
    rrf_k: int = Field(
        default=2,
        ge=1,
        le=1000,
        description=(
            "RRF rank constant k in score = 1 / (k + rank). "
            "Lower k (e.g. 2) strongly boosts top-ranked results from each branch. "
            "Higher k (e.g. 60) flattens the curve, weighting all ranks more equally. "
            "Qdrant default is 2. Tune against real queries — there is no universal optimum."
        ),
    )
    prefetch_limit_multiplier: int = Field(
        default=10,
        ge=2,
        description=(
            "Multiplier applied to the final search limit to determine the candidate pool "
            "size for each prefetch branch in hybrid/fusion search. Higher values improve "
            "recall at the cost of more server-side fusion work. "
            "E.g. limit=10 with multiplier=10 prefetches 100 candidates per branch."
        ),
    )
    qdrant_search_hnsw_ef: int | None = Field(
        default=None,
        ge=1,
        description=(
            "HNSW ef for the dense text vector search: how many candidates the "
            "graph search keeps. Higher is more accurate and slower. Unset uses "
            "Qdrant's default."
        ),
    )
    qdrant_search_rescore: bool | None = Field(
        default=None,
        description=(
            "Re-rank the dense text vector's quantized candidates with the original "
            "vectors. Qdrant does not rescore scalar-quantized search by default. "
            "Unset uses Qdrant's default."
        ),
    )
    qdrant_search_oversampling: float | None = Field(
        default=None,
        ge=1.0,
        description=(
            "How many times more quantized candidates to fetch for rescoring on the "
            "dense text vector (e.g. 4.0 fetches 4x the branch limit). Only useful "
            "with rescoring. Unset uses Qdrant's default."
        ),
    )
    qdrant_prefer_grpc: bool = Field(
        default=False,
        description=(
            "Talk to Qdrant over gRPC instead of HTTP. Applies to every search: "
            "faster when searches ask for IDs and scores only (with_payload=false), "
            "slower when they return payloads"
        ),
    )
    qdrant_grpc_port: int = Field(
        default=6334, ge=1, le=65535, description="Qdrant gRPC port"
    )
    qdrant_query_batch_max_size: int = Field(
        default=1,
        ge=1,
        le=256,
        description=(
            "Most concurrent searches sent to Qdrant in one query_batch_points "
            "call; 1 sends each search on its own with query_points"
        ),
    )
    qdrant_query_batch_max_wait_ms: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description=(
            "How long a batch of searches that is not full waits for more before "
            "it is sent, in milliseconds; 0 sends whatever has queued at once"
        ),
    )
    qdrant_query_batch_concurrency: int = Field(
        default=4,
        ge=1,
        le=64,
        description="query_batch_points calls that may run at the same time",
    )
    vector_priorities: dict[Priority, list[str]] = Field(
        default={
            "high": [
                "text_vector",
                "image_vector",
            ],
            "medium": [],
            "low": [],
        },
        description=(
            "Vector priority classification for performance optimization; "
            "a vector listed under no priority is treated as medium"
        ),
    )

    # Quantization
    qdrant_enable_quantization: bool = Field(
        default=False, description="Enable quantization for performance"
    )
    qdrant_quantization_type: QuantizationType = Field(
        default="scalar", description="Quantization type: binary, scalar, product"
    )
    qdrant_quantization_always_ram: bool | None = Field(
        default=None, description="Keep quantized vectors in RAM"
    )
    quantization_config: dict[Priority, QuantizationSettings] = Field(
        default={
            "high": QuantizationSettings(type="scalar", always_ram=True),
            "medium": QuantizationSettings(type="scalar", always_ram=False),
            "low": QuantizationSettings(type="binary", always_ram=False),
        },
        description="Quantization configuration per vector priority for memory optimization",
    )

    # HNSW Index
    qdrant_hnsw_ef_construct: int | None = Field(
        default=None, description="HNSW ef_construct parameter"
    )
    qdrant_hnsw_m: int | None = Field(default=None, description="HNSW M parameter")
    qdrant_hnsw_max_indexing_threads: int | None = Field(
        default=None, description="Maximum indexing threads"
    )
    hnsw_config: dict[Priority, HnswSettings] = Field(
        default={
            "high": HnswSettings(ef_construct=256, m=64),
            "medium": HnswSettings(ef_construct=200, m=48),
            "low": HnswSettings(ef_construct=128, m=32),
        },
        description="Anime-optimized HNSW parameters per vector priority for similarity matching",
    )

    # Optimizer
    default_segment_number: int = Field(
        default=4,
        ge=1,
        description=(
            "Number of segments Qdrant maintains per collection. "
            "More segments improve parallel indexing throughput; "
            "fewer reduce merge overhead on small collections."
        ),
    )
    indexing_threshold: int = Field(
        default=20000,
        ge=0,
        description=(
            "Minimum number of vectors in a segment before HNSW indexing starts. "
            "Vectors below this threshold are searched by brute-force. "
            "Lower values index sooner (higher RAM use); higher values defer indexing "
            "until the segment is large enough to benefit."
        ),
    )

    # Memory & Storage
    qdrant_memory_mapping_threshold: int | None = Field(
        default=None, description="Memory mapping threshold in KB"
    )
    memory_mapping_threshold_mb: int = Field(
        default=50,
        description="Memory mapping threshold in MB for large collection optimization",
    )
    qdrant_enable_wal: bool | None = Field(
        default=None, description="Enable Write-Ahead Logging"
    )
    wal_capacity_mb: int = Field(
        default=32,
        ge=1,
        description=(
            "Maximum WAL segment size in MB before Qdrant rolls to a new segment. "
            "Larger values reduce segment rotation frequency at the cost of more "
            "disk space per uncommitted segment."
        ),
    )
    wal_segments_ahead: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of WAL segments to pre-allocate ahead of the current write position. "
            "0 disables pre-allocation (default). Higher values smooth write latency "
            "spikes on slow disks by avoiding on-demand allocation during writes."
        ),
    )

    # Indexing
    qdrant_enable_payload_indexing: bool = Field(
        default=True, description="Enable payload field indexing"
    )
    qdrant_indexed_payload_fields: dict[str, str] = Field(
        default={
            # Core searchable fields
            "id": "keyword",
            "entity_type": "keyword",
            "anime_id": "keyword",
            "anime_ids": "keyword",
            "title": "keyword",
            "type": "keyword",
            "status": "keyword",
            "rating": "keyword",
            "source_material": "keyword",
            "nsfw": "bool",
            # Categorical fields
            "genres": "keyword",
            "tags": "keyword",
            "demographics": "keyword",
            "content_warnings": "keyword",
            # Temporal fields (flattened)
            "year": "integer",
            "season": "keyword",
            "duration": "integer",
            # Platform fields
            "sources": "keyword",
            # Statistics - MAL
            "statistics.mal.score": "float",
            "statistics.mal.scored_by": "integer",
            # Statistics - AniList
            "statistics.anilist.score": "float",
            "statistics.anilist.scored_by": "integer",
            # Statistics - AniDB
            "statistics.anidb.score": "float",
            "statistics.anidb.scored_by": "integer",
            # Statistics - Anime-Planet
            "statistics.animeplanet.score": "float",
            "statistics.animeplanet.scored_by": "integer",
            # Statistics - Kitsu
            "statistics.kitsu.score": "float",
            "statistics.kitsu.scored_by": "integer",
            # Statistics - AnimeSchedule
            "statistics.animeschedule.score": "float",
            "statistics.animeschedule.scored_by": "integer",
            # Aggregate score
            "score.mean": "float",
            "score.weighted": "float",
        },
        description="Payload fields with their types for optimized indexing (excludes operational metadata)",
    )

    @field_validator("multivector_vectors")
    @classmethod
    def validate_multivector_vectors(
        cls, v: list[str], info: ValidationInfo
    ) -> list[str]:
        """Validate multivector_vectors against vector_names."""
        vector_names = (info.data or {}).get("vector_names", {})
        unknown = [name for name in v if name not in vector_names]
        if unknown:
            raise ValueError(  # noqa: TRY003
                f"Unknown multivector vectors: {unknown}. "
                f"Valid vectors: {list(vector_names.keys())}"
            )
        return v

    @model_validator(mode="after")
    def validate_primary_vector_names(self) -> QdrantConfig:
        """Validate explicit primary vector names against vector_names."""
        if self.primary_text_vector_name not in self.vector_names:
            raise ValueError(  # noqa: TRY003
                "primary_text_vector_name must be a key in vector_names"
            )
        if self.primary_image_vector_name not in self.vector_names:
            raise ValueError(  # noqa: TRY003
                "primary_image_vector_name must be a key in vector_names"
            )
        if not self.sparse_vector_names:
            raise ValueError(  # noqa: TRY003
                "sparse_vector_names must contain at least one entry"
            )
        if self.primary_sparse_vector_name not in self.sparse_vector_names:
            raise ValueError(  # noqa: TRY003
                "primary_sparse_vector_name must be a key in sparse_vector_names"
            )
        return self

    @model_validator(mode="after")
    def validate_priorities(self) -> QdrantConfig:
        """Check priority vectors exist and every priority has index settings."""
        unknown = [
            name
            for names in self.vector_priorities.values()
            for name in names
            if name not in self.vector_names
        ]
        if unknown:
            raise ValueError(  # noqa: TRY003
                f"Unknown vectors in vector_priorities: {unknown}. "
                f"Valid vectors: {list(self.vector_names)}"
            )
        for field_name in ("hnsw_config", "quantization_config"):
            missing = [
                priority
                for priority in get_args(Priority)
                if priority not in getattr(self, field_name)
            ]
            if missing:
                raise ValueError(  # noqa: TRY003
                    f"{field_name} has no settings for priorities: {missing}"
                )
        return self
