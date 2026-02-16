"""Qdrant database configuration model."""

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator


class QdrantConfig(BaseModel):
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
    qdrant_distance_metric: str = Field(
        default="cosine", description="Distance metric for similarity"
    )

    # Vector Architecture & Dimensions
    vector_names: dict[str, int] = Field(
        default={
            "text_vector": 1024,
            "image_vector": 768,
        },
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
    sparse_vector_modifier: str = Field(
        default="none",
        description="Sparse vector modifier: none or idf",
    )
    sparse_index_on_disk: bool = Field(
        default=False,
        description="Store sparse index on disk instead of RAM",
    )
    vector_priorities: dict[str, list[str]] = Field(
        default={
            "high": [
                "text_vector",
                "image_vector",
            ],
            "medium": [],
            "low": [],
        },
        description="Vector priority classification for performance optimization",
    )

    # Quantization
    qdrant_enable_quantization: bool = Field(
        default=False, description="Enable quantization for performance"
    )
    qdrant_quantization_type: str = Field(
        default="scalar", description="Quantization type: binary, scalar, product"
    )
    qdrant_quantization_always_ram: bool | None = Field(
        default=None, description="Keep quantized vectors in RAM"
    )
    quantization_config: dict[str, dict[str, object]] = Field(
        default={
            "high": {"type": "scalar", "scalar_type": "int8", "always_ram": True},
            "medium": {"type": "scalar", "scalar_type": "int8", "always_ram": False},
            "low": {"type": "binary", "always_ram": False},
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
    hnsw_config: dict[str, dict[str, int]] = Field(
        default={
            "high": {"ef_construct": 256, "m": 64, "ef": 128},
            "medium": {"ef_construct": 200, "m": 48, "ef": 64},
            "low": {"ef_construct": 128, "m": 32, "ef": 32},
        },
        description="Anime-optimized HNSW parameters per vector priority for similarity matching",
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

    # Indexing
    qdrant_enable_payload_indexing: bool = Field(
        default=True, description="Enable payload field indexing"
    )
    qdrant_indexed_payload_fields: dict[str, str] = Field(
        default={
            # Core searchable fields
            "id": "keyword",
            "anime_id": "keyword",
            "anime_ids": "keyword",
            "title": "keyword",
            "title_text": "text",
            "type": "keyword",
            "status": "keyword",
            "episodes": "integer",
            "rating": "keyword",
            "source_material": "keyword",
            "nsfw": "bool",
            # Categorical fields
            "genres": "keyword",
            "tags": "keyword",
            "demographics": "text",
            "content_warnings": "text",
            # Character physical attributes (AnimePlanet)
            "characters.hair_color": "keyword",
            "characters.eye_color": "keyword",
            "characters.character_traits": "keyword",
            # Temporal fields (flattened)
            "year": "integer",
            "season": "keyword",
            "duration": "integer",
            # Platform fields
            "sources": "keyword",
            # Statistics - MAL
            "statistics.mal.score": "float",
            "statistics.mal.scored_by": "integer",
            "statistics.mal.members": "integer",
            "statistics.mal.favorites": "integer",
            "statistics.mal.rank": "integer",
            "statistics.mal.popularity_rank": "integer",
            # Statistics - AniList
            "statistics.anilist.score": "float",
            "statistics.anilist.favorites": "integer",
            "statistics.anilist.popularity_rank": "integer",
            # Statistics - AniDB
            "statistics.anidb.score": "float",
            "statistics.anidb.scored_by": "integer",
            # Statistics - Anime-Planet
            "statistics.animeplanet.score": "float",
            "statistics.animeplanet.scored_by": "integer",
            "statistics.animeplanet.rank": "integer",
            # Statistics - Kitsu
            "statistics.kitsu.score": "float",
            "statistics.kitsu.members": "integer",
            "statistics.kitsu.favorites": "integer",
            "statistics.kitsu.rank": "integer",
            "statistics.kitsu.popularity_rank": "integer",
            # Statistics - AnimeSchedule
            "statistics.animeschedule.score": "float",
            "statistics.animeschedule.scored_by": "integer",
            "statistics.animeschedule.members": "integer",
            "statistics.animeschedule.rank": "integer",
            # Aggregate score
            "score.arithmetic_mean": "float",
        },
        description="Payload fields with their types for optimized indexing (excludes operational metadata)",
    )

    @field_validator("qdrant_distance_metric")
    @classmethod
    def validate_distance_metric(cls, v: str) -> str:
        """Validate distance metric."""
        valid_metrics = ["cosine", "euclid", "dot"]
        if v.lower() not in valid_metrics:
            raise ValueError(f"Distance metric must be one of: {valid_metrics}")
        return v.lower()

    @field_validator("qdrant_quantization_type")
    @classmethod
    def validate_quantization_type(cls, v: str) -> str:
        """Validate quantization type."""
        valid_types = ["binary", "scalar", "product"]
        if v.lower() not in valid_types:
            raise ValueError(f"Quantization type must be one of: {valid_types}")
        return v.lower()

    @field_validator("sparse_vector_modifier")
    @classmethod
    def validate_sparse_vector_modifier(cls, v: str) -> str:
        """Validate sparse vector modifier."""
        valid_modifiers = ["none", "idf"]
        if v.lower() not in valid_modifiers:
            raise ValueError(
                f"Sparse vector modifier must be one of: {valid_modifiers}"
            )
        return v.lower()

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
    def validate_primary_vector_names(self) -> "QdrantConfig":
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
