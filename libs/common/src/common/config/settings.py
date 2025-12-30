"""Vector Service Configuration Settings."""

import os
from enum import Enum
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Application environment types."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


def get_environment() -> Environment:
    """Detect environment from APP_ENV variable.

    Returns:
        Environment: Detected environment based on APP_ENV.

    Raises:
        ValueError: If APP_ENV is not set or contains an invalid value.
    """
    env_str = os.getenv("APP_ENV")
    if not env_str:
        raise ValueError(
            "APP_ENV environment variable must be set to one of: "
            "development, staging, production"
        )

    env_str = env_str.lower()

    match env_str:
        case "production":
            return Environment.PRODUCTION
        case "staging":
            return Environment.STAGING
        case "development":
            return Environment.DEVELOPMENT
        case _:
            raise ValueError(
                f"Invalid APP_ENV value '{env_str}'. "
                "Must be one of: development, staging, production"
            )


class Settings(BaseSettings):
    """Vector service settings with validation and type safety."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # ============================================================================
    # ENVIRONMENT & APPLICATION
    # ============================================================================

    environment: Environment = Field(
        default_factory=get_environment,
        description="Application environment (development/staging/production)",
    )
    debug: bool = Field(default=True, description="Enable debug mode")

    # ============================================================================
    # SERVICE CONFIGURATION
    # ============================================================================

    vector_service_host: str = Field(
        default="0.0.0.0", description="Vector service host address"
    )
    vector_service_port: int = Field(
        default=8002, ge=1, le=65535, description="Vector service port"
    )

    # ============================================================================
    # QDRANT DATABASE CONNECTION
    # ============================================================================

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

    # ============================================================================
    # VECTOR ARCHITECTURE & DIMENSIONS
    # ============================================================================

    text_vector_size: int = Field(
        default=1024,
        description="Vector embedding dimensions for text vectors (BGE-M3)",
    )
    image_vector_size: int = Field(
        default=768,
        description="Image embedding dimensions (OpenCLIP ViT-L/14: 768, ViT-B/32: 512)",
    )

    # 11-Vector Semantic Architecture Configuration
    vector_names: dict[str, int] = Field(
        default={
            "title_vector": 1024,
            "character_vector": 1024,
            "genre_vector": 1024,
            "staff_vector": 1024,
            "temporal_vector": 1024,
            "streaming_vector": 1024,
            "related_vector": 1024,
            "franchise_vector": 1024,
            "episode_vector": 1024,
            "image_vector": 768,
            "character_image_vector": 768,
        },
        description="11-vector semantic architecture with named vectors and dimensions (BGE-M3: 1024-dim, OpenCLIP ViT-L/14: 768-dim)",
    )

    # Vector Priority Classification for Optimization
    vector_priorities: dict[str, list[str]] = Field(
        default={
            "high": [
                "title_vector",
                "character_vector",
                "genre_vector",
                "image_vector",
                "character_image_vector",
            ],
            "medium": [
                "staff_vector",
                "temporal_vector",
                "streaming_vector",
            ],
            "low": [
                "related_vector",
                "franchise_vector",
                "episode_vector",
            ],
        },
        description="Vector priority classification for performance optimization",
    )

    # ============================================================================
    # TEXT EMBEDDING MODELS
    # ============================================================================

    text_embedding_provider: str = Field(
        default="huggingface",
        description="Text embedding provider: fastembed, huggingface, sentence-transformers",
    )
    text_embedding_model: str = Field(
        default="BAAI/bge-m3", description="Modern text embedding model name"
    )

    # BGE Model-Specific Configuration
    bge_model_version: str = Field(
        default="m3", description="BGE model version: v1.5, m3, reranker"
    )
    bge_model_size: str = Field(
        default="base", description="BGE model size: small, base, large"
    )
    bge_max_length: int = Field(
        default=8192, description="BGE maximum input sequence length"
    )

    # ============================================================================
    # IMAGE EMBEDDING MODELS
    # ============================================================================

    image_embedding_provider: str = Field(
        default="openclip", description="Image embedding provider: openclip"
    )
    image_embedding_model: str = Field(
        default="ViT-L-14/laion2b_s32b_b82k",
        description="OpenCLIP ViT-L/14 model for high-quality image embeddings (768 dims)",
    )

    # OpenCLIP Model-Specific Configuration
    openclip_input_resolution: int = Field(
        default=224, description="OpenCLIP input image resolution"
    )
    openclip_text_max_length: int = Field(
        default=77, description="OpenCLIP maximum text sequence length"
    )

    # Image Processing Configuration
    image_batch_size: int = Field(
        default=16,
        ge=1,
        le=128,
        description="Batch size for image embedding processing (adjust based on GPU VRAM: 4-8 for 8GB, 16+ for 16GB+)",
    )

    # ============================================================================
    # MODEL MANAGEMENT
    # ============================================================================

    model_cache_dir: str | None = Field(
        default=None, description="Custom cache directory for embedding models"
    )
    model_warm_up: bool = Field(
        default=False, description="Pre-load and warm up models during initialization"
    )

    # ============================================================================
    # PERFORMANCE OPTIMIZATION - QUANTIZATION
    # ============================================================================

    qdrant_enable_quantization: bool = Field(
        default=False, description="Enable quantization for performance"
    )
    qdrant_quantization_type: str = Field(
        default="scalar", description="Quantization type: binary, scalar, product"
    )
    qdrant_quantization_always_ram: bool | None = Field(
        default=None, description="Keep quantized vectors in RAM"
    )

    # Advanced Quantization Configuration per Vector Priority
    quantization_config: dict[str, dict[str, object]] = Field(
        default={
            "high": {"type": "scalar", "scalar_type": "int8", "always_ram": True},
            "medium": {"type": "scalar", "scalar_type": "int8", "always_ram": False},
            "low": {"type": "binary", "always_ram": False},
        },
        description="Quantization configuration per vector priority for memory optimization",
    )

    # ============================================================================
    # PERFORMANCE OPTIMIZATION - HNSW INDEX
    # ============================================================================

    qdrant_hnsw_ef_construct: int | None = Field(
        default=None, description="HNSW ef_construct parameter"
    )
    qdrant_hnsw_m: int | None = Field(default=None, description="HNSW M parameter")
    qdrant_hnsw_max_indexing_threads: int | None = Field(
        default=None, description="Maximum indexing threads"
    )

    # Anime-Optimized HNSW Parameters per Vector Priority
    hnsw_config: dict[str, dict[str, int]] = Field(
        default={
            "high": {"ef_construct": 256, "m": 64, "ef": 128},
            "medium": {"ef_construct": 200, "m": 48, "ef": 64},
            "low": {"ef_construct": 128, "m": 32, "ef": 32},
        },
        description="Anime-optimized HNSW parameters per vector priority for similarity matching",
    )

    # ============================================================================
    # PERFORMANCE OPTIMIZATION - MEMORY & STORAGE
    # ============================================================================

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

    # ============================================================================
    # INDEXING CONFIGURATION
    # ============================================================================

    qdrant_enable_payload_indexing: bool = Field(
        default=True, description="Enable payload field indexing"
    )
    qdrant_indexed_payload_fields: dict[str, str] = Field(
        default={
            # Core searchable fields
            "id": "keyword",
            "title": "keyword",  # Exact title matching
            "title_text": "text",  # Full-text title search
            "type": "keyword",
            "status": "keyword",
            "episodes": "integer",
            "rating": "keyword",
            "source_material": "keyword",
            "nsfw": "bool",
            # Categorical fields
            "genres": "keyword",
            "tags": "keyword",
            "demographics": "text",  # Descriptive text content
            "content_warnings": "text",  # Descriptive text content
            # Character physical attributes (AnimePlanet)
            "characters.hair_color": "keyword",
            "characters.eye_color": "keyword",
            "characters.character_traits": "keyword",
            # Temporal fields (flattened)
            "year": "integer",
            "season": "keyword",
            "duration": "integer",  # Episode duration in seconds
            # Platform fields
            "sources": "keyword",
            # Statistics for numerical filtering - per-platform nested fields
            # MAL (MyAnimeList) statistics
            "statistics.mal.score": "float",
            "statistics.mal.scored_by": "integer",
            "statistics.mal.members": "integer",
            "statistics.mal.favorites": "integer",
            "statistics.mal.rank": "integer",
            "statistics.mal.popularity_rank": "integer",
            # AniList statistics
            "statistics.anilist.score": "float",
            "statistics.anilist.favorites": "integer",
            "statistics.anilist.popularity_rank": "integer",
            # AniDB statistics
            "statistics.anidb.score": "float",
            "statistics.anidb.scored_by": "integer",
            # Anime-Planet statistics
            "statistics.animeplanet.score": "float",
            "statistics.animeplanet.scored_by": "integer",
            "statistics.animeplanet.rank": "integer",
            # Kitsu statistics
            "statistics.kitsu.score": "float",
            "statistics.kitsu.members": "integer",
            "statistics.kitsu.favorites": "integer",
            "statistics.kitsu.rank": "integer",
            "statistics.kitsu.popularity_rank": "integer",
            # AnimeSchedule statistics
            "statistics.animeschedule.score": "float",
            "statistics.animeschedule.scored_by": "integer",
            "statistics.animeschedule.members": "integer",
            "statistics.animeschedule.rank": "integer",
            # Aggregate score field
            "score.arithmetic_mean": "float",
            # Note: enrichment_metadata intentionally excluded (non-indexed operational data)
        },
        description="Payload fields with their types for optimized indexing (excludes operational metadata)",
    )

    # ============================================================================
    # API CONFIGURATION
    # ============================================================================

    api_title: str = Field(default="Anime Vector Service", description="API title")
    api_version: str = Field(default="1.0.0", description="API version")
    api_description: str = Field(
        default="Microservice for anime vector database operations",
        description="API description",
    )

    # ============================================================================
    # REQUEST PROCESSING & LIMITS
    # ============================================================================

    # Batch Processing
    default_batch_size: int = Field(
        default=100, ge=1, le=1000, description="Default batch size for operations"
    )
    max_batch_size: int = Field(
        default=500, ge=1, le=2000, description="Maximum allowed batch size"
    )

    # Search & Request Limits
    max_search_limit: int = Field(
        default=100, ge=1, le=1000, description="Maximum search results limit"
    )
    request_timeout: int = Field(
        default=30, ge=1, le=300, description="Request timeout in seconds"
    )

    # ============================================================================
    # LOGGING CONFIGURATION
    # ============================================================================

    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string",
    )

    # ============================================================================
    # CORS CONFIGURATION
    # ============================================================================

    allowed_origins: list[str] = Field(
        default=["*"], description="Allowed CORS origins"
    )
    allowed_methods: list[str] = Field(
        default=["*"], description="Allowed HTTP methods"
    )
    allowed_headers: list[str] = Field(
        default=["*"], description="Allowed HTTP headers"
    )

    # ============================================================================
    # LIFECYCLE & VALIDATION
    # ============================================================================

    def model_post_init(self, __context) -> None:
        """Apply environment-specific overrides after initialization."""
        self.apply_environment_settings()

    def apply_environment_settings(self) -> None:
        """Apply environment-specific settings with smart defaults.

        DEVELOPMENT:
            - Sets debug=True, log_level=DEBUG as defaults
            - Respects user-provided values

        STAGING:
            - Sets debug=True, log_level=INFO, wal=True as defaults
            - Respects user-provided values

        PRODUCTION (ENFORCED):
            - ALWAYS enforces debug=False, log_level=WARNING
            - ALWAYS enforces wal=True, model_warm_up=True
            - Security: Cannot be bypassed by user configuration
        """
        if self.environment == Environment.DEVELOPMENT:
            # Apply defaults only if user didn't explicitly set values
            if os.getenv("DEBUG") is None:
                self.debug = True
            if os.getenv("LOG_LEVEL") is None:
                self.log_level = "DEBUG"

        elif self.environment == Environment.STAGING:
            # Apply defaults only if user didn't explicitly set values
            if os.getenv("DEBUG") is None:
                self.debug = True
            if os.getenv("LOG_LEVEL") is None:
                self.log_level = "INFO"
            if os.getenv("QDRANT_ENABLE_WAL") is None:
                self.qdrant_enable_wal = True

        elif self.environment == Environment.PRODUCTION:
            # ENFORCED - cannot be bypassed (security feature)
            self.debug = False
            self.log_level = "WARNING"
            self.qdrant_enable_wal = True
            self.model_warm_up = True

    @field_validator("qdrant_distance_metric")
    @classmethod
    def validate_distance_metric(cls, v: str) -> str:
        """Validate distance metric."""
        valid_metrics = ["cosine", "euclid", "dot"]
        if v.lower() not in valid_metrics:
            raise ValueError(f"Distance metric must be one of: {valid_metrics}")
        return v.lower()

    @field_validator("text_embedding_provider")
    @classmethod
    def validate_text_provider(cls, v: str) -> str:
        """Validate text embedding provider."""
        valid_providers = ["fastembed", "huggingface", "sentence-transformers"]
        if v.lower() not in valid_providers:
            raise ValueError(
                f"Text embedding provider must be one of: {valid_providers}"
            )
        return v.lower()

    @field_validator("image_embedding_provider")
    @classmethod
    def validate_image_provider(cls, v: str) -> str:
        """Validate image embedding provider."""
        valid_providers = ["openclip"]
        if v.lower() not in valid_providers:
            raise ValueError(
                f"Image embedding provider must be one of: {valid_providers}"
            )
        return v.lower()

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
