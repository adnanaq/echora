"""Vector Service Configuration Settings."""

from functools import lru_cache
from typing import Dict, List, Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Vector service settings with validation and type safety."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # Vector Service Configuration
    vector_service_host: str = Field(
        default="0.0.0.0", description="Vector service host address"
    )
    vector_service_port: int = Field(
        default=8002, ge=1, le=65535, description="Vector service port"
    )
    debug: bool = Field(default=True, description="Enable debug mode")

    # Qdrant Vector Database Configuration
    qdrant_url: str = Field(
        default="http://localhost:6333", description="Qdrant server URL"
    )
    qdrant_api_key: Optional[str] = Field(
        default=None, description="Qdrant API key for cloud authentication"
    )
    qdrant_collection_name: str = Field(
        default="anime_database", description="Qdrant collection name"
    )
    qdrant_vector_size: int = Field(
        default=1024,
        description="Vector embedding dimensions for text vectors (BGE-M3)",
    )
    qdrant_distance_metric: str = Field(
        default="cosine", description="Distance metric for similarity"
    )

    # Multi-Vector Configuration
    image_vector_size: int = Field(
        default=768,
        description="Image embedding dimensions (OpenCLIP ViT-L/14: 768, ViT-B/32: 512)",
    )

    # 11-Vector Semantic Architecture Configuration
    vector_names: Dict[str, int] = Field(
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
    vector_priorities: Dict[str, List[str]] = Field(
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

    # Modern Embedding Configuration
    text_embedding_provider: str = Field(
        default="huggingface",
        description="Text embedding provider: fastembed, huggingface, sentence-transformers",
    )
    text_embedding_model: str = Field(
        default="BAAI/bge-m3", description="Modern text embedding model name"
    )

    image_embedding_provider: str = Field(
        default="openclip", description="Image embedding provider: openclip"
    )
    image_embedding_model: str = Field(
        default="ViT-L-14/laion2b_s32b_b82k",
        description="OpenCLIP ViT-L/14 model for high-quality image embeddings (768 dims)",
    )

    # Model-Specific Configuration
    bge_model_version: str = Field(
        default="m3", description="BGE model version: v1.5, m3, reranker"
    )
    bge_model_size: str = Field(
        default="base", description="BGE model size: small, base, large"
    )
    bge_max_length: int = Field(
        default=8192, description="BGE maximum input sequence length"
    )

    openclip_input_resolution: int = Field(
        default=224, description="OpenCLIP input image resolution"
    )
    openclip_text_max_length: int = Field(
        default=77, description="OpenCLIP maximum text sequence length"
    )

    model_cache_dir: Optional[str] = Field(
        default=None, description="Custom cache directory for embedding models"
    )
    model_warm_up: bool = Field(
        default=False, description="Pre-load and warm up models during initialization"
    )

    # LoRA Fine-tuning Configuration
    lora_enabled: bool = Field(
        default=False, description="Enable LoRA (Low-Rank Adaptation) fine-tuning"
    )
    lora_rank: int = Field(
        default=16,
        ge=1,
        le=256,
        description="LoRA rank (r) parameter - higher values = more parameters",
    )
    lora_alpha: int = Field(
        default=32,
        ge=1,
        le=512,
        description="LoRA alpha parameter - scaling factor (typically 2*rank)",
    )
    lora_dropout: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="LoRA dropout probability for regularization",
    )
    lora_target_modules: List[str] = Field(
        default=[
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",  # Attention layers
            "fc1",
            "fc2",  # MLP layers
            "to_q",
            "to_k",
            "to_v",
            "to_out",  # Alternative naming
        ],
        description="Target modules for LoRA adaptation in vision transformers",
    )
    lora_bias: Literal["none", "all", "lora_only"] = Field(
        default="none", description="LoRA bias handling: none, all, lora_only"
    )
    lora_task_type: str = Field(
        default="FEATURE_EXTRACTION", description="LoRA task type for vision models"
    )

    # Image Processing Configuration
    image_batch_size: int = Field(
        default=16,
        ge=1,
        le=128,
        description="Batch size for image embedding processing (adjust based on GPU VRAM: 4-8 for 8GB, 16+ for 16GB+)",
    )

    # Qdrant Performance Optimization
    qdrant_enable_quantization: bool = Field(
        default=False, description="Enable quantization for performance"
    )
    qdrant_quantization_type: str = Field(
        default="scalar", description="Quantization type: binary, scalar, product"
    )
    qdrant_quantization_always_ram: Optional[bool] = Field(
        default=None, description="Keep quantized vectors in RAM"
    )

    # Advanced Quantization Configuration per Vector Priority
    quantization_config: Dict[str, Dict[str, object]] = Field(
        default={
            "high": {"type": "scalar", "scalar_type": "int8", "always_ram": True},
            "medium": {"type": "scalar", "scalar_type": "int8", "always_ram": False},
            "low": {"type": "binary", "always_ram": False},
        },
        description="Quantization configuration per vector priority for memory optimization",
    )

    # HNSW Configuration
    qdrant_hnsw_ef_construct: Optional[int] = Field(
        default=None, description="HNSW ef_construct parameter"
    )
    qdrant_hnsw_m: Optional[int] = Field(default=None, description="HNSW M parameter")
    qdrant_hnsw_max_indexing_threads: Optional[int] = Field(
        default=None, description="Maximum indexing threads"
    )

    # Anime-Optimized HNSW Parameters per Vector Priority
    hnsw_config: Dict[str, Dict[str, int]] = Field(
        default={
            "high": {"ef_construct": 256, "m": 64, "ef": 128},
            "medium": {"ef_construct": 200, "m": 48, "ef": 64},
            "low": {"ef_construct": 128, "m": 32, "ef": 32},
        },
        description="Anime-optimized HNSW parameters per vector priority for similarity matching",
    )

    # Memory and Storage Configuration
    qdrant_memory_mapping_threshold: Optional[int] = Field(
        default=None, description="Memory mapping threshold in KB"
    )

    # Advanced Memory Management for Million-Query Optimization
    memory_mapping_threshold_mb: int = Field(
        default=50,
        description="Memory mapping threshold in MB for large collection optimization",
    )
    qdrant_enable_wal: Optional[bool] = Field(
        default=None, description="Enable Write-Ahead Logging"
    )

    # Payload Indexing
    qdrant_enable_payload_indexing: bool = Field(
        default=True, description="Enable payload field indexing"
    )
    qdrant_indexed_payload_fields: Dict[str, str] = Field(
        default={
            # Core searchable fields
            "id": "keyword",
            "title": "keyword",           # Exact title matching
            "title_text": "text",        # Full-text title search
            "type": "keyword",
            "status": "keyword",
            "episodes": "integer",
            "rating": "keyword",
            "source_material": "keyword",
            "nsfw": "bool",
            # Categorical fields
            "genres": "keyword",
            "tags": "keyword",
            "demographics": "text",       # Descriptive text content
            "content_warnings": "text",   # Descriptive text content
            # Character physical attributes (AnimePlanet)
            "characters.hair_color": "keyword",
            "characters.eye_color": "keyword",
            "characters.character_traits": "keyword",
            # Temporal fields (flattened)
            "year": "integer",
            "season": "keyword",
            "duration": "integer",        # Episode duration in seconds
            # Platform fields
            "sources": "keyword",
            # Statistics for numerical filtering
            "statistics": "keyword",      # Keep as-is for now
            "score.median": "float",      # Representative score for range queries
            # Note: enrichment_metadata intentionally excluded (non-indexed operational data)
        },
        description="Payload fields with their types for optimized indexing (excludes operational metadata)",
    )

    # API Configuration
    api_title: str = Field(default="Anime Vector Service", description="API title")
    api_version: str = Field(default="1.0.0", description="API version")
    api_description: str = Field(
        default="Microservice for anime vector database operations",
        description="API description",
    )

    # Batch Processing Configuration
    default_batch_size: int = Field(
        default=100, ge=1, le=1000, description="Default batch size for operations"
    )
    max_batch_size: int = Field(
        default=500, ge=1, le=2000, description="Maximum allowed batch size"
    )

    # Request Limits
    max_search_limit: int = Field(
        default=100, ge=1, le=1000, description="Maximum search results limit"
    )
    request_timeout: int = Field(
        default=30, ge=1, le=300, description="Request timeout in seconds"
    )

    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string",
    )

    # CORS Configuration
    allowed_origins: List[str] = Field(
        default=["*"], description="Allowed CORS origins"
    )
    allowed_methods: List[str] = Field(
        default=["*"], description="Allowed HTTP methods"
    )
    allowed_headers: List[str] = Field(
        default=["*"], description="Allowed HTTP headers"
    )

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


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
