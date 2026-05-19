"""Service-level configuration."""

from pydantic import BaseModel, Field, field_validator, model_validator


class ServiceConfig(BaseModel):
    """Configuration for service API and runtime settings."""

    # Host & Port
    vector_service_host: str = Field(
        default="0.0.0.0", description="Vector service host address"
    )
    enrichment_service_host: str = Field(
        default="0.0.0.0", description="Enrichment service host address"
    )
    vector_service_port: int = Field(
        default=8002, ge=1, le=65535, description="Vector service port"
    )
    enrichment_service_port: int = Field(
        default=8010, ge=1, le=65535, description="Enrichment service port"
    )
    enable_gpu: bool = Field(
        default=False, description="Enable GPU usage for embedding models"
    )
    enrichment_default_file_path: str = Field(
        default="data/qdrant_storage/anime-offline-database.json",
        description="Default enrichment input JSON/JSONL file path",
    )
    enrichment_output_dir: str = Field(
        default="assets/seed_data",
        description="Directory for enrichment output artifacts",
    )

    # API Metadata
    api_title: str = Field(default="Anime Vector Service", description="API title")
    api_version: str = Field(default="1.0.0", description="API version")
    api_description: str = Field(
        default="Microservice for anime vector database operations",
        description="API description",
    )

    # Batch Processing & Limits
    default_batch_size: int = Field(
        default=100, ge=1, le=1000, description="Default batch size for operations"
    )
    max_batch_size: int = Field(
        default=500, ge=1, le=2000, description="Maximum allowed batch size"
    )
    max_search_limit: int = Field(
        default=100, ge=1, le=1000, description="Maximum search results limit"
    )
    request_timeout: int = Field(
        default=30, ge=1, le=300, description="Request timeout in seconds"
    )

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string",
    )

    # CORS
    allowed_origins: list[str] = Field(
        default=["*"], description="Allowed CORS origins"
    )
    allowed_methods: list[str] = Field(
        default=["*"], description="Allowed HTTP methods"
    )
    allowed_headers: list[str] = Field(
        default=["*"], description="Allowed HTTP headers"
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()

    @model_validator(mode="after")
    def validate_batch_sizes(self) -> "ServiceConfig":
        """Ensure default_batch_size does not exceed max_batch_size."""
        if self.default_batch_size > self.max_batch_size:
            raise ValueError(
                f"default_batch_size ({self.default_batch_size}) must not exceed "
                f"max_batch_size ({self.max_batch_size})"
            )
        return self
