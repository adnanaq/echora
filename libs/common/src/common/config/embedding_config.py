"""Embedding model configuration."""

from pydantic import BaseModel, Field, field_validator


class EmbeddingConfig(BaseModel):
    """Configuration for text and image embedding models."""

    # Text Embedding
    text_embedding_provider: str = Field(
        default="huggingface",
        description="Text embedding provider: fastembed, huggingface, sentence-transformers",
    )
    text_embedding_model: str = Field(
        default="BAAI/bge-m3", description="Modern text embedding model name"
    )
    bge_model_version: str = Field(
        default="m3", description="BGE model version: v1.5, m3, reranker"
    )
    bge_model_size: str = Field(
        default="base", description="BGE model size: small, base, large"
    )
    bge_max_length: int = Field(
        default=8192, description="BGE maximum input sequence length"
    )

    # Image Embedding
    image_embedding_provider: str = Field(
        default="openclip", description="Image embedding provider: openclip"
    )
    image_embedding_model: str = Field(
        default="ViT-L-14/laion2b_s32b_b82k",
        description="OpenCLIP ViT-L/14 model for high-quality image embeddings (768 dims)",
    )
    openclip_input_resolution: int = Field(
        default=224, description="OpenCLIP input image resolution"
    )
    openclip_text_max_length: int = Field(
        default=77, description="OpenCLIP maximum text sequence length"
    )

    # Image Processing
    image_batch_size: int = Field(
        default=16,
        ge=1,
        le=128,
        description="Batch size for image embedding processing (adjust based on GPU VRAM: 4-8 for 8GB, 16+ for 16GB+)",
    )
    max_concurrent_image_downloads: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum concurrent image downloads in batch processing (adjust based on bandwidth and rate limits)",
    )

    # Model Management
    model_cache_dir: str | None = Field(
        default=None, description="Custom cache directory for embedding models"
    )
    model_warm_up: bool = Field(
        default=False, description="Pre-load and warm up models during initialization"
    )

    # Concurrency
    embed_max_concurrency: int = Field(
        default=2,
        ge=1,
        le=16,
        description="Maximum concurrent embedding tasks per process",
    )

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

    @field_validator("bge_model_version")
    @classmethod
    def validate_bge_model_version(cls, v: str) -> str:
        """Validate BGE model version."""
        valid_versions = ["v1.5", "m3", "reranker"]
        if v.lower() not in valid_versions:
            raise ValueError(f"BGE model version must be one of: {valid_versions}")
        return v.lower()

    @field_validator("bge_model_size")
    @classmethod
    def validate_bge_model_size(cls, v: str) -> str:
        """Validate BGE model size."""
        valid_sizes = ["small", "base", "large"]
        if v.lower() not in valid_sizes:
            raise ValueError(f"BGE model size must be one of: {valid_sizes}")
        return v.lower()
