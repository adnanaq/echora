"""Shared schemas for Qdrant search tools.

This module contains all input/output schemas and configuration classes
used across the different search tools.
"""

from typing import Any, Dict, List, Literal, Optional, Union

from atomic_agents import BaseIOSchema, BaseToolConfig  # type: ignore[import-untyped]
from pydantic import Field

from src.models.anime import AnimeEntry
from src.vector.client.qdrant_client import QdrantClient

# ============================================================================
# Tool Configuration
# ============================================================================


class QdrantToolConfig(BaseToolConfig):  # type: ignore[misc]
    """Configuration for Qdrant search tools."""

    model_config = {"arbitrary_types_allowed": True}

    qdrant_client: QdrantClient = Field(..., description="Qdrant client instance")


# ============================================================================
# Input Schemas
# ============================================================================


class TextSearchInputSchema(BaseIOSchema):  # type: ignore[misc]
    """Input schema for text-based anime search."""

    tool_type: Literal["text_search"] = Field(default="text_search", description="Tool type discriminator")
    query: str = Field(..., description="Text query to search for anime content")
    limit: int = Field(
        default=10, description="Maximum number of results to return", ge=1, le=100
    )
    fusion_method: str = Field(
        default="rrf",
        description="Fusion algorithm to use: 'rrf' (Reciprocal Rank Fusion) or 'dbsf' (Distribution-Based Score Fusion)",
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional filters for statistics, genres, year, type, etc. Example: {'statistics.mal.score': {'gte': 7.0}, 'genres': ['Action', 'Adventure']}",
    )


class ImageSearchInputSchema(BaseIOSchema):  # type: ignore[misc]
    """Input schema for image-based anime search."""

    tool_type: Literal["image_search"] = Field(default="image_search", description="Tool type discriminator")
    image_data: str = Field(..., description="Base64 encoded image data")
    limit: int = Field(
        default=10, description="Maximum number of results to return", ge=1, le=100
    )
    fusion_method: str = Field(
        default="rrf",
        description="Fusion algorithm to use: 'rrf' or 'dbsf'",
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional filters for statistics, genres, year, type, etc.",
    )


class MultimodalSearchInputSchema(BaseIOSchema):  # type: ignore[misc]
    """Input schema for multimodal (text + image) anime search."""

    tool_type: Literal["multimodal_search"] = Field(default="multimodal_search", description="Tool type discriminator")
    query: str = Field(..., description="Text query to search for anime content")
    image_data: Optional[str] = Field(
        default=None, description="Optional base64 encoded image data"
    )
    limit: int = Field(
        default=10, description="Maximum number of results to return", ge=1, le=100
    )
    fusion_method: str = Field(
        default="rrf",
        description="Fusion algorithm to use: 'rrf' or 'dbsf'",
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional filters for statistics, genres, year, type, etc.",
    )


class CharacterSearchInputSchema(BaseIOSchema):  # type: ignore[misc]
    """Input schema for character-focused anime search."""

    tool_type: Literal["character_search"] = Field(default="character_search", description="Tool type discriminator")
    query: str = Field(
        ..., description="Text query focused on character names or descriptions"
    )
    image_data: Optional[str] = Field(
        default=None, description="Optional base64 encoded character image"
    )
    limit: int = Field(
        default=10, description="Maximum number of results to return", ge=1, le=100
    )
    fusion_method: str = Field(
        default="rrf",
        description="Fusion algorithm to use: 'rrf' or 'dbsf'",
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional filters for statistics, genres, year, type, etc.",
    )


# ============================================================================
# Output Schema
# ============================================================================


class SearchOutputSchema(BaseIOSchema):  # type: ignore[misc]
    """Output schema for all search operations."""

    results: List[AnimeEntry] = Field(
        default_factory=list, description="List of search results"
    )
    count: int = Field(default=0, description="Number of results returned")
    search_type: str = Field(
        ..., description="Type of search performed (text/image/multimodal/character)"
    )
