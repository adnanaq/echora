"""Admin API endpoints for vector service management.

Provides database statistics and collection information for administrative
dashboards and operational monitoring.
"""

import logging
from typing import Any

from common.config import get_settings
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from vector_db_interface import VectorDBClient
from vector_processing import TextProcessor, VisionProcessor

from ..dependencies import (
    get_text_processor,
    get_vector_db_client,
    get_vision_processor,
)

settings = get_settings()
logger = logging.getLogger(__name__)

router = APIRouter()


class StatsResponse(BaseModel):
    """Response model for database statistics."""

    collection_name: str = Field(..., description="Collection name")
    total_documents: int = Field(..., description="Total number of documents")
    vector_size: int = Field(..., description="Vector embedding size")
    distance_metric: str = Field(..., description="Distance metric used")
    status: str = Field(..., description="Collection status")
    additional_stats: dict[str, Any] = Field(
        default_factory=dict, description="Additional statistics"
    )


@router.get("/stats", response_model=StatsResponse)
async def get_database_stats(
    db_client: VectorDBClient = Depends(get_vector_db_client),
) -> StatsResponse:
    """Return comprehensive database statistics.

    Queries the vector database for document counts, collection status, and
    optimizer metrics.

    Args:
        db_client: Injected vector database client.

    Returns:
        Structured statistics including collection name, document count,
        vector configuration, and optimizer state.

    Raises:
        HTTPException: 500 if statistics cannot be retrieved.
    """
    try:
        # Get database statistics
        stats = await db_client.get_stats()

        if "error" in stats:
            raise HTTPException(
                status_code=500, detail=f"Failed to retrieve stats: {stats['error']}"
            )

        return StatsResponse(
            collection_name=stats.get("collection_name", "unknown"),
            total_documents=stats.get("total_documents", 0),
            vector_size=stats.get("vector_size", 0),
            distance_metric=stats.get("distance_metric", "unknown"),
            status=stats.get("status", "unknown"),
            additional_stats={
                "optimizer_status": stats.get("optimizer_status"),
                "indexed_vectors_count": stats.get("indexed_vectors_count"),
                "points_count": stats.get("points_count"),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get stats")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve database statistics: {str(e)}"
        ) from e


@router.get("/collection")
async def get_collection_info(
    db_client: VectorDBClient = Depends(get_vector_db_client),
    text_processor: TextProcessor = Depends(get_text_processor),
    vision_processor: VisionProcessor = Depends(get_vision_processor),
) -> dict[str, Any]:
    """Return detailed vector collection configuration and processor info.

    Combines database-level statistics with embedding model metadata from
    the text and vision processors.

    Args:
        db_client: Injected vector database client.
        text_processor: Injected text embedding processor.
        vision_processor: Injected vision embedding processor.

    Returns:
        Dict with collection config (name, URL, dimensions, metric),
        database stats, and processor model info.

    Raises:
        HTTPException: 500 if collection info cannot be retrieved.
    """
    try:
        stats = await db_client.get_stats()

        return {
            "collection_name": db_client.collection_name,
            "vector_size": db_client.vector_size,
            "image_vector_size": db_client.image_vector_size,
            "distance_metric": db_client.distance_metric,
            "stats": stats,
            "processors": {
                "text_processor": text_processor.get_model_info(),
                "vision_processor": vision_processor.get_model_info(),
            },
        }

    except Exception as e:
        logger.exception("Failed to get collection info")
        raise HTTPException(
            status_code=500, detail=f"Failed to get collection information: {str(e)}"
        ) from e
