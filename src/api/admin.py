"""
Admin API endpoints for vector service management.

Provides database statistics, health monitoring, and administrative operations.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from ..config import get_settings
from ..dependencies import get_qdrant_client, get_text_processor, get_vision_processor
from ..vector.client.qdrant_client import QdrantClient
from ..vector.processors.text_processor import TextProcessor
from ..vector.processors.vision_processor import VisionProcessor

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
    additional_stats: Dict[str, Any] = Field(
        default_factory=dict, description="Additional statistics"
    )


@router.get("/stats", response_model=StatsResponse)
async def get_database_stats(qdrant_client: QdrantClient = Depends(get_qdrant_client)) -> StatsResponse:
    """
    Get comprehensive database statistics.

    Returns information about the vector database including document counts,
    collection status, and performance metrics.
    """
    try:
        # from ..main import qdrant_client # Removed global import

        # if not qdrant_client: # No longer needed due to Depends
        #     raise HTTPException(status_code=503, detail="Vector database not available")

        # Get database statistics
        stats = await qdrant_client.get_stats()

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
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve database statistics: {str(e)}"
        )


@router.get("/health")
async def admin_health_check(qdrant_client: QdrantClient = Depends(get_qdrant_client)) -> Dict[str, Any]:
    """
    Detailed health check for admin purposes.

    Provides more comprehensive health information than the basic health endpoint.
    """
    try:
        # from ..main import qdrant_client # Removed global import

        # if not qdrant_client: # No longer needed due to Depends
        #     return {
        #         "status": "unhealthy",
        #         "qdrant_client": "not_initialized",
        #         "details": "Vector database client not available",
        #     }

        # Perform health check
        qdrant_healthy = await qdrant_client.health_check()

        # Get basic stats for additional health info
        stats = {}
        try:
            stats = await qdrant_client.get_stats()
        except Exception as e:
            logger.warning(f"Could not retrieve stats for health check: {e}")

        return {
            "status": "healthy" if qdrant_healthy else "unhealthy",
            "qdrant_client": "initialized",
            "qdrant_status": "healthy" if qdrant_healthy else "unhealthy",
            "collection_name": stats.get("collection_name", "unknown"),
            "document_count": stats.get("total_documents", 0),
            "details": (
                "All systems operational"
                if qdrant_healthy
                else "Vector database issues detected"
            ),
        }

    except Exception as e:
        logger.error(f"Admin health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "details": "Health check encountered an error",
        }


@router.delete("/vectors/{anime_id}")
async def delete_vector(anime_id: str, qdrant_client: QdrantClient = Depends(get_qdrant_client)) -> Dict[str, Any]:
    """
    Delete a vector from the database.

    Removes the specified anime from the vector collection.
    """
    try:
        # from ..main import qdrant_client # Removed global import

        # if not qdrant_client: # No longer needed due to Depends
        #     raise HTTPException(status_code=503, detail="Vector database not available")

        # Check if anime exists
        existing = await qdrant_client.get_by_id(anime_id)
        if not existing:
            raise HTTPException(status_code=404, detail=f"Anime not found: {anime_id}")

        # For now, we don't have a direct delete method in QdrantClient
        # This would need to be implemented
        # await qdrant_client.delete_by_id(anime_id)

        # Placeholder response
        return {
            "deleted": False,
            "anime_id": anime_id,
            "message": "Delete operation not yet implemented",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete operation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Delete operation failed: {str(e)}"
        )


@router.post("/reindex")
async def reindex_collection(qdrant_client: QdrantClient = Depends(get_qdrant_client)) -> Dict[str, Any]:
    """
    Rebuild the vector index.

    WARNING: This operation will clear the existing collection and recreate it.
    Use with caution in production environments.
    """
    try:
        # from ..main import qdrant_client # Removed global import

        # if not qdrant_client: # No longer needed due to Depends
        #     raise HTTPException(status_code=503, detail="Vector database not available")

        # Clear and recreate the index
        success = await qdrant_client.clear_index()

        if not success:
            raise HTTPException(status_code=500, detail="Failed to reindex collection")

        return {
            "reindexed": True,
            "message": "Collection successfully cleared and recreated",
            "warning": "All existing data has been removed",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reindex operation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Reindex operation failed: {str(e)}"
        )


@router.get("/collection/info")
async def get_collection_info(
    qdrant_client: QdrantClient = Depends(get_qdrant_client),
    text_processor: TextProcessor = Depends(get_text_processor),
    vision_processor: VisionProcessor = Depends(get_vision_processor),
) -> Dict[str, Any]:
    """
    Get detailed collection information.

    Returns comprehensive information about the vector collection configuration.
    """
    try:
        stats = await qdrant_client.get_stats()

        return {
            "collection_name": qdrant_client.collection_name,
            "qdrant_url": qdrant_client.url,
            "vector_size": qdrant_client._vector_size,
            "image_vector_size": qdrant_client._image_vector_size,
            "distance_metric": qdrant_client._distance_metric,
            "stats": stats,
            "processors": {
                "text_processor": text_processor.get_model_info() if text_processor else None,
                "vision_processor": vision_processor.get_model_info() if vision_processor else None,
            },
        }

    except Exception as e:
        logger.error(f"Failed to get collection info: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get collection information: {str(e)}"
        )
