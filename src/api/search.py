"""
API endpoint for performing searches using the AnimeQueryAgent.
"""

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.poc.atomic_agents_poc import FormattedResultsSchema

logger = logging.getLogger(__name__)

router = APIRouter()


class SearchQuery(BaseModel):
    """Request model for a search query."""

    query: str = Field(..., description="The natural language search query.")
    image_data: Optional[str] = Field(
        default=None,
        description="Optional base64 encoded image data for multimodal search.",
    )
    format_results: bool = Field(
        default=True, description="Whether to format the results."
    )


@router.get("/search", response_model=FormattedResultsSchema)
async def search(
    query: str = Query(..., description="The natural language search query."),
    image_data: Optional[str] = Query(
        default=None,
        description="Optional base64 encoded image data for multimodal search.",
    ),
) -> Any:
    """
    Performs a search using the AnimeQueryAgent.

    This endpoint takes a natural language query, passes it to the agent for parsing
    and execution, and returns the search results.
    """
    try:
        from ..main import query_parser_agent, qdrant_client

        if not qdrant_client:
            raise HTTPException(status_code=503, detail="Vector database not available")

        if not query_parser_agent:
            raise HTTPException(status_code=503, detail="Agent not initialized")

        results = await query_parser_agent.parse_and_search(
            user_query=query,
            image_data=image_data,
            format_results=True,
        )

        if not results:
            raise HTTPException(status_code=404, detail="No results found.")

        return results

    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
