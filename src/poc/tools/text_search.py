"""Text Search Tool for Anime Vector Database

Provides semantic text search using BGE-M3 embeddings across multiple
text-based vectors including titles, characters, genres, and more.
"""

import asyncio
import logging

from atomic_agents import BaseTool  # type: ignore[import-untyped]

from src.poc.tools.schemas import (
    QdrantToolConfig,
    SearchOutputSchema,
    TextSearchInputSchema,
)
from src.vector.client.qdrant_client import QdrantClient

logger = logging.getLogger(__name__)


class TextSearchTool(BaseTool[TextSearchInputSchema, SearchOutputSchema]):  # type: ignore[misc]
    """Tool for text-based anime search using BGE-M3 embeddings.

    This tool performs semantic search across 9 text-based vectors including:
    - title_vector: Anime titles and alternative names
    - character_vector: Character names and descriptions
    - genre_vector: Genre and theme information
    - staff_vector: Production staff information
    - temporal_vector: Release dates and scheduling
    - streaming_vector: Streaming platform availability
    - related_vector: Related anime and franchises
    - franchise_vector: Series and franchise relationships
    - episode_vector: Episode information and summaries

    Supports filtering by:
    - Statistics (MAL, AniList, AniDB, etc. scores and popularity)
    - Genres and tags
    - Type (TV, Movie, OVA, Special)
    - Status (FINISHED, RELEASING, NOT_YET_RELEASED)

    Returns anime ranked by semantic similarity using RRF or DBSF fusion.

    Attributes:
        input_schema: TextSearchInputSchema defining required/optional parameters
        output_schema: SearchOutputSchema defining return structure
        client: QdrantClient instance for database operations
    """

    input_schema = TextSearchInputSchema
    output_schema = SearchOutputSchema

    def __init__(self, config: QdrantToolConfig):
        """Initialize text search tool with Qdrant client.

        Args:
            config: QdrantToolConfig containing the Qdrant client instance
        """
        super().__init__(config)
        self.client: QdrantClient = config.qdrant_client

    async def run(self, params: TextSearchInputSchema) -> SearchOutputSchema:
        """Execute text search with filters.

        Args:
            params: Text search parameters including:
                - query: Search text
                - limit: Max results (1-100, default 10)
                - fusion_method: 'rrf' or 'dbsf' (default 'rrf')
                - filters: Optional dict with filter conditions

        Returns:
            SearchOutputSchema containing:
                - results: List of matching anime with scores
                - count: Number of results
                - search_type: "text"

        Example:
            >>> tool = TextSearchTool(config)
            >>> result = await tool.run(TextSearchInputSchema(
            ...     query="action anime",
            ...     filters={"statistics.mal.score": {"gte": 7.5}}
            ... ))
            >>> print(f"Found {result.count} anime")
        """
        try:
            # Build Qdrant filter from filter dictionary
            qdrant_filter = None
            if params.filters:
                qdrant_filter = self.client._build_filter(params.filters)

            results = await self.client.search_text_comprehensive(
                query=params.query,
                limit=params.limit,
                fusion_method=params.fusion_method,
                filters=qdrant_filter,
            )

            return SearchOutputSchema(
                results=results, count=len(results), search_type="text"
            )

        except Exception as e:
            logger.error(f"Text search failed: {e}")
            return SearchOutputSchema(results=[], count=0, search_type="text")
