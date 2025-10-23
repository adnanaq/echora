"""Multimodal Search Tool for Anime Vector Database

Provides combined text and image search using both BGE-M3 and OpenCLIP
embeddings for comprehensive multimodal queries.
"""

import asyncio
import logging

from atomic_agents import BaseTool  # type: ignore[import-untyped]

from src.poc.tools.schemas import (
    MultimodalSearchInputSchema,
    QdrantToolConfig,
    SearchOutputSchema,
)
from src.vector.client.qdrant_client import QdrantClient

logger = logging.getLogger(__name__)


class MultimodalSearchTool(BaseTool[MultimodalSearchInputSchema, SearchOutputSchema]):  # type: ignore[misc]
    """Tool for multimodal (text + image) anime search.

    Combines semantic text search and visual similarity search to find
    anime that match both textual descriptions and visual characteristics.

    Searches across all 11 vectors (9 text + 2 image) simultaneously,
    using Qdrant's native fusion to combine similarity scores.

    Ideal for queries like:
    - "Dark fantasy anime with gothic art style" (text + image of gothic art)
    - "Action shows similar to this poster" (text intent + visual reference)
    - Character searches with both name and appearance

    Attributes:
        input_schema: MultimodalSearchInputSchema defining required/optional parameters
        output_schema: SearchOutputSchema defining return structure
        client: QdrantClient instance for database operations
    """

    input_schema = MultimodalSearchInputSchema
    output_schema = SearchOutputSchema

    def __init__(self, config: QdrantToolConfig):
        """Initialize multimodal search tool with Qdrant client.

        Args:
            config: QdrantToolConfig containing the Qdrant client instance
        """
        super().__init__(config)
        self.client: QdrantClient = config.qdrant_client

    def run(self, params: MultimodalSearchInputSchema) -> SearchOutputSchema:
        """Execute multimodal search with filters.

        Args:
            params: Multimodal search parameters including:
                - query: Search text
                - image_data: Optional base64 encoded image
                - limit: Max results (1-100, default 10)
                - fusion_method: 'rrf' or 'dbsf' (default 'rrf')
                - filters: Optional dict with filter conditions

        Returns:
            SearchOutputSchema containing:
                - results: List of anime matching text and/or visual criteria
                - count: Number of results
                - search_type: "multimodal"

        Example:
            >>> tool = MultimodalSearchTool(config)
            >>> result = tool.run(MultimodalSearchInputSchema(
            ...     query="dark fantasy",
            ...     image_data="base64_gothic_art...",
            ...     filters={"statistics.mal.score": {"gte": 7.0}}
            ... ))
            >>> print(f"Found {result.count} matching anime")
        """
        try:
            # Build Qdrant filter from filter dictionary
            qdrant_filter = None
            if params.filters:
                qdrant_filter = self.client._build_filter(params.filters)

            # Run async search in sync context
            loop = asyncio.get_event_loop()
            results = loop.run_until_complete(
                self.client.search_complete(
                    query=params.query,
                    image_data=params.image_data,
                    limit=params.limit,
                    fusion_method=params.fusion_method,
                    filters=qdrant_filter,
                )
            )

            return SearchOutputSchema(
                results=results, count=len(results), search_type="multimodal"
            )

        except Exception as e:
            logger.error(f"Multimodal search failed: {e}")
            return SearchOutputSchema(results=[], count=0, search_type="multimodal")
