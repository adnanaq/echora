"""Character Search Tool for Anime Vector Database

Provides character-focused search using both character name embeddings
and character image embeddings.
"""

import asyncio
import logging

from atomic_agents import BaseTool  # type: ignore[import-untyped]

from src.poc.tools.schemas import (
    CharacterSearchInputSchema,
    QdrantToolConfig,
    SearchOutputSchema,
)
from src.vector.client.qdrant_client import QdrantClient

logger = logging.getLogger(__name__)


class CharacterSearchTool(BaseTool[CharacterSearchInputSchema, SearchOutputSchema]):  # type: ignore[misc]
    """Tool for character-focused anime search.

    Specialized search targeting character information specifically:
    - character_vector: Character names, personalities, roles
    - character_image_vector: Character visual designs and appearances

    Optimized for queries about specific characters or character types.
    Returns anime featuring characters that match the search criteria.

    Examples:
    - "Anime with strong female protagonists"
    - "Shows featuring characters like Spike Spiegel"
    - Character image search with visual reference

    Attributes:
        input_schema: CharacterSearchInputSchema defining required/optional parameters
        output_schema: SearchOutputSchema defining return structure
        client: QdrantClient instance for database operations
    """

    input_schema = CharacterSearchInputSchema
    output_schema = SearchOutputSchema

    def __init__(self, config: QdrantToolConfig):
        """Initialize character search tool with Qdrant client.

        Args:
            config: QdrantToolConfig containing the Qdrant client instance
        """
        super().__init__(config)
        self.client: QdrantClient = config.qdrant_client

    def run(self, params: CharacterSearchInputSchema) -> SearchOutputSchema:
        """Execute character search with filters.

        Args:
            params: Character search parameters including:
                - query: Character-focused search text
                - image_data: Optional base64 encoded character image
                - limit: Max results (1-100, default 10)
                - fusion_method: 'rrf' or 'dbsf' (default 'rrf')
                - filters: Optional dict with filter conditions

        Returns:
            SearchOutputSchema containing:
                - results: List of anime with matching characters
                - count: Number of results
                - search_type: "character"

        Example:
            >>> tool = CharacterSearchTool(config)
            >>> result = tool.run(CharacterSearchInputSchema(
            ...     query="strong female protagonist",
            ...     filters={"type": "TV"}
            ... ))
            >>> print(f"Found {result.count} anime with matching characters")
        """
        try:
            # Build Qdrant filter from filter dictionary
            qdrant_filter = None
            if params.filters:
                qdrant_filter = self.client._build_filter(params.filters)

            # Run async search in sync context
            loop = asyncio.get_event_loop()
            results = loop.run_until_complete(
                self.client.search_characters(
                    query=params.query,
                    image_data=params.image_data,
                    limit=params.limit,
                    fusion_method=params.fusion_method,
                    filters=qdrant_filter,
                )
            )

            return SearchOutputSchema(
                results=results, count=len(results), search_type="character"
            )

        except Exception as e:
            logger.error(f"Character search failed: {e}")
            return SearchOutputSchema(results=[], count=0, search_type="character")
