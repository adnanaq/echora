"""Image Search Tool for Anime Vector Database

Provides visual similarity search using OpenCLIP ViT-L/14 embeddings
for cover art, posters, and character images.
"""

import asyncio
import logging

from atomic_agents import BaseTool  # type: ignore[import-untyped]

from src.poc.tools.schemas import (
    ImageSearchInputSchema,
    QdrantToolConfig,
    SearchOutputSchema,
)
from src.vector.client.qdrant_client import QdrantClient

logger = logging.getLogger(__name__)


class ImageSearchTool(BaseTool[ImageSearchInputSchema, SearchOutputSchema]):  # type: ignore[misc]
    """Tool for image-based anime search using OpenCLIP embeddings.

    This tool performs visual similarity search across 2 image vectors:
    - image_vector: Anime cover art, posters, key visuals
    - character_image_vector: Character designs and appearances

    Uses OpenCLIP ViT-L/14 model for high-quality image embeddings.
    Supports the same filtering capabilities as text search.

    Returns visually similar anime ranked by cosine similarity.

    Attributes:
        input_schema: ImageSearchInputSchema defining required/optional parameters
        output_schema: SearchOutputSchema defining return structure
        client: QdrantClient instance for database operations
    """

    input_schema = ImageSearchInputSchema
    output_schema = SearchOutputSchema

    def __init__(self, config: QdrantToolConfig):
        """Initialize image search tool with Qdrant client.

        Args:
            config: QdrantToolConfig containing the Qdrant client instance
        """
        super().__init__(config)
        self.client: QdrantClient = config.qdrant_client

    def run(self, params: ImageSearchInputSchema) -> SearchOutputSchema:
        """Execute image search with filters.

        Args:
            params: Image search parameters including:
                - image_data: Base64 encoded image
                - limit: Max results (1-100, default 10)
                - fusion_method: 'rrf' or 'dbsf' (default 'rrf')
                - filters: Optional dict with filter conditions

        Returns:
            SearchOutputSchema containing:
                - results: List of visually similar anime with scores
                - count: Number of results
                - search_type: "image"

        Example:
            >>> tool = ImageSearchTool(config)
            >>> result = tool.run(ImageSearchInputSchema(
            ...     image_data="base64_encoded_image...",
            ...     limit=5
            ... ))
            >>> print(f"Found {result.count} similar anime")
        """
        try:
            # Build Qdrant filter from filter dictionary
            qdrant_filter = None
            if params.filters:
                logger.info(f"Building Qdrant Filter object from: {params.filters}")
                qdrant_filter = self.client._build_filter(params.filters)
                logger.info(f"Built Qdrant Filter: {qdrant_filter}")

            # Log the call to Qdrant
            logger.info(f"Calling Qdrant search_visual_comprehensive:")
            logger.info(f"  - Image data length: {len(params.image_data) if params.image_data else 0} bytes")
            logger.info(f"  - Limit: {params.limit}")
            logger.info(f"  - Fusion: {params.fusion_method}")
            logger.info(f"  - Filter: {'Applied' if qdrant_filter else 'None'}")

            # Run async search in sync context
            loop = asyncio.get_event_loop()
            results = loop.run_until_complete(
                self.client.search_visual_comprehensive(
                    image_data=params.image_data,
                    limit=params.limit,
                    fusion_method=params.fusion_method,
                    filters=qdrant_filter,
                )
            )

            logger.info(f"Image search returned {len(results)} results")

            return SearchOutputSchema(
                results=results, count=len(results), search_type="image"
            )

        except Exception as e:
            logger.error(f"Image search failed: {e}")
            return SearchOutputSchema(results=[], count=0, search_type="image")
