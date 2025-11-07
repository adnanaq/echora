"""
gRPC service implementation for AI Agent tasks.

This service handles natural language queries and multimodal search requests,
using the AnimeQueryAgent to parse queries and return ranked anime results.
"""
import grpc
import logging

from protos import agent_pb2
from protos import agent_pb2_grpc

logger = logging.getLogger(__name__)


class AgentService(agent_pb2_grpc.AgentServiceServicer):
    """
    Provides the gRPC implementation for the AgentService.

    This service uses the AnimeQueryAgent to process natural language queries
    and optional images, returning a ranked list of anime IDs from the vector database.
    """

    async def Search(
        self, request: agent_pb2.SearchRequest, context
    ) -> agent_pb2.SearchResponse:
        """
        Handles the gRPC Search request.

        Processes natural language queries (with optional image data) through the
        AnimeQueryAgent to perform semantic search over the anime vector database.

        Args:
            request: SearchRequest containing query text and optional image_data
            context: gRPC context for setting response metadata

        Returns:
            SearchResponse containing ranked list of anime IDs
        """
        # Import globals module for shared state
        from src import globals as app_globals

        if not app_globals.query_parser_agent:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("Query parser agent not available.")
            return agent_pb2.SearchResponse()

        try:
            logger.info(f"gRPC Search received request: query='{request.query[:50]}...'")

            # Extract query and optional image data
            query = request.query
            image_data = request.image_data if request.HasField("image_data") else None

            # Use the AnimeQueryAgent to parse query and execute search
            search_results = await app_globals.query_parser_agent.parse_and_search(
                user_query=query,
                image_data=image_data,
                format_results=True  # Get formatted results
            )

            # Extract anime IDs from formatted results
            anime_ids = self._extract_anime_ids(search_results)

            logger.info(f"gRPC Search returned {len(anime_ids)} results")

            # Get summary as reasoning
            reasoning = getattr(search_results, "summary", "Search completed successfully")

            return agent_pb2.SearchResponse(
                anime_ids=anime_ids,
                reasoning=reasoning,
            )

        except Exception as e:
            logger.error(f"gRPC Search failed: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"An internal error occurred: {e}")
            return agent_pb2.SearchResponse()

    def _extract_anime_ids(self, search_results) -> list[str]:
        """
        Extract anime IDs from agent search results.

        Args:
            search_results: FormattedResultsSchema or SearchOutputSchema from AnimeQueryAgent

        Returns:
            List of anime ID strings
        """
        # Handle FormattedResultsSchema (has results list with anime_id)
        if hasattr(search_results, "results"):
            anime_ids = []
            for result in search_results.results:
                # Each result should have anime_id
                if hasattr(result, "anime_id") and result.anime_id:
                    anime_ids.append(str(result.anime_id))
                elif isinstance(result, dict) and "anime_id" in result:
                    anime_ids.append(str(result["anime_id"]))
            return anime_ids

        logger.warning(f"Unexpected search results format: {type(search_results)}")
        return []
