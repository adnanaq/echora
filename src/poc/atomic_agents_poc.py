"""Atomic Agents POC for Query Parsing and Qdrant Integration

This POC demonstrates using Atomic Agents framework with Ollama (qwen2.5:32b)
to parse natural language queries and route them to appropriate Qdrant search
methods with filters.

The agent acts as an intelligent intermediary between user queries and the
Qdrant vector database, parsing search intent and filter requirements using
a local LLM instead of OpenAI API.

Requirements:
    - Ollama running locally (http://localhost:11434)
    - Model: qwen3:30b (pull with: ollama pull qwen3:30b)
    - instructor package: pip install instructor
    - atomic-agents package: pip install atomic-agents
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

from atomic_agents import (  # type: ignore[import-untyped]
    AgentConfig,
    AtomicAgent,
    BaseIOSchema,
    BaseTool,
    BaseToolConfig,
)
from atomic_agents.context import SystemPromptGenerator  # type: ignore[import-untyped]
from openai import OpenAI
from pydantic import Field

try:
    import instructor
except ImportError:
    raise ImportError(
        "instructor package required. Install with: pip install instructor"
    )

from src.config.settings import get_settings
from src.vector.client.qdrant_client import QdrantClient
from src.poc.tools import (
    CharacterSearchTool,
    CharacterSearchInputSchema,
    ImageSearchTool,
    ImageSearchInputSchema,
    MultimodalSearchTool,
    MultimodalSearchInputSchema,
    QdrantToolConfig,
    SearchOutputSchema,
    TextSearchTool,
    TextSearchInputSchema,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Agent Input/Output Schemas
# ============================================================================


# Agent Input/Output Schemas
# ============================================================================


class QueryParsingInputSchema(BaseIOSchema):
    """Input schema for the query parsing agent."""

    user_query: str = Field(..., description="Natural language query from the user")
    image_data: Optional[str] = Field(
        default=None, description="Optional base64 encoded image data"
    )


class QueryParsingOutputSchema(BaseIOSchema):
    """Output schema for the query parsing agent.

    Uses Union type pattern from orchestrator example for efficient type-driven dispatch.
    """

    tool_parameters: Union[
        TextSearchInputSchema,
        ImageSearchInputSchema,
        MultimodalSearchInputSchema,
        CharacterSearchInputSchema,
    ] = Field(
        ..., description="Parameters for the selected tool (type determines which tool to use)"
    )
    reasoning: str = Field(
        ..., description="Explanation of why this tool and parameters were chosen"
    )


# ============================================================================
# Query Parsing Agent
# ============================================================================


class AnimeQueryAgent:
    """Agent that parses natural language queries and routes to Qdrant search tools.

    This agent uses LLM reasoning to:
    1. Understand the user's search intent
    2. Extract filter requirements from the query
    3. Select the appropriate search tool
    4. Format parameters correctly for the Qdrant client
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        model: str = "qwen3:30b",
        ollama_base_url: str = "http://localhost:11434/v1",
    ):
        """Initialize the query parsing agent with Ollama.

        Args:
            qdrant_client: Qdrant client instance for search operations
            model: Ollama model name (default: qwen3:30b)
            ollama_base_url: Ollama API base URL (default: http://localhost:11434/v1)
        """
        self.qdrant_client = qdrant_client
        self.model = model

        # Initialize Ollama client with Instructor
        ollama_client = OpenAI(
            base_url=ollama_base_url,
            api_key="ollama",  # Dummy API key for Ollama
        )

        self.client = instructor.from_openai(
            ollama_client,
            mode=instructor.Mode.JSON,
        )

        # Initialize search tools as direct instance variables (orchestrator pattern)
        tool_config = QdrantToolConfig(qdrant_client=qdrant_client)
        self.text_search_tool = TextSearchTool(tool_config)
        self.image_search_tool = ImageSearchTool(tool_config)
        self.multimodal_search_tool = MultimodalSearchTool(tool_config)
        self.character_search_tool = CharacterSearchTool(tool_config)

        # Create system prompt
        self.system_prompt = self._create_system_prompt()

        # Initialize agent configuration
        # model_api_parameters can include temperature, max_tokens, etc.
        self.agent_config = AgentConfig(
            client=self.client,
            model=self.model,
            system_prompt_generator=self.system_prompt,
            model_api_parameters={
                "temperature": 0.7,
                "max_tokens": 2000,
            },
        )

        # Create the agent with input/output schemas as type parameters
        self.agent = AtomicAgent[QueryParsingInputSchema, QueryParsingOutputSchema](  # type: ignore[misc]
            config=self.agent_config
        )

    def _create_system_prompt(self) -> SystemPromptGenerator:
        """Create the system prompt for query parsing.

        Returns:
            SystemPromptGenerator configured for anime search query parsing
        """
        return SystemPromptGenerator(
            background=[
                "You are an expert anime search query parser.",
                "Your job is to analyze user queries and determine the best search approach.",
                "You understand anime terminology, genres, statistics, and filter requirements.",
                "You MUST return a JSON response with exactly three fields: tool_name, parameters, and reasoning.",
                "",
                "=== AVAILABLE TOOLS ===",
                "",
                "TOOL 1: text_search",
                "Description: Semantic text search using BGE-M3 embeddings across 9 text vectors",
                "Parameters:",
                "  - query (string, REQUIRED): The search query text",
                "  - limit (integer, optional): Max results to return (1-100, default: 10)",
                "  - fusion_method (string, optional): 'rrf' or 'dbsf' (default: 'rrf')",
                "  - filters (object, optional): Filter conditions (see FILTER FORMAT below)",
                "Returns: List of anime with similarity scores",
                "",
                "TOOL 2: image_search",
                "Description: Visual search using OpenCLIP ViT-L/14 embeddings",
                "Parameters:",
                "  - image_data (string, REQUIRED): Base64 encoded image",
                "  - limit (integer, optional): Max results (default: 10)",
                "  - fusion_method (string, optional): 'rrf' or 'dbsf' (default: 'rrf')",
                "  - filters (object, optional): Filter conditions",
                "Returns: List of visually similar anime",
                "",
                "TOOL 3: multimodal_search",
                "Description: Combined text + image semantic search",
                "Parameters:",
                "  - query (string, REQUIRED): The search query text",
                "  - image_data (string, optional): Base64 encoded image",
                "  - limit (integer, optional): Max results (default: 10)",
                "  - fusion_method (string, optional): 'rrf' or 'dbsf' (default: 'rrf')",
                "  - filters (object, optional): Filter conditions",
                "Returns: List of anime matching text and/or visual similarity",
                "",
                "TOOL 4: character_search",
                "Description: Search focused on character names and appearances",
                "Parameters:",
                "  - query (string, REQUIRED): Character-focused query",
                "  - image_data (string, optional): Base64 character image",
                "  - limit (integer, optional): Max results (default: 10)",
                "  - fusion_method (string, optional): 'rrf' or 'dbsf' (default: 'rrf')",
                "  - filters (object, optional): Filter conditions",
                "Returns: List of anime with matching characters",
                "",
                "=== FILTER FORMAT ===",
                "",
                "Filters support the following field types:",
                "",
                "1. STATISTICS FILTERS (range queries with gte/lte/gt/lt):",
                "   - statistics.mal.score: float (0-10, MAL rating)",
                "   - statistics.mal.scored_by: integer (number of MAL ratings)",
                "   - statistics.mal.members: integer (MAL member count)",
                "   - statistics.mal.favorites: integer (MAL favorites)",
                "   - statistics.mal.rank: integer (MAL rank position, lower is better)",
                "   - statistics.mal.popularity_rank: integer (MAL popularity rank)",
                "   - statistics.anilist.score: float (0-10, AniList rating)",
                "   - statistics.anilist.favorites: integer (AniList favorites)",
                "   - statistics.anilist.popularity_rank: integer",
                "   - statistics.anidb.score: float (0-10, AniDB rating)",
                "   - statistics.anidb.scored_by: integer",
                "   - statistics.animeplanet.score: float (0-10)",
                "   - statistics.animeplanet.scored_by: integer",
                "   - statistics.animeplanet.rank: integer",
                "   - statistics.kitsu.score: float (0-10)",
                "   - statistics.kitsu.members: integer",
                "   - statistics.kitsu.favorites: integer",
                "   - statistics.kitsu.rank: integer",
                "   - statistics.kitsu.popularity_rank: integer",
                "   - statistics.animeschedule.score: float (0-10)",
                "   - statistics.animeschedule.scored_by: integer",
                "   - statistics.animeschedule.members: integer",
                "   - statistics.animeschedule.rank: integer",
                "   - score.arithmetic_mean: float (0-10, cross-platform average)",
                "",
                "2. LIST FILTERS (match any):",
                "   - genres: array of strings (e.g., ['Action', 'Adventure'])",
                "   - tags: array of strings",
                "   - type: string (e.g., 'TV', 'Movie', 'OVA', 'Special')",
                "   - status: string (e.g., 'FINISHED', 'RELEASING', 'NOT_YET_RELEASED')",
                "",
                "3. RANGE FILTER OPERATORS:",
                "   - gte: Greater than or equal (>=)",
                "   - lte: Less than or equal (<=)",
                "   - gt: Greater than (>)",
                "   - lt: Less than (<)",
                "",
                "FILTER EXAMPLES:",
                "  High rated: {'statistics.mal.score': {'gte': 8.0}}",
                "  Popular: {'statistics.mal.members': {'gte': 100000}}",
                "  Top ranked: {'statistics.mal.rank': {'lte': 100}}",
                "  Multiple platforms: {'statistics.mal.score': {'gte': 7.5}, 'statistics.anilist.score': {'gte': 7.5}}",
                "  Genre filter: {'genres': ['Action', 'Shounen']}",
                "  Combined: {'statistics.mal.score': {'gte': 7.0}, 'genres': ['Action'], 'type': 'TV'}",
            ],
            steps=[
                "Analyze the user's natural language query",
                "Identify the search intent (text-only, image-only, multimodal, character-focused)",
                "Extract any filter requirements from the query (score thresholds, genres, year ranges, etc.)",
                "Map filter requirements to Qdrant filter format",
                "Select the appropriate search tool",
                "Format parameters correctly for the selected tool",
                "Return JSON with tool_name, parameters, and reasoning fields",
            ],
            output_instructions=[
                "YOU MUST RETURN ONLY JSON WITH EXACTLY THESE TWO FIELDS:",
                "",
                "REQUIRED OUTPUT STRUCTURE:",
                "{",
                '  "tool_parameters": {',
                '    "query": string (required for text/multimodal/character search),',
                '    "image_data": string (required for image search, optional for multimodal/character),',
                '    "limit": number (optional, default 10),',
                '    "fusion_method": "rrf" | "dbsf" (optional, default "rrf"),',
                '    "filters": {} (optional)',
                "  },",
                '  "reasoning": "your explanation here"',
                "}",
                "",
                "DO NOT return the input. DO NOT include user_query in your response.",
                "ONLY return tool_parameters and reasoning.",
                "The structure of tool_parameters determines which tool will be used automatically.",
                "",
                "Filter format examples:",
                "  - Single score: {'statistics.mal.score': {'gte': 8.0}}",
                "  - Multiple scores: {'statistics.mal.score': {'gte': 7.0}, 'statistics.anilist.score': {'gte': 7.0}}",
                "  - Genre filter: {'genres': ['Action', 'Adventure']}",
                "  - Combined: {'statistics.mal.score': {'gte': 7.0}, 'genres': ['Drama'], 'type': 'MOVIE'}",
                "",
                "COMPLETE VALID EXAMPLE:",
                '{"tool_parameters": {"query": "action anime", "limit": 10, "fusion_method": "rrf", "filters": {"statistics.mal.score": {"gte": 7.5}}}, "reasoning": "Text search with MAL score filter for highly rated action anime"}',
            ],
        )

    def parse_and_search(
        self, user_query: str, image_data: Optional[str] = None
    ) -> SearchOutputSchema:
        """Parse a natural language query and execute the appropriate search.

        Args:
            user_query: Natural language query from the user
            image_data: Optional base64 encoded image data

        Returns:
            Search results from the appropriate tool

        Example queries:
            - "Find highly rated action anime" -> text search with score filter
            - "Show me anime similar to this poster" -> image search
            - "Steampunk anime with robots" -> text search with genre filter
            - "Characters like this from popular shows" -> character search with filters
        """
        try:
            # Run the agent to parse the query
            logger.info(f"Parsing query: {user_query}")

            agent_input = QueryParsingInputSchema(
                user_query=user_query, image_data=image_data
            )

            agent_output: QueryParsingOutputSchema = self.agent.run(agent_input)

            # Type-driven dispatch (orchestrator pattern)
            tool_params = agent_output.tool_parameters
            if isinstance(tool_params, TextSearchInputSchema):
                tool_name = "text_search"
                result = self.text_search_tool.run(tool_params)  # type: ignore[attr-defined]
            elif isinstance(tool_params, ImageSearchInputSchema):
                tool_name = "image_search"
                result = self.image_search_tool.run(tool_params)  # type: ignore[attr-defined]
            elif isinstance(tool_params, MultimodalSearchInputSchema):
                tool_name = "multimodal_search"
                result = self.multimodal_search_tool.run(tool_params)  # type: ignore[attr-defined]
            elif isinstance(tool_params, CharacterSearchInputSchema):
                tool_name = "character_search"
                result = self.character_search_tool.run(tool_params)  # type: ignore[attr-defined]
            else:
                logger.error(f"Unknown tool parameters type: {type(tool_params)}")
                return SearchOutputSchema(results=[], count=0, search_type="error")

            # Log agent decision with clear formatting
            logger.info("=" * 80)
            logger.info("AGENT DECISION")
            logger.info("=" * 80)
            logger.info(f"Selected Tool: {tool_name}")
            logger.info(f"Reasoning: {agent_output.reasoning}")
            logger.info("-" * 80)
            logger.info("Parsed Parameters:")
            logger.info(json.dumps(tool_params.model_dump(), indent=2))
            logger.info("=" * 80)
            logger.info(f"Executing {tool_name} with parsed parameters...")

            # Log the actual values being passed to Qdrant
            logger.info("-" * 80)
            logger.info("VALUES PASSED TO QDRANT:")
            logger.info(f"  Query: {tool_params.query if hasattr(tool_params, 'query') else 'N/A'}")
            logger.info(f"  Limit: {tool_params.limit}")
            logger.info(f"  Fusion Method: {tool_params.fusion_method}")
            logger.info(f"  Filters: {json.dumps(tool_params.filters, indent=4) if tool_params.filters else 'None'}")
            logger.info("-" * 80)

            logger.info(f"Tool returned {result.count} results")
            logger.info("=" * 80 + "\n")

            return result

        except Exception as e:
            logger.error(f"Query parsing and search failed: {e}", exc_info=True)
            return SearchOutputSchema(
                results=[], count=0, search_type="error"
            )


# ============================================================================
# Example Usage
# ============================================================================


def main() -> None:
    """Demonstrate the atomic agents POC with example queries."""
    # Initialize settings and Qdrant client
    settings = get_settings()
    qdrant_client = QdrantClient(
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection_name,
        settings=settings,
    )

    # Initialize the agent with Ollama
    # Note: Ensure Ollama is running with qwen3:30b model
    # Run: ollama pull qwen3:30b
    agent = AnimeQueryAgent(
        qdrant_client=qdrant_client,
        model="qwen3:30b",
        ollama_base_url="http://localhost:11434/v1",
    )

    # Example queries demonstrating different search patterns
    example_queries = [
        # Text search with filters
        "Find highly rated action anime with scores above 7.5 on MAL",

        # Simple text search
        "Anime about pirates and adventure",

        # Character search
        "Shows featuring characters like Spike Spiegel",

        # Complex filters
        "Popular romance anime from 2020-2023 with at least 50k members on MAL",

        # Genre-based search
        "Psychological thriller anime with good ratings",
    ]

    print("\n" + "=" * 80)
    print("ATOMIC AGENTS POC - ANIME SEARCH QUERY PARSING")
    print("=" * 80 + "\n")

    for i, query in enumerate(example_queries, 1):
        print(f"\n{'-' * 80}")
        print(f"Example {i}: {query}")
        print(f"{'-' * 80}\n")

        result = agent.parse_and_search(query)

        print(f"Search Type: {result.search_type}")
        print(f"Results Count: {result.count}")

        if result.count > 0:
            print("\nTop 3 Results:")
            for j, anime in enumerate(result.results[:3], 1):
                title = anime.get("title", {})
                if isinstance(title, dict):
                    title_str = title.get("english") or title.get("romaji") or "Unknown"
                else:
                    title_str = str(title)

                score = anime.get("score", anime.get("_score", 0))
                mal_score = anime.get("statistics", {}).get("mal", {}).get("score", "N/A")

                print(f"  {j}. {title_str}")
                print(f"     Similarity: {score:.4f}, MAL Score: {mal_score}")

        print()

    print("\n" + "=" * 80)
    print("POC Demonstration Complete")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
