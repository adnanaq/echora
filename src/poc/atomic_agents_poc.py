"""Atomic Agents POC for Query Parsing and Qdrant Integration

This POC demonstrates using Atomic Agents framework with Ollama (qwen2.5:32b)
to parse natural language queries and route them to appropriate Qdrant search
methods with filters.

The agent acts as an intelligent intermediary between user queries and the
Qdrant vector database, parsing search intent and filter requirements using
a local LLM instead of OpenAI API.

Requirements:
    - Ollama running locally (http://localhost:11434)
    - Model: qwen3:8b (pull with: ollama pull qwen3:8b)
    - instructor package: pip install instructor
    - atomic-agents package: pip install atomic-agents
"""

import asyncio
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
from src.poc.tools import (
    CharacterSearchInputSchema,
    CharacterSearchTool,
    ImageSearchInputSchema,
    ImageSearchTool,
    MultimodalSearchInputSchema,
    MultimodalSearchTool,
    QdrantToolConfig,
    SearchOutputSchema,
    TextSearchInputSchema,
    TextSearchTool,
)
from src.vector.client.qdrant_client import QdrantClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        ...,
        description="Parameters for the selected tool (type determines which tool to use)",
        discriminator="tool_type",
    )
    reasoning: str = Field(
        ..., description="Explanation of why this tool and parameters were chosen"
    )


class FormattedAnimeResult(BaseIOSchema):
    """Single formatted anime result."""

    title: str = Field(..., description="English or Romaji title of the anime")
    anime_id: Optional[str] = Field(None, description="Unique anime identifier")
    year: Optional[int] = Field(None, description="Release year")
    similarity_score: float = Field(
        ..., description="Search similarity/relevance score"
    )
    average_score: Optional[float] = Field(
        None, description="Cross-platform arithmetic mean score (0-10)"
    )
    mal_score: Optional[float] = Field(None, description="MyAnimeList score (0-10)")
    anilist_score: Optional[float] = Field(None, description="AniList score (0-10)")
    animeplanet_score: Optional[float] = Field(
        None, description="AnimePlanet score (0-10)"
    )


class FormattedResultsSchema(BaseIOSchema):
    """Output schema for formatted search results."""

    results: List[FormattedAnimeResult] = Field(
        ..., description="List of formatted anime search results"
    )
    total_count: int = Field(..., description="Total number of results found")
    summary: str = Field(
        ..., description="Brief summary of the search results and key findings"
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

        # Use TOOLS mode for Qwen3 (supports function calling for better JSON adherence)
        self.client = instructor.from_openai(
            ollama_client,
            mode=instructor.Mode.JSON_SCHEMA,  # Most reliable mode for Ollama with JSON schema
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
                "temperature": 0,  # Deterministic output for consistent JSON
                "max_tokens": 2000,
            },
        )

        # Create the agent with input/output schemas as type parameters
        self.agent = AtomicAgent[QueryParsingInputSchema, QueryParsingOutputSchema](  # type: ignore[misc]
            config=self.agent_config
        )

        # Note: Formatting is done with simple Python code, no second agent needed

    def _create_system_prompt(self) -> SystemPromptGenerator:
        """Create the system prompt for query parsing.

        Returns:
            SystemPromptGenerator configured for anime search query parsing
        """
        return SystemPromptGenerator(
            background=[
                "You are an AI agent that parses anime search queries and extracts structured filters for search tools.",
                "Your task is to determine user intent, map filters to a predefined JSON schema, and select the appropriate search tool.",
                "Understand anime terminology, genres, statistics, score sources, and filtering requirements.",
                "Vague/adjective-based score interpretations (apply when no explicit source or numeric threshold is mentioned):",
                "  - 'highly rated', 'excellent': score.arithmetic_mean >= 8.0",
                "  - 'well-rated', 'good': score.arithmetic_mean >= 7.0",
                "  - 'average', 'decent': score.arithmetic_mean >= 5.0",
                # Filters
                "Filters include:",
                "  1. Statistics filters: range queries (gte, lte, gt, lt).",
                "     - Generic score (cross-platform average): score.arithmetic_mean (0-10).",
                "       Use this by default when the query does not specify a source.",
                "     - Source-specific scores used only when the query explicitly mentions a source:",
                "       - statistics.mal.{score|scored_by|members|favorites|rank|popularity_rank}",
                "       - statistics.anilist.{score|favorites|popularity_rank}",
                "       - statistics.anidb.{score|scored_by}",
                "       - statistics.animeplanet.{score|scored_by|rank}",
                "       - statistics.kitsu.{score|members|favorites|rank|popularity_rank}",
                "       - statistics.animeschedule.{score|scored_by|members|rank}",
                "  2. List filters: genres, tags, type, status",
                "     - type: uppercase string ['TV','MOVIE','OVA','ONA','SPECIAL','MUSIC','PV','UNKNOWN']",
                "     - status: uppercase string ['FINISHED','ONGOING','UPCOMING','UNKNOWN']",
                "     - genres: list of strings ['Action', 'Adventure', ...]",
                "  3. Range operators: gte, lte, gt, lt",
                # Examples
                "Example filters:",
                "  Generic score (no source mentioned):",
                "    {'score.arithmetic_mean': {'gte': 8.0}}",
                "    {'score.arithmetic_mean': {'gte': 7.5}, 'genres': ['Action']}",
                "  Source-specific (source explicitly mentioned):",
                "    {'statistics.mal.score': {'gte': 8.0}}",
                "    {'statistics.mal.score': {'gte': 7.5}, 'statistics.anilist.score': {'gte': 7.5}}",
                "  Other filters:",
                "    {'genres': ['Action','Adventure'], 'type': 'MOVIE', 'status': 'FINISHED'}",
                # Tool selection instructions
                "You must select the appropriate tool based on the query:",
                "  - text_search: semantic search for anime titles, genres, themes, or staff",
                "  - image_search: visual similarity search using cover art or posters",
                "  - multimodal_search: combined text + image search",
                "  - character_search: search specifically for character names or character-focused queries",
                "Always set the tool_type field in tool_parameters accordingly.",
                # Additional guidance
                "For queries mentioning generic terms like 'highly rated', 'popular', or 'quality', interpret them as needing a score filter using score.arithmetic_mean unless a specific platform is mentioned.",
                "Do not return the input or user query in your response.",
                "Only return tool_parameters and reasoning in the output JSON.",
            ],
            steps=[
                "Analyze the user's query to clearly identify the intent (e.g., searching for anime, characters, or images).",
                "Determine whether the query is source-specific (mentions MAL, Anilist, etc.) or generic.",
                "Extract all relevant filters, including scores, genres, types, statuses, and tags.",
                "If no explicit source is mentioned, use the generic field 'score.arithmetic_mean' for score-based filters.",
                "Map all extracted filters to the predefined JSON schema exactly as defined in the background.",
                "Select the appropriate tool_type based on the query intent:",
                "  - text_search: for anime title, genre, or thematic queries.",
                "  - image_search: for visual similarity or cover/poster-based queries.",
                "  - multimodal_search: when both text and image inputs are relevant.",
                "  - character_search: for queries that explicitly reference character names or character-specific searches.",
                "Populate tool_parameters with all required fields: tool_type, query (text), image_data (if applicable), limit (default 10), fusion_method (default 'rrf'), and filters.",
                "Ensure the final response strictly matches the JSON schema described in output_instructions, with only 'tool_parameters' and 'reasoning' fields.",
            ],
            output_instructions=[
                "Return ONLY a JSON object with exactly these fields: tool_parameters, reasoning.",
                "tool_parameters schema:",
                "  tool_type (required): 'text_search', 'image_search', 'multimodal_search', or 'character_search'.",
                "  query (required for text/multimodal/character searches): string.",
                "  image_data (required for image search, optional otherwise): string.",
                "  limit (optional, default 10): number.",
                "  fusion_method (optional, default 'rrf'): 'rrf' | 'dbsf'.",
                "  filters (optional): dictionary of filter criteria (see background).",
                "reasoning: string explaining the choice of tool and filters.",
                "Important:",
                "  - Do NOT include user_query in the output.",
                "  - Do NOT return any text outside the JSON object.",
                "Examples (valid JSON, all in single-line to avoid parsing errors):",
                '{"tool_parameters":{"tool_type":"text_search","query":"highly rated action anime","limit":10,"fusion_method":"rrf","filters":{"score.arithmetic_mean":{"gte":7.5},"genres":["Action"]}},"reasoning":"generic text search with cross-platform average score filter"}',
                '{"tool_parameters":{"tool_type":"character_search","query":"masuki satou","limit":10,"fusion_method":"rrf","filters":{}},"reasoning":"character search for anime featuring the character masuki satou"}',
            ],
        )

    def format_results(
        self, search_results: SearchOutputSchema
    ) -> FormattedResultsSchema:
        """Format raw search results into clean, structured output.

        Args:
            search_results: Raw search results from Qdrant

        Returns:
            Formatted results with extracted key information
        """
        formatted_results = []

        for result in search_results.results:
            # Convert to dict if it's a Pydantic model
            if hasattr(result, "model_dump"):
                result_dict = result.model_dump()
            else:
                result_dict = dict(result)

            # Extract title
            title = result_dict.get("title", "Unknown")

            # Extract scores
            score_data = result_dict.get("score", {})
            avg_score = (
                score_data.get("arithmetic_mean")
                if isinstance(score_data, dict)
                else None
            )

            # Get similarity score from Qdrant search results
            similarity = result_dict.get("similarity_score", 0.0)

            # Extract platform scores
            stats = result_dict.get("statistics", {})
            mal_score = stats.get("mal", {}).get("score") if stats else None
            anilist_score = stats.get("anilist", {}).get("score") if stats else None
            animeplanet_score = (
                stats.get("animeplanet", {}).get("score") if stats else None
            )

            formatted_results.append(
                FormattedAnimeResult(
                    title=title,
                    anime_id=result_dict.get("id"),
                    year=result_dict.get("year"),
                    similarity_score=similarity,
                    average_score=avg_score,
                    mal_score=mal_score,
                    anilist_score=anilist_score,
                    animeplanet_score=animeplanet_score,
                )
            )

        # Create simple summary
        summary = f"Found {len(formatted_results)} {search_results.search_type} search results"

        return FormattedResultsSchema(
            results=formatted_results,
            total_count=len(formatted_results),
            summary=summary,
        )

    def parse_and_search(
        self,
        user_query: str,
        image_data: Optional[str] = None,
        format_results: bool = True,
    ) -> Union[SearchOutputSchema, FormattedResultsSchema]:
        """Parse a natural language query and execute the appropriate search.

        Args:
            user_query: Natural language query from the user
            image_data: Optional base64 encoded image data
            format_results: Whether to format results using LLM (default: True)

        Returns:
            Raw search results (SearchOutputSchema) or formatted results (FormattedResultsSchema)

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

            # Log essential decision info
            logger.info(
                f"Tool: {tool_name} | Filters: {tool_params.filters} | Count: {result.count}"
            )
            logger.info(f"Reasoning: {agent_output.reasoning}")
            # logger.info(f"Results: {result.results}")  # Commented out - too verbose

            # Optionally format results with Python extraction
            if format_results:
                return self.format_results(result)

            return result

        except Exception as e:
            logger.error(f"Query parsing and search failed: {e}", exc_info=True)
            return SearchOutputSchema(results=[], count=0, search_type="error")
