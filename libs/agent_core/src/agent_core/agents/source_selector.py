"""Source-selector agent construction for bounded retrieval planning."""

from __future__ import annotations

from atomic_agents import AgentConfig, AtomicAgent, BasicChatInputSchema
from atomic_agents.context import ChatHistory, SystemPromptGenerator

from agent_core.context import RetrievedContextProvider
from agent_core.schemas import NextStep


def build_source_selector_agent(
    client,
    model: str,
    retrieved_context: RetrievedContextProvider,
) -> AtomicAgent[BasicChatInputSchema, NextStep]:
    """Builds the source-selection stage agent.

    Args:
        client: Instructor/OpenAI-compatible client used by Atomic Agents.
        model: Model identifier passed to Atomic Agents.
        retrieved_context: Dynamic context provider injected into prompts.

    Returns:
        AtomicAgent configured to emit bounded ``NextStep`` actions.
    """
    system = SystemPromptGenerator(
        background=[
            "You are a bounded source selector for anime search.",
            "Choose exactly one next action: qdrant_search or pg_graph.",
            "For qdrant_search, map user intent to the correct SearchIntent fields (query/image_query/filters).",
            "Use short operational rationale only (no chain-of-thought).",
        ],
        steps=[
            "Read rewritten query and retrieved context.",
            "Prefer qdrant_search for entity resolution/candidate gathering.",
            "Use pg_graph only for relationship/path/comparison operations.",
        ],
        output_instructions=[
            "Return NextStep only.",
            "Choose only action=qdrant_search or action=pg_graph.",
            "If action=qdrant_search, populate search_intent.",
            "For text-only requests: set search_intent.query and leave image_query null.",
            "For image-only requests: set search_intent.image_query and leave query null.",
            "For multimodal requests: set both query and image_query.",
            "Only use search_intent.filters for deterministic constraints (IDs, type/year constraints) supported by index payload fields.",
            "Do not encode metadata as free text inside query (avoid synthetic prefixes like 'anime_id:', 'uuid:', 'title:').",
            "If action=pg_graph, populate graph_intent.",
        ],
        context_providers={"retrieved": retrieved_context},
    )

    cfg = AgentConfig(
        client=client,
        model=model,
        history=ChatHistory(max_messages=20),
        system_prompt_generator=system,
        assistant_role="assistant",
    )
    return AtomicAgent[BasicChatInputSchema, NextStep](cfg)
