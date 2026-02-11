"""Planner agent construction for structured next-step decisions."""

from __future__ import annotations

from atomic_agents import AtomicAgent
from atomic_agents import AgentConfig, BasicChatInputSchema
from atomic_agents.context import ChatHistory, SystemPromptGenerator

from agent_core.context import RetrievedContextProvider
from agent_core.schemas import NextStep


def build_planner_agent(
    client,
    model: str,
    retrieved_context: RetrievedContextProvider,
) -> AtomicAgent[BasicChatInputSchema, NextStep]:
    """Builds the planner agent used by the orchestrator loop.

    Args:
        client: Instructor/OpenAI-compatible client used by Atomic Agents.
        model: Model identifier passed to Atomic Agents.
        retrieved_context: Dynamic context provider injected into planner prompts.

    Returns:
        AtomicAgent configured to emit only ``NextStep`` outputs.
    """
    system = SystemPromptGenerator(
        background=[
            "You are an internal planning agent for an anime search system.",
            "You must decide exactly ONE next action per turn: qdrant_search, pg_graph, or final.",
            "Do not generate SQL. Use only the provided schema fields.",
            "Do not include chain-of-thought. Keep rationales short and operational.",
        ],
        steps=[
            "Read the user's query.",
            "Review Retrieved Context if present.",
            "Choose the next action. Prefer qdrant_search first for entity resolution and candidates.",
            "If Postgres graph is unavailable, avoid pg_graph and finalize with warnings.",
        ],
        output_instructions=[
            "Return a NextStep object only.",
            "If action is qdrant_search, populate search_intent.",
            "If action is pg_graph, populate graph_intent.",
            "If action is final, populate final with an AgentResponse.",
            "For final.source_entities include all evidence entities used to justify the answer.",
            "For final.result_entities include only the primary entities to display to the user.",
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
