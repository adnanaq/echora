"""Source-selector agent construction for bounded retrieval planning."""

from __future__ import annotations

from atomic_agents import AgentConfig, AtomicAgent
from atomic_agents.context import ChatHistory, SystemPromptGenerator

from agent_core.context import RetrievedContextProvider
from agent_core.schemas import SourceSelectionOutput, SourceSelectionInput


def build_source_selector_agent(
    client,
    model: str,
    retrieved_context: RetrievedContextProvider,
) -> AtomicAgent[SourceSelectionInput, SourceSelectionOutput]:
    """Builds the source-selection stage agent.

    Args:
        client: Instructor/OpenAI-compatible client used by Atomic Agents.
        model: Model identifier passed to Atomic Agents.
        retrieved_context: Dynamic context provider injected into prompts.

    Returns:
        AtomicAgent configured to emit bounded ``SourceSelectionOutput`` actions.
    """
    system = SystemPromptGenerator(
        background=[
            "You are a source selector for anime search.",
            "Choose ONE action per turn: qdrant_search (semantic) OR pg_graph (relationships).",
            "Use last_action, attempted_actions, and warnings to avoid repeating failed strategies.",
        ],
        steps=[
            "1. Read rewritten_query and Retrieved Context cards.",
            "2. Check attempted_actions: what was already tried?",
            "3. Check warnings: why did previous attempts fail?",
            "4. Decision logic:",
            "   - First turn + semantic query → qdrant_search",
            "   - First turn + relationship/comparison query → consider pg_graph (but qdrant may find candidates first)",
            "   - Previous Qdrant failed (score <threshold) → try pg_graph if relationship query, else rewrite",
            "   - Previous PG unavailable → fall back to qdrant_search",
            "   - Both tried with different intents → switch strategies intelligently",
            "5. Populate search_intent OR graph_intent based on action.",
        ],
        output_instructions=[
            "Return SourceSelectionOutput with ONE action.",
            "rationale: one sentence explaining why this action (max 20 words).",
            "For qdrant_search: populate search_intent with query/image_query/filters.",
            "For pg_graph: populate graph_intent with query_type/start/end refs.",
            "Avoid synthetic prefixes in query (no 'anime_id:', 'uuid:', etc.).",
        ],
        context_providers={"retrieved": retrieved_context},
    )

    cfg = AgentConfig(
        client=client,
        model=model,
        history=ChatHistory(max_messages=20),
        system_prompt_generator=system,
        assistant_role="assistant",
        model_api_parameters={
            "temperature": 0,
            "reasoning_effort": "minimal",
            "max_completion_tokens": 260,
            "max_retries": 1,
        },
    )
    return AtomicAgent[SourceSelectionInput, SourceSelectionOutput](cfg)
