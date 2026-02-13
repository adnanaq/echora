"""Rewrite agent construction for normalized query and retrieval gating."""

from __future__ import annotations

from atomic_agents import AgentConfig, AtomicAgent
from atomic_agents.context import ChatHistory, SystemPromptGenerator

from agent_core.schemas import RewriteOutput, RewriteInput


def build_rewrite_agent(client, model: str) -> AtomicAgent[RewriteInput, RewriteOutput]:
    """Builds the rewrite/gating stage agent.

    Args:
        client: Instructor/OpenAI-compatible client used by Atomic Agents.
        model: Model identifier passed to Atomic Agents.

    Returns:
        AtomicAgent configured to emit ``RewriteOutput``.
    """
    system = SystemPromptGenerator(
        background=[
            "You are an Anime Query Rewriter for a hybrid search system (Qdrant semantic + PostgreSQL graph).",
            "Normalize user queries for retrieval and classify intent (semantic search vs. graph traversal).",
            "Output must conform to RewriteOutput schema with concrete, search-optimized phrasing.",
        ],
        steps=[
            "1. Preserve exact user-provided titles/names/entities.",
            "2. Classify query intent (be conservative - most queries are semantic search):",
            "   - Graph traversal (requires_graph_traversal=True) ONLY for:",
            "     * Explicit relationship questions: 'how is X related to Y', 'what connects A to B'",
            "     * Cross-entity comparisons: 'compare Naruto vs Bleach', 'differences between X and Y'",
            "     * Family trees or character lineage: 'Naruto's family', 'who are X's descendants'",
            "   - Semantic search (requires_graph_traversal=False) for everything else:",
            "     * Content/theme search: 'anime with X and Y', 'anime about Z'",
            "     * Character search: 'anime with dog character', 'shows featuring X'",
            "     * Recommendations: 'anime like X', 'similar to Y'",
            "3. Rewrite into concise, keyword-rich phrasing (no fluff, just search terms + constraints).",
            "4. If current_rewritten_query + missing_information exist, refine to address missing_information.",
            "5. Use last_retrieval_summary to avoid repeating unproductive phrasing.",
            "6. Set needs_external_context=True for lookups/comparisons/recommendations; False for general knowledge.",
        ],
        output_instructions=[
            "Return RewriteOutput matching the schema exactly.",
            "rationale: one sentence, max 20 words.",
            "requires_graph_traversal: True only for relationship/comparison/traversal queries.",
            "Never invent specific anime titles not mentioned by the user.",
        ],
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
            "max_completion_tokens": 220,
            "max_retries": 1,
        },
    )
    return AtomicAgent[RewriteInput, RewriteOutput](cfg)
