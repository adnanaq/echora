"""Sufficiency agent construction for context completeness checks."""

from __future__ import annotations

from atomic_agents import AtomicAgent
from atomic_agents import AgentConfig
from atomic_agents.context import ChatHistory, SystemPromptGenerator

from agent_core.context import RetrievedContextProvider
from agent_core.schemas import SufficiencyOutput, SufficiencyInput


def build_sufficiency_agent(
    client,
    model: str,
    retrieved_context: RetrievedContextProvider,
) -> AtomicAgent[SufficiencyInput, SufficiencyOutput]:
    """Builds the sufficiency checker agent used after retrieval steps.

    Args:
        client: Instructor/OpenAI-compatible client used by Atomic Agents.
        model: Model identifier passed to Atomic Agents.
        retrieved_context: Dynamic context provider injected into sufficiency prompts.

    Returns:
        AtomicAgent configured to emit only ``SufficiencyOutput`` outputs.
    """
    system = SystemPromptGenerator(
        background=[
            "You are a sufficiency checker for anime search.",
            "Decide if retrieved context is enough to answer the user's query.",
            "If relationship/comparison queries only tried Qdrant, request pg_graph.",
        ],
        steps=[
            "1. Read user_query carefully (what is the user actually asking?).",
            "2. Review draft_answer: does it fully address the query?",
            "3. Check Retrieved Context cards: what sources were used?",
            "4. Check attempted_actions: was only 'qdrant_search' tried?",
            "5. Check last_search_similarity_score:",
            "   - Score >0.7: high semantic match (good for entity lookups)",
            "   - Score 0.5-0.7: moderate match (may need refinement)",
            "   - Score <0.5: weak match (likely needs rewrite or different retrieval method)",
            "6. Decision logic:",
            "   - If query asks EXPLICITLY about relationships ('how is X related to Y', 'compare A vs B') AND only Qdrant tried:",
            "     → sufficient=False, need_graph_traversal=True, missing=['pg_graph traversal for relationships']",
            "   - If draft_answer is vague/incomplete despite high score (>0.7):",
            "     → sufficient=False, list specific missing facts",
            "   - If score >0.7 and draft_answer addresses the query (even partially):",
            "     → sufficient=True (don't request graph for content/theme searches)",
            "   - If draft_answer fully addresses query with supporting evidence:",
            "     → sufficient=True",
        ],
        output_instructions=[
            "Return SufficiencyOutput matching schema.",
            "rationale: one sentence, max 25 words, operational (not subjective).",
            "missing: list concrete gaps (max 5 items, each under 10 words).",
            "need_graph_traversal: True only if relationship query AND pg_graph not in attempted_actions.",
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
            "max_completion_tokens": 240,
            "max_retries": 1,
        },
    )
    return AtomicAgent[SufficiencyInput, SufficiencyOutput](cfg)
