"""Rewrite agent construction for normalized query and retrieval gating."""

from __future__ import annotations

from atomic_agents import AgentConfig, AtomicAgent, BasicChatInputSchema
from atomic_agents.context import ChatHistory, SystemPromptGenerator

from agent_core.schemas import QueryRewrite


def build_rewrite_agent(client, model: str) -> AtomicAgent[BasicChatInputSchema, QueryRewrite]:
    """Builds the rewrite/gating stage agent.

    Args:
        client: Instructor/OpenAI-compatible client used by Atomic Agents.
        model: Model identifier passed to Atomic Agents.

    Returns:
        AtomicAgent configured to emit ``QueryRewrite``.
    """
    system = SystemPromptGenerator(
        background=[
            "You are a query rewrite and routing-gate stage for anime search.",
            "Rewrite user queries for retrieval and decide if external context is required.",
            "Do not include chain-of-thought. Keep rationale short and operational.",
        ],
        steps=[
            "Normalize the query into concise retrieval-friendly wording.",
            "Preserve title/entity text as written by the user; do not invent metadata key prefixes.",
            "Set needs_external_context=true when facts, relationships, comparisons, or recommendations are requested.",
            "Set needs_external_context=false only when a direct answer is possible without retrieval.",
        ],
        output_instructions=[
            "Return QueryRewrite only.",
            "Do not answer the user question here.",
            "Do not synthesize labels like 'anime_id:', 'uuid:', or 'title:' in rewritten_query unless they were explicitly in user text.",
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
    return AtomicAgent[BasicChatInputSchema, QueryRewrite](cfg)
