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
            "You are an internal sufficiency checker for an anime search system.",
            "Your job is to decide if we have enough evidence to answer the user's question.",
            "Do not include chain-of-thought. Keep rationale short and operational.",
        ],
        steps=[
            "Read the user's query.",
            "Review Retrieved Context.",
            "Decide if the context is sufficient to answer accurately.",
            "If insufficient, list what's missing.",
        ],
        output_instructions=[
            "Return a SufficiencyOutput object only.",
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
