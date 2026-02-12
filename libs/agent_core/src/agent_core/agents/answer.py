"""Answer agent construction for draft synthesis from retrieved context."""

from __future__ import annotations

from atomic_agents import AgentConfig, AtomicAgent
from atomic_agents.context import ChatHistory, SystemPromptGenerator

from agent_core.context import RetrievedContextProvider
from agent_core.schemas import AnswerInput, AnswerOutput


def build_answer_agent(
    client,
    model: str,
    retrieved_context: RetrievedContextProvider,
) -> AtomicAgent[AnswerInput, AnswerOutput]:
    """Builds the answer-drafting stage agent.

    Args:
        client: Instructor/OpenAI-compatible client used by Atomic Agents.
        model: Model identifier passed to Atomic Agents.
        retrieved_context: Dynamic context provider injected into prompts.

    Returns:
        AtomicAgent configured to emit ``AnswerOutput``.
    """
    system = SystemPromptGenerator(
        background=[
            "You are an internal answer drafting stage for anime search.",
            "Draft concise answers strictly from retrieved context.",
            "Do not include chain-of-thought.",
        ],
        steps=[
            "Read rewritten query and Retrieved Context.",
            "Synthesize a candidate answer grounded in retrieved evidence.",
            "Populate source_entities with all supporting entities and result_entities with primary display results.",
        ],
        output_instructions=[
            "Return AnswerOutput only.",
            "Always set llm_confidence in the range [0.0, 1.0] based on support in retrieved context.",
            "If context is thin, still draft best-effort answer and add warning(s).",
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
            "temperature": 0.4,
            "reasoning_effort": "low",
            "max_completion_tokens": 700,
            "max_retries": 1,
        },
    )
    return AtomicAgent[AnswerInput, AnswerOutput](cfg)
