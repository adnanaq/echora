"""Typed stage-input builders for the agent orchestrator."""

from __future__ import annotations

from agent_core.schemas import (
    AnswerInput,
    SourceSelectionInput,
    SufficiencyInput,
    RewriteInput,
    SourceSelectionOutput,
)


def build_rewrite_stage_input(
    *,
    query: str,
    image_query: str | None,
) -> RewriteInput:
    """Build rewrite-stage input from request-level fields.

    Args:
        query: Original user query text.
        image_query: Optional image payload.

    Returns:
        Structured input for the rewrite stage.
    """
    return RewriteInput(
        user_query=query,
        has_image_query=bool(image_query),
    )


def build_source_selection_stage_input(
    *,
    query: str,
    rewritten_query: str,
    turn: int,
    image_query: str | None,
    warnings: list[str],
    last_step: SourceSelectionOutput | None,
    last_summary: str | None,
) -> SourceSelectionInput:
    """Build source-selection input for the current turn.

    Args:
        query: Original user query text.
        rewritten_query: Query normalized by rewrite stage.
        turn: Current 1-based turn index.
        image_query: Optional image payload.
        warnings: Accumulated orchestration warnings.
        last_step: Previous step, if any.
        last_summary: Previous retrieval summary, if any.

    Returns:
        Structured input for source selection.
    """
    return SourceSelectionInput(
        user_query=query,
        rewritten_query=rewritten_query,
        turn=turn,
        has_image_query=bool(image_query),
        warnings=warnings[-5:],
        last_action=last_step.action if last_step is not None else None,
        last_summary=last_summary,
    )


def build_answer_stage_input(*, query: str, rewritten_query: str) -> AnswerInput:
    """Build answer-stage input from query and rewritten query.

    Args:
        query: Original user query text.
        rewritten_query: Query normalized by rewrite stage.

    Returns:
        Structured input for answer drafting.
    """
    return AnswerInput(user_query=query, rewritten_query=rewritten_query)


def build_sufficiency_stage_input(
    *,
    query: str,
    draft_answer: str,
) -> SufficiencyInput:
    """Build sufficiency-stage input from query and draft answer.

    Args:
        query: Original user query text.
        draft_answer: Latest draft answer produced by answer stage.

    Returns:
        Structured input for sufficiency validation.
    """
    return SufficiencyInput(user_query=query, draft_answer=draft_answer)
