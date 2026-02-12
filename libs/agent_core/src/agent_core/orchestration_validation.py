"""Validation/normalization helpers for source-selector outputs."""

from __future__ import annotations

from agent_core.schemas import SourceSelectionOutput


def normalize_step_for_turn(
    *,
    step: SourceSelectionOutput,
    rewritten_query: str,
    request_image_query: str | None,
) -> tuple[SourceSelectionOutput | None, str | None]:
    """Validate and normalize one source-selector output.

    Args:
        step: Source-selector output for the turn.
        rewritten_query: Rewritten query from rewrite stage.
        request_image_query: Request-level image payload.

    Returns:
        Tuple ``(normalized_step, warning)`` where warning is set on validation
        failure. ``normalized_step`` is ``None`` when the step is invalid.
    """
    if step.action == "qdrant_search":
        if step.search_intent is None:
            return None, (
                "Source selector emitted qdrant_search without search_intent; "
                "skipping."
            )

        id_filter = step.search_intent.filters.get("id")
        has_id_filter = isinstance(id_filter, list) and bool(id_filter)

        if request_image_query and not step.search_intent.image_query and not has_id_filter:
            step.search_intent.image_query = request_image_query

        if rewritten_query and step.search_intent.query is None and not has_id_filter:
            step.search_intent.query = rewritten_query

        return step, None

    if step.action == "pg_graph":
        if step.graph_intent is None:
            return None, "Source selector emitted pg_graph without graph_intent; skipping."
        return step, None

    return None, f"Unknown source action '{step.action}'; skipping."
