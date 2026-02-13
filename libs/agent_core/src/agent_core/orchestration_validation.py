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
    # Pure validation first
    validation_error = _validate_step(step)
    if validation_error:
        return None, validation_error
    
    # Enrich search intent if present
    if step.action == "qdrant_search" and step.search_intent is not None:
        step.search_intent = _enrich_search_intent(
            step.search_intent,
            rewritten_query,
            request_image_query
        )
    
    return step, None


def _validate_step(step: SourceSelectionOutput) -> str | None:
    """Pure validation - returns error message or None.
    
    Args:
        step: Source-selector output to validate.
        
    Returns:
        Error message if validation fails, None if valid.
    """
    if step.action == "qdrant_search":
        if step.search_intent is None:
            return "Source selector emitted qdrant_search without search_intent; skipping."
    elif step.action == "pg_graph":
        if step.graph_intent is None:
            return "Source selector emitted pg_graph without graph_intent; skipping."
    else:
        return f"Unknown source action '{step.action}'; skipping."
    
    return None


def _enrich_search_intent(
    intent: SearchIntent,
    rewritten_query: str,
    request_image_query: str | None,
):
    """Fills in missing query fields from request context.
    
    Args:
        intent: SearchIntent to enrich.
        rewritten_query: Fallback text query.
        request_image_query: Fallback image query.
        
    Returns:
        Enriched SearchIntent (mutated in place for compatibility).
    """
    # Check if ID filter exists - if so, queries are optional
    id_filter = intent.filters.get("id")
    has_id_filter = isinstance(id_filter, list) and bool(id_filter)
    
    # Fill in missing queries if no ID filter
    if not has_id_filter:
        if request_image_query and not intent.image_query:
            intent.image_query = request_image_query
        if rewritten_query and intent.query is None:
            intent.query = rewritten_query
    
    return intent
