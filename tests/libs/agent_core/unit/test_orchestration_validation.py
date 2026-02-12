"""Unit tests for source-step validation and normalization helpers."""

from agent_core.orchestration_validation import normalize_step_for_turn
from agent_core.schemas import (
    EntityRef,
    EntityType,
    GraphIntent,
    SourceSelectionOutput,
    SearchIntent,
)


def test_normalize_step_qdrant_requires_search_intent() -> None:
    """Reject qdrant action when search_intent is missing."""
    step = SourceSelectionOutput(action="qdrant_search", rationale="missing intent")

    normalized_step, warning = normalize_step_for_turn(
        step=step,
        rewritten_query="test",
        request_image_query=None,
    )

    assert normalized_step is None
    assert warning is not None
    assert "without search_intent" in warning


def test_normalize_step_qdrant_applies_request_defaults() -> None:
    """Fill missing query/image_query defaults when no id filter is present."""
    step = SourceSelectionOutput(
        action="qdrant_search",
        rationale="lookup",
        search_intent=SearchIntent(
            rationale="lookup anime",
            entity_type=EntityType.ANIME,
            query=None,
            image_query=None,
            filters={},
        ),
    )

    normalized_step, warning = normalize_step_for_turn(
        step=step,
        rewritten_query="id invaded synopsis",
        request_image_query="data:image/png;base64,abcd",
    )

    assert warning is None
    assert normalized_step is not None
    assert normalized_step.search_intent is not None
    assert normalized_step.search_intent.query == "id invaded synopsis"
    assert normalized_step.search_intent.image_query == "data:image/png;base64,abcd"


def test_normalize_step_qdrant_keeps_id_lookup_unchanged() -> None:
    """Do not override query/image on deterministic ID lookups."""
    step = SourceSelectionOutput(
        action="qdrant_search",
        rationale="id lookup",
        search_intent=SearchIntent(
            rationale="by id",
            entity_type=EntityType.ANIME,
            query=None,
            image_query=None,
            filters={"id": ["019bce3b-d48e-3d81-61ba-518ea655b2de"]},
        ),
    )

    normalized_step, warning = normalize_step_for_turn(
        step=step,
        rewritten_query="should not apply",
        request_image_query="data:image/png;base64,abcd",
    )

    assert warning is None
    assert normalized_step is not None
    assert normalized_step.search_intent is not None
    assert normalized_step.search_intent.query is None
    assert normalized_step.search_intent.image_query is None


def test_normalize_step_pg_graph_requires_graph_intent() -> None:
    """Reject pg_graph action when graph_intent is missing."""
    step = SourceSelectionOutput(action="pg_graph", rationale="missing graph")

    normalized_step, warning = normalize_step_for_turn(
        step=step,
        rewritten_query="graph",
        request_image_query=None,
    )

    assert normalized_step is None
    assert warning is not None
    assert "without graph_intent" in warning


def test_normalize_step_pg_graph_valid_intent() -> None:
    """Accept pg_graph action when graph_intent is present."""
    step = SourceSelectionOutput(
        action="pg_graph",
        rationale="find path",
        graph_intent=GraphIntent(
            rationale="path",
            query_type="path",
            start=EntityRef(entity_type=EntityType.CHARACTER, id="char-a"),
            end=EntityRef(entity_type=EntityType.ANIME, id="anime-b"),
            max_hops=3,
            edge_types=[],
            filters={},
            limits={"max_results": 50, "max_fanout_per_hop": 50},
        ),
    )

    normalized_step, warning = normalize_step_for_turn(
        step=step,
        rewritten_query="path",
        request_image_query=None,
    )

    assert warning is None
    assert normalized_step is not None
    assert normalized_step.action == "pg_graph"
