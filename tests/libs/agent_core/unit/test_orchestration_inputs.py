"""Unit tests for typed orchestration stage-input builders."""

from agent_core.orchestration_inputs import (
    build_answer_stage_input,
    build_rewrite_stage_input,
    build_source_selection_stage_input,
    build_sufficiency_stage_input,
)
from agent_core.schemas import EntityRef, EntityType, GraphIntent, SourceSelectionOutput


def test_build_rewrite_stage_input() -> None:
    """Build rewrite input with image presence metadata."""
    stage_input = build_rewrite_stage_input(
        query="What is ID: INVADED about?",
        image_query="data:image/png;base64,abcd",
    )

    assert stage_input.user_query == "What is ID: INVADED about?"
    assert stage_input.has_image_query is True


def test_build_source_selection_stage_input_trims_warning_tail() -> None:
    """Keep only the recent warning tail and previous action metadata."""
    previous_step = SourceSelectionOutput(
        action="pg_graph",
        rationale="graph hop",
        graph_intent=GraphIntent(
            rationale="path",
            query_type="path",
            start=EntityRef(entity_type=EntityType.ANIME, id="a"),
            end=EntityRef(entity_type=EntityType.ANIME, id="b"),
            max_hops=3,
            edge_types=[],
            filters={},
            limits={"max_results": 50, "max_fanout_per_hop": 50},
        ),
    )

    stage_input = build_source_selection_stage_input(
        query="Compare two anime",
        rewritten_query="compare anime graph path",
        turn=3,
        image_query=None,
        warnings=["w1", "w2", "w3", "w4", "w5", "w6"],
        last_step=previous_step,
        last_summary="graph summary",
    )

    assert stage_input.turn == 3
    assert stage_input.last_action == "pg_graph"
    assert stage_input.warnings == ["w2", "w3", "w4", "w5", "w6"]
    assert stage_input.last_summary == "graph summary"


def test_build_answer_and_sufficiency_inputs() -> None:
    """Build answer and sufficiency stage inputs from canonical fields."""
    answer_input = build_answer_stage_input(
        query="Tell me about Cowboy Bebop",
        rewritten_query="cowboy bebop synopsis",
    )
    sufficiency_input = build_sufficiency_stage_input(
        query="Tell me about Cowboy Bebop",
        draft_answer="Cowboy Bebop is a space western anime.",
    )

    assert answer_input.user_query == "Tell me about Cowboy Bebop"
    assert answer_input.rewritten_query == "cowboy bebop synopsis"
    assert sufficiency_input.user_query == "Tell me about Cowboy Bebop"
    assert sufficiency_input.draft_answer == "Cowboy Bebop is a space western anime."
