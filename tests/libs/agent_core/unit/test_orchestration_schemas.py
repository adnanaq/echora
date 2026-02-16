"""Unit tests for orchestration schema enhancements."""

import pytest
from agent_core.schemas import (
    RewriteOutput,
    SufficiencyInput,
    SufficiencyOutput,
    SourceSelectionInput,
)


def test_rewrite_output_requires_graph_traversal() -> None:
    """RewriteOutput should detect relationship/comparison intent."""
    # Relationship query
    rewrite = RewriteOutput(
        rewritten_query="Boruto AND Naruto relationship",
        needs_external_context=True,
        requires_graph_traversal=True,
        rationale="Relationship query needs graph traversal.",
    )
    assert rewrite.requires_graph_traversal is True

    # Semantic search query
    rewrite2 = RewriteOutput(
        rewritten_query="psychological thriller anime",
        needs_external_context=True,
        requires_graph_traversal=False,
        rationale="Semantic search for genre.",
    )
    assert rewrite2.requires_graph_traversal is False


def test_sufficiency_input_with_similarity_score() -> None:
    """SufficiencyInput should accept similarity score and attempted actions."""
    suff_input = SufficiencyInput(
        user_query="What's the relationship between Boruto and One Piece?",
        draft_answer="Both are popular anime series.",
        last_search_similarity_score=0.85,
        attempted_actions=["qdrant_search"],
    )

    assert suff_input.last_search_similarity_score == 0.85
    assert suff_input.attempted_actions == ["qdrant_search"]


def test_sufficiency_input_defaults() -> None:
    """SufficiencyInput should have sensible defaults."""
    suff_input = SufficiencyInput(
        user_query="Tell me about Cowboy Bebop",
        draft_answer="Cowboy Bebop is a space western anime.",
    )

    assert suff_input.last_search_similarity_score is None
    assert suff_input.attempted_actions == []


def test_sufficiency_output_need_graph_traversal() -> None:
    """SufficiencyOutput should be able to request graph traversal."""
    suff_output = SufficiencyOutput(
        sufficient=False,
        rationale="Relationship query needs pg_graph but only qdrant_search attempted.",
        missing=["pg_graph traversal for relationships"],
        need_graph_traversal=True,
    )

    assert suff_output.sufficient is False
    assert suff_output.need_graph_traversal is True
    assert "pg_graph" in suff_output.missing[0]


def test_sufficiency_output_sufficient_with_high_score() -> None:
    """SufficiencyOutput should mark as sufficient when appropriate."""
    suff_output = SufficiencyOutput(
        sufficient=True,
        rationale="High similarity (0.85) and draft covers all entities.",
        missing=[],
        need_graph_traversal=False,
    )

    assert suff_output.sufficient is True
    assert suff_output.need_graph_traversal is False
    assert len(suff_output.missing) == 0


def test_source_selection_input_with_graph_hint() -> None:
    """SourceSelectionInput should receive graph traversal hint from rewrite."""
    source_input = SourceSelectionInput(
        user_query="Compare Naruto and Bleach",
        rewritten_query="Naruto Bleach comparison",
        turn=1,
        has_image_query=False,
        warnings=[],
        last_action=None,
        last_summary=None,
        attempted_actions=[],
        requires_graph_traversal=True,
    )

    assert source_input.requires_graph_traversal is True
    assert source_input.attempted_actions == []


def test_sufficiency_input_type_safety() -> None:
    """SufficiencyInput.attempted_actions should be properly typed."""
    suff_input = SufficiencyInput(
        user_query="Test query",
        draft_answer="Test answer",
        attempted_actions=["qdrant_search", "pg_graph"],
    )

    # Type should be Literal["qdrant_search", "pg_graph"]
    assert all(action in ["qdrant_search", "pg_graph"] for action in suff_input.attempted_actions)


def test_similarity_score_validation() -> None:
    """SufficiencyInput should validate similarity score range."""
    # Valid score
    suff_input = SufficiencyInput(
        user_query="Test",
        draft_answer="Answer",
        last_search_similarity_score=0.75,
    )
    assert suff_input.last_search_similarity_score == 0.75

    # Test boundaries
    suff_input_min = SufficiencyInput(
        user_query="Test",
        draft_answer="Answer",
        last_search_similarity_score=0.0,
    )
    assert suff_input_min.last_search_similarity_score == 0.0

    suff_input_max = SufficiencyInput(
        user_query="Test",
        draft_answer="Answer",
        last_search_similarity_score=1.0,
    )
    assert suff_input_max.last_search_similarity_score == 1.0

    # Out of range should fail validation
    with pytest.raises(Exception):  # Pydantic ValidationError
        SufficiencyInput(
            user_query="Test",
            draft_answer="Answer",
            last_search_similarity_score=1.5,
        )

    with pytest.raises(Exception):  # Pydantic ValidationError
        SufficiencyInput(
            user_query="Test",
            draft_answer="Answer",
            last_search_similarity_score=-0.1,
        )
