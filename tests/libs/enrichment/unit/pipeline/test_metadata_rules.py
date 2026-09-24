"""Tests for the per-field merge rules.

The weighted score is the part worth pinning down: it has two constants, and a
regression in either would move every score in the database without failing
anything else.
"""

import pytest
from enrichment.pipeline.metadata_rules import merge_score

_BASELINE = {"baseline_score": 6.2, "baseline_votes": 1000}


def test_few_voters_pull_the_score_toward_the_baseline() -> None:
    score = merge_score({"mal": {"score": 9.0, "scored_by": 50}}, **_BASELINE)
    assert score is not None
    assert score["mean"] == 9.0
    assert score["weighted"] == pytest.approx(6.33, abs=0.01)


def test_many_voters_leave_the_score_alone() -> None:
    score = merge_score({"mal": {"score": 9.0, "scored_by": 200_000}}, **_BASELINE)
    assert score is not None
    assert score["weighted"] == pytest.approx(8.99, abs=0.01)


def test_an_ordinary_score_is_unmoved_however_few_voted() -> None:
    score = merge_score({"mal": {"score": 6.2, "scored_by": 3}}, **_BASELINE)
    assert score is not None
    assert score["weighted"] == pytest.approx(6.2, abs=0.01)


def test_weighted_is_absent_when_no_provider_counted_votes() -> None:
    score = merge_score({"mal": {"score": 8.0}, "kitsu": {"score": 7.0}}, **_BASELINE)
    assert score == {"mean": 7.5, "median": 7.5}


def test_votes_pool_across_providers() -> None:
    split = merge_score(
        {
            "mal": {"score": 9.0, "scored_by": 500},
            "kitsu": {"score": 9.0, "scored_by": 500},
        },
        **_BASELINE,
    )
    single = merge_score({"mal": {"score": 9.0, "scored_by": 1000}}, **_BASELINE)
    assert split is not None and single is not None
    assert split["weighted"] == single["weighted"]


def test_every_provider_counts_once_toward_the_mean() -> None:
    score = merge_score(
        {
            "mal": {"score": 9.0, "scored_by": 900_000},
            "anidb": {"score": 6.0, "scored_by": 100},
        },
        **_BASELINE,
    )
    assert score is not None
    assert score["mean"] == 7.5


def test_providers_without_a_score_are_ignored() -> None:
    score = merge_score(
        {
            "mal": {"score": 8.0, "scored_by": 10_000},
            "anisearch": {"rank": 125},
        },
        **_BASELINE,
    )
    assert score is not None
    assert score["mean"] == 8.0


def test_no_score_anywhere_returns_nothing() -> None:
    assert merge_score({"anisearch": {"rank": 125}}, **_BASELINE) is None
    assert merge_score({}, **_BASELINE) is None
