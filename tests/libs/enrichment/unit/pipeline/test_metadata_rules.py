"""Tests for the per-field merge rules.

The weighted score and the company merge are the parts worth pinning down. The
score has two constants, and a regression in either would move every score in
the database without failing anything else. The company merge decides which
spelling is stored and which provider's role is believed, and both rules were
settled by measurement rather than by what reads well.
"""

import pytest
from enrichment.pipeline.metadata_rules import merge_companies, merge_score

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


def _record(*companies: dict[str, object]) -> dict[str, object]:
    return {"companies": list(companies)}


def test_one_company_under_two_spellings_becomes_one_entry() -> None:
    merged = merge_companies(
        [
            ("mal", _record({"name": "Toei Animation", "roles": ["STUDIO"]})),
            (
                "anisearch",
                _record({"name": "Toei Animation Co., Ltd.", "roles": ["STUDIO"]}),
            ),
        ]
    )
    assert [entry["name"] for entry in merged] == ["Toei Animation"]


def test_the_spelling_most_providers_used_is_stored() -> None:
    merged = merge_companies(
        [
            ("mal", _record({"name": "TOKYO MX", "roles": ["PRODUCER"]})),
            ("anilist", _record({"name": "Tokyo MX", "roles": ["PRODUCER"]})),
            ("kitsu", _record({"name": "Tokyo MX", "roles": ["PRODUCER"]})),
        ]
    )
    assert [entry["name"] for entry in merged] == ["Tokyo MX"]


def test_a_tie_on_spelling_goes_to_the_most_trusted_provider() -> None:
    merged = merge_companies(
        [
            ("mal", _record({"name": "Visual Art's", "roles": ["PRODUCER"]})),
            ("kitsu", _record({"name": "Visual Art’s", "roles": ["PRODUCER"]})),
        ]
    )
    assert [entry["name"] for entry in merged] == ["Visual Art's"]


def test_roles_from_different_providers_are_pooled() -> None:
    merged = merge_companies(
        [
            ("mal", _record({"name": "Madhouse", "roles": ["STUDIO"]})),
            ("kitsu", _record({"name": "Madhouse", "roles": ["PRODUCER"]})),
        ]
    )
    assert merged[0]["roles"] == ["STUDIO", "PRODUCER"]


def test_a_single_field_provider_does_not_decide_the_role() -> None:
    merged = merge_companies(
        [
            ("mal", _record({"name": "GKIDS", "roles": ["LICENSOR"]})),
            ("anime_planet", _record({"name": "GKIDS", "roles": ["STUDIO"]})),
        ]
    )
    assert merged[0]["roles"] == ["LICENSOR"]


def test_a_single_field_provider_decides_when_nobody_else_saw_the_company() -> None:
    merged = merge_companies(
        [("anime_planet", _record({"name": "Studio Sota", "roles": ["STUDIO"]}))]
    )
    assert merged[0]["roles"] == ["STUDIO"]


def test_sources_are_pooled_and_deduplicated() -> None:
    merged = merge_companies(
        [
            (
                "mal",
                _record(
                    {
                        "name": "Toei Animation",
                        "roles": ["STUDIO"],
                        "sources": ["https://myanimelist.net/anime/producer/18"],
                    }
                ),
            ),
            (
                "anilist",
                _record(
                    {
                        "name": "Toei Animation",
                        "roles": ["STUDIO"],
                        "sources": [
                            "https://myanimelist.net/anime/producer/18/",
                            "https://anilist.co/studio/18",
                        ],
                    }
                ),
            ),
        ]
    )
    assert merged[0]["sources"] == [
        "https://myanimelist.net/anime/producer/18",
        "https://anilist.co/studio/18",
    ]


def test_a_company_without_a_name_is_dropped() -> None:
    merged = merge_companies(
        [
            (
                "mal",
                _record(
                    {"name": "", "roles": ["STUDIO"]},
                    {"name": "Bones", "roles": ["STUDIO"]},
                ),
            )
        ]
    )
    assert [entry["name"] for entry in merged] == ["Bones"]


def test_no_companies_anywhere_returns_nothing() -> None:
    assert merge_companies([("mal", {}), ("kitsu", _record())]) == []
