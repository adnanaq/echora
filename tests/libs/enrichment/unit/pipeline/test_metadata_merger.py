"""Tests for cross-provider anime metadata consolidation.

Focus is arbitration: which provider's value survives, and what counts as a
value at all. Mappers emit a key for every model field, so "absent" arrives as
an empty container rather than a missing key — the case that silently discarded
AniDB's `titles` before `_provider_supplied` existed.
"""

import pytest
from enrichment.pipeline.metadata_merger import (
    merge_provider_records,
    merged_anime_id,
)

_MAL_URL = "https://myanimelist.net/anime/21/One_Piece"


def _record(**fields: object) -> dict[str, object]:
    return {"sources": [_MAL_URL], **fields}


def test_first_signal_takes_the_most_trusted_concrete_value() -> None:
    merged = merge_provider_records(
        {
            "kitsu": _record(type="OVA", year=2001),
            "mal": _record(type="TV", year=1999),
        }
    )
    assert merged["type"] == "TV"
    assert merged["year"] == 1999


def test_sentinel_never_beats_a_concrete_value() -> None:
    # A real value from the lowest-ranked provider outranks UNKNOWN from the
    # highest, so "no data" cannot masquerade as an answer.
    merged = merge_provider_records(
        {"mal": _record(type="UNKNOWN"), "kitsu": _record(type="TV")}
    )
    assert merged["type"] == "TV"


def test_sentinel_survives_when_every_provider_agrees_on_it() -> None:
    merged = merge_provider_records(
        {"mal": _record(type="UNKNOWN"), "kitsu": _record(type="UNKNOWN")}
    )
    assert merged["type"] == "UNKNOWN"


def test_empty_container_does_not_outrank_a_populated_one() -> None:
    # Six providers emit `titles: {}`; only AniDB fills it. Treating {} as a
    # value let MAL's empty dict win and dropped all 27 titles.
    merged = merge_provider_records(
        {"mal": _record(titles={}), "anidb": _record(titles={"de": "One Piece"})}
    )
    assert merged["titles"] == {"de": "One Piece"}


def test_disagreement_on_a_uniform_field_is_logged(caplog) -> None:
    merge_provider_records({"mal": _record(year=1999), "kitsu": _record(year=2001)})
    assert "disagree on year" in caplog.text


def test_statistics_are_kept_per_provider_without_arbitration() -> None:
    merged = merge_provider_records(
        {
            "mal": _record(statistics={"mal": {"score": 8.73}}),
            "anisearch": _record(statistics={"anisearch": {"score": 8.36}}),
        }
    )
    assert merged["statistics"] == {
        "anisearch": {"score": 8.36},
        "mal": {"score": 8.73},
    }


def test_sources_keep_one_url_per_work_preferring_the_slug() -> None:
    merged = merge_provider_records(
        {
            "mal": _record(),
            "anilist": {"sources": ["https://myanimelist.net/anime/21"]},
        }
    )
    assert merged["sources"] == [_MAL_URL]


def test_sources_union_the_offline_seed_and_the_providers() -> None:
    # Neither side is complete: animeschedule is absent from the seed, and
    # livechart is absent from every provider.
    merged = merge_provider_records(
        {
            "animeschedule": _record(
                sources=["https://animeschedule.net/anime/one-piece"]
            )
        },
        {"sources": ["https://livechart.me/anime/321"]},
    )
    assert merged["sources"] == [
        "https://animeschedule.net/anime/one-piece",
        "https://livechart.me/anime/321",
    ]


def test_numeric_url_drops_when_a_slug_names_the_same_work() -> None:
    # Kitsu is addressable both ways and the two halves disagree on which to
    # use, so the same anime arrives as two URLs.
    merged = merge_provider_records(
        {"kitsu": _record(sources=["https://kitsu.io/anime/one-piece"])},
        {"sources": ["https://kitsu.app/anime/12"]},
    )
    assert merged["sources"] == ["https://kitsu.io/anime/one-piece"]


def test_numeric_url_survives_when_no_slug_rivals_it() -> None:
    # MAL and AniList identify every work numerically; dropping those would
    # empty `sources`.
    merged = merge_provider_records({"mal": _record()})
    assert merged["sources"] == [_MAL_URL]


def test_id_is_stable_across_differing_provider_coverage() -> None:
    full = merge_provider_records(
        {
            "mal": _record(),
            "kitsu": _record(sources=["https://kitsu.io/anime/one-piece"]),
        }
    )
    mal_only = merge_provider_records({"mal": _record()})
    assert full["id"] == mal_only["id"]


def test_id_ignores_slug_decoration() -> None:
    assert merged_anime_id([_MAL_URL]) == merged_anime_id(
        ["https://myanimelist.net/anime/21"]
    )


def test_id_requires_at_least_one_source() -> None:
    with pytest.raises(ValueError, match="Cannot derive an anime id"):
        merged_anime_id([])
