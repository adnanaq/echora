"""Tests for cross-provider relationship consolidation.

Focus is the identity boundary: which entries fuse into one work and which stay
apart. Every franchise title shares a long prefix, so the interesting cases are
the near misses where a high similarity score and a distinct work coincide.
"""

import json

import pytest
from enrichment.pipeline.relationship_merger import (
    load_agent_providers,
    merge_agent_relationships,
    merge_relation_field,
    titles_match,
    validate,
)


class _StubResolver:
    """Resolver over a fixed URL-to-work-id map."""

    def __init__(self, mapping: dict[str, str]) -> None:
        self._mapping = mapping

    def resolve(self, url: str) -> str | None:
        return self._mapping.get(url)


def _entry(title: str, url: str, **extra: object) -> dict[str, object]:
    return {"title": title, "type": "MOVIE", "sources": [url], **extra}


def _url(anime_id: int) -> str:
    return f"https://myanimelist.net/anime/{anime_id}"


def _count(merged: dict[str, list[dict[str, object]]]) -> int:
    return sum(len(entries) for entries in merged.values())


@pytest.mark.parametrize(
    ("left", "right"),
    [
        # One differing token is the entire meaning, despite a 0.91 score.
        ("One Piece: Episode of Sabo", "One Piece: Episode of Skypiea"),
        # 0.94 score; "ace" and "law" are the only distinguishing tokens.
        ("One Piece Ace's Story (Light Novel)", "One Piece Law's Story (Light Novel)"),
        # A qualifier on one side only marks a companion work, not the same one.
        (
            "One Piece: Adventures in Alabasta",
            "One Piece: Adventures in Alabasta Prologue",
        ),
        # A differing trailing sequence number vetoes regardless of score.
        ("Collabo Special", "Collabo Special 2"),
        # Nothing left to compare once punctuation is stripped.
        ("", "One Piece"),
        ("---", "One Piece"),
    ],
)
def test_titles_match_rejects_distinct_works(left: str, right: str) -> None:
    assert titles_match(left, right) is False


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("One Piece Movie 01", "One Piece Movie 1"),  # zero padding
        ("One Piece: Episode of Luffy", "One Piece Episode of Luffy"),  # punctuation
        ("Mamore! Mahou no Pump", "mamore-mahou-no-pump"),  # slug form
        ("Kyutai Panic Adventure!", "Kyuutai Panic Adventure"),  # long vowel
        ("CHOPPER's", "choppers"),  # apostrophe splits a token
        (
            "One Piece Film: Gold Episode 0 - 711 ver.",
            "one-piece-film-gold-episode-0-711ver",
        ),
        ("ONE PIECE 東の海編", "ONE PIECE　東の海編"),  # full-width space between kanji
    ],
)
def test_titles_match_accepts_spelling_variants(left: str, right: str) -> None:
    assert titles_match(left, right) is True


def test_same_title_different_works_stay_apart() -> None:
    """A remake sharing its original's title must not fuse with it.

    Exact-key grouping unions on the normalized title with no resolver check,
    so the title key is withheld once the resolver has placed the entry.
    """
    per_provider = {
        "mal": {
            "side_story": [_entry("Same Title", "https://myanimelist.net/anime/1")]
        },
        "anidb": {"side_story": [_entry("Same Title", "https://anidb.net/anime/2")]},
    }
    resolver = _StubResolver(
        {
            "https://myanimelist.net/anime/1": "work-a",
            "https://anidb.net/anime/2": "work-b",
        }
    )
    merged = merge_relation_field(
        per_provider, is_source_material=False, resolver=resolver
    )
    assert _count(merged) == 2


def test_same_work_id_fuses_across_providers() -> None:
    per_provider = {
        "mal": {
            "side_story": [_entry("Romaji Title", "https://myanimelist.net/anime/1")]
        },
        "anidb": {"side_story": [_entry("English Title", "https://anidb.net/anime/2")]},
    }
    resolver = _StubResolver(
        {
            "https://myanimelist.net/anime/1": "work-a",
            "https://anidb.net/anime/2": "work-a",
        }
    )
    merged = merge_relation_field(
        per_provider, is_source_material=False, resolver=resolver
    )
    assert _count(merged) == 1
    entry = next(iter(merged.values()))[0]
    assert len(entry["sources"]) == 2


def test_unresolved_entry_still_reaches_a_resolved_one() -> None:
    """Withholding the title key must not strand entries the resolver misses."""
    per_provider = {
        "mal": {
            "side_story": [_entry("Shared Title", "https://myanimelist.net/anime/1")]
        },
        "animeschedule": {
            "side_story": [_entry("Shared Title", "https://animeschedule.net/anime/x")]
        },
    }
    resolver = _StubResolver({"https://myanimelist.net/anime/1": "work-a"})
    merged = merge_relation_field(
        per_provider, is_source_material=False, resolver=resolver
    )
    assert _count(merged) == 1


def test_concrete_value_beats_sentinel_regardless_of_rank() -> None:
    per_provider = {
        "mal": {
            "side_story": [
                {
                    "title": "Work",
                    "type": "UNKNOWN",
                    "sources": ["https://myanimelist.net/anime/1"],
                }
            ]
        },
        "kitsu": {
            "side_story": [
                {
                    "title": "Work",
                    "type": "MOVIE",
                    "sources": ["https://myanimelist.net/anime/1"],
                }
            ]
        },
    }
    merged = merge_relation_field(per_provider, is_source_material=False)
    assert next(iter(merged.values()))[0]["type"] == "MOVIE"


def test_merged_output_validates_against_the_models() -> None:
    per_provider = {
        "mal": {
            "side_story": [
                _entry(
                    "Work",
                    "https://myanimelist.net/anime/1",
                    year=1999,
                    episode_count=12,
                    status="FINISHED",
                )
            ]
        }
    }
    merged = merge_relation_field(per_provider, is_source_material=False)
    entry = next(iter(merged.values()))[0]
    assert entry["year"] == 1999
    assert entry["episode_count"] == 12
    assert entry["status"] == "FINISHED"
    assert validate({"related_anime": merged, "related_source_material": {}}) == []


def test_untitled_entry_is_dropped_not_emitted() -> None:
    """RelatedAnime.title is required, so an untitled group cannot be shipped."""
    per_provider = {
        "mal": {"side_story": [{"sources": ["https://myanimelist.net/anime/1"]}]}
    }
    merged = merge_relation_field(per_provider, is_source_material=False)
    assert _count(merged) == 0
    assert validate({"related_anime": merged, "related_source_material": {}}) == []


def test_missing_type_falls_back_to_the_unknown_sentinel() -> None:
    per_provider = {
        "mal": {
            "side_story": [
                {"title": "Work", "sources": ["https://myanimelist.net/anime/1"]}
            ]
        }
    }
    merged = merge_relation_field(per_provider, is_source_material=False)
    assert next(iter(merged.values()))[0]["type"] == "UNKNOWN"
    assert validate({"related_anime": merged, "related_source_material": {}}) == []


def test_validate_rejects_a_non_canonical_relation_key() -> None:
    """The relation enums fold anything unknown into OTHER via _missing_.

    Constructing one therefore never raises, so the check must compare values.
    """
    merged = {
        "related_anime": {
            "TOTALLY_BOGUS": [
                {"title": "X", "type": "TV", "sources": ["u"], "images": []}
            ]
        },
        "related_source_material": {
            "ALSO_BOGUS": [
                {"title": "Y", "type": "MANGA", "sources": ["v"], "images": []}
            ]
        },
    }
    errors = validate(merged)
    assert len(errors) == 2
    assert any("bad AnimeRelationType" in e for e in errors)
    assert any("bad SourceMaterialRelationType" in e for e in errors)


def test_load_agent_providers_reads_the_latest_record(tmp_path) -> None:
    """Helpers append on re-run, so the first line is the stale earlier fetch."""
    (tmp_path / "mal_anime.jsonl").write_text(
        '{"title": "stale", "entity_type": "anime"}\n'
        '{"title": "fresh", "entity_type": "anime"}\n'
    )
    assert load_agent_providers(tmp_path)["mal"]["title"] == "fresh"


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('{"title": "a"}\n\n\n', "a"),
        ('{"title": "a"}\n\n{"title": "b"}\n', "b"),
        ('{"title": "a"}\n{"title": "b"}', "b"),
    ],
)
def test_load_agent_providers_ignores_blank_lines(
    tmp_path, content: str, expected: str
) -> None:
    (tmp_path / "mal_anime.jsonl").write_text(content)
    assert load_agent_providers(tmp_path)["mal"]["title"] == expected


@pytest.mark.parametrize("content", ["", "   \n\n"])
def test_load_agent_providers_skips_empty_file(tmp_path, content: str) -> None:
    (tmp_path / "mal_anime.jsonl").write_text(content)
    assert load_agent_providers(tmp_path) == {}


def test_sentinel_type_survives_when_no_source_names_a_real_one() -> None:
    per_provider = {
        "mal": {
            "side_story": [
                {
                    "title": "Work",
                    "type": "UNKNOWN",
                    "sources": ["https://myanimelist.net/anime/1"],
                }
            ]
        }
    }
    merged = merge_relation_field(per_provider, is_source_material=False)
    assert next(iter(merged.values()))[0]["type"] == "UNKNOWN"


def test_the_sentinel_relation_is_kept_when_it_is_all_anyone_reported() -> None:
    per_provider = {
        "mal": {"OTHER": [_entry("Work", "https://myanimelist.net/anime/1")]}
    }
    merged = merge_relation_field(per_provider, is_source_material=False)
    assert list(merged) == ["OTHER"]


def test_three_spellings_of_one_title_fuse_into_a_single_work() -> None:
    per_provider = {
        "mal": {"side_story": [_entry("CHOPPER's Big Adventure", _url(1))]},
        "anidb": {"side_story": [_entry("choppers big adventure", _url(2))]},
        "kitsu": {"side_story": [_entry("Chopper's Big Adventures", _url(3))]},
    }
    merged = merge_relation_field(per_provider, is_source_material=False)
    assert _count(merged) == 1
    assert len(next(iter(merged.values()))[0]["sources"]) == 3


def test_merge_agent_relationships_reads_both_fields_from_a_directory(
    tmp_path,
) -> None:
    (tmp_path / "mal_anime.jsonl").write_text(
        json.dumps(
            {
                "related_anime": {
                    "side_story": [_entry("Side Story", _url(1))],
                },
                "related_source_material": {
                    "adaptation": [
                        {
                            "title": "The Manga",
                            "type": "MANGA",
                            "sources": ["https://myanimelist.net/manga/13"],
                        }
                    ]
                },
            }
        )
        + "\n"
    )
    merged = merge_agent_relationships(tmp_path)
    assert _count(merged["related_anime"]) == 1
    assert _count(merged["related_source_material"]) == 1


def test_validate_reports_an_entry_the_models_reject() -> None:
    merged = {
        "related_anime": {
            "SIDE_STORY": [{"title": "X", "type": "TV", "sources": "not-a-list"}]
        },
        "related_source_material": {
            "ADAPTATION": [{"title": "Y", "type": "MANGA", "sources": "not-a-list"}]
        },
    }
    errors = validate(merged)
    assert len(errors) == 2
    assert any("SIDE_STORY/'X'" in error for error in errors)
    assert any("ADAPTATION/'Y'" in error for error in errors)
