"""Unit tests for MalAnime and related anime models."""

import pytest
from enrichment.sources.mal.mal_models import (
    MalAnime,
    MalCompanyRef,
    MalRelatedEntry,
    MalThemeSong,
)
from pydantic import ValidationError

# =============================================================================
# MalCompanyRef
# =============================================================================


def test_mal_company_ref_fields() -> None:
    ref = MalCompanyRef(
        name="Toei Animation", source="https://myanimelist.net/anime/producer/18"
    )
    assert ref.name == "Toei Animation"
    assert ref.source == "https://myanimelist.net/anime/producer/18"


# =============================================================================
# MalRelatedEntry
# =============================================================================


def test_mal_related_entry_anime() -> None:
    entry = MalRelatedEntry(
        relation="Side Story",
        title="One Piece Film: Gold",
        source="https://myanimelist.net/anime/28933",
        entry_type="Movie",
        is_anime=True,
    )
    assert entry.relation == "Side Story"
    assert entry.is_anime is True


def test_mal_related_entry_manga() -> None:
    entry = MalRelatedEntry(
        relation="Adaptation",
        title="One Piece",
        source="https://myanimelist.net/manga/103",
        entry_type="Manga",
        is_anime=False,
    )
    assert entry.is_anime is False


def test_mal_related_entry_optional_fields() -> None:
    entry = MalRelatedEntry(
        relation="Other",
        title="Some Title",
        source="https://myanimelist.net/anime/99999",
        is_anime=True,
    )
    assert entry.entry_type is None


# =============================================================================
# MalThemeSong
# =============================================================================


def test_mal_theme_song_full() -> None:
    song = MalThemeSong(title="We Are!", artist="Kitadani Hiroshi")
    assert song.title == "We Are!"
    assert song.artist == "Kitadani Hiroshi"
    assert song.episodes == []


def test_mal_theme_song_no_artist() -> None:
    song = MalThemeSong(title="We Are!")
    assert song.artist is None


# =============================================================================
# MalAnime
# =============================================================================


def test_mal_scraped_anime_minimal() -> None:
    """Minimal required fields only."""
    anime = MalAnime(
        source="https://myanimelist.net/anime/21",
        title="One Piece",
    )
    assert anime.title == "One Piece"
    assert anime.synonyms == []
    assert anime.genres == []
    assert anime.related_entries == []
    assert anime.picture_urls == []


def test_mal_scraped_anime_extra_field_rejected() -> None:
    """extra='forbid' rejects unknown fields."""
    with pytest.raises(ValidationError):
        MalAnime(
            source="...",
            title="One Piece",
            unknown_field="value",  # type: ignore[call-arg]
        )


def test_mal_scraped_anime_full() -> None:
    anime = MalAnime(
        source="https://myanimelist.net/anime/21",
        title="One Piece",
        title_english="One Piece",
        title_japanese="ワンピース",
        type="TV",
        status="Currently Airing",
        score=8.7,
        scored_by=2644378,
        rank=54,
        popularity=17,
        episode_count=1122,
        year=1999,
        season="fall",
        genres=["Action", "Adventure"],
        studios=[MalCompanyRef(name="Toei Animation", source="...")],
    )
    assert anime.score == 8.7
    assert anime.rank == 54
    assert len(anime.studios) == 1
    assert anime.studios[0].name == "Toei Animation"


def test_mal_scraped_anime_related_entries() -> None:
    entry = MalRelatedEntry(
        relation="Side Story",
        title="One Piece Film: Gold",
        source="https://myanimelist.net/anime/28933",
        entry_type="Movie",
        is_anime=True,
    )
    anime = MalAnime(source="...", title="One Piece", related_entries=[entry])
    assert len(anime.related_entries) == 1
    assert anime.related_entries[0].relation == "Side Story"
    assert anime.related_entries[0].is_anime is True


def test_mal_scraped_anime_images_dict() -> None:
    anime = MalAnime(
        source="...",
        title="One Piece",
        images={"jpg": "https://cdn.myanimelist.net/images/anime/6/73245.jpg"},
    )
    assert "jpg" in anime.images


def test_mal_scraped_anime_external_and_streaming_links() -> None:
    from enrichment.sources.mal.mal_models import MalExternalLink

    anime = MalAnime(
        source="...",
        title="One Piece",
        external_sources=[
            MalExternalLink(name="Official Site", source="https://example.com")
        ],
        streaming=[
            MalExternalLink(
                name="Crunchyroll", source="https://crunchyroll.com/one-piece"
            )
        ],
    )
    assert len(anime.external_sources) == 1
    assert anime.external_sources[0].name == "Official Site"
    assert len(anime.streaming) == 1
    assert anime.streaming[0].name == "Crunchyroll"


def _scraped(**fields: object) -> MalAnime:
    return MalAnime.model_validate(
        {"source": "https://myanimelist.net/anime/21", "title": "One Piece", **fields}
    )


def test_mal_anime_title_is_stripped() -> None:
    assert _scraped(title="  One Piece ").title == "One Piece"


def test_mal_anime_synonyms_text_split_on_commas() -> None:
    assert _scraped(synonyms="OP, Wan Pisu").synonyms == ["OP", "Wan Pisu"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1,155", 1155), ("12", 12), ("Unknown", None), ("?", None), (None, None)],
)
def test_mal_anime_episode_count_text_parses_to_number(
    raw: str | None, expected: int | None
) -> None:
    assert _scraped(episode_count=raw).episode_count == expected


def test_mal_anime_duration_text_parses_to_seconds() -> None:
    assert _scraped(duration="1 hr. 30 min.").duration == 5400


@pytest.mark.parametrize(("raw", "expected"), [("8.73", 8.73), ("N/A", None)])
def test_mal_anime_score_text_parses_to_number(
    raw: str, expected: float | None
) -> None:
    assert _scraped(score=raw).score == expected


def test_mal_anime_counts_in_millions_parse_to_numbers() -> None:
    anime = _scraped(
        scored_by="1,234,567",
        members="2,644,378",
        favorites="256,728",
        popularity="#17",
    )

    assert (anime.scored_by, anime.members, anime.favorites, anime.popularity) == (
        1234567,
        2644378,
        256728,
        17,
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('<div data-id="info2"><span>Ranked:</span> #4536<sup>2</sup></div>', 4536),
        ("Ranked: #1,234", 1234),
        ('<div data-id="info2"><span>Ranked:</span> N/A</div>', None),
    ],
)
def test_mal_anime_rank_html_parses_to_rank(raw: str, expected: int | None) -> None:
    assert _scraped(rank=raw).rank == expected


def test_mal_anime_broadcast_text_parses_to_day_time_and_zone() -> None:
    anime = _scraped(broadcast="Sundays at 23:15 (JST)")

    assert (anime.broadcast_day, anime.broadcast_time, anime.broadcast_timezone) == (
        "Sundays",
        "23:15",
        "JST",
    )


def test_mal_anime_aired_and_premiered_give_dates_season_and_year() -> None:
    anime = _scraped(aired="Oct 20, 1999 to ?", premiered="Fall 1999")

    assert (
        anime.aired_from,
        anime.aired_to,
        anime.season,
        anime.year,
        anime.month,
    ) == (
        "1999-10-20",
        None,
        "fall",
        1999,
        None,
    )


def test_mal_anime_aired_without_day_gives_month_and_year() -> None:
    anime = _scraped(aired="Apr 2026 to ?", premiered=None)

    assert (anime.aired_from, anime.year, anime.month, anime.season) == (
        None,
        2026,
        "April",
        None,
    )


def test_mal_anime_zero_rank_and_popularity_mean_not_ranked() -> None:
    anime = _scraped(
        rank="<span>Ranked:</span> #0", popularity="#0", members="0", favorites="0"
    )

    assert (anime.rank, anime.popularity, anime.members, anime.favorites) == (
        None,
        None,
        0,
        0,
    )
