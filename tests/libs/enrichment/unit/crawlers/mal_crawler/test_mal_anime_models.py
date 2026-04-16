"""Unit tests for MalAnime and related anime models."""

import pytest
from pydantic import ValidationError

from enrichment.crawlers.mal_crawler.mal_models import (
    MalCompanyRef,
    MalExternalLink,
    MalRelatedEntry,
    MalAnime,
    MalThemeSong,
)


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
    from enrichment.crawlers.mal_crawler.mal_models import MalExternalLink

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
