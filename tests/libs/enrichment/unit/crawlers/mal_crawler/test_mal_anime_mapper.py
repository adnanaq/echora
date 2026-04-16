"""Unit tests for mal_mapper.py — anime_from_mal value normalization."""

from enrichment.crawlers.mal_crawler.mal_models import (
    MalCompanyRef,
    MalRelatedEntry,
    MalAnime,
    MalThemeSong,
    MalTrailer,
)
from enrichment.crawlers.mal_crawler.mal_mapper import anime_from_mal


def _make_sample_anime() -> MalAnime:
    return MalAnime(
        source="https://myanimelist.net/anime/21",
        title="One Piece",
        title_english="One Piece",
        title_japanese="ワンピース",
        type="TV",
        status="Currently Airing",
        source_material="Manga",
        score=8.7,
        scored_by=2644378,
        rank=54,
        popularity=17,
        members=2644378,
        episode_count=1122,
        season="fall",
        year=1999,
        synopsis="The story of Monkey D. Luffy...",
        genres=["Action", "Adventure"],
        broadcast_day="Sundays",
        broadcast_time="23:15",
        broadcast_timezone="JST",
        studios=[
            MalCompanyRef(
                name="Toei Animation",
                source="https://myanimelist.net/anime/producer/18",
            )
        ],
        producers=[
            MalCompanyRef(
                name="Fuji TV", source="https://myanimelist.net/anime/producer/29"
            )
        ],
        aired_from="1999-10-20",
        duration=1440,
        opening_themes=[MalThemeSong(title="We Are!", artist="Kitadani Hiroshi")],
        related_entries=[
            MalRelatedEntry(
                relation="Side Story",
                title="One Piece Film: Gold",
                source="https://myanimelist.net/anime/28933",
                entry_type="Movie",
                is_anime=True,
            ),
        ],
    )


def test_anime_from_mal_title() -> None:
    result = anime_from_mal(_make_sample_anime())
    assert result["title"] == "One Piece"


def test_anime_from_mal_type_normalization() -> None:
    result = anime_from_mal(_make_sample_anime())
    assert result["type"] == "TV"


def test_anime_from_mal_status_normalization() -> None:
    result = anime_from_mal(_make_sample_anime())
    assert result["status"] == "ONGOING"


def test_anime_from_mal_source_material_normalization() -> None:
    result = anime_from_mal(_make_sample_anime())
    assert result["source_material"] == "MANGA"


def test_anime_from_mal_duration_passed_through() -> None:
    result = anime_from_mal(_make_sample_anime())
    assert result["duration"] == 1440


def test_anime_from_mal_statistics_built() -> None:
    result = anime_from_mal(_make_sample_anime())
    assert "statistics" in result
    assert "mal" in result["statistics"]
    mal_stats = result["statistics"]["mal"]
    assert mal_stats["score"] == 8.7
    assert mal_stats["popularity"] == 17


def test_anime_from_mal_season_uppercased() -> None:
    result = anime_from_mal(_make_sample_anime())
    assert result["season"] == "FALL"


def test_anime_from_mal_broadcast_built() -> None:
    result = anime_from_mal(_make_sample_anime())
    assert "broadcast" in result
    broadcast = result["broadcast"]
    assert broadcast["day"] == "Sundays"
    assert broadcast["time"] == "23:15"
    assert broadcast["timezone"] == "JST"


def test_anime_from_mal_studios_mapped() -> None:
    result = anime_from_mal(_make_sample_anime())
    assert "studios" in result
    assert result["studios"][0]["name"] == "Toei Animation"
    assert (
        "https://myanimelist.net/anime/producer/18" in result["studios"][0]["sources"]
    )


def test_anime_from_mal_related_anime_split() -> None:
    result = anime_from_mal(_make_sample_anime())
    assert "related_anime" in result
    assert result["related_anime"]["SIDE_STORY"][0]["title"] == "One Piece Film: Gold"


def test_anime_from_mal_opening_themes_parsed() -> None:
    result = anime_from_mal(_make_sample_anime())
    assert "opening_themes" in result
    assert result["opening_themes"][0]["title"] == "We Are!"


def test_anime_from_mal_field_names_valid() -> None:
    """All output keys should be valid Anime model field names."""
    from common.models.anime import Anime

    result = anime_from_mal(_make_sample_anime())
    valid_fields = set(Anime.model_fields.keys())
    for key in result:
        assert key in valid_fields, f"Mapper output key '{key}' not in Anime model"


def test_anime_from_mal_sources() -> None:
    sample = _make_sample_anime()
    result = anime_from_mal(sample)
    assert result["sources"] == [sample.source]


def test_anime_from_mal_trailer_watch_url_passed_through() -> None:
    """Watch URL from MalTrailer is passed through to TrailerEntry.source."""
    sample = _make_sample_anime()
    sample.trailer = MalTrailer(source="https://www.youtube.com/watch?v=gAX3Zj-JGE0")
    result = anime_from_mal(sample)
    assert len(result["trailers"]) == 1
    assert (
        result["trailers"][0]["source"] == "https://www.youtube.com/watch?v=gAX3Zj-JGE0"
    )


def test_anime_from_mal_trailer_thumbnail_passed_through() -> None:
    """Thumbnail from MalTrailer is passed through to TrailerEntry.thumbnail."""
    sample = _make_sample_anime()
    sample.trailer = MalTrailer(
        source="https://www.youtube.com/watch?v=gAX3Zj-JGE0",
        thumbnail="https://img.youtube.com/vi/gAX3Zj-JGE0/maxresdefault.jpg",
    )
    result = anime_from_mal(sample)
    assert (
        result["trailers"][0]["thumbnail"]
        == "https://img.youtube.com/vi/gAX3Zj-JGE0/maxresdefault.jpg"
    )


def test_anime_from_mal_trailer_title_passed_through() -> None:
    """Title from MalTrailer is passed through to TrailerEntry.title."""
    sample = _make_sample_anime()
    sample.trailer = MalTrailer(
        source="https://www.youtube.com/watch?v=gAX3Zj-JGE0", title="PV 1"
    )
    result = anime_from_mal(sample)
    assert result["trailers"][0]["title"] == "PV 1"


def test_anime_from_mal_trailer_none_when_absent() -> None:
    """No trailer entry when MalAnime.trailer is None."""
    sample = _make_sample_anime()
    sample.trailer = None
    result = anime_from_mal(sample)
    assert result["trailers"] == []
