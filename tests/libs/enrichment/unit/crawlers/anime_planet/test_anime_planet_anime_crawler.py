"""Unit tests for the refactored anime-planet crawler.

Tests cover:
- Pure utility functions (_normalize_anime_url, _extract_slug_from_url, _extract_json_ld)
- Raw-data helper functions (_build_related_anime_entries, _build_related_manga_entries)
- Integration flow mocked at crawl_single_url (not AsyncWebCrawler)
- CLI main() function
"""

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from common.models.anime import AnimeSeason
from common.utils.datetime_utils import determine_anime_season
from enrichment.crawlers.anime_planet.anime_planet_anime_crawler import (
    _build_anime_from_raw,
    _build_related_anime_entries,
    _build_related_manga_entries,
    _extract_json_ld,
    _extract_slug_from_url,
    _normalize_anime_url,
    _parse_aggregate_rating,
    _parse_alt_title,
    _parse_rank,
    _parse_season,
    fetch_animeplanet_anime,
)
from enrichment.crawlers.anime_planet.anime_planet_models import AnimePlanetAnime
from enrichment.crawlers.anime_planet.animeplanet_mapper import anime_from_animeplanet

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# _normalize_anime_url
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "identifier, expected",
    [
        ("dandadan", "https://www.anime-planet.com/anime/dandadan"),
        ("/anime/dandadan", "https://www.anime-planet.com/anime/dandadan"),
        (
            "https://www.anime-planet.com/anime/dandadan",
            "https://www.anime-planet.com/anime/dandadan",
        ),
        ("anime/one-piece", "https://www.anime-planet.com/anime/one-piece"),
        (
            "https://anime-planet.com/anime/one-piece",
            "https://www.anime-planet.com/anime/one-piece",
        ),
    ],
)
def test_normalize_anime_url_valid(identifier, expected):
    assert _normalize_anime_url(identifier) == expected


def test_normalize_anime_url_invalid():
    with pytest.raises(ValueError, match="Invalid URL"):
        _normalize_anime_url("https://www.google.com/anime/dandadan")


# ---------------------------------------------------------------------------
# _extract_slug_from_url
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url, expected_slug",
    [
        ("https://www.anime-planet.com/anime/dandadan", "dandadan"),
        ("https://www.anime-planet.com/anime/one-piece?foo=bar", "one-piece"),
    ],
)
def test_extract_slug_from_url_valid(url, expected_slug):
    assert _extract_slug_from_url(url) == expected_slug


def test_extract_slug_from_url_invalid():
    with pytest.raises(ValueError, match="Could not extract slug"):
        _extract_slug_from_url("https://www.anime-planet.com/manga/dandadan")


# ---------------------------------------------------------------------------
# _extract_json_ld
# ---------------------------------------------------------------------------


def test_extract_json_ld_valid():
    html = """
    <html><script type="application/ld+json">
    {
        "@context": "https://schema.org",
        "@type": "TVSeries",
        "name": "Dandadan",
        "description": "This is a &lt;b&gt;great&lt;/b&gt; show.",
        "image": "https://www.anime-planet.comhttps://s4.anilist.co/file/anilistcdn/media/anime/cover/large/bx158822-DbJ2c82s35jA.jpg"
    }
    </script></html>
    """
    json_ld = _extract_json_ld(html)
    assert json_ld is not None
    assert json_ld["name"] == "Dandadan"
    assert json_ld["description"] == "This is a <b>great</b> show."
    assert "anime-planet.comhttps" not in json_ld["image"]


def test_extract_json_ld_malformed_image_url():
    html = """
    <html><script type="application/ld+json">
    {
        "image": "https://www.anime-planet.comhttps://s4.anilist.co/file/anilistcdn/media/anime/cover/large/bx158822-DbJ2c82s35jA.jpg"
    }
    </script></html>
    """
    json_ld = _extract_json_ld(html)
    assert json_ld is not None
    assert "anime-planet.comhttps" not in json_ld["image"]


def test_extract_json_ld_no_description():
    html = """
    <html><script type="application/ld+json">
    {"name": "Dandadan"}
    </script></html>
    """
    json_ld = _extract_json_ld(html)
    assert json_ld is not None
    assert "description" not in json_ld


@pytest.mark.parametrize(
    "html_input",
    [
        "<html></html>",
        '<html><script type="application/ld+json">{invalid json}</script></html>',
    ],
)
def test_extract_json_ld_invalid(html_input):
    assert _extract_json_ld(html_input) is None


# ---------------------------------------------------------------------------
# _parse_aggregate_rating
# ---------------------------------------------------------------------------


def test_parse_aggregate_rating_none_input():
    assert _parse_aggregate_rating(None) is None


def test_parse_aggregate_rating_empty_dict():
    assert _parse_aggregate_rating({}) is None


def test_parse_aggregate_rating_valid():
    result = _parse_aggregate_rating({"ratingValue": "4.5", "ratingCount": "1000"})
    assert result is not None
    assert result.rating_value == pytest.approx(4.5)
    assert result.rating_count == 1000


def test_parse_aggregate_rating_invalid_value_types():
    """Non-numeric ratingValue/ratingCount are silently ignored."""
    result = _parse_aggregate_rating({"ratingValue": "not-a-float", "ratingCount": "not-an-int"})
    assert result is None


def test_parse_aggregate_rating_both_none_after_parse():
    """Explicit None values produce no rating model."""
    result = _parse_aggregate_rating({"ratingValue": None, "ratingCount": None})
    assert result is None


def test_parse_aggregate_rating_partial_valid():
    """Only ratingValue present — still returns a model."""
    result = _parse_aggregate_rating({"ratingValue": "3.2"})
    assert result is not None
    assert result.rating_value == pytest.approx(3.2)
    assert result.rating_count is None


# ---------------------------------------------------------------------------
# _parse_season
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "season_url, expected",
    [
        (None, None),
        ("", None),
        ("/anime/seasons/fall-2024", "fall"),
        ("/anime/seasons/winter-1999", "winter"),
        ("/anime/seasons/spring-2023", "spring"),
        ("/anime/seasons/summer-2020", "summer"),
        # slug with only one segment (no year suffix)
        ("/anime/seasons/fall", "fall"),
        # no /seasons/ segment → no match
        ("/anime/not-a-season-url", None),
        # full URL form
        ("https://www.anime-planet.com/anime/seasons/fall-2024", "fall"),
    ],
)
def test_parse_season(season_url, expected):
    assert _parse_season(season_url) == expected


# ---------------------------------------------------------------------------
# _parse_rank
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rank_text, expected",
    [
        (None, None),
        ("", None),
        ("Rank #157", 157),
        ("Rank #1", 1),
        ("Rank #99999", 99999),
        # text without '#' digit → no match
        ("no hash here", None),
    ],
)
def test_parse_rank(rank_text, expected):
    assert _parse_rank(rank_text) == expected


# ---------------------------------------------------------------------------
# _parse_alt_title
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "aka, expected",
    [
        (None, None),
        ("", None),
        ("   ", None),  # whitespace only → None
        ("Alt title: ダンダダン", "ダンダダン"),
        ("alt title: ワンピース", "ワンピース"),  # case-insensitive prefix strip
        ("ALT TITLE:  Bleach  ", "Bleach"),  # extra spaces stripped
        # No prefix → returned as-is
        ("ダンダダン", "ダンダダン"),
        ("Plain title", "Plain title"),
    ],
)
def test_parse_alt_title(aka, expected):
    assert _parse_alt_title(aka) == expected


# ---------------------------------------------------------------------------
# _build_related_anime_entries
# ---------------------------------------------------------------------------


def test_build_related_anime_entries_basic():
    raw = [
        {
            "url": "/anime/one-piece-film-red",
            "title": "One Piece Film: Red",
            "relation_subtype": "Sequel",
            "type": "Movie",
            "image": "https://example.com/red.jpg",
        },
        {"url": "", "title": "No URL"},              # skipped
        {"url": "/manga/dandadan", "title": "Wrong"}, # skipped (no /anime/ segment)
    ]
    entries = _build_related_anime_entries(raw)
    assert len(entries) == 1
    assert entries[0].slug == "one-piece-film-red"
    assert entries[0].title == "One Piece Film: Red"
    assert entries[0].relation_subtype == "Sequel"
    assert entries[0].type == "Movie"


def test_build_related_anime_entries_empty_subtype_becomes_none():
    raw = [{"url": "/anime/slug", "title": "Title", "relation_subtype": ""}]
    entries = _build_related_anime_entries(raw)
    assert entries[0].relation_subtype is None


def test_build_related_anime_entries_empty():
    assert _build_related_anime_entries([]) == []


@pytest.mark.parametrize(
    "raw_type, expected_type, expected_ep_count",
    [
        # Plain format strings (no episode count)
        ("Movie",          "Movie",       None),
        ("Web",            "Web",         None),
        ("",               None,          None),
        (None,             None,          None),
        # Format + episode count strings from fa-tv span
        ("OVA: 1 ep",      "OVA",         1),
        ("TV Special: 1 ep","TV Special",  1),
        ("TV Special: 9 ep","TV Special",  9),
        ("DVD Special: 5 ep","DVD Special",5),
        ("Web: 2 ep",      "Web",         2),
        ("Web: 12 ep",     "Web",         12),
        ("TV: 21 ep",      "TV",          21),
        ("Other: 1 ep",    "Other",       1),
        ("Other: 2 ep",    "Other",       2),
        ("Other: 3 ep",    "Other",       3),
        ("Other: 9 ep",    "Other",       9),
        # Multi-colon title edge case: split on FIRST colon only
        ("Music Video: 1 ep", "Music Video", 1),
    ],
)
def test_build_related_anime_entries_type_and_episode_parsing(
    raw_type, expected_type, expected_ep_count
):
    """type and episode_count are parsed from the fa-tv metadata span string."""
    raw = [{"url": "/anime/some-slug", "title": "Some Title", "type": raw_type}]
    entries = _build_related_anime_entries(raw)
    assert len(entries) == 1
    assert entries[0].type == expected_type
    assert entries[0].episode_count == expected_ep_count


def test_build_related_anime_entries_type_stripped_of_whitespace():
    """Whitespace around the type string is stripped before parsing."""
    raw = [{"url": "/anime/slug", "title": "T", "type": "  OVA: 1 ep  "}]
    entries = _build_related_anime_entries(raw)
    assert entries[0].type == "OVA"
    assert entries[0].episode_count == 1


# ---------------------------------------------------------------------------
# _build_related_manga_entries
# ---------------------------------------------------------------------------


def test_build_related_manga_entries_basic():
    raw = [
        {
            "url": "/manga/one-piece",
            "title": "One Piece Manga",
            "relation_subtype": "Original Manga",
        },
        {"url": "", "title": "No URL"},               # skipped
        {"url": "/anime/dandadan", "title": "Wrong"},  # skipped (no /manga/ segment)
    ]
    entries = _build_related_manga_entries(raw)
    assert len(entries) == 1
    assert entries[0].slug == "one-piece"
    assert entries[0].relation_subtype == "Original Manga"


def test_build_related_manga_entries_empty():
    assert _build_related_manga_entries([]) == []


@pytest.mark.parametrize(
    "vol_ch, expected_type, expected_volumes, expected_chapters",
    [
        # One Shot — format type with implicit 1 chapter
        ("One Shot",             "One Shot", None, 1),
        ("one shot",             "One Shot", None, 1),   # case-insensitive
        # Full vol + ch
        ("Vol: 114 - Ch: 1179+", None,       114,  1179),
        ("Vol: 1 - Ch: 3",       None,       1,    3),
        ("Vol: 8 - Ch: 141+",    None,       8,    141),
        ("Vol: 5 - Ch: 25",      None,       5,    25),
        # Vol only
        ("Vol: 1",               None,       1,    None),
        ("Vol: 2",               None,       2,    None),
        # Ch only
        ("Ch: 19",               None,       None, 19),
        # Empty / missing — no data available
        ("",                     None,       None, None),
        (None,                   None,       None, None),
        # Date bleed-through guard: "- ?" must NOT be treated as vol/ch
        ("- ?",                  None,       None, None),
    ],
)
def test_build_related_manga_entries_vol_ch_parsing(
    vol_ch, expected_type, expected_volumes, expected_chapters
):
    """vol_ch field is parsed into type, volumes, and chapters."""
    raw = [{"url": "/manga/some-manga", "title": "Some Manga", "vol_ch": vol_ch}]
    entries = _build_related_manga_entries(raw)
    assert len(entries) == 1
    assert entries[0].type == expected_type
    assert entries[0].volumes == expected_volumes
    assert entries[0].chapters == expected_chapters


def test_build_related_manga_entries_vol_ch_stripped_of_whitespace():
    raw = [{"url": "/manga/slug", "title": "T", "vol_ch": "  Vol: 3 - Ch: 12  "}]
    entries = _build_related_manga_entries(raw)
    assert entries[0].volumes == 3
    assert entries[0].chapters == 12


# ---------------------------------------------------------------------------
# Mapper: related entry field passthrough
# ---------------------------------------------------------------------------


def _make_anime_with_related(
    related_anime: list | None = None,
    related_manga: list | None = None,
) -> "AnimePlanetAnime":
    """Construct a minimal AnimePlanetAnime with custom related entries."""
    from enrichment.crawlers.anime_planet.anime_planet_models import (
        AnimePlanetMangaEntry,
        AnimePlanetRelatedEntry,
    )

    return AnimePlanetAnime(
        name="Test Anime",
        slug="test-anime",
        schema_type="TVSeries",
        related_anime=related_anime or [],
        related_anime_other=[],
        related_manga=related_manga or [],
    )


def test_mapper_related_anime_episode_count_passthrough():
    """episode_count from AnimePlanetRelatedEntry reaches RelatedAnime in canonical output."""
    from enrichment.crawlers.anime_planet.anime_planet_models import AnimePlanetRelatedEntry

    entry = AnimePlanetRelatedEntry(
        url="/anime/special",
        slug="special",
        title="Test Special",
        relation_subtype="Same Franchise",
        type="TV Special",
        episode_count=3,
    )
    anime = _make_anime_with_related(related_anime=[entry])
    data = anime_from_animeplanet(anime)

    side_stories = data["related_anime"].get("SIDE_STORY", [])
    assert any(e["title"] == "Test Special" for e in side_stories)
    match = next(e for e in side_stories if e["title"] == "Test Special")
    assert match["episode_count"] == 3


def test_mapper_related_anime_no_episode_count_is_absent():
    """episode_count is omitted from output when not available (exclude_none=True)."""
    from enrichment.crawlers.anime_planet.anime_planet_models import AnimePlanetRelatedEntry

    entry = AnimePlanetRelatedEntry(
        url="/anime/film",
        slug="film",
        title="Test Movie",
        relation_subtype="Same Franchise",
        type="Movie",
        episode_count=None,
    )
    anime = _make_anime_with_related(related_anime=[entry])
    data = anime_from_animeplanet(anime)

    side_stories = data["related_anime"].get("SIDE_STORY", [])
    match = next(e for e in side_stories if e["title"] == "Test Movie")
    assert "episode_count" not in match


def test_mapper_related_source_material_volumes_and_chapters():
    """volumes and chapters from AnimePlanetMangaEntry reach RelatedSourceMaterial."""
    from enrichment.crawlers.anime_planet.anime_planet_models import AnimePlanetMangaEntry

    entry = AnimePlanetMangaEntry(
        url="/manga/some-manga",
        slug="some-manga",
        title="Some Manga",
        relation_subtype="Original Manga",
        volumes=7,
        chapters=62,
    )
    anime = _make_anime_with_related(related_manga=[entry])
    data = anime_from_animeplanet(anime)

    all_manga = [
        e
        for entries in data["related_source_material"].values()
        for e in entries
    ]
    match = next(e for e in all_manga if e["title"] == "Some Manga")
    assert match["volumes"] == 7
    assert match["chapters"] == 62


def test_mapper_manga_type_unknown_when_not_exposed():
    """Manga with no type from AP resolves to UNKNOWN — we don't assume."""
    from enrichment.crawlers.anime_planet.anime_planet_models import AnimePlanetMangaEntry

    entry = AnimePlanetMangaEntry(
        url="/manga/plain-manga",
        slug="plain-manga",
        title="Plain Manga",
        volumes=3,
        chapters=20,
    )
    anime = _make_anime_with_related(related_manga=[entry])
    data = anime_from_animeplanet(anime)

    all_manga = [e for entries in data["related_source_material"].values() for e in entries]
    match = next(e for e in all_manga if e["title"] == "Plain Manga")
    assert match["type"] == "UNKNOWN"


def test_mapper_manga_type_one_shot():
    """One Shot manga entries map to SourceMaterialType.ONE_SHOT."""
    from enrichment.crawlers.anime_planet.anime_planet_models import AnimePlanetMangaEntry

    entry = AnimePlanetMangaEntry(
        url="/manga/one-shot-manga",
        slug="one-shot-manga",
        title="Romance Dawn",
        type="One Shot",
        chapters=1,
    )
    anime = _make_anime_with_related(related_manga=[entry])
    data = anime_from_animeplanet(anime)

    all_manga = [e for entries in data["related_source_material"].values() for e in entries]
    match = next(e for e in all_manga if e["title"] == "Romance Dawn")
    assert match["type"] == "ONE SHOT"


def test_mapper_manga_type_light_novel_title_still_unknown():
    """Title containing '(Light Novel)' does not affect type — we don't infer from titles."""
    from enrichment.crawlers.anime_planet.anime_planet_models import AnimePlanetMangaEntry

    entry = AnimePlanetMangaEntry(
        url="/manga/ln-entry",
        slug="ln-entry",
        title="Some Story (Light Novel)",
        volumes=2,
        chapters=8,
    )
    anime = _make_anime_with_related(related_manga=[entry])
    data = anime_from_animeplanet(anime)

    all_manga = [e for entries in data["related_source_material"].values() for e in entries]
    match = next(e for e in all_manga if "Light Novel" in e["title"])
    assert match["type"] == "UNKNOWN"


def test_mapper_manga_vol_ch_absent_when_not_available():
    """volumes and chapters are omitted from output when AP exposes no count data."""
    from enrichment.crawlers.anime_planet.anime_planet_models import AnimePlanetMangaEntry

    entry = AnimePlanetMangaEntry(
        url="/manga/no-count",
        slug="no-count",
        title="No Count Manga",
    )
    anime = _make_anime_with_related(related_manga=[entry])
    data = anime_from_animeplanet(anime)

    all_manga = [e for entries in data["related_source_material"].values() for e in entries]
    match = next(e for e in all_manga if e["title"] == "No Count Manga")
    assert "volumes" not in match
    assert "chapters" not in match


# ---------------------------------------------------------------------------
# Season derivation (utility, independent of crawler)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "date_str, expected_season",
    [
        ("2024-01-15", AnimeSeason.WINTER),
        ("2024-04-20", AnimeSeason.SPRING),
        ("2024-08-01", AnimeSeason.SUMMER),
        ("2024-11-30", AnimeSeason.FALL),
        ("2024-12-01", AnimeSeason.WINTER),
        ("invalid-date", None),
        ("", None),
    ],
)
def test_determine_season_from_date(date_str, expected_season):
    assert determine_anime_season(date_str) == expected_season


# ---------------------------------------------------------------------------
# fetch_animeplanet_anime — integration tests mocked at crawl_single_url
# ---------------------------------------------------------------------------

_BASE_JSON_LD = {
    "@type": "TVSeries",
    "name": "Dandadan",
    "description": "A story of ghosts and aliens.",
    "url": "https://www.anime-planet.com/anime/dandadan",
    "startDate": "2024-10-01",
    "endDate": "2024-12-25",
    "numberOfEpisodes": 12,
    "genre": ["Action", "Comedy"],
    "aggregateRating": {"ratingValue": "4.5", "ratingCount": "1000"},
}

_BASE_XPATH = {
    "type_raw": "TV",
    "season_url": "/anime/seasons/fall-2024",
    "rank_text": "Rank #123",
    "aka": "Alt title: ダンダダン",
    "cover": "https://example.com/cover.jpg",
    "studios": [{"name": "Science SARU"}],
    "tags": [{"name": "Action"}, {"name": "Supernatural"}],
    "related_anime_raw": [
        {
            "url": "/anime/dandadan-season-2",
            "title": "Dandadan Season 2",
            "relation_subtype": "Sequel",
            "type": "TV: 12 ep",
            "image": None,
        }
    ],
    "related_anime_other_raw": [],
    "related_manga_raw": [
        {
            "url": "/manga/dandadan",
            "title": "Dandadan Manga",
            "relation_subtype": "Original Manga",
            "vol_ch": "Vol: 15 - Ch: 170",
            "image": None,
        }
    ],
}


def _make_crawl_result(xpath_data: dict, json_ld: dict) -> dict:
    html = f'<html><script type="application/ld+json">{json.dumps(json_ld)}</script></html>'
    return {
        "success": True,
        "status_code": 200,
        "extracted_content": json.dumps([xpath_data]),
        "html": html,
        "error_message": None,
    }


@pytest.mark.usefixtures("mock_redis_cache_miss")
@patch("enrichment.crawlers.anime_planet.anime_planet_anime_crawler.crawl_single_url")
async def test_fetch_animeplanet_anime_success(mock_crawl):
    mock_crawl.return_value = _make_crawl_result(_BASE_XPATH, _BASE_JSON_LD)

    anime = await fetch_animeplanet_anime("https://www.anime-planet.com/anime/dandadan")

    assert anime is not None
    assert anime.name == "Dandadan"
    assert anime.start_date == "2024-10-01"
    assert anime.end_date == "2024-12-25"
    assert anime.number_of_episodes == 12
    assert anime.aggregate_rating is not None
    assert anime.aggregate_rating.rating_value == pytest.approx(4.5)
    assert anime.aggregate_rating.rating_count == 1000
    assert anime.rank == 123
    assert anime.alt_title == "ダンダダン"
    assert "Action" in anime.genres
    assert "Action" in anime.tags or "Supernatural" in anime.tags
    assert any(s == "Science SARU" for s in anime.studios)
    assert any(e.slug == "dandadan-season-2" for e in anime.related_anime)
    assert any(e.slug == "dandadan" for e in anime.related_manga)

    # Verify mapper output is correct
    data = anime_from_animeplanet(anime)
    assert data["title"] == "Dandadan"
    assert data["year"] == 2024
    assert data["season"] == "FALL"
    assert data["status"] == "FINISHED"
    assert data["episode_count"] == 12
    # Score: 4.5 × 2 = 9.0
    assert data["statistics"]["anime_planet"]["score"] == pytest.approx(9.0)
    assert data["statistics"]["anime_planet"]["scored_by"] == 1000
    assert data["statistics"]["anime_planet"]["rank"] == 123
    assert data["title_japanese"] == "ダンダダン"
    assert "Action" in data["tags"]
    assert "Supernatural" in data["tags"]
    assert any(p["name"] == "Science SARU" for p in data["producers"])
    assert "SEQUEL" in data["related_anime"]
    sequel = data["related_anime"]["SEQUEL"][0]
    assert sequel["type"] == "TV"
    assert sequel["episode_count"] == 12

    assert "ADAPTATION" in data["related_source_material"]
    manga = data["related_source_material"]["ADAPTATION"][0]
    assert manga["type"] == "UNKNOWN"
    assert manga["volumes"] == 15
    assert manga["chapters"] == 170


@pytest.mark.usefixtures("mock_redis_cache_miss")
@patch("enrichment.crawlers.anime_planet.anime_planet_anime_crawler.crawl_single_url")
async def test_fetch_animeplanet_anime_season_from_url(mock_crawl):
    """Season URL slug takes precedence over start_date derivation."""
    xpath = {**_BASE_XPATH, "season_url": "/anime/seasons/winter-2024"}
    mock_crawl.return_value = _make_crawl_result(xpath, _BASE_JSON_LD)

    anime = await fetch_animeplanet_anime("https://www.anime-planet.com/anime/dandadan")
    assert anime is not None
    data = anime_from_animeplanet(anime)
    assert data["season"] == "WINTER"


@pytest.mark.usefixtures("mock_redis_cache_miss")
@patch("enrichment.crawlers.anime_planet.anime_planet_anime_crawler.crawl_single_url")
async def test_fetch_animeplanet_anime_season_fallback_from_start_date(mock_crawl):
    """When season_url is absent, season is derived from start_date."""
    xpath = {**_BASE_XPATH, "season_url": None}
    json_ld = {**_BASE_JSON_LD, "startDate": "2024-04-05"}  # spring
    mock_crawl.return_value = _make_crawl_result(xpath, json_ld)

    anime = await fetch_animeplanet_anime("https://www.anime-planet.com/anime/dandadan")
    assert anime is not None
    data = anime_from_animeplanet(anime)
    assert data["season"] == "SPRING"


@pytest.mark.usefixtures("mock_redis_cache_miss")
@patch("enrichment.crawlers.anime_planet.anime_planet_anime_crawler.crawl_single_url")
async def test_fetch_animeplanet_anime_season_url_no_match_falls_back_to_date(mock_crawl):
    """Malformed season_url (no regex match) falls back to start_date derivation."""
    xpath = {**_BASE_XPATH, "season_url": "/anime/not-a-season-url"}
    json_ld = {**_BASE_JSON_LD, "startDate": "2024-07-10"}  # summer
    mock_crawl.return_value = _make_crawl_result(xpath, json_ld)

    anime = await fetch_animeplanet_anime("https://www.anime-planet.com/anime/dandadan")
    assert anime is not None
    data = anime_from_animeplanet(anime)
    assert data["season"] == "SUMMER"


@pytest.mark.usefixtures("mock_redis_cache_miss")
@patch("enrichment.crawlers.anime_planet.anime_planet_anime_crawler.crawl_single_url")
async def test_fetch_animeplanet_anime_no_json_ld_returns_none(mock_crawl):
    mock_crawl.return_value = {
        "success": True,
        "status_code": 200,
        "extracted_content": json.dumps([_BASE_XPATH]),
        "html": "<html></html>",
        "error_message": None,
    }

    data = await fetch_animeplanet_anime("https://www.anime-planet.com/anime/dandadan")
    assert data is None


@pytest.mark.usefixtures("mock_redis_cache_miss")
@patch("enrichment.crawlers.anime_planet.anime_planet_anime_crawler.crawl_single_url")
async def test_fetch_animeplanet_anime_404(mock_crawl):
    mock_crawl.return_value = {
        "success": False,
        "status_code": 404,
        "extracted_content": None,
        "html": "",
        "error_message": "Not found",
    }
    assert await fetch_animeplanet_anime("https://www.anime-planet.com/anime/nonexistent") is None


@pytest.mark.usefixtures("mock_redis_cache_miss")
@patch("enrichment.crawlers.anime_planet.anime_planet_anime_crawler.crawl_single_url")
async def test_fetch_animeplanet_anime_crawl_failure(mock_crawl):
    mock_crawl.return_value = {
        "success": False,
        "status_code": 200,
        "extracted_content": None,
        "html": "",
        "error_message": "Browser crashed",
    }
    assert await fetch_animeplanet_anime("https://www.anime-planet.com/anime/dandadan") is None


@pytest.mark.usefixtures("mock_redis_cache_miss")
@patch("enrichment.crawlers.anime_planet.anime_planet_anime_crawler.crawl_single_url")
async def test_fetch_animeplanet_anime_empty_extraction(mock_crawl):
    """Empty extracted_content list → None."""
    mock_crawl.return_value = {
        "success": True,
        "status_code": 200,
        "extracted_content": "[]",
        "html": "",
        "error_message": None,
    }
    assert await fetch_animeplanet_anime("https://www.anime-planet.com/anime/dandadan") is None


@pytest.mark.usefixtures("mock_redis_cache_miss")
@patch("enrichment.crawlers.anime_planet.anime_planet_anime_crawler.crawl_single_url")
async def test_fetch_animeplanet_anime_http_500(mock_crawl):
    """Non-200/non-404 HTTP status → None."""
    mock_crawl.return_value = {
        "success": True,
        "status_code": 500,
        "extracted_content": None,
        "html": "",
        "error_message": None,
    }
    assert await fetch_animeplanet_anime("https://www.anime-planet.com/anime/dandadan") is None


@pytest.mark.usefixtures("mock_redis_cache_miss")
@patch("enrichment.crawlers.anime_planet.anime_planet_anime_crawler._build_anime_from_raw")
@patch("enrichment.crawlers.anime_planet.anime_planet_anime_crawler.crawl_single_url")
async def test_fetch_animeplanet_anime_build_raises(mock_crawl, mock_build):
    """Exception in _build_anime_from_raw is caught and returns None."""
    mock_crawl.return_value = _make_crawl_result(_BASE_XPATH, _BASE_JSON_LD)
    mock_build.side_effect = ValueError("malformed data")
    assert await fetch_animeplanet_anime("https://www.anime-planet.com/anime/dandadan") is None


@pytest.mark.parametrize(
    "start_date, end_date, expected_status",
    [
        ("2024-01-01", "2024-03-31", "FINISHED"),
        ("1999-10-20", None, "ONGOING"),
        ("2099-01-01", None, "UPCOMING"),
        (None, None, "UNKNOWN"),
    ],
)
@pytest.mark.usefixtures("mock_redis_cache_miss")
@patch("enrichment.crawlers.anime_planet.anime_planet_anime_crawler.crawl_single_url")
async def test_fetch_anime_status_derivation(mock_crawl, start_date, end_date, expected_status):
    slug = f"status-test-{start_date}-{end_date}"
    json_ld = {**_BASE_JSON_LD, "startDate": start_date, "endDate": end_date}
    mock_crawl.return_value = _make_crawl_result(_BASE_XPATH, json_ld)

    anime = await fetch_animeplanet_anime(f"https://www.anime-planet.com/anime/{slug}")
    assert anime is not None
    data = anime_from_animeplanet(anime)
    assert data["status"] == expected_status


# ---------------------------------------------------------------------------
# fetch_animeplanet_anime — URL contract
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("mock_redis_cache_miss")
@patch(
    "enrichment.crawlers.anime_planet.anime_planet_anime_crawler._fetch_animeplanet_anime_data",
    new_callable=AsyncMock,
)
async def test_fetch_animeplanet_anime_extracts_slug_from_url_for_cache(mock_inner):
    """Crawler extracts the slug from the URL and uses it as the cache key."""
    mock_inner.return_value = None  # short-circuit; we only care about the call arg
    await fetch_animeplanet_anime("https://www.anime-planet.com/anime/one-piece")
    mock_inner.assert_called_once_with("one-piece")


@pytest.mark.usefixtures("mock_redis_cache_miss")
@patch("enrichment.crawlers.anime_planet.anime_planet_anime_crawler.crawl_single_url")
async def test_fetch_animeplanet_anime_accepts_non_www_url(mock_crawl):
    """Non-www AP URL works — slug is extracted by regex regardless of subdomain."""
    mock_crawl.return_value = _make_crawl_result(_BASE_XPATH, _BASE_JSON_LD)
    anime = await fetch_animeplanet_anime("https://anime-planet.com/anime/dandadan")
    assert anime is not None
    assert anime.slug == "dandadan"

