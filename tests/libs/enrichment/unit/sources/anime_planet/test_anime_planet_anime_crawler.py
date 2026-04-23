"""Unit tests for the anime-planet anime crawler.

Baseline tests use real processed output from
https://www.anime-planet.com/anime/one-piece (2026-04-17)
via the ap_anime_raw session fixture.

Edge-case tests use {**ap_anime_raw, "field": override} or synthetic
inline dicts to isolate specific parsing branches.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest
from common.models.anime import AnimeSeason
from common.utils.datetime_utils import determine_anime_season
from enrichment.sources.anime_planet.anime_planet_anime_crawler import (
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
from enrichment.sources.anime_planet.anime_planet_models import AnimePlanetAnime
from enrichment.sources.anime_planet.animeplanet_mapper import anime_from_animeplanet

pytestmark = pytest.mark.asyncio

_ONE_PIECE_URL = "https://www.anime-planet.com/anime/one-piece"


# ---------------------------------------------------------------------------
# _normalize_anime_url
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "identifier, expected",
    [
        ("dandadan", "https://www.anime-planet.com/anime/dandadan"),
        ("/anime/dandadan", "https://www.anime-planet.com/anime/dandadan"),
        ("https://www.anime-planet.com/anime/dandadan", "https://www.anime-planet.com/anime/dandadan"),
        ("anime/one-piece", "https://www.anime-planet.com/anime/one-piece"),
        ("https://anime-planet.com/anime/one-piece", "https://www.anime-planet.com/anime/one-piece"),
    ],
)
def test_normalize_anime_url_valid(identifier, expected):
    assert _normalize_anime_url(identifier) == expected


def test_normalize_anime_url_invalid():
    with pytest.raises(ValueError, match="anime-planet"):
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
    with pytest.raises(ValueError, match="No anime slug"):
        _extract_slug_from_url("https://www.anime-planet.com/manga/dandadan")


# ---------------------------------------------------------------------------
# _extract_json_ld
# ---------------------------------------------------------------------------


def test_extract_json_ld_valid():
    html = """
    <html><script type="application/ld+json">
    {
        "@type": "TVSeries",
        "name": "Dandadan",
        "description": "This is a &lt;b&gt;great&lt;/b&gt; show.",
        "image": "https://www.anime-planet.comhttps://s4.anilist.co/file/cover.jpg"
    }
    </script></html>
    """
    json_ld = _extract_json_ld(html)
    assert json_ld is not None
    assert json_ld["name"] == "Dandadan"
    assert json_ld["description"] == "This is a <b>great</b> show."
    assert "anime-planet.comhttps" not in json_ld["image"]


def test_extract_json_ld_html_entities_unescaped():
    html = '<html><script type="application/ld+json">{"name": "Test", "description": "&amp;&lt;&gt;"}</script></html>'
    json_ld = _extract_json_ld(html)
    assert json_ld is not None
    assert json_ld["description"] == "&<>"


def test_extract_json_ld_no_description():
    html = '<html><script type="application/ld+json">{"name": "Dandadan"}</script></html>'
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


def test_parse_aggregate_rating_from_fixture(ap_anime_raw) -> None:
    result = _parse_aggregate_rating(ap_anime_raw["aggregate_rating"])
    assert result is not None
    assert result.rating_value == pytest.approx(4.315)
    assert result.rating_count == 64725


def test_parse_aggregate_rating_none_and_empty():
    assert _parse_aggregate_rating(None) is None
    assert _parse_aggregate_rating({}) is None


def test_parse_aggregate_rating_valid():
    result = _parse_aggregate_rating({"ratingValue": "4.5", "ratingCount": "1000"})
    assert result is not None
    assert result.rating_value == pytest.approx(4.5)
    assert result.rating_count == 1000


def test_parse_aggregate_rating_invalid_types():
    assert _parse_aggregate_rating({"ratingValue": "not-a-float", "ratingCount": "not-an-int"}) is None
    assert _parse_aggregate_rating({"ratingValue": None, "ratingCount": None}) is None


def test_parse_aggregate_rating_partial_valid():
    result = _parse_aggregate_rating({"ratingValue": "3.2"})
    assert result is not None
    assert result.rating_value == pytest.approx(3.2)
    assert result.rating_count is None


# ---------------------------------------------------------------------------
# _parse_season / _parse_rank / _parse_alt_title
# ---------------------------------------------------------------------------


def test_parse_scalar_helpers_from_fixture(ap_anime_raw) -> None:
    assert _parse_season(ap_anime_raw["season_url"]) == "fall"
    assert _parse_rank(ap_anime_raw["rank_text"]) == 158
    assert _parse_alt_title(ap_anime_raw["aka"]) == "ワンピース"


@pytest.mark.parametrize(
    "season_url, expected",
    [
        (None, None),
        ("", None),
        ("/anime/seasons/fall-2024", "fall"),
        ("/anime/seasons/winter-1999", "winter"),
        ("/anime/seasons/spring-2023", "spring"),
        ("/anime/seasons/summer-2020", "summer"),
        ("/anime/seasons/fall", "fall"),
        ("/anime/not-a-season-url", None),
        ("https://www.anime-planet.com/anime/seasons/fall-2024", "fall"),
    ],
)
def test_parse_season(season_url, expected):
    assert _parse_season(season_url) == expected


@pytest.mark.parametrize(
    "rank_text, expected",
    [
        (None, None),
        ("", None),
        ("Rank #157", 157),
        ("Rank #1", 1),
        ("no hash here", None),
    ],
)
def test_parse_rank(rank_text, expected):
    assert _parse_rank(rank_text) == expected


@pytest.mark.parametrize(
    "aka, expected",
    [
        (None, None),
        ("", None),
        ("   ", None),
        ("Alt title: ダンダダン", "ダンダダン"),
        ("alt title: ワンピース", "ワンピース"),
        ("ALT TITLE:  Bleach  ", "Bleach"),
        ("ダンダダン", "ダンダダン"),
    ],
)
def test_parse_alt_title(aka, expected):
    assert _parse_alt_title(aka) == expected


# ---------------------------------------------------------------------------
# _build_related_anime_entries — fixture-grounded
# ---------------------------------------------------------------------------


def test_build_related_anime_from_fixture(ap_anime_raw) -> None:
    entries = _build_related_anime_entries(ap_anime_raw["related_anime_raw"])
    assert len(entries) == 67
    first = entries[0]
    assert first.title == "The One Piece"
    assert first.slug == "the-one-piece"
    assert first.type == "Web"
    ova = next(e for e in entries if e.slug == "one-piece-defeat-the-pirate-ganzak")
    assert ova.type == "OVA"
    assert ova.episode_count == 1


def test_build_related_anime_other_count_from_fixture(ap_anime_raw) -> None:
    assert len(_build_related_anime_entries(ap_anime_raw["related_anime_other_raw"])) == 17


def test_build_related_anime_entries_basic():
    raw = [
        {"url": "/anime/one-piece-film-red", "title": "One Piece Film: Red", "relation_subtype": "Sequel", "type": "Movie"},
        {"url": "", "title": "No URL"},
        {"url": "/manga/dandadan", "title": "Wrong"},
    ]
    entries = _build_related_anime_entries(raw)
    assert len(entries) == 1
    assert entries[0].slug == "one-piece-film-red"
    assert entries[0].type == "Movie"
    assert _build_related_anime_entries([])[0:] == []


def test_build_related_anime_entries_empty_subtype_becomes_none():
    entries = _build_related_anime_entries([{"url": "/anime/slug", "title": "T", "relation_subtype": ""}])
    assert entries[0].relation_subtype is None


@pytest.mark.parametrize(
    "raw_type, expected_type, expected_ep_count",
    [
        ("Movie", "Movie", None),
        ("Web", "Web", None),
        ("", None, None),
        (None, None, None),
        ("OVA: 1 ep", "OVA", 1),
        ("TV Special: 9 ep", "TV Special", 9),
        ("Web: 12 ep", "Web", 12),
        ("TV: 21 ep", "TV", 21),
        ("Music Video: 1 ep", "Music Video", 1),
    ],
)
def test_build_related_anime_type_and_episode_parsing(raw_type, expected_type, expected_ep_count):
    entries = _build_related_anime_entries([{"url": "/anime/slug", "title": "T", "type": raw_type}])
    assert entries[0].type == expected_type
    assert entries[0].episode_count == expected_ep_count


# ---------------------------------------------------------------------------
# _build_related_manga_entries — fixture-grounded
# ---------------------------------------------------------------------------


def test_build_related_manga_from_fixture(ap_anime_raw) -> None:
    entries = _build_related_manga_entries(ap_anime_raw["related_manga_raw"])
    assert len(entries) == 24
    romance_dawn = next(e for e in entries if e.slug == "romance-dawn")
    assert romance_dawn.type == "One Shot"
    assert romance_dawn.chapters == 1
    main_manga = next(e for e in entries if e.slug == "one-piece")
    assert main_manga.volumes == 114
    assert main_manga.chapters == 1179


def test_build_related_manga_entries_basic():
    raw = [
        {"url": "/manga/one-piece", "title": "One Piece", "relation_subtype": "Original Manga"},
        {"url": "", "title": "No URL"},
        {"url": "/anime/dandadan", "title": "Wrong"},
    ]
    entries = _build_related_manga_entries(raw)
    assert len(entries) == 1
    assert entries[0].slug == "one-piece"
    assert entries[0].relation_subtype == "Original Manga"


@pytest.mark.parametrize(
    "vol_ch, expected_type, expected_volumes, expected_chapters",
    [
        ("One Shot", "One Shot", None, 1),
        ("one shot", "One Shot", None, 1),
        ("Vol: 114 - Ch: 1179+", None, 114, 1179),
        ("Vol: 1 - Ch: 3", None, 1, 3),
        ("Vol: 1", None, 1, None),
        ("Ch: 19", None, None, 19),
        ("", None, None, None),
        (None, None, None, None),
        ("- ?", None, None, None),
    ],
)
def test_build_related_manga_vol_ch_parsing(vol_ch, expected_type, expected_volumes, expected_chapters):
    entries = _build_related_manga_entries([{"url": "/manga/slug", "title": "T", "vol_ch": vol_ch}])
    assert entries[0].type == expected_type
    assert entries[0].volumes == expected_volumes
    assert entries[0].chapters == expected_chapters


# ---------------------------------------------------------------------------
# _build_anime_from_raw — fixture-grounded
# ---------------------------------------------------------------------------


def test_build_anime_from_raw_from_fixture(ap_anime_raw) -> None:
    anime = _build_anime_from_raw(ap_anime_raw)
    assert anime.name == "One Piece"
    assert anime.slug == "one-piece"
    assert anime.season == "fall"
    assert anime.rank == 158
    assert anime.alt_title == "ワンピース"
    assert anime.number_of_episodes == 1157
    assert anime.studios == ["Toei Animation"]
    assert "Shounen" in anime.tags
    assert "Action" in anime.genres
    assert anime.aggregate_rating is not None
    assert anime.aggregate_rating.rating_value == pytest.approx(4.315)
    assert anime.aggregate_rating.rating_count == 64725
    assert len(anime.related_anime) == 67
    assert len(anime.related_anime_other) == 17
    assert len(anime.related_manga) == 24
    assert anime.cover is not None and "one-piece" in anime.cover


def test_build_anime_from_raw_field_overrides(ap_anime_raw) -> None:
    assert _build_anime_from_raw({**ap_anime_raw, "rank_text": "Rank #42"}).rank == 42
    assert _build_anime_from_raw({**ap_anime_raw, "season_url": None}).season is None
    assert _build_anime_from_raw({**ap_anime_raw, "aka": None}).alt_title is None
    assert _build_anime_from_raw({**ap_anime_raw, "aggregate_rating": None}).aggregate_rating is None


# ---------------------------------------------------------------------------
# Canonical mapper — fixture-grounded
# ---------------------------------------------------------------------------


def test_mapper_from_fixture(ap_anime_raw) -> None:
    canonical = anime_from_animeplanet(_build_anime_from_raw(ap_anime_raw))
    assert canonical["title"] == "One Piece"
    assert canonical["year"] == 1999
    assert canonical["season"] == "FALL"
    assert canonical["status"] == "ONGOING"
    assert canonical["episode_count"] == 1157
    assert canonical["title_japanese"] == "ワンピース"
    assert any(p["name"] == "Toei Animation" for p in canonical["producers"])
    stats = canonical["statistics"]["anime_planet"]
    assert stats["score"] == pytest.approx(8.63)
    assert stats["scored_by"] == 64725
    assert stats["rank"] == 158
    all_manga = [e for entries in canonical["related_source_material"].values() for e in entries]
    assert len(all_manga) == 24
    romance_dawn = next(e for e in all_manga if "Romance Dawn" in e["title"])
    assert romance_dawn["type"] == "ONE SHOT"


# ---------------------------------------------------------------------------
# Mapper: related entry field passthrough (edge-case coverage)
# ---------------------------------------------------------------------------


def _make_anime_with_related(related_anime=None, related_manga=None) -> AnimePlanetAnime:
    return AnimePlanetAnime(
        name="Test Anime",
        slug="test-anime",
        schema_type="TVSeries",
        related_anime=related_anime or [],
        related_anime_other=[],
        related_manga=related_manga or [],
    )


def test_mapper_related_anime_episode_count_passthrough():
    from enrichment.sources.anime_planet.anime_planet_models import AnimePlanetRelatedEntry
    entry = AnimePlanetRelatedEntry(url="/anime/special", slug="special", title="Test Special", relation_subtype="Same Franchise", type="TV Special", episode_count=3)
    data = anime_from_animeplanet(_make_anime_with_related(related_anime=[entry]))
    match = next(e for e in data["related_anime"].get("SIDE_STORY", []) if e["title"] == "Test Special")
    assert match["episode_count"] == 3


def test_mapper_related_anime_no_episode_count_is_absent():
    from enrichment.sources.anime_planet.anime_planet_models import AnimePlanetRelatedEntry
    entry = AnimePlanetRelatedEntry(url="/anime/film", slug="film", title="Test Movie", relation_subtype="Same Franchise", type="Movie", episode_count=None)
    data = anime_from_animeplanet(_make_anime_with_related(related_anime=[entry]))
    match = next(e for e in data["related_anime"].get("SIDE_STORY", []) if e["title"] == "Test Movie")
    assert "episode_count" not in match


def test_mapper_related_source_material_volumes_and_chapters():
    from enrichment.sources.anime_planet.anime_planet_models import AnimePlanetMangaEntry
    entry = AnimePlanetMangaEntry(url="/manga/some-manga", slug="some-manga", title="Some Manga", relation_subtype="Original Manga", volumes=7, chapters=62)
    data = anime_from_animeplanet(_make_anime_with_related(related_manga=[entry]))
    all_manga = [e for entries in data["related_source_material"].values() for e in entries]
    match = next(e for e in all_manga if e["title"] == "Some Manga")
    assert match["volumes"] == 7
    assert match["chapters"] == 62


def test_mapper_manga_type_edge_cases():
    from enrichment.sources.anime_planet.anime_planet_models import AnimePlanetMangaEntry
    unknown = AnimePlanetMangaEntry(url="/manga/plain", slug="plain", title="Plain Manga", volumes=3, chapters=20)
    one_shot = AnimePlanetMangaEntry(url="/manga/os", slug="os", title="Romance Dawn", type="One Shot", chapters=1)
    ln = AnimePlanetMangaEntry(url="/manga/ln", slug="ln", title="Some Story (Light Novel)", volumes=2, chapters=8)
    no_count = AnimePlanetMangaEntry(url="/manga/nc", slug="nc", title="No Count Manga")

    data = anime_from_animeplanet(_make_anime_with_related(related_manga=[unknown, one_shot, ln, no_count]))
    all_manga = {e["title"]: e for entries in data["related_source_material"].values() for e in entries}

    assert all_manga["Plain Manga"]["type"] == "UNKNOWN"
    assert all_manga["Romance Dawn"]["type"] == "ONE SHOT"
    assert all_manga["Some Story (Light Novel)"]["type"] == "UNKNOWN"
    assert "volumes" not in all_manga["No Count Manga"]
    assert "chapters" not in all_manga["No Count Manga"]


# ---------------------------------------------------------------------------
# Season derivation utility
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
        {"url": "/anime/dandadan-season-2", "title": "Dandadan Season 2", "relation_subtype": "Sequel", "type": "TV: 12 ep", "image": None}
    ],
    "related_anime_other_raw": [],
    "related_manga_raw": [
        {"url": "/manga/dandadan", "title": "Dandadan Manga", "relation_subtype": "Original Manga", "vol_ch": "Vol: 15 - Ch: 170", "image": None}
    ],
}


def _make_crawl_result(xpath_data: dict, json_ld: dict) -> dict:
    html = f'<html><script type="application/ld+json">{json.dumps(json_ld)}</script></html>'
    return {"success": True, "status_code": 200, "extracted_content": json.dumps([xpath_data]), "html": html, "error_message": None}


@pytest.mark.usefixtures("mock_redis_cache_miss")
@patch("enrichment.sources.anime_planet.anime_planet_anime_crawler.crawl_single_url")
async def test_fetch_animeplanet_anime_success(mock_crawl):
    mock_crawl.return_value = _make_crawl_result(_BASE_XPATH, _BASE_JSON_LD)
    anime = await fetch_animeplanet_anime("https://www.anime-planet.com/anime/dandadan")
    assert anime is not None
    assert anime["title"] == "Dandadan"
    assert anime["year"] == 2024
    assert anime["season"] == "FALL"
    assert anime["status"] == "FINISHED"
    assert anime["episode_count"] == 12
    assert anime["statistics"]["anime_planet"]["score"] == pytest.approx(9.0)
    assert anime["statistics"]["anime_planet"]["scored_by"] == 1000
    assert anime["statistics"]["anime_planet"]["rank"] == 123
    assert anime["title_japanese"] == "ダンダダン"
    assert "Supernatural" in anime["tags"]
    assert any(p["name"] == "Science SARU" for p in anime["producers"])
    sequel = anime["related_anime"]["SEQUEL"][0]
    assert sequel["type"] == "TV" and sequel["episode_count"] == 12
    manga = anime["related_source_material"]["ADAPTATION"][0]
    assert manga["volumes"] == 15 and manga["chapters"] == 170


@pytest.mark.usefixtures("mock_redis_cache_miss")
@patch("enrichment.sources.anime_planet.anime_planet_anime_crawler.crawl_single_url")
async def test_fetch_animeplanet_anime_one_piece_fixture_via_mock(mock_crawl, ap_anime_raw) -> None:
    """Full round-trip using one-piece fixture data injected via mock crawl result."""
    html = f'<html><script type="application/ld+json">{json.dumps({"@type": ap_anime_raw["schema_type"], "name": ap_anime_raw["name"], "url": ap_anime_raw["url"], "startDate": ap_anime_raw["start_date"], "numberOfEpisodes": ap_anime_raw["number_of_episodes"], "genre": ap_anime_raw["genres"], "aggregateRating": ap_anime_raw["aggregate_rating"]})}</script></html>'
    xpath = {
        "type_raw": ap_anime_raw["type_raw"],
        "season_url": ap_anime_raw["season_url"],
        "rank_text": ap_anime_raw["rank_text"],
        "aka": ap_anime_raw["aka"],
        "cover": ap_anime_raw["cover"],
        "studios": [{"name": s} for s in ap_anime_raw["studios"]],
        "tags": [{"name": t} for t in ap_anime_raw["tags"]],
        "related_anime_raw": ap_anime_raw["related_anime_raw"],
        "related_anime_other_raw": ap_anime_raw["related_anime_other_raw"],
        "related_manga_raw": ap_anime_raw["related_manga_raw"],
    }
    mock_crawl.return_value = {"success": True, "status_code": 200, "extracted_content": json.dumps([xpath]), "html": html, "error_message": None}

    anime = await fetch_animeplanet_anime(_ONE_PIECE_URL)
    assert anime is not None
    assert anime["title"] == "One Piece"
    assert anime["year"] == 1999
    assert anime["season"] == "FALL"
    assert anime["status"] == "ONGOING"
    assert anime["episode_count"] == 1157
    assert anime["title_japanese"] == "ワンピース"


@pytest.mark.usefixtures("mock_redis_cache_miss")
@patch("enrichment.sources.anime_planet.anime_planet_anime_crawler.crawl_single_url")
async def test_fetch_animeplanet_anime_season_priority(mock_crawl):
    """season_url slug takes precedence; absent season_url falls back to start_date."""
    mock_crawl.return_value = _make_crawl_result({**_BASE_XPATH, "season_url": "/anime/seasons/winter-2024"}, _BASE_JSON_LD)
    assert (await fetch_animeplanet_anime("https://www.anime-planet.com/anime/dandadan"))["season"] == "WINTER"

    mock_crawl.return_value = _make_crawl_result({**_BASE_XPATH, "season_url": None}, {**_BASE_JSON_LD, "startDate": "2024-04-05"})
    assert (await fetch_animeplanet_anime("https://www.anime-planet.com/anime/dandadan"))["season"] == "SPRING"

    mock_crawl.return_value = _make_crawl_result({**_BASE_XPATH, "season_url": "/anime/not-a-season-url"}, {**_BASE_JSON_LD, "startDate": "2024-07-10"})
    assert (await fetch_animeplanet_anime("https://www.anime-planet.com/anime/dandadan"))["season"] == "SUMMER"


@pytest.mark.usefixtures("mock_redis_cache_miss")
@patch("enrichment.sources.anime_planet.anime_planet_anime_crawler.crawl_single_url")
async def test_fetch_animeplanet_anime_failure_cases(mock_crawl):
    """No JSON-LD, 404, crawl failure, empty extraction, and HTTP 500 all return None."""
    no_json_ld = {"success": True, "status_code": 200, "extracted_content": json.dumps([_BASE_XPATH]), "html": "<html></html>", "error_message": None}
    mock_crawl.return_value = no_json_ld
    assert await fetch_animeplanet_anime("https://www.anime-planet.com/anime/dandadan") is None

    mock_crawl.return_value = {"success": False, "status_code": 404, "extracted_content": None, "html": "", "error_message": "Not found"}
    assert await fetch_animeplanet_anime("https://www.anime-planet.com/anime/nonexistent") is None

    mock_crawl.return_value = {"success": False, "status_code": 200, "extracted_content": None, "html": "", "error_message": "Browser crashed"}
    assert await fetch_animeplanet_anime("https://www.anime-planet.com/anime/dandadan") is None

    mock_crawl.return_value = {"success": True, "status_code": 200, "extracted_content": "[]", "html": "", "error_message": None}
    assert await fetch_animeplanet_anime("https://www.anime-planet.com/anime/dandadan") is None

    mock_crawl.return_value = {"success": True, "status_code": 500, "extracted_content": None, "html": "", "error_message": None}
    assert await fetch_animeplanet_anime("https://www.anime-planet.com/anime/dandadan") is None


@pytest.mark.usefixtures("mock_redis_cache_miss")
@patch("enrichment.sources.anime_planet.anime_planet_anime_crawler._fetch_animeplanet_anime_data", new_callable=AsyncMock)
async def test_fetch_animeplanet_anime_extracts_slug_from_url_for_cache(mock_inner):
    mock_inner.return_value = None
    await fetch_animeplanet_anime(_ONE_PIECE_URL)
    mock_inner.assert_called_once_with("one-piece")


@pytest.mark.usefixtures("mock_redis_cache_miss")
@patch("enrichment.sources.anime_planet.anime_planet_anime_crawler.crawl_single_url")
async def test_fetch_animeplanet_anime_accepts_non_www_url(mock_crawl):
    mock_crawl.return_value = _make_crawl_result(_BASE_XPATH, _BASE_JSON_LD)
    anime = await fetch_animeplanet_anime("https://anime-planet.com/anime/dandadan")
    assert anime is not None
    assert any("dandadan" in s for s in anime.get("sources", []))


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
@patch("enrichment.sources.anime_planet.anime_planet_anime_crawler.crawl_single_url")
async def test_fetch_anime_status_derivation(mock_crawl, start_date, end_date, expected_status):
    json_ld = {**_BASE_JSON_LD, "startDate": start_date, "endDate": end_date}
    mock_crawl.return_value = _make_crawl_result(_BASE_XPATH, json_ld)
    anime = await fetch_animeplanet_anime("https://www.anime-planet.com/anime/status-test")
    assert anime is not None
    assert anime["status"] == expected_status
