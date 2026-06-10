"""Unit tests for anime_planet_anime_crawler.py.

Fixture-grounded tests use the one-piece HTML fixture (2026-06-10).
Edge-case tests use synthetic inline HTML or dict overrides.
"""

import json
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from common.models.anime import AnimeSeason
from common.utils.datetime_utils import determine_anime_season
from enrichment.sources.anime_planet.anime_planet_anime_crawler import (
    _XPATHS,
    _build_anime_from_raw,
    _build_related_anime_entries,
    _build_related_manga_entries,
    _extract_anime_from_html,
    _extract_json_ld,
    _extract_slug_from_url,
    _fetch_anime_html,
    _normalize_anime_url,
    _parse_aggregate_rating,
    _parse_alt_title,
    _parse_rank,
    _parse_related_entry_element,
    _parse_season,
    fetch_animeplanet_anime,
)
from enrichment.sources.anime_planet.anime_planet_models import (
    AnimePlanetMangaEntry,
    AnimePlanetRelatedEntry,
)
from enrichment.sources.anime_planet.animeplanet_mapper import anime_from_animeplanet
from lxml import etree

pytestmark = pytest.mark.asyncio

_ONE_PIECE_URL = "https://www.anime-planet.com/anime/one-piece"
_PATCH_FETCH_DATA = (
    "enrichment.sources.anime_planet.anime_planet_anime_crawler._fetch_animeplanet_anime_data"
)
_PATCH_FETCH_HTML = (
    "enrichment.sources.anime_planet.anime_planet_anime_crawler._fetch_anime_html"
)


def _make_html(json_ld: dict) -> str:
    """Minimal valid AP page HTML with the given JSON-LD block."""
    return (
        '<html><body>'
        '<section class="entryBar">'
        '<span class="type">TV</span>'
        '<a href="/anime/seasons/fall-2024">Fall 2024</a>'
        '</section>'
        f'<script type="application/ld+json">{json.dumps(json_ld)}</script>'
        '</body></html>'
    )


_BASE_JSON_LD: dict[str, Any] = {
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


# =============================================================================
# _XPATHS invariant
# =============================================================================


def test_xpaths_cover_required_fields() -> None:
    for key in ("type_raw", "season_url", "rank_text", "studios", "aka", "tags",
                "cover", "related_anime", "related_anime_other", "related_manga"):
        assert key in _XPATHS, f"_XPATHS missing key: {key!r}"
    assert all("entryBar" in v or "RelatedEntry" in v or "/" in v for v in _XPATHS.values())


# =============================================================================
# _normalize_anime_url
# =============================================================================


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
def test_normalize_anime_url_valid(identifier: str, expected: str) -> None:
    assert _normalize_anime_url(identifier) == expected


def test_normalize_anime_url_invalid() -> None:
    with pytest.raises(ValueError, match="anime-planet"):
        _normalize_anime_url("https://www.google.com/anime/dandadan")


# =============================================================================
# _extract_slug_from_url
# =============================================================================


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://www.anime-planet.com/anime/dandadan", "dandadan"),
        ("https://www.anime-planet.com/anime/one-piece?foo=bar", "one-piece"),
    ],
)
def test_extract_slug_from_url_valid(url: str, expected: str) -> None:
    assert _extract_slug_from_url(url) == expected


def test_extract_slug_from_url_invalid() -> None:
    with pytest.raises(ValueError, match="No anime slug"):
        _extract_slug_from_url("https://www.anime-planet.com/manga/dandadan")


# =============================================================================
# _extract_json_ld
# =============================================================================


def test_extract_json_ld_valid() -> None:
    html = (
        '<html><script type="application/ld+json">'
        '{"@type":"TVSeries","name":"Dandadan",'
        '"description":"A &lt;b&gt;great&lt;/b&gt; show.",'
        '"image":"https://www.anime-planet.comhttps://s4.anilist.co/cover.jpg"}'
        "</script></html>"
    )
    result = _extract_json_ld(html)
    assert result is not None
    assert result["name"] == "Dandadan"
    assert result["description"] == "A <b>great</b> show."
    assert "anime-planet.comhttps" not in result["image"]


def test_extract_json_ld_no_description() -> None:
    html = '<html><script type="application/ld+json">{"name":"Test"}</script></html>'
    assert _extract_json_ld(html) is not None
    assert "description" not in _extract_json_ld(html)  # type: ignore[index]


@pytest.mark.parametrize(
    "html",
    [
        "<html></html>",
        '<html><script type="application/ld+json">{invalid}</script></html>',
    ],
)
def test_extract_json_ld_invalid(html: str) -> None:
    assert _extract_json_ld(html) is None


# =============================================================================
# _parse_aggregate_rating
# =============================================================================


def test_parse_aggregate_rating_from_fixture(ap_anime_extracted: dict) -> None:
    result = _parse_aggregate_rating(ap_anime_extracted["aggregate_rating"])
    assert result is not None
    assert result.rating_value == pytest.approx(4.315)
    assert result.rating_count == 64986


@pytest.mark.parametrize(
    "ar, expected_value, expected_count",
    [
        ({"ratingValue": "4.5", "ratingCount": "1000"}, 4.5, 1000),
        ({"ratingValue": "3.2"}, 3.2, None),
        ({"ratingCount": "500"}, None, 500),
    ],
)
def test_parse_aggregate_rating_valid(ar: dict, expected_value: Any, expected_count: Any) -> None:
    result = _parse_aggregate_rating(ar)
    assert result is not None
    assert (result.rating_value == pytest.approx(expected_value)) if expected_value else result.rating_value is None
    assert result.rating_count == expected_count


@pytest.mark.parametrize("ar", [None, {}, {"ratingValue": None, "ratingCount": None},
                                 {"ratingValue": "bad", "ratingCount": "bad"}])
def test_parse_aggregate_rating_returns_none(ar: Any) -> None:
    assert _parse_aggregate_rating(ar) is None


# =============================================================================
# _parse_season / _parse_rank / _parse_alt_title
# =============================================================================


def test_parse_scalar_helpers_from_fixture(ap_anime_extracted: dict) -> None:
    assert _parse_season(ap_anime_extracted["season_url"]) == "fall"
    assert _parse_rank(ap_anime_extracted["rank_text"]) == 161
    assert _parse_alt_title(ap_anime_extracted["aka"]) == "ワンピース"


@pytest.mark.parametrize(
    "season_url, expected",
    [
        (None, None), ("", None),
        ("/anime/seasons/fall-2024", "fall"),
        ("/anime/seasons/winter-1999", "winter"),
        ("/anime/seasons/spring-2023", "spring"),
        ("/anime/seasons/summer-2020", "summer"),
        ("/anime/not-a-season-url", None),
        ("https://www.anime-planet.com/anime/seasons/fall-2024", "fall"),
    ],
)
def test_parse_season(season_url: str | None, expected: str | None) -> None:
    assert _parse_season(season_url) == expected


@pytest.mark.parametrize(
    "rank_text, expected",
    [
        (None, None), ("", None),
        ("Rank #157", 157), ("Rank #1", 1), ("no hash here", None),
    ],
)
def test_parse_rank(rank_text: str | None, expected: int | None) -> None:
    assert _parse_rank(rank_text) == expected


@pytest.mark.parametrize(
    "aka, expected",
    [
        (None, None), ("", None), ("   ", None),
        ("Alt title: ダンダダン", "ダンダダン"),
        ("alt title: ワンピース", "ワンピース"),
        ("ALT TITLE:  Bleach  ", "Bleach"),
        ("ダンダダン", "ダンダダン"),
    ],
)
def test_parse_alt_title(aka: str | None, expected: str | None) -> None:
    assert _parse_alt_title(aka) == expected


# =============================================================================
# _parse_related_entry_element
# =============================================================================


def _parse_el(html: str, *, is_manga: bool = False) -> dict[str, Any]:
    tree = etree.fromstring(html, etree.HTMLParser())
    els = cast(list[Any], tree.xpath("//a[contains(@class,'RelatedEntry')]"))
    return _parse_related_entry_element(els[0], is_manga=is_manga)


def test_parse_anime_entry_element() -> None:
    html = """<html><body><a href="/anime/test-sequel" class="RelatedEntry">
      <p class="RelatedEntry__name">Test Sequel</p>
      <span class="RelatedEntry__subtitle">Sequel</span>
      <ul>
        <li><i class="fa-tv"></i><span class="RelatedEntry__metadata_item">TV: 12 ep</span></li>
        <li><i class="fa-calendar"></i><span class="RelatedEntry__metadata_item">2024</span></li>
      </ul>
      <img class="RelatedEntry__image" src="https://example.com/img.jpg" />
    </a></body></html>"""
    entry = _parse_el(html)
    assert entry["url"] == "/anime/test-sequel"
    assert entry["title"] == "Test Sequel"
    assert entry["relation_subtype"] == "Sequel"
    assert entry["type"] == "TV: 12 ep"
    assert entry["image"] == "https://example.com/img.jpg"
    assert "vol_ch" not in entry


def test_parse_manga_entry_element() -> None:
    html = """<html><body><a href="/manga/dandadan" class="RelatedEntry">
      <p class="RelatedEntry__name">Dandadan</p>
      <ul>
        <li><i class="fa-book-open"></i><span class="RelatedEntry__metadata_item">Vol: 24 - Ch: 236</span></li>
      </ul>
    </a></body></html>"""
    entry = _parse_el(html, is_manga=True)
    assert entry["url"] == "/manga/dandadan"
    assert entry["title"] == "Dandadan"
    assert entry["vol_ch"] == "Vol: 24 - Ch: 236"
    assert entry["relation_subtype"] is None
    assert "type" not in entry


# =============================================================================
# _extract_anime_from_html
# =============================================================================


def test_extract_from_html_fixture(ap_anime_html: str) -> None:
    raw = _extract_anime_from_html(ap_anime_html)
    assert raw is not None
    assert raw["name"] == "One Piece"
    assert raw["schema_type"] == "TVSeries"
    assert raw["start_date"] == "1999-10-20"
    assert raw["end_date"] is None
    assert raw["number_of_episodes"] == 1165
    assert "Action" in raw["genres"]
    assert raw["aggregate_rating"]["ratingValue"] == pytest.approx(4.315)
    assert raw["type_raw"] is not None and "TV" in raw["type_raw"]
    assert raw["season_url"] == "/anime/seasons/fall-1999"
    assert "161" in raw["rank_text"]
    assert raw["aka"] == "Alt title: ワンピース"
    assert raw["studios"] == ["Toei Animation"]
    assert "Shounen" in raw["tags"]
    assert raw["cover"] is not None and "one-piece" in raw["cover"]
    assert len(raw["related_anime_raw"]) == 67
    assert len(raw["related_anime_other_raw"]) == 17
    assert len(raw["related_manga_raw"]) == 24
    assert "slug" not in raw


@pytest.mark.parametrize(
    "html",
    [
        "",
        "<html><body><p>no json-ld here</p></body></html>",
        '<html><body><script type="application/ld+json">{"description": "no name"}</script></body></html>',
    ],
)
def test_extract_returns_none_on_bad_html(html: str) -> None:
    assert _extract_anime_from_html(html) is None


# =============================================================================
# _build_related_anime_entries
# =============================================================================


def test_build_related_anime_from_fixture(ap_anime_extracted: dict) -> None:
    entries = _build_related_anime_entries(ap_anime_extracted["related_anime_raw"])
    assert len(entries) == 67
    ganzak = next(e for e in entries if "ganzak" in e.slug)
    assert ganzak.type == "OVA"
    assert ganzak.episode_count == 1
    the_one_piece = next(e for e in entries if e.slug == "the-one-piece")
    assert the_one_piece.type == "Web"


def test_build_related_anime_other_count_from_fixture(ap_anime_extracted: dict) -> None:
    assert len(_build_related_anime_entries(ap_anime_extracted["related_anime_other_raw"])) == 17


@pytest.mark.parametrize(
    "raw_type, expected_type, expected_ep",
    [
        ("Movie", "Movie", None),
        ("Web", "Web", None),
        ("", None, None),
        (None, None, None),
        ("OVA: 1 ep", "OVA", 1),
        ("TV Special: 9 ep", "TV Special", 9),
        ("Music Video: 1 ep", "Music Video", 1),
    ],
)
def test_build_related_anime_type_parsing(raw_type: str | None, expected_type: str | None, expected_ep: int | None) -> None:
    entries = _build_related_anime_entries([{"url": "/anime/slug", "title": "T", "type": raw_type}])
    assert entries[0].type == expected_type
    assert entries[0].episode_count == expected_ep


def test_build_related_anime_entries_filters_invalid() -> None:
    raw = [
        {"url": "/anime/valid", "title": "Valid", "type": "Movie"},
        {"url": "", "title": "No URL"},
        {"url": "/manga/wrong", "title": "Wrong domain"},
    ]
    entries = _build_related_anime_entries(raw)
    assert len(entries) == 1
    assert entries[0].slug == "valid"


def test_build_related_anime_empty_subtype_becomes_none() -> None:
    entries = _build_related_anime_entries([{"url": "/anime/slug", "title": "T", "relation_subtype": ""}])
    assert entries[0].relation_subtype is None


# =============================================================================
# _build_related_manga_entries
# =============================================================================


def test_build_related_manga_from_fixture(ap_anime_extracted: dict) -> None:
    entries = _build_related_manga_entries(ap_anime_extracted["related_manga_raw"])
    assert len(entries) == 24
    romance_dawn = next(e for e in entries if e.slug == "romance-dawn")
    assert romance_dawn.type == "One Shot"
    assert romance_dawn.chapters == 1
    main = next(e for e in entries if e.slug == "one-piece")
    assert main.volumes == 114
    assert main.chapters == 1184


def test_build_related_manga_entries_filters_invalid() -> None:
    raw = [
        {"url": "/manga/valid", "title": "Valid Manga", "vol_ch": "Vol: 1"},
        {"url": "", "title": "No URL"},
        {"url": "/anime/wrong-domain", "title": "Wrong domain"},
    ]
    entries = _build_related_manga_entries(raw)
    assert len(entries) == 1
    assert entries[0].slug == "valid"


@pytest.mark.parametrize(
    "vol_ch, expected_type, expected_volumes, expected_chapters",
    [
        ("One Shot", "One Shot", None, 1),
        ("one shot", "One Shot", None, 1),
        ("Vol: 114 - Ch: 1184+", None, 114, 1184),
        ("Vol: 1 - Ch: 3", None, 1, 3),
        ("Vol: 1", None, 1, None),
        ("Ch: 19", None, None, 19),
        ("", None, None, None),
        (None, None, None, None),
        ("- ?", None, None, None),
    ],
)
def test_build_related_manga_vol_ch_parsing(
    vol_ch: str | None, expected_type: str | None, expected_volumes: int | None, expected_chapters: int | None
) -> None:
    entries = _build_related_manga_entries([{"url": "/manga/slug", "title": "T", "vol_ch": vol_ch}])
    assert entries[0].type == expected_type
    assert entries[0].volumes == expected_volumes
    assert entries[0].chapters == expected_chapters


# =============================================================================
# AnimePlanetAnimeCrawler
# =============================================================================


def test_crawler_get_extraction_schema() -> None:
    from enrichment.sources.anime_planet.anime_planet_anime_crawler import AnimePlanetAnimeCrawler
    from enrichment.sources.base.framework import DockerTransport, NullRepository
    crawler = AnimePlanetAnimeCrawler(DockerTransport(), NullRepository())
    assert crawler.get_extraction_schema() is _XPATHS


# =============================================================================
# _build_anime_from_raw
# =============================================================================


def test_build_anime_from_raw_from_fixture(ap_anime_extracted: dict) -> None:
    anime = _build_anime_from_raw(ap_anime_extracted)
    assert anime.name == "One Piece"
    assert anime.slug == "one-piece"
    assert anime.season == "fall"
    assert anime.rank == 161
    assert anime.alt_title == "ワンピース"
    assert anime.number_of_episodes == 1165
    assert anime.studios == ["Toei Animation"]
    assert "Shounen" in anime.tags
    assert "Action" in anime.genres
    assert anime.aggregate_rating is not None
    assert anime.aggregate_rating.rating_value == pytest.approx(4.315)
    assert anime.aggregate_rating.rating_count == 64986
    assert len(anime.related_anime) == 67
    assert len(anime.related_anime_other) == 17
    assert len(anime.related_manga) == 24
    assert anime.cover is not None and "one-piece" in anime.cover


def test_build_anime_from_raw_field_overrides(ap_anime_extracted: dict) -> None:
    assert _build_anime_from_raw({**ap_anime_extracted, "rank_text": "Rank #42"}).rank == 42
    assert _build_anime_from_raw({**ap_anime_extracted, "season_url": None}).season is None
    assert _build_anime_from_raw({**ap_anime_extracted, "aka": None}).alt_title is None
    assert _build_anime_from_raw({**ap_anime_extracted, "aggregate_rating": None}).aggregate_rating is None


# =============================================================================
# _fetch_anime_html
# =============================================================================


async def test_fetch_html_success() -> None:
    page_mock = AsyncMock()
    page_mock.wait_for = AsyncMock()
    page_mock.get_content = AsyncMock(return_value="<html>content</html>")
    browser_mock = AsyncMock()
    browser_mock.get = AsyncMock(return_value=page_mock)
    browser_mock.stop = AsyncMock()

    import zendriver as zd
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(zd, "start", AsyncMock(return_value=browser_mock))
        result = await _fetch_anime_html(_ONE_PIECE_URL)

    assert result == "<html>content</html>"
    page_mock.wait_for.assert_awaited_once()


async def test_fetch_html_navigation_failure_returns_none() -> None:
    page_mock = AsyncMock()
    page_mock.wait_for = AsyncMock(side_effect=Exception("timeout"))
    browser_mock = AsyncMock()
    browser_mock.get = AsyncMock(return_value=page_mock)
    browser_mock.stop = AsyncMock()

    import zendriver as zd
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(zd, "start", AsyncMock(return_value=browser_mock))
        result = await _fetch_anime_html(_ONE_PIECE_URL)

    assert result is None


async def test_fetch_html_stop_exception_swallowed() -> None:
    page_mock = AsyncMock()
    page_mock.wait_for = AsyncMock()
    page_mock.get_content = AsyncMock(return_value="<html>ok</html>")
    browser_mock = AsyncMock()
    browser_mock.get = AsyncMock(return_value=page_mock)
    browser_mock.stop = AsyncMock(side_effect=Exception("stop failed"))

    import zendriver as zd
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(zd, "start", AsyncMock(return_value=browser_mock))
        result = await _fetch_anime_html(_ONE_PIECE_URL)

    assert result == "<html>ok</html>"


# =============================================================================
# Canonical mapper
# =============================================================================


def test_mapper_from_fixture(ap_anime_extracted: dict) -> None:
    canonical = anime_from_animeplanet(_build_anime_from_raw(ap_anime_extracted))
    assert canonical["title"] == "One Piece"
    assert canonical["year"] == 1999
    assert canonical["season"] == "FALL"
    assert canonical["status"] == "ONGOING"
    assert canonical["episode_count"] == 1165
    assert canonical["title_japanese"] == "ワンピース"
    assert any(p["name"] == "Toei Animation" for p in canonical["producers"])
    stats = canonical["statistics"]["anime_planet"]
    assert stats["score"] == pytest.approx(8.63)
    assert stats["scored_by"] == 64986
    assert stats["rank"] == 161
    all_manga = [e for entries in canonical["related_source_material"].values() for e in entries]
    assert len(all_manga) == 24
    romance_dawn = next(e for e in all_manga if "Romance Dawn" in e["title"])
    assert romance_dawn["type"] == "ONE SHOT"


def _make_anime_with_related(
    related_anime: list | None = None,
    related_manga: list | None = None,
) -> Any:
    from enrichment.sources.anime_planet.anime_planet_models import AnimePlanetAnime
    return AnimePlanetAnime(
        name="Test Anime", slug="test-anime", schema_type="TVSeries",
        related_anime=related_anime or [], related_anime_other=[], related_manga=related_manga or [],
    )


def test_mapper_related_anime_episode_count_passthrough() -> None:
    entry = AnimePlanetRelatedEntry(
        url="/anime/special", slug="special", title="Test Special",
        relation_subtype="Same Franchise", type="TV Special", episode_count=3,
    )
    data = anime_from_animeplanet(_make_anime_with_related(related_anime=[entry]))
    match = next(e for e in data["related_anime"].get("SIDE_STORY", []) if e["title"] == "Test Special")
    assert match["episode_count"] == 3


def test_mapper_related_anime_no_episode_count_absent() -> None:
    entry = AnimePlanetRelatedEntry(
        url="/anime/film", slug="film", title="Test Movie",
        relation_subtype="Same Franchise", type="Movie", episode_count=None,
    )
    data = anime_from_animeplanet(_make_anime_with_related(related_anime=[entry]))
    match = next(e for e in data["related_anime"].get("SIDE_STORY", []) if e["title"] == "Test Movie")
    assert "episode_count" not in match


def test_mapper_related_source_material_volumes_and_chapters() -> None:
    entry = AnimePlanetMangaEntry(
        url="/manga/some-manga", slug="some-manga", title="Some Manga",
        relation_subtype="Original Manga", volumes=7, chapters=62,
    )
    data = anime_from_animeplanet(_make_anime_with_related(related_manga=[entry]))
    all_manga = [e for entries in data["related_source_material"].values() for e in entries]
    match = next(e for e in all_manga if e["title"] == "Some Manga")
    assert match["volumes"] == 7
    assert match["chapters"] == 62


def test_mapper_manga_type_edge_cases() -> None:
    entries = [
        AnimePlanetMangaEntry(url="/manga/plain", slug="plain", title="Plain Manga", volumes=3, chapters=20),
        AnimePlanetMangaEntry(url="/manga/os", slug="os", title="Romance Dawn", type="One Shot", chapters=1),
        AnimePlanetMangaEntry(url="/manga/nc", slug="nc", title="No Count Manga"),
    ]
    data = anime_from_animeplanet(_make_anime_with_related(related_manga=entries))
    by_title = {e["title"]: e for entries in data["related_source_material"].values() for e in entries}
    assert by_title["Plain Manga"]["type"] == "UNKNOWN"
    assert by_title["Romance Dawn"]["type"] == "ONE SHOT"
    assert "volumes" not in by_title["No Count Manga"]
    assert "chapters" not in by_title["No Count Manga"]


# =============================================================================
# Season derivation utility
# =============================================================================


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
def test_determine_season_from_date(date_str: str, expected_season: AnimeSeason | None) -> None:
    assert determine_anime_season(date_str) == expected_season


# =============================================================================
# fetch_animeplanet_anime
# =============================================================================


@pytest.mark.usefixtures("mock_redis_cache_miss")
@patch(_PATCH_FETCH_HTML)
async def test_fetch_success_with_html_fixture(mock_fetch: AsyncMock, ap_anime_html: str) -> None:
    mock_fetch.return_value = ap_anime_html
    anime = await fetch_animeplanet_anime(_ONE_PIECE_URL)
    assert anime is not None
    assert anime["title"] == "One Piece"
    assert anime["year"] == 1999
    assert anime["season"] == "FALL"
    assert anime["status"] == "ONGOING"
    assert anime["episode_count"] == 1165
    assert anime["title_japanese"] == "ワンピース"
    assert any(p["name"] == "Toei Animation" for p in anime["producers"])


@pytest.mark.usefixtures("mock_redis_cache_miss")
@patch(_PATCH_FETCH_HTML)
async def test_fetch_accepts_non_www_url(mock_fetch: AsyncMock, ap_anime_html: str) -> None:
    mock_fetch.return_value = ap_anime_html
    anime = await fetch_animeplanet_anime("https://anime-planet.com/anime/one-piece")
    assert anime is not None
    assert any("one-piece" in s for s in anime.get("sources", []))


@pytest.mark.usefixtures("mock_redis_cache_miss")
@patch(_PATCH_FETCH_DATA, new_callable=AsyncMock)
async def test_fetch_extracts_slug_for_cache(mock_inner: AsyncMock) -> None:
    mock_inner.return_value = None
    await fetch_animeplanet_anime(_ONE_PIECE_URL)
    mock_inner.assert_called_once_with("one-piece")


@pytest.mark.usefixtures("mock_redis_cache_miss")
@patch(_PATCH_FETCH_HTML)
@pytest.mark.parametrize(
    "html",
    [None, "<html></html>", '<html><script type="application/ld+json">{"description":"no name"}</script></html>'],
)
async def test_fetch_returns_none_on_failure(mock_fetch: AsyncMock, html: str | None) -> None:
    mock_fetch.return_value = html
    assert await fetch_animeplanet_anime("https://www.anime-planet.com/anime/dandadan") is None


@pytest.mark.usefixtures("mock_redis_cache_miss")
@patch(_PATCH_FETCH_HTML)
async def test_fetch_season_from_season_url(mock_fetch: AsyncMock) -> None:
    mock_fetch.return_value = _make_html({**_BASE_JSON_LD, "startDate": "2024-07-10"})
    # entryBar in _make_html has /anime/seasons/fall-2024 → season = FALL (not SUMMER from date)
    anime = await fetch_animeplanet_anime("https://www.anime-planet.com/anime/dandadan")
    assert anime is not None
    assert anime["season"] == "FALL"


@pytest.mark.usefixtures("mock_redis_cache_miss")
@patch(_PATCH_FETCH_HTML)
async def test_fetch_season_falls_back_to_start_date(mock_fetch: AsyncMock) -> None:
    # HTML with no season link in entryBar → falls back to startDate
    html = (
        '<html><body><section class="entryBar"><span class="type">TV</span></section>'
        f'<script type="application/ld+json">{json.dumps({**_BASE_JSON_LD, "startDate": "2024-04-05"})}</script>'
        '</body></html>'
    )
    mock_fetch.return_value = html
    anime = await fetch_animeplanet_anime("https://www.anime-planet.com/anime/dandadan")
    assert anime is not None
    assert anime["season"] == "SPRING"


@pytest.mark.usefixtures("mock_redis_cache_miss")
@patch(_PATCH_FETCH_HTML)
@pytest.mark.parametrize(
    "start_date, end_date, expected_status",
    [
        ("2024-01-01", "2024-03-31", "FINISHED"),
        ("1999-10-20", None, "ONGOING"),
        ("2099-01-01", None, "UPCOMING"),
        (None, None, "UNKNOWN"),
    ],
)
async def test_fetch_status_derivation(
    mock_fetch: AsyncMock,
    start_date: str | None,
    end_date: str | None,
    expected_status: str,
) -> None:
    jld = {**_BASE_JSON_LD, "startDate": start_date, "endDate": end_date}
    mock_fetch.return_value = _make_html(jld)
    anime = await fetch_animeplanet_anime("https://www.anime-planet.com/anime/status-test")
    assert anime is not None
    assert anime["status"] == expected_status
