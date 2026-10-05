import json
import logging
from typing import Any, cast
from unittest.mock import create_autospec

import pytest
import zendriver
from enrichment.sources.anime_planet import anime_planet_anime_crawler
from enrichment.sources.anime_planet.anime_planet_anime_crawler import (
    _XPATHS,
    AnimePlanetAnimeCrawler,
    _build_anime_from_raw,
    _build_related_anime_entries,
    _build_related_manga_entries,
    _extract_anime_from_html,
    _extract_json_ld,
    _extract_slug_from_url,
    _fetch_anime_html,
    _fetch_animeplanet_anime_data,
    _normalize_anime_url,
    _parse_aggregate_rating,
    _parse_alt_title,
    _parse_rank,
    _parse_related_entry_element,
    _parse_season,
    _parse_start_year,
    fetch_animeplanet_anime,
)
from enrichment.sources.base.framework import NullRepository
from lxml import etree

ONE_PIECE_URL = "https://www.anime-planet.com/anime/one-piece"
DANDADAN_URL = "https://www.anime-planet.com/anime/dandadan"
DANDADAN_JSON_LD: dict[str, Any] = {
    "@type": "TVSeries",
    "name": "Dandadan",
    "description": "A story of ghosts and aliens.",
    "url": DANDADAN_URL,
    "startDate": "2024-10-01",
    "endDate": "2024-12-25",
    "numberOfEpisodes": 12,
    "genre": ["Action", "Comedy"],
    "aggregateRating": {"ratingValue": "4.5", "ratingCount": "1000"},
}


def _page(json_ld: dict, entry_bar: str | None = None) -> str:
    bar = (
        entry_bar
        if entry_bar is not None
        else '<span class="type">TV</span><a href="/anime/seasons/fall-2024">Fall 2024</a>'
    )
    return (
        "<html><body>"
        f'<section class="entryBar">{bar}</section>'
        f'<script type="application/ld+json">{json.dumps(json_ld)}</script>'
        "</body></html>"
    )


def _related_element(html: str, *, is_manga: bool = False) -> dict[str, Any]:
    tree = etree.fromstring(html, etree.HTMLParser())
    elements = cast(list[Any], tree.xpath("//a[contains(@class,'RelatedEntry')]"))
    return _parse_related_entry_element(elements[0], is_manga=is_manga)


@pytest.fixture
def anime_planet_browser(open_browser):
    return lambda browser: open_browser(anime_planet_anime_crawler, browser)


def test_xpaths_include_every_extracted_page_field() -> None:
    assert {
        "type_raw",
        "season_url",
        "year_text",
        "rank_text",
        "studios",
        "aka",
        "tags",
        "cover",
        "related_anime",
        "related_anime_other",
        "related_manga",
    } <= _XPATHS.keys()


@pytest.mark.parametrize(
    "identifier",
    [
        "dandadan",
        "/anime/dandadan",
        "anime/dandadan",
        DANDADAN_URL,
        "https://anime-planet.com/anime/dandadan",
    ],
)
def test_normalize_anime_url_slug_path_or_address_gives_full_www_address(
    identifier: str,
) -> None:
    assert _normalize_anime_url(identifier) == DANDADAN_URL


def test_normalize_anime_url_other_site_raises_value_error() -> None:
    with pytest.raises(ValueError, match="anime-planet"):
        _normalize_anime_url("https://www.google.com/anime/dandadan")


@pytest.mark.parametrize(
    "url", [DANDADAN_URL, "https://www.anime-planet.com/anime/dandadan?foo=bar"]
)
def test_extract_slug_from_url_anime_address_returns_slug(url: str) -> None:
    assert _extract_slug_from_url(url) == "dandadan"


def test_extract_slug_from_url_manga_address_raises_value_error() -> None:
    with pytest.raises(ValueError, match="No anime slug"):
        _extract_slug_from_url("https://www.anime-planet.com/manga/dandadan")


def test_extract_json_ld_decodes_description_and_fixes_doubled_image_host() -> None:
    page = (
        '<html><script type="application/ld+json">'
        '{"@type":"TVSeries","name":"Dandadan",'
        '"description":"A &lt;b&gt;great&lt;/b&gt; show.",'
        '"image":"https://www.anime-planet.comhttps://s4.anilist.co/cover.jpg"}'
        "</script></html>"
    )
    json_ld = _extract_json_ld(page)
    assert (json_ld["name"], json_ld["description"]) == (
        "Dandadan",
        "A <b>great</b> show.",
    )
    assert json_ld["image"] == "https://s4.anilist.co/cover.jpg"


def test_extract_json_ld_without_description_leaves_it_out() -> None:
    page = '<html><script type="application/ld+json">{"name":"Test"}</script></html>'
    assert _extract_json_ld(page) == {"name": "Test"}


@pytest.mark.parametrize(
    "page",
    [
        "<html></html>",
        '<html><script type="application/ld+json">{invalid}</script></html>',
    ],
)
def test_extract_json_ld_missing_or_invalid_block_returns_none(page: str) -> None:
    assert _extract_json_ld(page) is None


def test_parse_aggregate_rating_one_piece_page_gives_value_and_count(
    ap_anime_extracted: dict,
) -> None:
    rating = _parse_aggregate_rating(ap_anime_extracted["aggregate_rating"])
    assert rating.rating_value == pytest.approx(4.315)
    assert rating.rating_count == 64986


def test_parse_aggregate_rating_value_and_count_parsed() -> None:
    rating = _parse_aggregate_rating({"ratingValue": "4.5", "ratingCount": "1000"})
    assert (rating.rating_value, rating.rating_count) == (pytest.approx(4.5), 1000)


def test_parse_aggregate_rating_value_alone_gives_no_count() -> None:
    rating = _parse_aggregate_rating({"ratingValue": "3.2"})
    assert (rating.rating_value, rating.rating_count) == (pytest.approx(3.2), None)


def test_parse_aggregate_rating_count_alone_gives_no_value() -> None:
    rating = _parse_aggregate_rating({"ratingCount": "500"})
    assert (rating.rating_value, rating.rating_count) == (None, 500)


@pytest.mark.parametrize(
    "rating",
    [
        None,
        {},
        {"ratingValue": None, "ratingCount": None},
        {"ratingValue": "bad", "ratingCount": "bad"},
    ],
)
def test_parse_aggregate_rating_without_usable_numbers_returns_none(
    rating: Any,
) -> None:
    assert _parse_aggregate_rating(rating) is None


def test_parse_season_one_piece_page_gives_fall(ap_anime_extracted: dict) -> None:
    assert _parse_season(ap_anime_extracted["season_url"]) == "fall"


@pytest.mark.parametrize(
    ("season_url", "expected"),
    [
        ("/anime/seasons/fall-2024", "fall"),
        ("/anime/seasons/winter-1999", "winter"),
        ("/anime/seasons/spring-2023", "spring"),
        ("/anime/seasons/summer-2020", "summer"),
        ("https://www.anime-planet.com/anime/seasons/fall-2024", "fall"),
    ],
)
def test_parse_season_season_link_gives_season(season_url: str, expected: str) -> None:
    assert _parse_season(season_url) == expected


@pytest.mark.parametrize("season_url", [None, "", "/anime/not-a-season-url"])
def test_parse_season_without_season_link_returns_none(season_url: str | None) -> None:
    assert _parse_season(season_url) is None


def test_parse_rank_one_piece_page_gives_rank(ap_anime_extracted: dict) -> None:
    assert _parse_rank(ap_anime_extracted["rank_text"]) == 161


@pytest.mark.parametrize(
    ("rank_text", "expected"), [("Rank #157", 157), ("Rank #1", 1)]
)
def test_parse_rank_rank_text_gives_number(rank_text: str, expected: int) -> None:
    assert _parse_rank(rank_text) == expected


@pytest.mark.parametrize("rank_text", [None, "", "no hash here"])
def test_parse_rank_without_rank_number_returns_none(rank_text: str | None) -> None:
    assert _parse_rank(rank_text) is None


def test_parse_start_year_one_piece_page_gives_1999(ap_anime_extracted: dict) -> None:
    assert _parse_start_year(ap_anime_extracted["year_text"]) == 1999


@pytest.mark.parametrize(
    ("year_text", "expected"),
    [(" 2002 ", 2002), (" 1999 - ? ", 1999), ("2019 - 2021", 2019)],
)
def test_parse_start_year_year_or_range_gives_first_year(
    year_text: str, expected: int
) -> None:
    assert _parse_start_year(year_text) == expected


@pytest.mark.parametrize("year_text", [None, ""])
def test_parse_start_year_without_year_returns_none(year_text: str | None) -> None:
    assert _parse_start_year(year_text) is None


def test_parse_alt_title_one_piece_page_gives_japanese_title(
    ap_anime_extracted: dict,
) -> None:
    assert _parse_alt_title(ap_anime_extracted["aka"]) == "ワンピース"


@pytest.mark.parametrize(
    ("aka", "expected"),
    [
        ("Alt title: ダンダダン", "ダンダダン"),
        ("alt title: ワンピース", "ワンピース"),
        ("ALT TITLE:  Bleach  ", "Bleach"),
        ("ダンダダン", "ダンダダン"),
    ],
)
def test_parse_alt_title_label_removed_from_title(aka: str, expected: str) -> None:
    assert _parse_alt_title(aka) == expected


@pytest.mark.parametrize("aka", [None, "", "   "])
def test_parse_alt_title_without_title_returns_none(aka: str | None) -> None:
    assert _parse_alt_title(aka) is None


def test_parse_related_entry_element_anime_entry_reads_type_and_image() -> None:
    entry = _related_element(
        """<html><body><a href="/anime/test-sequel" class="RelatedEntry">
          <p class="RelatedEntry__name">Test Sequel</p>
          <span class="RelatedEntry__subtitle">Sequel</span>
          <ul>
            <li><i class="fa-tv"></i><span class="RelatedEntry__metadata_item">TV: 12 ep</span></li>
            <li><i class="fa-calendar"></i><span class="RelatedEntry__metadata_item">2024</span></li>
          </ul>
          <img class="RelatedEntry__image" src="https://example.com/img.jpg" />
        </a></body></html>"""
    )
    assert entry == {
        "url": "/anime/test-sequel",
        "title": "Test Sequel",
        "relation_subtype": "Sequel",
        "type": "TV: 12 ep",
        "image": "https://example.com/img.jpg",
    }


def test_parse_related_entry_element_manga_entry_reads_volumes_and_chapters() -> None:
    entry = _related_element(
        """<html><body><a href="/manga/dandadan" class="RelatedEntry">
          <p class="RelatedEntry__name">Dandadan</p>
          <ul>
            <li><i class="fa-book-open"></i><span class="RelatedEntry__metadata_item">Vol: 24 - Ch: 236</span></li>
          </ul>
        </a></body></html>""",
        is_manga=True,
    )
    assert (entry["url"], entry["title"], entry["vol_ch"]) == (
        "/manga/dandadan",
        "Dandadan",
        "Vol: 24 - Ch: 236",
    )
    assert entry["relation_subtype"] is None
    assert "type" not in entry


def test_extract_anime_from_html_one_piece_page_reads_page_data(
    ap_anime_html: str,
) -> None:
    raw = _extract_anime_from_html(ap_anime_html)
    assert (raw["name"], raw["schema_type"], raw["number_of_episodes"]) == (
        "One Piece",
        "TVSeries",
        1165,
    )
    assert (raw["start_date"], raw["end_date"]) == ("1999-10-20", None)
    assert "Action" in raw["genres"]
    assert raw["aggregate_rating"]["ratingValue"] == pytest.approx(4.315)
    assert "slug" not in raw


def test_extract_anime_from_html_one_piece_page_reads_entry_bar(
    ap_anime_html: str,
) -> None:
    raw = _extract_anime_from_html(ap_anime_html)
    assert "TV" in raw["type_raw"]
    assert raw["season_url"] == "/anime/seasons/fall-1999"
    assert "161" in raw["rank_text"]
    assert raw["studios"] == [
        {
            "name": "Toei Animation",
            "url": "https://www.anime-planet.com/anime/studios/toei-animation",
        }
    ]


def test_extract_anime_from_html_one_piece_page_reads_title_tags_cover_and_relations(
    ap_anime_html: str,
) -> None:
    raw = _extract_anime_from_html(ap_anime_html)
    assert raw["aka"] == "Alt title: ワンピース"
    assert "Shounen" in raw["tags"]
    assert "one-piece" in raw["cover"]
    assert (
        len(raw["related_anime_raw"]),
        len(raw["related_anime_other_raw"]),
        len(raw["related_manga_raw"]),
    ) == (67, 17, 24)


def test_extract_anime_from_html_studio_link_without_name_skipped() -> None:
    raw = _extract_anime_from_html(
        _page(
            DANDADAN_JSON_LD,
            entry_bar='<a href="/anime/studios/blank"> </a>'
            '<a href="/anime/studios/science-saru">Science SARU</a>',
        )
    )
    assert raw["studios"] == [
        {
            "name": "Science SARU",
            "url": "https://www.anime-planet.com/anime/studios/science-saru",
        }
    ]


@pytest.mark.parametrize(
    "page",
    [
        "",
        "<html><body><p>no json-ld here</p></body></html>",
        '<html><body><script type="application/ld+json">{"description": "no name"}</script></body></html>',
    ],
)
def test_extract_anime_from_html_without_named_page_data_returns_none(
    page: str,
) -> None:
    assert _extract_anime_from_html(page) is None


def test_extract_anime_from_html_unencodable_character_logs_parse_failure(
    caplog,
) -> None:
    page = _page(DANDADAN_JSON_LD).replace("<body>", "<body>\ud800")
    with caplog.at_level(logging.ERROR):
        assert _extract_anime_from_html(page) is None
    assert "Failed to parse anime page HTML" in caplog.messages


def test_build_related_anime_entries_one_piece_page_reads_types_and_episodes(
    ap_anime_extracted: dict,
) -> None:
    entries = _build_related_anime_entries(ap_anime_extracted["related_anime_raw"])
    assert len(entries) == 67
    ganzak = next(entry for entry in entries if "ganzak" in entry.slug)
    assert (ganzak.type, ganzak.episode_count) == ("OVA", 1)
    assert (
        next(entry for entry in entries if entry.slug == "the-one-piece").type == "Web"
    )


def test_build_related_anime_entries_one_piece_other_franchise_reads_every_entry(
    ap_anime_extracted: dict,
) -> None:
    entries = _build_related_anime_entries(
        ap_anime_extracted["related_anime_other_raw"]
    )
    assert len(entries) == 17


@pytest.mark.parametrize(
    ("raw_type", "expected_type", "expected_episodes"),
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
def test_build_related_anime_entries_type_text_gives_type_and_episode_count(
    raw_type: str | None, expected_type: str | None, expected_episodes: int | None
) -> None:
    entry = _build_related_anime_entries(
        [{"url": "/anime/slug", "title": "T", "type": raw_type}]
    )[0]
    assert (entry.type, entry.episode_count) == (expected_type, expected_episodes)


def test_build_related_anime_entries_without_anime_address_skipped() -> None:
    entries = _build_related_anime_entries(
        [
            {"url": "/anime/valid", "title": "Valid", "type": "Movie"},
            {"url": "", "title": "No URL"},
            {"url": "/manga/wrong", "title": "Wrong domain"},
        ]
    )
    assert [entry.slug for entry in entries] == ["valid"]


def test_build_related_anime_entries_empty_relation_gives_none() -> None:
    entry = _build_related_anime_entries(
        [{"url": "/anime/slug", "title": "T", "relation_subtype": ""}]
    )[0]
    assert entry.relation_subtype is None


def test_build_related_manga_entries_one_piece_page_reads_volumes_and_chapters(
    ap_anime_extracted: dict,
) -> None:
    entries = _build_related_manga_entries(ap_anime_extracted["related_manga_raw"])
    assert len(entries) == 24
    romance_dawn = next(entry for entry in entries if entry.slug == "romance-dawn")
    assert (romance_dawn.type, romance_dawn.chapters) == ("One Shot", 1)
    main = next(entry for entry in entries if entry.slug == "one-piece")
    assert (main.volumes, main.chapters) == (114, 1184)


def test_build_related_manga_entries_without_manga_address_skipped() -> None:
    entries = _build_related_manga_entries(
        [
            {"url": "/manga/valid", "title": "Valid Manga", "vol_ch": "Vol: 1"},
            {"url": "", "title": "No URL"},
            {"url": "/anime/wrong-domain", "title": "Wrong domain"},
        ]
    )
    assert [entry.slug for entry in entries] == ["valid"]


@pytest.mark.parametrize(
    ("vol_ch", "expected"),
    [
        ("One Shot", ("One Shot", None, 1)),
        ("one shot", ("One Shot", None, 1)),
        ("Vol: 114 - Ch: 1184+", (None, 114, 1184)),
        ("Vol: 1 - Ch: 3", (None, 1, 3)),
        ("Vol: 1", (None, 1, None)),
        ("Ch: 19", (None, None, 19)),
        ("", (None, None, None)),
        (None, (None, None, None)),
        ("- ?", (None, None, None)),
    ],
)
def test_build_related_manga_entries_volume_text_gives_type_volumes_and_chapters(
    vol_ch: str | None, expected: tuple
) -> None:
    entry = _build_related_manga_entries(
        [{"url": "/manga/slug", "title": "T", "vol_ch": vol_ch}]
    )[0]
    assert (entry.type, entry.volumes, entry.chapters) == expected


def test_build_anime_from_raw_one_piece_page_builds_model(
    ap_anime_extracted: dict,
) -> None:
    anime = _build_anime_from_raw(ap_anime_extracted)
    assert (anime.name, anime.slug, anime.season, anime.rank) == (
        "One Piece",
        "one-piece",
        "fall",
        161,
    )
    assert (anime.alt_title, anime.number_of_episodes, anime.start_year) == (
        "ワンピース",
        1165,
        1999,
    )
    assert [(studio.name, studio.url) for studio in anime.studios] == [
        ("Toei Animation", "https://www.anime-planet.com/anime/studios/toei-animation")
    ]
    assert "Shounen" in anime.tags and "Action" in anime.genres
    assert anime.aggregate_rating.rating_count == 64986
    assert (
        len(anime.related_anime),
        len(anime.related_anime_other),
        len(anime.related_manga),
    ) == (67, 17, 24)
    assert "one-piece" in anime.cover


def test_build_anime_from_raw_rank_text_gives_rank(ap_anime_extracted: dict) -> None:
    assert (
        _build_anime_from_raw({**ap_anime_extracted, "rank_text": "Rank #42"}).rank
        == 42
    )


@pytest.mark.parametrize(
    ("field", "attribute"),
    [
        ("season_url", "season"),
        ("aka", "alt_title"),
        ("aggregate_rating", "aggregate_rating"),
    ],
)
def test_build_anime_from_raw_missing_page_value_gives_none(
    ap_anime_extracted: dict, field: str, attribute: str
) -> None:
    anime = _build_anime_from_raw({**ap_anime_extracted, field: None})
    assert getattr(anime, attribute) is None


def test_get_extraction_schema_returns_xpaths() -> None:
    assert AnimePlanetAnimeCrawler(NullRepository()).get_extraction_schema() is _XPATHS


def test_normalize_identifier_slug_gives_full_address() -> None:
    crawler = AnimePlanetAnimeCrawler(NullRepository())
    assert crawler.normalize_identifier("dandadan") == DANDADAN_URL


async def test_fetch_anime_html_page_with_entry_bar_returns_page_content(
    anime_planet_browser, browser_serving
) -> None:
    page = _page(DANDADAN_JSON_LD)
    with anime_planet_browser(browser_serving({DANDADAN_URL: page})):
        assert await _fetch_anime_html(DANDADAN_URL) == page


async def test_fetch_anime_html_navigation_error_logs_and_returns_none(
    anime_planet_browser, caplog
) -> None:
    browser = create_autospec(zendriver.Browser, instance=True)
    browser.get.side_effect = ConnectionError("connection refused")
    with anime_planet_browser(browser), caplog.at_level(logging.WARNING):
        assert await _fetch_anime_html(DANDADAN_URL) is None
    assert any(
        message.startswith(f"navigation failed for {DANDADAN_URL}")
        for message in caplog.messages
    )


async def test_fetch_animeplanet_anime_data_page_gives_raw_data_with_slug(
    anime_planet_browser, browser_serving
) -> None:
    with anime_planet_browser(browser_serving({DANDADAN_URL: _page(DANDADAN_JSON_LD)})):
        raw = await _fetch_animeplanet_anime_data("dandadan")
    assert (raw["name"], raw["slug"]) == ("Dandadan", "dandadan")


async def test_fetch_animeplanet_anime_data_navigation_error_returns_none(
    anime_planet_browser, caplog
) -> None:
    browser = create_autospec(zendriver.Browser, instance=True)
    browser.get.side_effect = ConnectionError("connection refused")
    with anime_planet_browser(browser), caplog.at_level(logging.WARNING):
        assert await _fetch_animeplanet_anime_data("dandadan") is None
    assert f"Navigation returned no HTML for {DANDADAN_URL}" in caplog.messages


async def test_fetch_animeplanet_anime_data_page_without_data_returns_none(
    anime_planet_browser, browser_serving, caplog
) -> None:
    page = '<html><body><section class="entryBar"></section></body></html>'
    with (
        anime_planet_browser(browser_serving({DANDADAN_URL: page})),
        caplog.at_level(logging.WARNING),
    ):
        assert await _fetch_animeplanet_anime_data("dandadan") is None
    assert f"No data extracted from {DANDADAN_URL}" in caplog.messages


async def test_fetch_animeplanet_anime_one_piece_page_returns_canonical_anime(
    anime_planet_browser, browser_serving, ap_anime_html: str
) -> None:
    with anime_planet_browser(browser_serving({ONE_PIECE_URL: ap_anime_html})):
        anime = await fetch_animeplanet_anime(ONE_PIECE_URL)
    assert (anime["title"], anime["year"], anime["season"], anime["status"]) == (
        "One Piece",
        1999,
        "FALL",
        "ONGOING",
    )
    assert (anime["episode_count"], anime["title_japanese"]) == (1165, "ワンピース")
    assert any(company["name"] == "Toei Animation" for company in anime["companies"])


async def test_fetch_animeplanet_anime_address_without_www_fetches_www_page(
    anime_planet_browser, browser_serving, ap_anime_html: str
) -> None:
    with anime_planet_browser(browser_serving({ONE_PIECE_URL: ap_anime_html})):
        anime = await fetch_animeplanet_anime(
            "https://anime-planet.com/anime/one-piece"
        )
    assert anime["title"] == "One Piece"


@pytest.mark.parametrize(
    "page",
    [
        '<html><body><section class="entryBar"></section></body></html>',
        '<html><body><section class="entryBar"></section><script type="application/ld+json">{"description":"no name"}</script></body></html>',
    ],
)
async def test_fetch_animeplanet_anime_unusable_page_returns_none(
    anime_planet_browser, browser_serving, page: str
) -> None:
    with anime_planet_browser(browser_serving({DANDADAN_URL: page})):
        assert await fetch_animeplanet_anime(DANDADAN_URL) is None


async def test_fetch_animeplanet_anime_season_link_wins_over_start_date(
    anime_planet_browser, browser_serving
) -> None:
    page = _page({**DANDADAN_JSON_LD, "startDate": "2024-07-10"})
    with anime_planet_browser(browser_serving({DANDADAN_URL: page})):
        anime = await fetch_animeplanet_anime(DANDADAN_URL)
    assert anime["season"] == "FALL"


async def test_fetch_animeplanet_anime_without_season_link_derives_season_from_start_date(
    anime_planet_browser, browser_serving
) -> None:
    page = _page(
        {**DANDADAN_JSON_LD, "startDate": "2024-04-05"},
        entry_bar='<span class="type">TV</span>',
    )
    with anime_planet_browser(browser_serving({DANDADAN_URL: page})):
        anime = await fetch_animeplanet_anime(DANDADAN_URL)
    assert anime["season"] == "SPRING"


@pytest.mark.parametrize(
    ("start_date", "end_date", "expected"),
    [
        ("2024-01-01", "2024-03-31", "FINISHED"),
        ("1999-10-20", None, "ONGOING"),
        ("2099-01-01", None, "UPCOMING"),
        (None, None, "UNKNOWN"),
    ],
)
async def test_fetch_animeplanet_anime_dates_give_status(
    anime_planet_browser,
    browser_serving,
    start_date: str | None,
    end_date: str | None,
    expected: str,
) -> None:
    page = _page({**DANDADAN_JSON_LD, "startDate": start_date, "endDate": end_date})
    with anime_planet_browser(browser_serving({DANDADAN_URL: page})):
        anime = await fetch_animeplanet_anime(DANDADAN_URL)
    assert anime["status"] == expected
