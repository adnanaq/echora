from collections.abc import Awaitable, Callable
from unittest.mock import patch

import pytest
from enrichment.sources.anisearch import anisearch_anime_crawler as crawler_module
from enrichment.sources.anisearch.anisearch_anime_crawler import (
    _XPATHS,
    BASE_ANIME_URL,
    AniSearchAnimeCrawler,
    _build_anime_from_raw,
    _extract_anime_from_html,
    _extract_path_from_url,
    _extract_relations_from_html,
    _fetch_anisearch_anime_data,
    _parse_relations,
    _process_relation_tooltips,
    fetch_anisearch_anime,
)
from enrichment.sources.anisearch.anisearch_anime_models import AniSearchAnime
from enrichment.sources.base.exceptions import (
    ServiceBlockedError,
    ServiceNetworkError,
    ServiceParseError,
)
from enrichment.sources.base.framework import NullRepository
from enrichment.sources.base.polite_http import FetchedPage

_URL = "https://www.anisearch.com/anime/2227,one-piece"


@pytest.fixture(scope="session")
def one_piece_processed(one_piece_main_raw, one_piece_relations_raw):
    anime_rels, manga_rels = _parse_relations(one_piece_relations_raw)
    return {
        **one_piece_main_raw,
        "anime_relations": anime_rels,
        "manga_relations": manga_rels,
    }


def test_xpaths_has_required_main_keys() -> None:
    assert {
        "cover_image",
        "title_blocks",
        "type",
        "status",
        "published",
        "studio",
        "studio_url",
        "broadcast_raw",
        "source_material",
        "synonyms",
        "description",
        "genres",
        "tags",
        "rating_score",
        "rank_toplist",
        "rank_trending",
        "websites",
    } <= set(_XPATHS)


def test_xpaths_has_relation_keys() -> None:
    assert {"anime_relation_rows", "manga_relation_rows"} <= set(_XPATHS)


def test_xpaths_cover_image_targets_details_cover() -> None:
    assert "details-cover" in _XPATHS["cover_image"]
    assert _XPATHS["cover_image"].endswith("/@src")


def test_xpaths_title_blocks_targets_information_titles_with_language() -> None:
    assert "information" in _XPATHS["title_blocks"]
    assert "@lang" in _XPATHS["title_blocks"]


def test_xpaths_genres_anchor_on_genre_href() -> None:
    assert (
        "/genre/main/" in _XPATHS["genres"] or "/genre/subsidiary/" in _XPATHS["genres"]
    )


def test_xpaths_tags_anchor_on_tag_href() -> None:
    assert "/genre/tag/" in _XPATHS["tags"]


def test_xpaths_relations_target_correct_sections() -> None:
    assert "relations_anime" in _XPATHS["anime_relation_rows"]
    assert "relations_manga" in _XPATHS["manga_relation_rows"]
    assert "tbody" in _XPATHS["anime_relation_rows"]


def test_extract_anime_from_html_one_piece_page_reads_every_title_block(
    one_piece_main_html,
) -> None:
    raw = _extract_anime_from_html(one_piece_main_html)
    assert raw is not None
    assert raw["title_blocks"][0] == {
        "language": "ja",
        "title": "One Piece",
        "native_titles": ["ワンピース"],
    }
    assert [block["language"] for block in raw["title_blocks"]] == [
        "ja",
        "en",
        "de",
        "es",
        "fr",
        "it",
    ]


def test_extract_anime_from_html_chinese_work_with_japanese_title_reads_native_lines_in_page_order(
    mo_dao_zu_shi_main_html,
) -> None:
    raw = _extract_anime_from_html(mo_dao_zu_shi_main_html)
    assert raw is not None
    assert raw["title_blocks"][:2] == [
        {
            "language": "zh",
            "title": "Mo Dao Zu Shi: Qianchen Pian",
            "native_titles": ["魔道祖师 前尘篇"],
        },
        {
            "language": "ja",
            "title": "Madou Soshi: Zenjin Hen",
            "native_titles": ["Madō Soshi: Zenjin Hen", "魔道祖師 前塵編"],
        },
    ]


def test_extract_anime_from_html_one_piece_page_reads_cover_address(
    one_piece_main_html,
) -> None:
    raw = _extract_anime_from_html(one_piece_main_html)
    assert raw is not None
    assert raw["cover_image"] is not None
    assert raw["cover_image"].startswith("https://")


def test_extract_anime_from_html_one_piece_page_reads_type(one_piece_main_html) -> None:
    raw = _extract_anime_from_html(one_piece_main_html)
    assert raw is not None
    assert "TV-Series" in (raw["type"] or "")


def test_extract_anime_from_html_one_piece_page_reads_genres(
    one_piece_main_html,
) -> None:
    raw = _extract_anime_from_html(one_piece_main_html)
    assert raw is not None
    assert len(raw["genres"]) > 0
    assert all(isinstance(genre, str) for genre in raw["genres"])


def test_extract_anime_from_html_one_piece_page_reads_tags(one_piece_main_html) -> None:
    raw = _extract_anime_from_html(one_piece_main_html)
    assert raw is not None
    assert len(raw["tags"]) > 0


def test_extract_anime_from_html_one_piece_page_reads_websites(
    one_piece_main_html,
) -> None:
    raw = _extract_anime_from_html(one_piece_main_html)
    assert raw is not None
    assert len(raw["websites"]) > 0
    assert all(w["url"] for w in raw["websites"])


def test_extract_anime_from_html_one_piece_page_reads_studio(
    one_piece_main_html,
) -> None:
    raw = _extract_anime_from_html(one_piece_main_html)
    assert raw is not None
    assert raw["studio"] == "Toei Animation Co., Ltd."
    assert "toei-animation" in (raw["studio_url"] or "")


def test_extract_anime_from_html_one_piece_page_reads_rating_and_votes(
    one_piece_main_html,
) -> None:
    raw = _extract_anime_from_html(one_piece_main_html)
    assert raw is not None
    assert raw["rating_score"] is not None
    assert "." in raw["rating_score"]
    assert raw["rating_votes"] == 7902


def test_extract_anime_from_html_empty_returns_none() -> None:
    assert _extract_anime_from_html("") is None


def test_extract_anime_from_html_malformed_page_returns_empty_fields() -> None:
    raw = _extract_anime_from_html("<not valid xml at all >>>")
    assert (raw["title_blocks"], raw["genres"], raw["websites"]) == ([], [], [])


def test_extract_relations_from_html_one_piece_page_reads_every_anime_relation(
    one_piece_relations_html,
) -> None:
    raw = _extract_relations_from_html(one_piece_relations_html)
    assert raw is not None
    assert len(raw["anime_relations"]) == 79


def test_extract_relations_from_html_one_piece_page_reads_every_manga_relation(
    one_piece_relations_html,
) -> None:
    raw = _extract_relations_from_html(one_piece_relations_html)
    assert raw is not None
    assert len(raw["manga_relations"]) == 2


def test_extract_relations_from_html_one_piece_page_reads_relation_fields(
    one_piece_relations_html,
) -> None:
    raw = _extract_relations_from_html(one_piece_relations_html)
    assert raw is not None
    entry = raw["anime_relations"][0]
    assert entry["relation_type"] is not None
    assert entry["title"] is not None
    assert entry["url"] is not None
    assert entry["details"] is not None


def test_extract_relations_from_html_one_piece_page_lists_original_manga(
    one_piece_relations_html,
) -> None:
    raw = _extract_relations_from_html(one_piece_relations_html)
    assert raw is not None
    titles = [r["title"] for r in raw["manga_relations"]]
    assert "One Piece" in titles


def test_extract_relations_from_html_one_piece_page_keeps_image_tooltips(
    one_piece_relations_html,
) -> None:
    raw = _extract_relations_from_html(one_piece_relations_html)
    assert raw is not None
    images = [r["image"] for r in raw["anime_relations"] if r.get("image")]
    assert len(images) > 0
    assert all("<img" in img for img in images)


def test_extract_relations_from_html_empty_returns_none() -> None:
    assert _extract_relations_from_html("") is None


def test_extract_path_from_url_anime_address_returns_path() -> None:
    assert _extract_path_from_url(_URL) == "2227,one-piece"


def test_extract_path_from_url_trailing_slash_removed() -> None:
    assert _extract_path_from_url(_URL + "/") == "2227,one-piece"


def test_extract_path_from_url_other_site_raises_value_error() -> None:
    with pytest.raises(ValueError, match="URL must start with"):
        _extract_path_from_url("https://myanimelist.net/anime/21")


def test_extract_path_from_url_without_anime_path_raises_value_error() -> None:
    with pytest.raises(ValueError, match="does not contain anime path"):
        _extract_path_from_url(BASE_ANIME_URL)


def test_process_relation_tooltips_image_tag_becomes_address() -> None:
    rel = {
        "image": '<img src="https://cdn.anisearch.com/images/anime/cover/2/2227.webp" />'
    }
    _process_relation_tooltips([rel])
    assert rel["image"] == "https://cdn.anisearch.com/images/anime/cover/2/2227.webp"


def test_process_relation_tooltips_escaped_image_tag_becomes_address() -> None:
    escaped = "&lt;img src=&quot;https://cdn.anisearch.com/cover.webp&quot;&gt;"
    rel = {"image": escaped}
    _process_relation_tooltips([rel])
    assert rel["image"] == "https://cdn.anisearch.com/cover.webp"


def test_process_relation_tooltips_without_image_leaves_relation_unchanged() -> None:
    rel = {"title": "Test"}
    _process_relation_tooltips([rel])
    assert rel == {"title": "Test"}


def test_process_relation_tooltips_text_without_image_tag_left_unchanged() -> None:
    rel = {"image": "no img tag here"}
    _process_relation_tooltips([rel])
    assert rel["image"] == "no img tag here"


def test_process_relation_tooltips_empty_list_stays_empty() -> None:
    relations: list[dict] = []
    _process_relation_tooltips(relations)
    assert relations == []


def test_process_relation_tooltips_one_piece_page_gives_image_addresses(
    one_piece_relations_raw,
) -> None:
    rels = list(one_piece_relations_raw["anime_relations"])
    _process_relation_tooltips(rels)
    for rel in rels:
        if rel.get("image"):
            assert not rel["image"].startswith("<")
            assert rel["image"].startswith("https://")


def _anime_from_page(raw: dict) -> AniSearchAnime:
    return _build_anime_from_raw(raw, _URL)


def test_build_anime_from_raw_chinese_work_reads_title_fields_from_title_blocks(
    balala_the_fairies_2_main_html,
) -> None:
    raw = _extract_anime_from_html(balala_the_fairies_2_main_html)
    assert raw is not None
    anime = _anime_from_page(raw)
    assert (anime.title, anime.title_english, anime.title_japanese, anime.titles) == (
        "Ba La La Xiao Mo Xian Zhi: Qiji Wubu",
        "Balala the Fairies 2",
        None,
        {"zh": "巴啦啦小魔仙之奇迹舞步"},
    )


def test_build_anime_from_raw_type_label_and_details_removed(
    one_piece_main_raw,
) -> None:
    assert _anime_from_page(one_piece_main_raw).type == "TV-Series"


@pytest.mark.parametrize(
    ("page_fixture", "episode_count"),
    [
        ("balala_the_fairies_2_main_html", 52),
        ("mo_dao_zu_shi_main_html", 15),
        ("twelve_sai_main_html", 12),
    ],
)
def test_build_anime_from_raw_saved_page_reads_episode_count(
    request, page_fixture: str, episode_count: int
) -> None:
    raw = _extract_anime_from_html(request.getfixturevalue(page_fixture))
    assert raw is not None
    assert _anime_from_page(raw).episode_count == episode_count


def test_build_anime_from_raw_estimated_episode_count_reads_unknown(
    one_piece_main_html,
) -> None:
    raw = _extract_anime_from_html(one_piece_main_html)
    assert raw is not None
    assert _anime_from_page(raw).episode_count is None


@pytest.mark.parametrize(
    ("page_fixture", "is_estimated"),
    [
        ("one_piece_main_html", True),
        ("balala_the_fairies_2_main_html", False),
        ("mo_dao_zu_shi_main_html", False),
        ("twelve_sai_main_html", False),
    ],
)
def test_extract_anime_from_html_saved_page_reports_estimated_episode_count(
    request, page_fixture: str, is_estimated: bool
) -> None:
    raw = _extract_anime_from_html(request.getfixturevalue(page_fixture))
    assert raw is not None
    assert raw["is_episode_count_estimated"] is is_estimated


def test_extract_anime_from_html_runtime_also_unconfirmed_reports_estimated_episode_count(
    balala_the_fairies_2_main_html,
) -> None:
    page_html = balala_the_fairies_2_main_html.replace(
        "TV-Series, 52",
        'TV-Series, <span class="issue-info showpop" data-tooltip="Number of '
        'episodes: preliminary estimate&lt;br&gt;Runtime: not confirmed"></span>52',
        1,
    )
    raw = _extract_anime_from_html(page_html)
    assert raw is not None
    assert raw["is_episode_count_estimated"] is True


@pytest.mark.parametrize(
    ("page_fixture", "synonyms"),
    [
        ("one_piece_main_html", ["OP", "OneP"]),
        (
            "balala_the_fairies_2_main_html",
            ["Balala the Fairies Season 2", "Balala the Fairies: Miracle Dance"],
        ),
        (
            "mo_dao_zu_shi_main_html",
            [
                "Founder of Diabolism",
                "Grandmaster of Demonic Cultivation",
                "Madou Soshi (魔道祖師)",
                "Modao Zushi",
                "The Founder of Evil Magic",
                "The Master of Diabolism",
                "Großmeister der dämonischen Kultivierung: Die Vergangenheit",
            ],
        ),
        ("twelve_sai_main_html", ["Age 12: Kiss, Hate, Love"]),
    ],
)
def test_extract_anime_from_html_saved_page_reads_one_synonym_per_name(
    request, page_fixture: str, synonyms: list[str]
) -> None:
    raw = _extract_anime_from_html(request.getfixturevalue(page_fixture))
    assert raw is not None
    assert raw["synonyms"] == synonyms


def test_build_anime_from_raw_status_label_removed(one_piece_main_raw) -> None:
    assert _anime_from_page(one_piece_main_raw).status == "Ongoing"


def test_build_anime_from_raw_open_range_gives_start_date_only(
    one_piece_main_raw,
) -> None:
    anime = _anime_from_page(one_piece_main_raw)
    assert anime.start_date == "1999-10-20"
    assert anime.end_date is None


def test_build_anime_from_raw_closed_range_gives_both_dates(one_piece_main_raw) -> None:
    raw = {**one_piece_main_raw, "published": "Published: 20.10.1999 - 31.03.2002"}
    anime = _anime_from_page(raw)
    assert anime.start_date == "1999-10-20"
    assert anime.end_date == "2002-03-31"


def test_build_anime_from_raw_single_date_gives_start_date_only(
    one_piece_main_raw,
) -> None:
    raw = {**one_piece_main_raw, "published": "Published: 05.04.2003"}
    anime = _anime_from_page(raw)
    assert anime.start_date == "2003-04-05"
    assert anime.end_date is None


def test_build_anime_from_raw_without_published_date_gives_no_dates(
    one_piece_main_raw,
) -> None:
    raw = {**one_piece_main_raw, "published": None}
    anime = _anime_from_page(raw)
    assert anime.start_date is None
    assert anime.end_date is None


@pytest.mark.parametrize(
    ("published", "expected"),
    [
        ("Published: 11.2008 ‑ ?", (None, None, 2008, "November")),
        ("Published: 01.2027 ‑ ?", (None, None, 2027, "January")),
        ("Published: 2027 ‑ ?", (None, None, 2027, None)),
        ("Published: 1983", (None, None, 1983, None)),
        ("Published: ?", (None, None, None, None)),
        ("Published: 20.10.1999 ‑ ?", ("1999-10-20", None, 1999, None)),
        ("Published: 20.10.1999‑31.03.2002", ("1999-10-20", "2002-03-31", 1999, None)),
        ("Published: 05.2001 ‑ 31.03.2002", (None, "2002-03-31", 2001, "May")),
    ],
)
def test_build_anime_from_raw_published_date_keeps_only_stated_precision(
    one_piece_main_raw, published: str, expected: tuple
) -> None:
    anime = _anime_from_page({**one_piece_main_raw, "published": published})
    assert (
        anime.start_date,
        anime.end_date,
        anime.start_year,
        anime.start_month,
    ) == expected


def test_build_anime_from_raw_broadcast_gives_day_time_and_zone(
    one_piece_main_raw,
) -> None:
    anime = _anime_from_page(one_piece_main_raw)
    assert (anime.broadcast_day, anime.broadcast_time, anime.broadcast_timezone) == (
        "Sunday",
        "23:15",
        "JST",
    )


def test_build_anime_from_raw_without_broadcast_gives_no_broadcast_parts(
    one_piece_main_raw,
) -> None:
    anime = _anime_from_page({**one_piece_main_raw, "broadcast_raw": None})
    assert (anime.broadcast_day, anime.broadcast_time, anime.broadcast_timezone) == (
        None,
        None,
        None,
    )


def test_build_anime_from_raw_relative_studio_path_gives_full_address(
    one_piece_main_raw,
) -> None:
    anime = _anime_from_page(one_piece_main_raw)
    assert (
        anime.studio_url
        == "https://www.anisearch.com/company/412,toei-animation-co-ltd"
    )


def test_build_anime_from_raw_studio_path_with_leading_slash_gives_full_address(
    one_piece_main_raw,
) -> None:
    raw = {**one_piece_main_raw, "studio_url": "/company/412,toei-animation-co-ltd"}
    anime = _anime_from_page(raw)
    assert (
        anime.studio_url
        == "https://www.anisearch.com/company/412,toei-animation-co-ltd"
    )


def test_build_anime_from_raw_without_studio_path_gives_no_studio_address(
    one_piece_main_raw,
) -> None:
    assert (
        _anime_from_page({**one_piece_main_raw, "studio_url": None}).studio_url is None
    )


def test_build_anime_from_raw_source_material_label_removed(one_piece_main_raw) -> None:
    assert _anime_from_page(one_piece_main_raw).source_material == "Manga"


def test_build_anime_from_raw_synonyms_kept_as_read(one_piece_main_raw) -> None:
    assert _anime_from_page(one_piece_main_raw).synonyms == ["OP", "OneP"]


def test_build_anime_from_raw_synonym_with_commas_kept_whole(
    one_piece_main_raw,
) -> None:
    raw = {**one_piece_main_raw, "synonyms": ["Age 12: Kiss, Hate, Love"]}
    assert _anime_from_page(raw).synonyms == ["Age 12: Kiss, Hate, Love"]


def test_build_anime_from_raw_without_synonyms_gives_empty_list(
    one_piece_main_raw,
) -> None:
    assert _anime_from_page({**one_piece_main_raw, "synonyms": None}).synonyms == []


def test_extract_anime_from_html_one_piece_page_gives_genre_names(
    one_piece_main_html,
) -> None:
    raw = _extract_anime_from_html(one_piece_main_html)
    assert raw is not None
    assert "Action" in raw["genres"]
    assert "Fighting-Shounen" in raw["genres"]
    assert all(isinstance(genre, str) for genre in raw["genres"])


def test_extract_anime_from_html_one_piece_page_gives_tag_names(
    one_piece_main_html,
) -> None:
    raw = _extract_anime_from_html(one_piece_main_html)
    assert raw is not None
    assert "Pirate" in raw["tags"]
    assert all(isinstance(tag, str) for tag in raw["tags"])


def test_build_anime_from_raw_websites_kept(one_piece_main_raw) -> None:
    anime = _anime_from_page(one_piece_main_raw)
    assert len(anime.websites) == len(one_piece_main_raw["websites"])
    assert all(website["url"] for website in anime.websites)


def test_build_anime_from_raw_rating_gives_score_and_votes(one_piece_main_raw) -> None:
    statistics = _anime_from_page(one_piece_main_raw).statistics
    assert statistics is not None
    assert statistics.score == pytest.approx(4.18)
    assert statistics.scored_by == 7902


def test_build_anime_from_raw_zero_votes_omits_votes(one_piece_main_raw) -> None:
    statistics = _anime_from_page({**one_piece_main_raw, "rating_votes": 0}).statistics
    assert statistics is not None
    assert statistics.scored_by is None


def test_build_anime_from_raw_unrated_anime_omits_score_and_votes(
    one_piece_main_raw,
) -> None:
    raw = {
        **one_piece_main_raw,
        "rating_score": "Calculated Value0.00 = 0%",
        "rating_votes": 0,
    }
    statistics = _anime_from_page(raw).statistics
    assert statistics is not None
    assert (statistics.score, statistics.scored_by) == (None, None)


def test_build_anime_from_raw_toplist_rank_gives_rank(one_piece_main_raw) -> None:
    statistics = _anime_from_page(one_piece_main_raw).statistics
    assert statistics is not None
    assert statistics.rank == 125


def test_build_anime_from_raw_rank_in_thousands_gives_whole_rank(
    one_piece_main_raw,
) -> None:
    statistics = _anime_from_page(
        {**one_piece_main_raw, "rank_toplist": "#12.829"}
    ).statistics
    assert statistics is not None
    assert statistics.rank == 12829


def test_build_anime_from_raw_votes_in_millions_kept(one_piece_main_raw) -> None:
    statistics = _anime_from_page(
        {**one_piece_main_raw, "rating_votes": 1234567}
    ).statistics
    assert statistics is not None
    assert statistics.scored_by == 1234567


def test_build_anime_from_raw_trending_rank_gives_trending(one_piece_main_raw) -> None:
    statistics = _anime_from_page(one_piece_main_raw).statistics
    assert statistics is not None
    assert statistics.trending == 26


def test_build_anime_from_raw_without_statistics_gives_none(one_piece_main_raw) -> None:
    raw = {
        **one_piece_main_raw,
        "rating_score": None,
        "rating_votes": None,
        "rank_toplist": None,
        "rank_trending": None,
    }
    assert _anime_from_page(raw).statistics is None


def test_build_anime_from_raw_without_score_keeps_rank(one_piece_main_raw) -> None:
    statistics = _anime_from_page(
        {**one_piece_main_raw, "rating_score": None}
    ).statistics
    assert statistics is not None
    assert statistics.score is None
    assert statistics.rank == 125


def test_build_anime_from_raw_description_trimmed(one_piece_main_raw) -> None:
    raw = {**one_piece_main_raw, "description": "  some synopsis  "}
    assert _anime_from_page(raw).synopsis == "some synopsis"


def test_parse_relations_without_relations_page_returns_empty_lists() -> None:
    assert _parse_relations(None) == ([], [])


def test_parse_relations_empty_relation_lists_return_empty_lists() -> None:
    assert _parse_relations({"anime_relations": [], "manga_relations": []}) == ([], [])


def test_parse_relations_missing_relation_keys_return_empty_lists() -> None:
    assert _parse_relations({}) == ([], [])


def test_parse_relations_one_piece_page_keeps_every_anime_relation(
    one_piece_relations_raw,
) -> None:
    anime, _ = _parse_relations(one_piece_relations_raw)
    assert len(anime) == len(one_piece_relations_raw["anime_relations"])


def test_parse_relations_one_piece_page_keeps_every_manga_relation(
    one_piece_relations_raw,
) -> None:
    _, manga = _parse_relations(one_piece_relations_raw)
    assert len(manga) == len(one_piece_relations_raw["manga_relations"])


def test_parse_relations_one_piece_page_gives_image_addresses(
    one_piece_relations_raw,
) -> None:
    anime, manga = _parse_relations(one_piece_relations_raw)
    for rel in anime + manga:
        if rel.get("image"):
            assert rel["image"].startswith("https://")


def test_parse_relations_one_piece_page_lists_original_manga(
    one_piece_relations_raw,
) -> None:
    _, manga = _parse_relations(one_piece_relations_raw)
    titles = [r["title"] for r in manga]
    assert "One Piece" in titles


def test_build_anime_from_raw_one_piece_page_maps_titles(one_piece_processed) -> None:
    anime = _build_anime_from_raw(one_piece_processed, _URL)
    assert (anime.title, anime.title_english, anime.title_japanese, anime.titles) == (
        "One Piece",
        "One Piece",
        "ワンピース",
        {},
    )


def test_build_anime_from_raw_one_piece_page_keeps_synonyms(
    one_piece_processed,
) -> None:
    anime = _build_anime_from_raw(one_piece_processed, _URL)
    assert "OP" in anime.synonyms
    assert "OneP" in anime.synonyms


def test_build_anime_from_raw_one_piece_page_gives_statistics(
    one_piece_processed,
) -> None:
    anime = _build_anime_from_raw(one_piece_processed, _URL)
    assert anime.statistics is not None
    assert anime.statistics.score == pytest.approx(4.18)
    assert anime.statistics.scored_by == 7902
    assert anime.statistics.rank == 125
    assert anime.statistics.trending == 26


def test_build_anime_from_raw_one_piece_page_keeps_every_relation(
    one_piece_processed, one_piece_relations_raw
) -> None:
    anime = _build_anime_from_raw(one_piece_processed, _URL)
    assert len(anime.anime_relations) == len(one_piece_relations_raw["anime_relations"])
    assert len(anime.manga_relations) == len(one_piece_relations_raw["manga_relations"])


def test_build_anime_from_raw_sets_page_address(one_piece_processed) -> None:
    assert _build_anime_from_raw(one_piece_processed, _URL).url == _URL


def test_build_anime_from_raw_one_piece_page_keeps_broadcast(
    one_piece_processed,
) -> None:
    anime = _build_anime_from_raw(one_piece_processed, _URL)
    assert anime.broadcast_day == "Sunday"
    assert anime.broadcast_time == "23:15"
    assert anime.broadcast_timezone == "JST"


def test_build_anime_from_raw_one_piece_page_keeps_studio(one_piece_processed) -> None:
    anime = _build_anime_from_raw(one_piece_processed, _URL)
    assert anime.studio == "Toei Animation Co., Ltd."
    assert "toei-animation" in (anime.studio_url or "")


def test_build_anime_from_raw_without_relations_gives_empty_lists(
    one_piece_processed,
) -> None:
    raw = {**one_piece_processed, "anime_relations": [], "manga_relations": []}
    anime = _build_anime_from_raw(raw, _URL)
    assert anime.anime_relations == []
    assert anime.manga_relations == []


def test_normalize_identifier_anime_address_returned_unchanged() -> None:
    crawler = AniSearchAnimeCrawler(NullRepository())
    assert crawler.normalize_identifier(_URL) == _URL


def test_normalize_identifier_other_site_raises_value_error() -> None:
    crawler = AniSearchAnimeCrawler(NullRepository())
    with pytest.raises(ValueError, match="Not an AniSearch anime URL"):
        crawler.normalize_identifier("https://myanimelist.net/anime/21")


def test_build_source_model_canonical_address_wins_over_requested_address(
    one_piece_processed,
) -> None:
    crawler = AniSearchAnimeCrawler(NullRepository())
    canonical = "https://www.anisearch.com/anime/2227,one-piece"
    raw = {**one_piece_processed, "_canonical_url": canonical}
    model = crawler.build_source_model(raw, "https://www.anisearch.com/anime/2227")
    assert model.url == canonical


def test_build_source_model_without_canonical_address_keeps_requested_address(
    one_piece_processed,
) -> None:
    crawler = AniSearchAnimeCrawler(NullRepository())
    model = crawler.build_source_model(one_piece_processed, _URL)
    assert model.url == _URL


def _site(
    pages: dict[str, str | None],
) -> Callable[[str], Awaitable[FetchedPage | None]]:
    async def fetch(url: str) -> FetchedPage | None:
        html = pages.get(url)
        return None if html is None else FetchedPage(url=_URL, html=html)

    return fetch


@pytest.mark.usefixtures("cache_off")
async def test_fetch_anisearch_anime_data_real_pages_returns_page_text(
    one_piece_main_html: str, one_piece_relations_html: str
) -> None:
    pages = {
        f"{BASE_ANIME_URL}2227": one_piece_main_html,
        f"{_URL}/relations?show=overall": one_piece_relations_html,
    }
    with patch.object(
        crawler_module, "fetch_anisearch_page", autospec=True, side_effect=_site(pages)
    ):
        result = await _fetch_anisearch_anime_data("2227")

    assert result is not None
    assert result["title_blocks"][0]["title"] == "One Piece"
    assert result["type"].startswith("Type: TV-Series")
    assert result["rank_toplist"] == "#125"
    assert len(result["anime_relations"]) == 79
    assert len(result["manga_relations"]) == 2
    assert result["_canonical_url"] == _URL


@pytest.mark.usefixtures("cache_off")
async def test_fetch_anisearch_anime_data_missing_main_page_returns_none() -> None:
    with patch.object(
        crawler_module, "fetch_anisearch_page", autospec=True, side_effect=_site({})
    ):
        assert await _fetch_anisearch_anime_data("2227,one-piece") is None


@pytest.mark.usefixtures("cache_off")
async def test_fetch_anisearch_anime_data_empty_main_page_raises_parse_error() -> None:
    pages = {f"{BASE_ANIME_URL}2227,one-piece": ""}
    with (
        patch.object(
            crawler_module,
            "fetch_anisearch_page",
            autospec=True,
            side_effect=_site(pages),
        ),
        pytest.raises(ServiceParseError),
    ):
        await _fetch_anisearch_anime_data("2227,one-piece")


@pytest.mark.usefixtures("cache_off")
async def test_fetch_anisearch_anime_data_missing_relations_page_returns_data_without_relations(
    one_piece_main_html: str,
) -> None:
    pages = {f"{BASE_ANIME_URL}2227,one-piece": one_piece_main_html}
    with patch.object(
        crawler_module, "fetch_anisearch_page", autospec=True, side_effect=_site(pages)
    ):
        result = await _fetch_anisearch_anime_data("2227,one-piece")

    assert result is not None
    assert result["anime_relations"] == []
    assert result["manga_relations"] == []
    assert "_canonical_url" not in result


@pytest.mark.usefixtures("cache_off")
async def test_fetch_anisearch_anime_data_relations_fetch_failure_raises_network_error(
    one_piece_main_html: str,
) -> None:
    async def fetch(url: str) -> FetchedPage:
        if url.endswith("/relations?show=overall"):
            raise ServiceNetworkError(service="anisearch", cause="HTTP 500")
        return FetchedPage(url=_URL, html=one_piece_main_html)

    with (
        patch.object(
            crawler_module, "fetch_anisearch_page", autospec=True, side_effect=fetch
        ),
        pytest.raises(ServiceNetworkError),
    ):
        await _fetch_anisearch_anime_data("2227,one-piece")


@pytest.mark.usefixtures("cache_off")
async def test_fetch_anisearch_anime_data_blocked_raises_service_blocked_error() -> (
    None
):
    with (
        patch.object(
            crawler_module,
            "fetch_anisearch_page",
            autospec=True,
            side_effect=ServiceBlockedError("HTTP 423", service="anisearch"),
        ),
        pytest.raises(ServiceBlockedError),
    ):
        await _fetch_anisearch_anime_data("2227,one-piece")


@pytest.mark.usefixtures("cache_off")
async def test_fetch_anisearch_anime_missing_page_returns_none() -> None:
    with patch.object(
        crawler_module, "fetch_anisearch_page", autospec=True, side_effect=_site({})
    ):
        assert await fetch_anisearch_anime(_URL) is None


@pytest.mark.usefixtures("cache_off")
async def test_fetch_anisearch_anime_bare_address_returns_canonical_anime_with_page_source(
    one_piece_main_html: str, one_piece_relations_html: str
) -> None:
    pages = {
        f"{BASE_ANIME_URL}2227": one_piece_main_html,
        f"{_URL}/relations?show=overall": one_piece_relations_html,
    }
    with patch.object(
        crawler_module, "fetch_anisearch_page", autospec=True, side_effect=_site(pages)
    ):
        result = await fetch_anisearch_anime(f"{BASE_ANIME_URL}2227")

    assert result["title"] == "One Piece"
    assert result["sources"] == [_URL]


def test_get_extraction_schema_returns_xpaths() -> None:
    crawler = AniSearchAnimeCrawler(NullRepository())
    schema = crawler.get_extraction_schema()
    assert schema == {"xpaths": _XPATHS}
