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
    _post_process_main,
    _process_relation_tooltips,
    fetch_anisearch_anime,
)
from enrichment.sources.base.exceptions import ServiceBlockedError
from enrichment.sources.base.framework import NullRepository
from enrichment.sources.base.polite_http import FetchedPage

_URL = "https://www.anisearch.com/anime/2227,one-piece"


@pytest.fixture(scope="session")
def one_piece_processed(one_piece_main_raw, one_piece_relations_raw):
    data = _post_process_main(one_piece_main_raw)
    anime_rels, manga_rels = _parse_relations(one_piece_relations_raw)
    data["anime_relations"] = anime_rels
    data["manga_relations"] = manga_rels
    return data


def test_xpaths_has_required_main_keys() -> None:
    assert {
        "cover_image",
        "title_alt",
        "title_ja",
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


def test_xpaths_title_alt_targets_grey_ja() -> None:
    assert "grey" in _XPATHS["title_alt"]
    assert "ja" in _XPATHS["title_alt"]


def test_xpaths_title_ja_targets_f16_strong() -> None:
    assert "f16" in _XPATHS["title_ja"]
    assert "strong" in _XPATHS["title_ja"]


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


def test_extract_anime_from_html_one_piece_page_reads_titles(
    one_piece_main_html,
) -> None:
    raw = _extract_anime_from_html(one_piece_main_html)
    assert raw is not None
    assert raw["title_ja"] == "One Piece"
    assert raw["title_alt"] == "ワンピース"


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
    assert all(isinstance(g["name"], str) for g in raw["genres"])


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
    assert (raw["title_ja"], raw["genres"], raw["websites"]) == (None, [], [])


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


def test_post_process_main_type_label_and_details_removed(one_piece_main_raw) -> None:
    assert _post_process_main(one_piece_main_raw)["type"] == "TV-Series"


def test_post_process_main_status_label_removed(one_piece_main_raw) -> None:
    assert _post_process_main(one_piece_main_raw)["status"] == "Ongoing"


def test_post_process_main_open_range_gives_start_date_only(one_piece_main_raw) -> None:
    data = _post_process_main(one_piece_main_raw)
    assert data["start_date"] == "1999-10-20"
    assert data["end_date"] is None


def test_post_process_main_closed_range_gives_both_dates(one_piece_main_raw) -> None:
    raw = {**one_piece_main_raw, "published": "Published: 20.10.1999 - 31.03.2002"}
    data = _post_process_main(raw)
    assert data["start_date"] == "1999-10-20"
    assert data["end_date"] == "2002-03-31"


def test_post_process_main_single_date_gives_start_date_only(
    one_piece_main_raw,
) -> None:
    raw = {**one_piece_main_raw, "published": "Published: 05.04.2003"}
    data = _post_process_main(raw)
    assert data["start_date"] == "2003-04-05"
    assert data["end_date"] is None


def test_post_process_main_without_published_date_gives_no_dates(
    one_piece_main_raw,
) -> None:
    raw = {**one_piece_main_raw, "published": None}
    data = _post_process_main(raw)
    assert data["start_date"] is None
    assert data["end_date"] is None


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
def test_post_process_main_published_date_keeps_only_stated_precision(
    one_piece_main_raw, published: str, expected: tuple
) -> None:
    data = _post_process_main({**one_piece_main_raw, "published": published})
    assert (
        data["start_date"],
        data["end_date"],
        data["start_year"],
        data["start_month"],
    ) == expected


def test_post_process_main_broadcast_gives_day_time_and_zone(
    one_piece_main_raw,
) -> None:
    data = _post_process_main(one_piece_main_raw)
    assert data["broadcast_day"] == "Sunday"
    assert data["broadcast_time"] == "23:15"
    assert data["broadcast_timezone"] == "JST"


def test_post_process_main_without_broadcast_gives_no_broadcast_parts(
    one_piece_main_raw,
) -> None:
    raw = {**one_piece_main_raw, "broadcast_raw": None}
    data = _post_process_main(raw)
    assert data["broadcast_day"] is None
    assert data["broadcast_time"] is None
    assert data["broadcast_timezone"] is None


def test_post_process_main_relative_studio_path_gives_full_address(
    one_piece_main_raw,
) -> None:
    data = _post_process_main(one_piece_main_raw)
    assert (
        data["studio_url"]
        == "https://www.anisearch.com/company/412,toei-animation-co-ltd"
    )


def test_post_process_main_studio_path_with_leading_slash_gives_full_address(
    one_piece_main_raw,
) -> None:
    raw = {**one_piece_main_raw, "studio_url": "/company/412,toei-animation-co-ltd"}
    data = _post_process_main(raw)
    assert (
        data["studio_url"]
        == "https://www.anisearch.com/company/412,toei-animation-co-ltd"
    )


def test_post_process_main_without_studio_path_gives_no_studio_address(
    one_piece_main_raw,
) -> None:
    raw = {**one_piece_main_raw, "studio_url": None}
    assert _post_process_main(raw)["studio_url"] is None


def test_post_process_main_source_material_label_removed(one_piece_main_raw) -> None:
    assert _post_process_main(one_piece_main_raw)["source_material"] == "Manga"


def test_post_process_main_synonyms_split_on_commas(one_piece_main_raw) -> None:
    assert _post_process_main(one_piece_main_raw)["synonyms"] == ["OP", "OneP"]


def test_post_process_main_without_synonyms_gives_empty_list(
    one_piece_main_raw,
) -> None:
    raw = {**one_piece_main_raw, "synonyms": None}
    assert _post_process_main(raw)["synonyms"] == []


def test_post_process_main_genres_become_names(one_piece_main_raw) -> None:
    data = _post_process_main(one_piece_main_raw)
    assert "Action" in data["genres"]
    assert "Fighting-Shounen" in data["genres"]
    assert all(isinstance(g, str) for g in data["genres"])


def test_post_process_main_tags_become_names(one_piece_main_raw) -> None:
    data = _post_process_main(one_piece_main_raw)
    assert "Pirate" in data["tags"]
    assert all(isinstance(t, str) for t in data["tags"])


def test_post_process_main_genre_without_name_skipped(one_piece_main_raw) -> None:
    raw = {**one_piece_main_raw, "genres": [{"name": ""}, {"name": "Action"}]}
    assert _post_process_main(raw)["genres"] == ["Action"]


def test_post_process_main_websites_kept(one_piece_main_raw) -> None:
    data = _post_process_main(one_piece_main_raw)
    assert len(data["websites"]) == len(one_piece_main_raw["websites"])
    assert all(w["url"] for w in data["websites"])


def test_post_process_main_website_without_address_skipped(one_piece_main_raw) -> None:
    raw = {
        **one_piece_main_raw,
        "websites": [
            {"name": "Empty", "url": ""},
            {"name": "Official", "url": "https://one-piece.com"},
        ],
    }
    data = _post_process_main(raw)
    assert len(data["websites"]) == 1
    assert data["websites"][0]["name"] == "Official"


def test_post_process_main_rating_gives_score_and_votes(one_piece_main_raw) -> None:
    data = _post_process_main(one_piece_main_raw)
    assert data["statistics"]["score"] == pytest.approx(4.18)
    assert data["statistics"]["scored_by"] == 7902


def test_post_process_main_zero_votes_omits_votes(one_piece_main_raw) -> None:
    raw = {**one_piece_main_raw, "rating_votes": 0}
    assert "scored_by" not in _post_process_main(raw)["statistics"]


def test_post_process_main_unrated_anime_omits_score_and_votes(
    one_piece_main_raw,
) -> None:
    raw = {
        **one_piece_main_raw,
        "rating_score": "Calculated Value0.00 = 0%",
        "rating_votes": 0,
    }
    stats = _post_process_main(raw)["statistics"]
    assert "score" not in stats
    assert "scored_by" not in stats


def test_post_process_main_toplist_rank_gives_rank(one_piece_main_raw) -> None:
    assert _post_process_main(one_piece_main_raw)["statistics"]["rank"] == 125


def test_post_process_main_trending_rank_gives_trending(one_piece_main_raw) -> None:
    assert _post_process_main(one_piece_main_raw)["statistics"]["trending"] == 26


def test_post_process_main_without_statistics_gives_none(one_piece_main_raw) -> None:
    raw = {
        **one_piece_main_raw,
        "rating_score": None,
        "rating_votes": None,
        "rank_toplist": None,
        "rank_trending": None,
    }
    assert _post_process_main(raw)["statistics"] is None


def test_post_process_main_without_score_keeps_rank(one_piece_main_raw) -> None:
    raw = {**one_piece_main_raw, "rating_score": None}
    stats = _post_process_main(raw)["statistics"]
    assert "score" not in stats
    assert stats["rank"] == 125


def test_post_process_main_description_trimmed(one_piece_main_raw) -> None:
    raw = {**one_piece_main_raw, "description": "  some synopsis  "}
    assert _post_process_main(raw)["description"] == "some synopsis"


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
    assert anime.title == "One Piece"
    assert anime.title_japanese == "ワンピース"


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


def test_build_anime_from_raw_without_statistics_gives_none(
    one_piece_processed,
) -> None:
    raw = {**one_piece_processed, "statistics": None}
    assert _build_anime_from_raw(raw, _URL).statistics is None


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
async def test_fetch_anisearch_anime_data_real_pages_returns_processed_data(
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
    assert result["title_ja"] == "One Piece"
    assert result["type"] == "TV-Series"
    assert result["broadcast_day"] == "Sunday"
    assert result["statistics"]["score"] == pytest.approx(4.18)
    assert len(result["anime_relations"]) == 79
    assert len(result["manga_relations"]) == 2
    assert result["_canonical_url"] == _URL


@pytest.mark.usefixtures("cache_off")
async def test_fetch_anisearch_anime_data_main_page_unreadable_returns_none() -> None:
    with patch.object(
        crawler_module, "fetch_anisearch_page", autospec=True, side_effect=_site({})
    ):
        assert await _fetch_anisearch_anime_data("2227,one-piece") is None


@pytest.mark.usefixtures("cache_off")
async def test_fetch_anisearch_anime_data_empty_main_page_returns_none() -> None:
    pages = {f"{BASE_ANIME_URL}2227,one-piece": ""}
    with patch.object(
        crawler_module, "fetch_anisearch_page", autospec=True, side_effect=_site(pages)
    ):
        assert await _fetch_anisearch_anime_data("2227,one-piece") is None


@pytest.mark.usefixtures("cache_off")
async def test_fetch_anisearch_anime_data_relations_unreadable_returns_data_without_relations(
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
async def test_fetch_anisearch_anime_unreadable_page_returns_none() -> None:
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
