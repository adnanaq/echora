"""Unit tests for anisearch_episode_crawler (zendriver + lxml XPath)."""

from unittest.mock import AsyncMock

import pytest
from unittest.mock import patch

from enrichment.sources.anisearch.anisearch_episode_crawler import (
    AniSearchEpisodeCrawler,
    _XPATHS,
    _extract_episodes_from_html,
    _fetch_anisearch_episode_data,
    _parse_episode_row,
    fetch_anisearch_episodes,
)
from enrichment.sources.base.framework import NullRepository

_URL = "https://www.anisearch.com/anime/2227,one-piece"


# ---------------------------------------------------------------------------
# _XPATHS
# ---------------------------------------------------------------------------


def test_xpaths_has_episode_rows_key() -> None:
    assert "episode_rows" in _XPATHS
    assert "episodes" in _XPATHS["episode_rows"]
    assert "@data-episode='true'" in _XPATHS["episode_rows"]


def test_xpaths_has_all_field_keys() -> None:
    assert {
        "episode_number_raw", "runtime", "release_date",
        "title_en", "title_ja", "title_de", "title_fr", "title_it",
    } <= set(_XPATHS)


def test_xpaths_episode_number_raw_targets_full_th() -> None:
    assert "episodeNumber" in _XPATHS["episode_number_raw"]


# ---------------------------------------------------------------------------
# _extract_episodes_from_html
# ---------------------------------------------------------------------------


def test_extract_episodes_empty_html_returns_none() -> None:
    assert _extract_episodes_from_html("") is None


def test_extract_episodes_no_table_returns_empty_list() -> None:
    raw = _extract_episodes_from_html("<html><body><p>nothing</p></body></html>")
    assert raw is not None
    assert raw["episodes"] == []


def test_extract_episodes_real_fixture(one_piece_episodes_html) -> None:
    raw = _extract_episodes_from_html(one_piece_episodes_html)
    assert raw is not None
    assert len(raw["episodes"]) > 0
    first = raw["episodes"][0]
    assert first["episode_number_raw"] is not None
    assert "1" in first["episode_number_raw"]



def test_extract_episodes_field_structure(one_piece_episodes_html) -> None:
    raw = _extract_episodes_from_html(one_piece_episodes_html)
    assert raw is not None
    for row in raw["episodes"]:
        assert set(row) >= {
            "episode_number_raw", "runtime", "release_date",
            "title_en", "title_ja", "title_de", "title_fr", "title_it",
        }


# ---------------------------------------------------------------------------
# normalize_identifier
# ---------------------------------------------------------------------------


def test_normalize_passes_through_valid_url() -> None:
    crawler = AniSearchEpisodeCrawler(NullRepository())
    url = "https://www.anisearch.com/anime/18878,dan-da-dan/episodes"
    assert crawler.normalize_identifier(url) == url


def test_normalize_rejects_non_anisearch_url() -> None:
    crawler = AniSearchEpisodeCrawler(NullRepository())
    with pytest.raises(ValueError, match="Not an AniSearch"):
        crawler.normalize_identifier("https://myanimelist.net/anime/123")


# ---------------------------------------------------------------------------
# _parse_episode_row
# ---------------------------------------------------------------------------


def test_parse_episode_row_normal_episode() -> None:
    raw = {
        "episode_number_raw": "1",
        "runtime": "24 min",
        "release_date": "20. Oct 1999",
        "title_en": "I'm Luffy! The Man Who's Gonna Be King Of The Pirates!",
        "title_ja": "Ore wa Luffy! Kaizoku Ou ni naru Otoko da! (俺はルフィ!海賊王になる男だ!)",
    }
    result = _parse_episode_row(raw)
    assert result is not None
    assert result["episode_number"] == 1
    assert result["is_filler"] is False
    assert result["is_recap"] is False
    assert result["duration"] == 1440
    assert result["aired"] == "1999-10-20"
    assert result["title"] == "I'm Luffy! The Man Who's Gonna Be King Of The Pirates!"
    assert result["title_romaji"] == "Ore wa Luffy! Kaizoku Ou ni naru Otoko da!"
    assert result["title_japanese"] == "俺はルフィ!海賊王になる男だ!"


def test_parse_episode_row_filler_only() -> None:
    raw = {"episode_number_raw": "50Filler", "runtime": "24 min", "release_date": None, "title_en": None, "title_ja": None}
    result = _parse_episode_row(raw)
    assert result is not None
    assert result["episode_number"] == 50
    assert result["is_filler"] is True
    assert result["is_recap"] is False


def test_parse_episode_row_recap_only() -> None:
    raw = {"episode_number_raw": "457Recap", "runtime": None, "release_date": None, "title_en": None, "title_ja": None}
    result = _parse_episode_row(raw)
    assert result is not None
    assert result["episode_number"] == 457
    assert result["is_filler"] is False
    assert result["is_recap"] is True


def test_parse_episode_row_filler_and_recap() -> None:
    raw = {"episode_number_raw": "279FillerRecap", "runtime": "24 min", "release_date": None, "title_en": None, "title_ja": None}
    result = _parse_episode_row(raw)
    assert result is not None
    assert result["episode_number"] == 279
    assert result["is_filler"] is True
    assert result["is_recap"] is True


def test_parse_episode_row_strips_dubbed_title_prefix() -> None:
    raw = {
        "episode_number_raw": "1",
        "runtime": "24 min",
        "release_date": "20. Oct 1999",
        "title_en": "I'm Gonna Be King of the Pirates! [4Kids Ep 1] | I'm Luffy! The Man Who's Gonna Be King Of The Pirates!",
        "title_ja": None,
    }
    result = _parse_episode_row(raw)
    assert result is not None
    assert result["title"] == "I'm Luffy! The Man Who's Gonna Be King Of The Pirates!"


def test_parse_episode_row_future_episode_nulls() -> None:
    raw = {
        "episode_number_raw": "1157",
        "runtime": None,
        "release_date": None,
        "title_en": None,
        "title_ja": None,
    }
    result = _parse_episode_row(raw)
    assert result is not None
    assert result["episode_number"] == 1157
    assert result["is_filler"] is False
    assert result["is_recap"] is False
    assert result["duration"] is None
    assert result["aired"] is None
    assert result["title"] is None


def test_parse_episode_row_question_mark_values_become_none() -> None:
    raw = {"episode_number_raw": "1157", "runtime": "?", "release_date": "?", "title_en": "", "title_ja": ""}
    result = _parse_episode_row(raw)
    assert result is not None
    assert result["duration"] is None
    assert result["aired"] is None
    assert result["title"] is None


def test_parse_episode_row_returns_none_without_episode_number() -> None:
    assert _parse_episode_row({"episode_number_raw": "", "title_en": "Title"}) is None


def test_parse_episode_row_title_ja_without_kanji() -> None:
    raw = {"episode_number_raw": "1", "runtime": None, "release_date": None, "title_en": None, "title_ja": "Ore wa Luffy"}
    result = _parse_episode_row(raw)
    assert result is not None
    assert result["title_romaji"] == "Ore wa Luffy"
    assert result["title_japanese"] is None


# ---------------------------------------------------------------------------
# fetch_anisearch_episodes — async, mocked
# ---------------------------------------------------------------------------


def _make_browser_mock(mocker, html: str | None):
    page_mock = mocker.AsyncMock()
    page_mock.wait_for = AsyncMock()
    if html is None:
        page_mock.wait_for.side_effect = Exception("timeout")
    else:
        page_mock.get_content = AsyncMock(return_value=html)
    browser_mock = mocker.AsyncMock()
    browser_mock.get = AsyncMock(return_value=page_mock)
    browser_mock.stop = AsyncMock()
    return browser_mock


@pytest.mark.asyncio
async def test_fetch_episodes_returns_parsed_list(mocker, one_piece_episodes_raw) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_episode_crawler._extract_episodes_from_html",
        return_value=one_piece_episodes_raw,
    )
    mocker.patch("zendriver.start", new_callable=AsyncMock,
                 return_value=_make_browser_mock(mocker, "<html></html>"))

    result = await fetch_anisearch_episodes(_URL)
    assert result is not None
    assert len(result) == 7
    assert result[0]["episode_number"] == 1
    assert result[0]["title"] == "I'm Luffy! The Man Who's Gonna Be King Of The Pirates!"
    # ep 279: both filler and recap
    assert result[2]["filler"] is True
    assert result[2]["recap"] is True
    # ep 457: recap only
    assert result[3]["recap"] is True
    assert result[3]["filler"] is False
    # ep 50: filler only
    assert result[4]["filler"] is True
    assert result[4]["recap"] is False
    assert result[0]["titles"] == {
        "de": "Hier kommt Ruffy, der künftige König der Piraten!",
        "fr": "Je suis Luffy ! Celui qui deviendra Roi des pirates !",
        "it": "Io sono Rufy! L'uomo che diventerà Re dei pirati!",
    }
    assert set(result[1]["titles"].keys()) == {"de", "fr"}
    assert result[5]["titles"] == {}
    assert result[5]["episode_number"] == 1144


@pytest.mark.asyncio
async def test_fetch_episodes_navigation_failure_returns_none(mocker) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    browser_mock = _make_browser_mock(mocker, html=None)
    browser_mock.stop.side_effect = Exception("stop failed")
    mocker.patch("zendriver.start", new_callable=AsyncMock, return_value=browser_mock)

    assert await fetch_anisearch_episodes(_URL) is None


@pytest.mark.asyncio
async def test_fetch_episodes_filters_out_unparseable_rows(mocker, one_piece_episodes_raw) -> None:
    bad_row = {"episode_number_raw": ""}
    fixture_with_bad = {"episodes": [one_piece_episodes_raw["episodes"][0], bad_row, one_piece_episodes_raw["episodes"][1]]}

    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_episode_crawler._extract_episodes_from_html",
        return_value=fixture_with_bad,
    )
    mocker.patch("zendriver.start", new_callable=AsyncMock,
                 return_value=_make_browser_mock(mocker, "<html></html>"))

    result = await fetch_anisearch_episodes(_URL)
    assert result is not None
    assert len(result) == 2
    assert result[0]["episode_number"] == 1
    assert result[1]["episode_number"] == 2


@pytest.mark.asyncio
async def test_fetch_episode_data_empty_content_returns_none(mocker) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    browser_mock = _make_browser_mock(mocker, html="")
    mocker.patch("zendriver.start", new_callable=AsyncMock, return_value=browser_mock)

    assert await _fetch_anisearch_episode_data(_URL) is None


def test_get_extraction_schema_returns_xpaths() -> None:
    crawler = AniSearchEpisodeCrawler(NullRepository())
    schema = crawler.get_extraction_schema()
    assert schema == {"xpaths": _XPATHS}


@pytest.mark.asyncio
async def test_fetch_anisearch_episodes_writes_output_path(mocker, tmp_path, one_piece_episodes_raw) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_episode_crawler._extract_episodes_from_html",
        return_value=one_piece_episodes_raw,
    )
    mocker.patch("zendriver.start", new_callable=AsyncMock,
                 return_value=_make_browser_mock(mocker, "<html></html>"))

    output = str(tmp_path / "episodes.jsonl")
    result = await fetch_anisearch_episodes(_URL, output_path=output)
    assert result is not None
    assert len(result) == 7
    import json
    lines = [json.loads(l) for l in open(output)]
    assert len(lines) == 7
