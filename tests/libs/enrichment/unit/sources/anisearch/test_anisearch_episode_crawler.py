"""Unit tests for anisearch_episode_crawler (Docker/XPath transport)."""

import json
from unittest.mock import AsyncMock, patch

import pytest
from enrichment.sources.anisearch.anisearch_episode_crawler import (
    AniSearchEpisodeCrawler,
    _parse_episode_row,
    fetch_anisearch_episodes,
)
from enrichment.sources.base.framework import DockerTransport, NullRepository

_URL = "https://www.anisearch.com/anime/2227,one-piece"


# ---------------------------------------------------------------------------
# normalize_identifier
# ---------------------------------------------------------------------------


def test_normalize_passes_through_valid_url() -> None:
    crawler = AniSearchEpisodeCrawler(DockerTransport(), NullRepository())
    url = "https://www.anisearch.com/anime/18878,dan-da-dan/episodes"
    assert crawler.normalize_identifier(url) == url


def test_normalize_rejects_non_anisearch_url() -> None:
    crawler = AniSearchEpisodeCrawler(DockerTransport(), NullRepository())
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
    assert result["duration"] == 1440
    assert result["aired"] == "1999-10-20"
    assert result["title"] == "I'm Luffy! The Man Who's Gonna Be King Of The Pirates!"
    assert result["title_romaji"] == "Ore wa Luffy! Kaizoku Ou ni naru Otoko da!"
    assert result["title_japanese"] == "俺はルフィ!海賊王になる男だ!"


def test_parse_episode_row_detects_filler() -> None:
    raw = {
        "episode_number_raw": "50\nFiller",
        "runtime": "24 min",
        "release_date": "29. Nov 2000",
        "title_en": "Usopp vs. Daddy the Parent! Showdown at High!",
        "title_ja": None,
    }
    result = _parse_episode_row(raw)
    assert result is not None
    assert result["episode_number"] == 50
    assert result["is_filler"] is True


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
    assert result["duration"] is None
    assert result["aired"] is None
    assert result["title"] is None
    assert result["title_romaji"] is None
    assert result["title_japanese"] is None


def test_parse_episode_row_question_mark_values_become_none() -> None:
    raw = {
        "episode_number_raw": "1157",
        "runtime": "?",
        "release_date": "?",
        "title_en": "",
        "title_ja": "",
    }
    result = _parse_episode_row(raw)
    assert result is not None
    assert result["duration"] is None
    assert result["aired"] is None
    assert result["title"] is None


def test_parse_episode_row_returns_none_without_episode_number() -> None:
    assert _parse_episode_row({"episode_number_raw": "", "title_en": "Title"}) is None


def test_parse_episode_row_title_ja_without_kanji() -> None:
    # title_ja has no (Kanji) parenthetical — entire string becomes title_romaji, title_japanese is None
    raw = {
        "episode_number_raw": "1",
        "runtime": None,
        "release_date": None,
        "title_en": None,
        "title_ja": "Ore wa Luffy",
    }
    result = _parse_episode_row(raw)
    assert result is not None
    assert result["title_romaji"] == "Ore wa Luffy"
    assert result["title_japanese"] is None


# ---------------------------------------------------------------------------
# fetch_anisearch_episodes — integration with crawl_single_url mock
# ---------------------------------------------------------------------------


def _crawl_result(extracted: list) -> dict:
    return {"status_code": 200, "extracted_content": json.dumps(extracted)}


@pytest.mark.asyncio
async def test_fetch_episodes_returns_parsed_list(
    mocker, one_piece_episodes_raw: list
) -> None:
    mock_crawl = mocker.patch(
        "enrichment.sources.anisearch.anisearch_episode_crawler.crawl_single_url",
        new_callable=AsyncMock,
        return_value=_crawl_result(one_piece_episodes_raw),
    )
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.setex = AsyncMock()

    with patch(
        "http_cache.result_cache.get_result_cache_redis_client", return_value=mock_redis
    ):
        result = await fetch_anisearch_episodes(_URL)

    assert result is not None
    # ep 1, 2, 279 (filler+recap), 457 (recap only), 50 (filler only), 1144 (partial), 1200 (future) → 7
    assert len(result) == 7
    assert result[0]["episode_number"] == 1
    # 4Kids dub prefix stripped
    assert (
        result[0]["title"] == "I'm Luffy! The Man Who's Gonna Be King Of The Pirates!"
    )
    # ep 279: both filler and recap
    assert result[2]["filler"] is True
    assert result[2]["recap"] is True
    # ep 457: recap only
    assert result[3]["recap"] is True
    assert result[3]["filler"] is False
    # ep 50: filler only
    assert result[4]["filler"] is True
    assert result[4]["recap"] is False
    # ep 1: all three extra languages present
    assert result[0]["titles"] == {
        "de": "Hier kommt Ruffy, der künftige König der Piraten!",
        "fr": "Je suis Luffy ! Celui qui deviendra Roi des pirates !",
        "it": "Io sono Rufy! L'uomo che diventerà Re dei pirati!",
    }
    # ep 2: only de+fr, no Italian
    assert set(result[1]["titles"].keys()) == {"de", "fr"}
    # ep 1144: no extra titles
    assert result[5]["titles"] == {}
    assert result[5]["episode_number"] == 1144
    mock_crawl.assert_awaited_once()


@pytest.mark.asyncio
async def test_fetch_episodes_returns_none_on_crawl_failure(
    mocker,
) -> None:
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_episode_crawler.crawl_single_url",
        new_callable=AsyncMock,
        return_value=None,
    )
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.setex = AsyncMock()

    with patch(
        "http_cache.result_cache.get_result_cache_redis_client", return_value=mock_redis
    ):
        result = await fetch_anisearch_episodes(_URL)

    assert result is None


@pytest.mark.asyncio
async def test_fetch_episodes_returns_none_on_404(
    mocker,
) -> None:
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_episode_crawler.crawl_single_url",
        new_callable=AsyncMock,
        return_value={"status_code": 404, "extracted_content": None},
    )
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.setex = AsyncMock()

    with patch(
        "http_cache.result_cache.get_result_cache_redis_client", return_value=mock_redis
    ):
        result = await fetch_anisearch_episodes(_URL)

    assert result is None


@pytest.mark.asyncio
async def test_fetch_episodes_returns_none_on_non_200_status(
    mocker,
) -> None:
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_episode_crawler.crawl_single_url",
        new_callable=AsyncMock,
        return_value={"status_code": 500, "extracted_content": None},
    )
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.setex = AsyncMock()

    with patch(
        "http_cache.result_cache.get_result_cache_redis_client", return_value=mock_redis
    ):
        result = await fetch_anisearch_episodes(_URL)

    assert result is None


@pytest.mark.asyncio
async def test_fetch_episodes_filters_out_unparseable_rows(
    mocker, one_piece_episodes_raw: list
) -> None:
    episodes = one_piece_episodes_raw[0]["episodes"]
    bad_row = {"episode_number_raw": ""}
    fixture_with_bad = [{"episodes": [episodes[0], bad_row, episodes[1]]}]

    mocker.patch(
        "enrichment.sources.anisearch.anisearch_episode_crawler.crawl_single_url",
        new_callable=AsyncMock,
        return_value=_crawl_result(fixture_with_bad),
    )
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.setex = AsyncMock()

    with patch(
        "http_cache.result_cache.get_result_cache_redis_client", return_value=mock_redis
    ):
        result = await fetch_anisearch_episodes(_URL)

    assert result is not None
    assert len(result) == 2
    assert result[0]["episode_number"] == 1
    assert result[1]["episode_number"] == 2


@pytest.mark.asyncio
async def test_cache_key_url_not_output_path(
    tmp_path, mocker, one_piece_episodes_raw: list
) -> None:
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_episode_crawler.crawl_single_url",
        new_callable=AsyncMock,
        return_value=_crawl_result(one_piece_episodes_raw),
    )
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.setex = AsyncMock()

    # Second get returns the cached raw body so crawl_single_url is not called again
    mock_redis.get = AsyncMock(
        side_effect=[None, json.dumps(one_piece_episodes_raw[0])]
    )

    with patch(
        "http_cache.result_cache.get_result_cache_redis_client", return_value=mock_redis
    ):
        await fetch_anisearch_episodes(_URL, output_path=str(tmp_path / "out1.jsonl"))
        await fetch_anisearch_episodes(_URL, output_path=str(tmp_path / "out2.jsonl"))

    assert mock_redis.setex.call_count == 1, "Same URL should produce one cache entry"
    key = mock_redis.setex.call_args[0][0]
    assert "output_path" not in key
