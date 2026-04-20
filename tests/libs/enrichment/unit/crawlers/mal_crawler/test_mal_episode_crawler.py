"""Unit tests for mal_episode_crawler.py — title parsing and data assembly.

Baseline tests use real XPath extraction output from:
- mal_episode_raw:             ep 1  (regular, characters + staff populated)
- mal_episode_filler_raw:      ep 50 (filler badge, no chars/staff)
- mal_episode_recap_raw:       ep 279 (recap badge, no chars/staff)
- mal_episode_no_synopsis_raw: ep 1152 (no synopsis, no duration field)

Edge-case tests use {**fixture, "field": override} to isolate specific branches.
"""

import json
from unittest.mock import AsyncMock

import pytest

from enrichment.crawlers.mal_crawler.mal_episode_crawler import (
    _build_episode_from_raw,
    _fetch_mal_episode_data,
    _parse_episode_characters,
    _parse_episode_staff,
    _parse_title_info,
    fetch_mal_episodes,
)

_EP1_URL = "https://myanimelist.net/anime/21/One_Piece/episode/1"
_EP50_URL = "https://myanimelist.net/anime/21/One_Piece/episode/50"
_EP279_URL = "https://myanimelist.net/anime/21/One_Piece/episode/279"
_EP1152_URL = "https://myanimelist.net/anime/21/One_Piece/episode/1152"


def _build(raw: dict, episode_number: int, url: str):
    return _build_episode_from_raw(dict(raw), episode_number, url)


# =============================================================================
# _parse_title_info — fixture-grounded
# =============================================================================


def test_parse_title_ep1_from_fixture(mal_episode_raw) -> None:
    title, jp, romaji, filler, recap = _parse_title_info(
        mal_episode_raw["title_header"], mal_episode_raw["subtitle_raw"], 1
    )
    assert title == "I'm Luffy! The Man Who's Gonna Be King of the Pirates!"
    assert jp == "俺はルフィ！海賊王になる男だ！"
    assert romaji == "Ore wa Luffy! Kaizoku Ou ni Naru Otoko Da!"
    assert filler is False
    assert recap is False


def test_parse_title_filler_from_fixture(mal_episode_filler_raw) -> None:
    title, jp, romaji, filler, recap = _parse_title_info(
        mal_episode_filler_raw["title_header"], mal_episode_filler_raw["subtitle_raw"], 50
    )
    assert title == "Usopp vs. Daddy the Parent! Showdown at High!"
    assert filler is True
    assert recap is False
    assert jp == "ウソップＶＳ子連れのダディ真昼の決闘"
    assert romaji == "Usopp vs Kozure no Dadi Mahiru no Kettou"


def test_parse_title_recap_from_fixture(mal_episode_recap_raw) -> None:
    title, jp, romaji, filler, recap = _parse_title_info(
        mal_episode_recap_raw["title_header"], mal_episode_recap_raw["subtitle_raw"], 279
    )
    assert title == "Jump Towards the Falls! Luffy's Feelings!"
    assert recap is True
    assert filler is False
    assert jp == "滝に向かって飛べ！ルフィの想い！！"
    assert romaji == "Taki ni Mukatte Tobe! Luffy no Omoi!!"


def test_parse_title_no_synopsis_ep_from_fixture(mal_episode_no_synopsis_raw) -> None:
    title, jp, romaji, filler, recap = _parse_title_info(
        mal_episode_no_synopsis_raw["title_header"], mal_episode_no_synopsis_raw["subtitle_raw"], 1152
    )
    assert title == "Her Father and Mother's Legacy! Bonney's Nika Punch"
    assert jp == "父と母の想い! ボニーの解放の拳[ニカパンチ]"
    assert filler is False
    assert recap is False


def test_parse_title_filler_newline_whitespace(mal_episode_filler_raw) -> None:
    """crawl4ai emits filler badge with leading newlines — whitespace collapse must handle it."""
    assert "\n" in mal_episode_filler_raw["title_header"]
    title, *_, filler, recap = _parse_title_info(mal_episode_filler_raw["title_header"], None, 50)
    assert filler is True
    assert "Filler" not in title


def test_parse_title_recap_newline_whitespace(mal_episode_recap_raw) -> None:
    assert "\n" in mal_episode_recap_raw["title_header"]
    title, *_, filler, recap = _parse_title_info(mal_episode_recap_raw["title_header"], None, 279)
    assert recap is True
    assert "Recap" not in title


def test_parse_title_filler_case_insensitive() -> None:
    title, *_, filler, _ = _parse_title_info("#1 - Title FILLER", None, 1)
    assert filler is True
    assert "FILLER" not in title


def test_parse_title_fallback_when_no_header() -> None:
    title, *_, filler, recap = _parse_title_info(None, None, 42)
    assert title == "Episode 42"
    assert filler is False
    assert recap is False


def test_parse_title_subtitle_no_kanji() -> None:
    _, jp, romaji, *_ = _parse_title_info("#1 - Test", "Romaji Title Only", 1)
    assert jp is None
    assert romaji == "Romaji Title Only"


# =============================================================================
# _parse_episode_characters — fixture-grounded
# =============================================================================


def test_parse_episode_characters_from_fixture(mal_episode_raw) -> None:
    result = _parse_episode_characters(mal_episode_raw["characters"])
    assert len(result) == 10
    luffy = result[0]
    assert luffy.name == "Monkey D., Luffy"
    assert luffy.mal_id == 40
    assert luffy.role == "Main"
    assert len(luffy.voice_actors) == 4
    ja_va = next(v for v in luffy.voice_actors if v.language == "Japanese")
    assert ja_va.person_id == 75
    assert ja_va.name == "Tanaka, Mayumi"


def test_parse_episode_characters_empty_for_filler_and_recap(
    mal_episode_filler_raw, mal_episode_recap_raw
) -> None:
    assert _parse_episode_characters(mal_episode_filler_raw["characters"]) == []
    assert _parse_episode_characters(mal_episode_recap_raw["characters"]) == []


_SAMPLE_VA_HTML = (
    '<a class="fw-b" href="https://myanimelist.net/people/70/Tanaka_Mayumi">'
    "Tanaka, Mayumi</a> (Japanese)<br>"
    '<a class="fw-b" href="https://myanimelist.net/people/81/Colleen_Clinkenbeard">'
    "Clinkenbeard, Colleen</a> (English)<br>"
)


def test_parse_episode_characters_empty_inputs() -> None:
    assert _parse_episode_characters(None) == []
    assert _parse_episode_characters([]) == []


def test_parse_episode_characters_invalid_urls_skipped() -> None:
    assert _parse_episode_characters([{"char_name": "Luffy", "char_url": ""}]) == []
    assert _parse_episode_characters([{"char_name": "Luffy", "char_url": "https://myanimelist.net/anime/21"}]) == []


def test_parse_episode_characters_role_defaults_to_supporting() -> None:
    items = [{"char_name": "Minor", "char_url": "https://myanimelist.net/character/999", "role": None, "voice_actors_html": ""}]
    assert _parse_episode_characters(items)[0].role == "Supporting"


def test_parse_episode_characters_multiple_vas() -> None:
    items = [{"char_name": "Luffy", "char_url": "https://myanimelist.net/character/40", "role": "Main", "voice_actors_html": _SAMPLE_VA_HTML}]
    vas = _parse_episode_characters(items)[0].voice_actors
    assert vas[0].person_id == 70
    assert vas[0].language == "Japanese"
    assert vas[1].person_id == 81
    assert vas[1].language == "English"


# =============================================================================
# _parse_episode_staff — fixture-grounded
# =============================================================================


def test_parse_episode_staff_from_fixture(mal_episode_raw) -> None:
    result = _parse_episode_staff(mal_episode_raw["staff"])
    assert len(result) == 9
    assert result[0].name == "Takegami, Junki"
    assert result[0].person_id == 5163
    assert result[0].role == "Script"


def test_parse_episode_staff_empty_for_filler_and_recap(
    mal_episode_filler_raw, mal_episode_recap_raw
) -> None:
    assert _parse_episode_staff(mal_episode_filler_raw["staff"]) == []
    assert _parse_episode_staff(mal_episode_recap_raw["staff"]) == []


def test_parse_episode_staff_empty_inputs() -> None:
    assert _parse_episode_staff(None) == []
    assert _parse_episode_staff([]) == []


def test_parse_episode_staff_invalid_entries_skipped() -> None:
    assert _parse_episode_staff([{"name": "X", "person_url": "https://myanimelist.net/people/999", "role": ""}]) == []
    assert _parse_episode_staff([{"name": "X", "person_url": "https://myanimelist.net/character/123", "role": "Script"}]) == []


def test_parse_episode_staff_full_item() -> None:
    items = [{"name": "Takegami, Junki", "person_url": "https://myanimelist.net/people/999/Takegami_Junki", "role": "Script"}]
    result = _parse_episode_staff(items)
    assert len(result) == 1
    assert result[0].person_id == 999
    assert result[0].name == "Takegami, Junki"
    assert result[0].role == "Script"


# =============================================================================
# _build_episode_from_raw — fixture-grounded
# =============================================================================


def test_build_ep1_from_fixture(mal_episode_raw) -> None:
    ep = _build(mal_episode_raw, 1, _EP1_URL)
    assert ep.title == "I'm Luffy! The Man Who's Gonna Be King of the Pirates!"
    assert ep.title_japanese == "俺はルフィ！海賊王になる男だ！"
    assert ep.title_romaji == "Ore wa Luffy! Kaizoku Ou ni Naru Otoko Da!"
    assert ep.aired == "1999-10-20"
    assert ep.duration == 1477
    assert ep.synopsis is not None and "Alvida" in ep.synopsis
    assert ep.filler is False
    assert ep.recap is False
    assert len(ep.characters) == 10
    assert len(ep.staff) == 9


def test_build_ep50_filler_from_fixture(mal_episode_filler_raw) -> None:
    ep = _build(mal_episode_filler_raw, 50, _EP50_URL)
    assert ep.title == "Usopp vs. Daddy the Parent! Showdown at High!"
    assert ep.filler is True
    assert ep.recap is False
    assert ep.aired == "2000-11-29"
    assert ep.characters == []
    assert ep.staff == []


def test_build_ep279_recap_from_fixture(mal_episode_recap_raw) -> None:
    ep = _build(mal_episode_recap_raw, 279, _EP279_URL)
    assert ep.title == "Jump Towards the Falls! Luffy's Feelings!"
    assert ep.recap is True
    assert ep.filler is False
    assert ep.aired == "2006-10-01"
    assert ep.duration == 1440


def test_build_ep1152_no_synopsis_from_fixture(mal_episode_no_synopsis_raw) -> None:
    assert "duration_raw" not in mal_episode_no_synopsis_raw
    ep = _build(mal_episode_no_synopsis_raw, 1152, _EP1152_URL)
    assert ep.title == "Her Father and Mother's Legacy! Bonney's Nika Punch"
    assert ep.synopsis is None
    assert ep.duration is None
    assert ep.aired == "2025-12-07"


def test_build_episode_no_title_falls_back() -> None:
    ep = _build({}, 42, "https://myanimelist.net/anime/21/One_Piece/episode/42")
    assert ep.title == "Episode 42"


def test_build_episode_url_from_raw_preferred(mal_episode_raw) -> None:
    raw = {**mal_episode_raw, "_url": _EP1_URL}
    ep = _build(raw, 1, "https://other.url/episode/1")
    assert ep.source == _EP1_URL


def test_build_episode_filler_and_recap_titles_have_no_badge(
    mal_episode_filler_raw, mal_episode_recap_raw
) -> None:
    ep50 = _build(mal_episode_filler_raw, 50, _EP50_URL)
    ep279 = _build(mal_episode_recap_raw, 279, _EP279_URL)
    assert "Filler" not in ep50.title and "Recap" not in ep50.title
    assert "Recap" not in ep279.title and "Filler" not in ep279.title


# =============================================================================
# fetch_mal_episodes — cache + chunking
# =============================================================================


@pytest.mark.asyncio
async def test_fetch_mal_episodes_uses_cache_and_merges_results(
    mocker, mal_episode_raw, mal_episode_filler_raw
) -> None:
    raw1 = dict(mal_episode_raw)
    raw2 = dict(mal_episode_filler_raw)

    mocker.patch.object(_fetch_mal_episode_data, "cache_batch_get", new=AsyncMock(return_value=([raw1, None], [1])))
    cache_set = AsyncMock()
    mocker.patch.object(_fetch_mal_episode_data, "cache_batch_set", new=cache_set)
    mocker.patch(
        "enrichment.crawlers.mal_crawler.mal_episode_crawler.crawl_batch_urls",
        new=AsyncMock(return_value=[{"url": _EP50_URL, "status_code": 200, "extracted_content": json.dumps([raw2])}]),
    )

    result = await fetch_mal_episodes([_EP1_URL, _EP50_URL])
    assert len(result) == 2
    assert result[0]["title"] == "I'm Luffy! The Man Who's Gonna Be King of the Pirates!"
    assert result[1]["title"] == "Usopp vs. Daddy the Parent! Showdown at High!"
    assert result[1]["filler"] is True
    cache_set.assert_awaited_once()


@pytest.mark.asyncio
async def test_fetch_mal_episodes_404_yields_none(mocker, mal_episode_raw) -> None:
    raw1 = dict(mal_episode_raw)
    url2 = "https://myanimelist.net/anime/21/One_Piece/episode/99999"

    mocker.patch.object(_fetch_mal_episode_data, "cache_batch_get", new=AsyncMock(return_value=([None, None], [0, 1])))
    mocker.patch.object(_fetch_mal_episode_data, "cache_batch_set", new=AsyncMock())
    mocker.patch(
        "enrichment.crawlers.mal_crawler.mal_episode_crawler.crawl_batch_urls",
        new=AsyncMock(return_value=[
            {"url": _EP1_URL, "status_code": 200, "extracted_content": json.dumps([raw1])},
            {"url": url2, "status_code": 404, "extracted_content": None},
        ]),
    )

    result = await fetch_mal_episodes([_EP1_URL, url2])
    assert result[0] is not None
    assert result[1] is None


@pytest.mark.asyncio
async def test_fetch_mal_episodes_chunks_requests(mocker, mal_episode_raw) -> None:
    urls = [f"https://myanimelist.net/anime/21/One_Piece/episode/{i}" for i in range(1, 37)]
    raw = dict(mal_episode_raw)

    mocker.patch.object(_fetch_mal_episode_data, "cache_batch_get", new=AsyncMock(return_value=([None] * len(urls), list(range(len(urls))))))
    cache_set = AsyncMock()
    mocker.patch.object(_fetch_mal_episode_data, "cache_batch_set", new=cache_set)

    async def _batch_result(batch_urls, **kwargs):
        return [{"url": u, "status_code": 200, "extracted_content": json.dumps([raw])} for u in batch_urls]

    mocker.patch("enrichment.crawlers.mal_crawler.mal_episode_crawler.crawl_batch_urls", side_effect=_batch_result)

    result = await fetch_mal_episodes(urls)
    assert len(result) == len(urls)
    assert all(item is not None for item in result)
    assert cache_set.await_count == 2


@pytest.mark.asyncio
async def test_fetch_mal_episodes_no_synopsis_ep_yields_none_synopsis(
    mocker, mal_episode_no_synopsis_raw
) -> None:
    raw = dict(mal_episode_no_synopsis_raw)
    mocker.patch.object(_fetch_mal_episode_data, "cache_batch_get", new=AsyncMock(return_value=([None], [0])))
    mocker.patch.object(_fetch_mal_episode_data, "cache_batch_set", new=AsyncMock())
    mocker.patch(
        "enrichment.crawlers.mal_crawler.mal_episode_crawler.crawl_batch_urls",
        new=AsyncMock(return_value=[{"url": _EP1152_URL, "status_code": 200, "extracted_content": json.dumps([raw])}]),
    )

    result = await fetch_mal_episodes([_EP1152_URL])
    assert result[0] is not None
    assert result[0].get("synopsis") is None
