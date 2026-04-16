"""Unit tests for mal_episode_crawler.py — title parsing and data assembly."""

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


# =============================================================================
# _parse_title_info — filler / recap suffix stripping
# =============================================================================


def test_parse_title_normal() -> None:
    """Regular episode — no flags set, title unchanged."""
    title, jp, romaji, filler, recap = _parse_title_info(
        "#1 - I'm Luffy! The Man Who Will Become the Pirate King!",
        None,
        1,
    )
    assert title == "I'm Luffy! The Man Who Will Become the Pirate King!"
    assert filler is False
    assert recap is False


def test_parse_title_filler_stripped() -> None:
    """'Filler' suffix is stripped from title and flag is set (ep 50)."""
    title, jp, romaji, filler, recap = _parse_title_info(
        "#50 - Usopp vs. Daddy the Parent! Showdown at High! Filler",
        None,
        50,
    )
    assert title == "Usopp vs. Daddy the Parent! Showdown at High!"
    assert filler is True
    assert recap is False


def test_parse_title_recap_stripped() -> None:
    """'Recap' suffix is stripped from title and flag is set (ep 279)."""
    title, jp, romaji, filler, recap = _parse_title_info(
        "#279 - Jump Towards the Falls! Luffy's Feelings! Recap",
        None,
        279,
    )
    assert title == "Jump Towards the Falls! Luffy's Feelings!"
    assert recap is True
    assert filler is False


def test_parse_title_filler_case_insensitive() -> None:
    """Suffix detection is case-insensitive."""
    title, *_, filler, recap = _parse_title_info("#1 - Title FILLER", None, 1)
    assert filler is True
    assert "FILLER" not in title
    assert "filler" not in title.lower()


def test_parse_title_subtitle_parsing() -> None:
    """Romaji and kanji are split from subtitle text correctly."""
    title, jp, romaji, filler, recap = _parse_title_info(
        "#1 - I'm Luffy!",
        "Ore wa Luffy! Kaizoku Ou ni Naru Otoko Da! (俺はルフィ！海賊王になる男だ！)",
        1,
    )
    assert jp == "俺はルフィ！海賊王になる男だ！"
    assert romaji == "Ore wa Luffy! Kaizoku Ou ni Naru Otoko Da!"
    assert filler is False


def test_parse_title_fallback_when_no_header() -> None:
    """Falls back to 'Episode N' when no header is provided."""
    title, *_, filler, recap = _parse_title_info(None, None, 42)
    assert title == "Episode 42"
    assert filler is False
    assert recap is False


def test_parse_title_filler_full_pipeline_ep50() -> None:
    """End-to-end with exact values from MAL ep 50 — filler flag + romaji + kanji
    all extracted correctly from the same real crawl4ai output."""
    header = (
        "#50 - Usopp vs. Daddy the Parent! Showdown at High!\n                  Filler"
    )
    subtitle = "Usopp vs Kozure no Dadi Mahiru no Kettou (ウソップＶＳ子連れのダディ真昼の決闘)"

    title, jp, romaji, filler, recap = _parse_title_info(header, subtitle, 50)

    assert title == "Usopp vs. Daddy the Parent! Showdown at High!"
    assert filler is True
    assert recap is False
    assert jp == "ウソップＶＳ子連れのダディ真昼の決闘"
    assert romaji == "Usopp vs Kozure no Dadi Mahiru no Kettou"


def test_parse_title_recap_full_pipeline_ep279() -> None:
    """End-to-end with exact values from MAL ep 279 — recap flag + romaji + kanji
    all extracted correctly from the same real crawl4ai output."""
    header = "#279 - Jump Towards the Falls! Luffy's Feelings!\n                  Recap"
    subtitle = (
        "Taki ni Mukatte Tobe! Luffy no Omoi!! (滝に向かって飛べ！ルフィの想い！！)"
    )

    title, jp, romaji, filler, recap = _parse_title_info(header, subtitle, 279)

    assert title == "Jump Towards the Falls! Luffy's Feelings!"
    assert recap is True
    assert filler is False
    assert jp == "滝に向かって飛べ！ルフィの想い！！"
    assert romaji == "Taki ni Mukatte Tobe! Luffy no Omoi!!"


# =============================================================================
# _parse_episode_characters
# =============================================================================

_SAMPLE_VA_HTML = (
    '<a class="fw-b" href="https://myanimelist.net/people/70/Tanaka_Mayumi">'
    "Tanaka, Mayumi</a> (Japanese)<br>"
    '<a class="fw-b" href="https://myanimelist.net/people/81/Colleen_Clinkenbeard">'
    "Clinkenbeard, Colleen</a> (English)<br>"
)


def test_parse_episode_characters_empty_returns_empty() -> None:
    assert _parse_episode_characters(None) == []
    assert _parse_episode_characters([]) == []


def test_parse_episode_characters_missing_url_skipped() -> None:
    items = [{"char_name": "Luffy", "char_url": ""}]
    assert _parse_episode_characters(items) == []


def test_parse_episode_characters_non_character_url_skipped() -> None:
    items = [{"char_name": "Luffy", "char_url": "https://myanimelist.net/anime/21"}]
    assert _parse_episode_characters(items) == []


def test_parse_episode_characters_full_item() -> None:
    items = [
        {
            "char_name": "Monkey D., Luffy",
            "char_url": "https://myanimelist.net/character/40/Monkey_D_Luffy",
            "role": "Main",
            "voice_actors_html": _SAMPLE_VA_HTML,
        }
    ]
    result = _parse_episode_characters(items)
    assert len(result) == 1
    char = result[0]
    assert char.mal_id == 40
    assert char.name == "Monkey D., Luffy"
    assert char.role == "Main"
    assert len(char.voice_actors) == 2


def test_parse_episode_characters_multiple_vas() -> None:
    items = [
        {
            "char_name": "Luffy",
            "char_url": "https://myanimelist.net/character/40",
            "role": "Main",
            "voice_actors_html": _SAMPLE_VA_HTML,
        }
    ]
    result = _parse_episode_characters(items)
    vas = result[0].voice_actors
    assert vas[0].person_id == 70
    assert vas[0].language == "Japanese"
    assert vas[1].person_id == 81
    assert vas[1].language == "English"


def test_parse_episode_characters_role_defaults_to_supporting() -> None:
    items = [
        {
            "char_name": "Minor Character",
            "char_url": "https://myanimelist.net/character/999",
            "role": None,
            "voice_actors_html": "",
        }
    ]
    result = _parse_episode_characters(items)
    assert result[0].role == "Supporting"


# =============================================================================
# _parse_episode_staff
# =============================================================================


def test_parse_episode_staff_empty_returns_empty() -> None:
    assert _parse_episode_staff(None) == []
    assert _parse_episode_staff([]) == []


def test_parse_episode_staff_missing_role_skipped() -> None:
    items = [
        {
            "name": "Takegami, Junki",
            "person_url": "https://myanimelist.net/people/999/Takegami_Junki",
            "role": "",
        }
    ]
    assert _parse_episode_staff(items) == []


def test_parse_episode_staff_non_people_url_skipped() -> None:
    items = [
        {
            "name": "Someone",
            "person_url": "https://myanimelist.net/character/123",
            "role": "Script",
        }
    ]
    assert _parse_episode_staff(items) == []


def test_parse_episode_staff_full_item() -> None:
    items = [
        {
            "name": "Takegami, Junki",
            "person_url": "https://myanimelist.net/people/999/Takegami_Junki",
            "role": "Script",
        }
    ]
    result = _parse_episode_staff(items)
    assert len(result) == 1
    assert result[0].person_id == 999
    assert result[0].name == "Takegami, Junki"
    assert result[0].role == "Script"


# =============================================================================
# _build_episode_from_raw
# =============================================================================


def test_build_episode_from_raw_minimal() -> None:
    raw = {"title_header": "#1 - I'm Luffy!"}
    ep = _build_episode_from_raw(
        raw, episode_number=1, url="https://myanimelist.net/anime/21/episode/1"
    )
    assert ep.episode_number == 1
    assert ep.title == "I'm Luffy!"
    assert ep.filler is False
    assert ep.recap is False


def test_build_episode_from_raw_filler_flag() -> None:
    raw = {"title_header": "#50 - Showdown at High! Filler"}
    ep = _build_episode_from_raw(raw, episode_number=50, url="...")
    assert ep.filler is True
    assert ep.recap is False


def test_build_episode_from_raw_recap_flag() -> None:
    raw = {"title_header": "#279 - Luffy's Feelings! Recap"}
    ep = _build_episode_from_raw(raw, episode_number=279, url="...")
    assert ep.recap is True
    assert ep.filler is False


def test_build_episode_from_raw_no_title_falls_back() -> None:
    """When title_header is absent, falls back to 'Episode N'."""
    raw: dict = {}
    ep = _build_episode_from_raw(raw, episode_number=42, url="...")
    assert ep.title == "Episode 42"


def test_build_episode_from_raw_parses_aired_and_duration() -> None:
    raw = {
        "title_header": "#1 - Test",
        "aired_raw": "Oct 20, 1999",
        "duration_raw": "00:24:37",
    }
    ep = _build_episode_from_raw(raw, 1, "...")
    assert ep.aired == "1999-10-20"
    assert ep.duration == 1477


def test_build_episode_from_raw_subtitle_parsed() -> None:
    raw = {
        "title_header": "#1 - I'm Luffy!",
        "subtitle_raw": "Ore wa Luffy! Kaizoku Ou ni Naru Otoko Da! (俺はルフィ！海賊王になる男だ！)",
    }
    ep = _build_episode_from_raw(raw, 1, "...")
    assert ep.title_japanese == "俺はルフィ！海賊王になる男だ！"
    assert ep.title_romaji == "Ore wa Luffy! Kaizoku Ou ni Naru Otoko Da!"


def test_build_episode_from_raw_url_from_raw_preferred() -> None:
    """_url key in raw takes precedence over the url argument."""
    raw = {
        "title_header": "#1 - Test",
        "_url": "https://myanimelist.net/anime/21/One_Piece/episode/1",
    }
    ep = _build_episode_from_raw(raw, 1, "https://other.url/episode/1")
    assert ep.source == "https://myanimelist.net/anime/21/One_Piece/episode/1"


def test_build_episode_from_raw_with_anime_id() -> None:
    """anime_id kwarg is not a parameter of _build_episode_from_raw — it's set by fetch_mal_episode."""
    raw = {"title_header": "#1 - Test"}
    ep = _build_episode_from_raw(raw, 1, "https://myanimelist.net/anime/21/episode/1")
    assert ep.episode_number == 1


def test_build_episode_from_raw_mal_placeholder_synopsis_becomes_none() -> None:
    """MAL's 'no synopsis yet' message must not be stored as synopsis text."""
    raw = {
        "title_header": "#12 - Episode Title",
        "synopsis_raw": "Sorry, this episode doesn't seem to have a synopsis yet. Maybe you can help us out? If you've watched this episode, you can easily add episode information to our database here.",
    }
    ep = _build_episode_from_raw(
        raw, 12, "https://myanimelist.net/anime/57334/Dandadan/episode/12"
    )
    assert ep.synopsis is None


def test_build_episode_from_raw_real_synopsis_preserved() -> None:
    raw = {
        "title_header": "#1 - Momo and Okarun",
        "synopsis_raw": "Momo is a high school girl born into a family of spirit mediums.",
    }
    ep = _build_episode_from_raw(
        raw, 1, "https://myanimelist.net/anime/57334/Dandadan/episode/1"
    )
    assert (
        ep.synopsis
        == "Momo is a high school girl born into a family of spirit mediums."
    )


# =============================================================================
# fetch_mal_episodes — cache + chunking
# =============================================================================


@pytest.mark.asyncio
async def test_fetch_mal_episodes_uses_cache_and_merges_results(mocker) -> None:
    url1 = "https://myanimelist.net/anime/21/One_Piece/episode/1"
    url2 = "https://myanimelist.net/anime/21/One_Piece/episode/2"
    raw1 = {"title_header": "#1 - I'm Luffy!", "_url": url1}
    raw2 = {"title_header": "#2 - Zoro!", "_url": url2}

    mocker.patch.object(
        _fetch_mal_episode_data,
        "cache_batch_get",
        new=AsyncMock(return_value=([raw1, None], [1])),
    )
    cache_set = AsyncMock()
    mocker.patch.object(
        _fetch_mal_episode_data,
        "cache_batch_set",
        new=cache_set,
    )
    mocker.patch(
        "enrichment.crawlers.mal_crawler.mal_episode_crawler.crawl_batch_urls",
        new=AsyncMock(
            return_value=[
                {
                    "url": url2,
                    "status_code": 200,
                    "extracted_content": json.dumps([raw2]),
                }
            ]
        ),
    )

    result = await fetch_mal_episodes([url1, url2])
    assert len(result) == 2
    assert result[0] is not None
    assert result[0]["title"] == "I'm Luffy!"
    assert result[1] is not None
    assert result[1]["title"] == "Zoro!"
    cache_set.assert_awaited_once()


@pytest.mark.asyncio
async def test_fetch_mal_episodes_chunks_requests(mocker) -> None:
    urls = [
        f"https://myanimelist.net/anime/21/One_Piece/episode/{i}"
        for i in range(1, 37)  # 36 URLs > _EPISODE_BATCH_SIZE=35 → forces 2 batches
    ]
    raw = {"title_header": "#1 - I'm Luffy!"}

    mocker.patch.object(
        _fetch_mal_episode_data,
        "cache_batch_get",
        new=AsyncMock(return_value=([None] * len(urls), list(range(len(urls))))),
    )
    cache_set = AsyncMock()
    mocker.patch.object(
        _fetch_mal_episode_data,
        "cache_batch_set",
        new=cache_set,
    )

    async def _batch_result(batch_urls: list[str], **kwargs) -> list[dict[str, str]]:
        return [
            {"url": u, "status_code": 200, "extracted_content": json.dumps([raw])}
            for u in batch_urls
        ]

    mocker.patch(
        "enrichment.crawlers.mal_crawler.mal_episode_crawler.crawl_batch_urls",
        side_effect=_batch_result,
    )

    result = await fetch_mal_episodes(urls)
    assert len(result) == len(urls)
    assert all(item is not None for item in result)
    assert cache_set.await_count == 2
