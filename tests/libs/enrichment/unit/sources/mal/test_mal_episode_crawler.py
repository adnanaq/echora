"""Unit tests for mal_episode_crawler.py — zendriver + lxml refactor.

Baseline tests use real XPath extraction output from:
- mal_episode_extracted:             ep 1  (regular, characters + staff populated)
- mal_episode_filler_extracted:      ep 50 (filler badge, no chars/staff)
- mal_episode_recap_extracted:       ep 279 (recap badge, no chars/staff)
- mal_episode_no_synopsis_extracted: ep 1152 (no synopsis, no duration)

Edge-case tests use {**fixture, "field": override} to isolate specific branches.
"""

from unittest.mock import AsyncMock

import pytest
from enrichment.sources.mal.mal_episode_crawler import (
    _XPATHS,
    _build_episode_from_raw,
    _extract_episode_from_html,
    _fetch_episode_html,
    _fetch_mal_episode_data,
    _parse_episode_characters,
    _parse_episode_staff,
    _parse_title_info,
    fetch_mal_episode,
    fetch_mal_episodes,
    main,
    MalEpisodeCrawler,
)

pytestmark = pytest.mark.asyncio

_EP1_URL = "https://myanimelist.net/anime/21/One_Piece/episode/1"
_EP50_URL = "https://myanimelist.net/anime/21/One_Piece/episode/50"
_EP279_URL = "https://myanimelist.net/anime/21/One_Piece/episode/279"
_EP1152_URL = "https://myanimelist.net/anime/21/One_Piece/episode/1152"


def _build(raw: dict, episode_number: int, url: str):
    return _build_episode_from_raw(dict(raw), episode_number, url)


# =============================================================================
# _XPATHS invariant
# =============================================================================


def test_xpaths_has_required_keys() -> None:
    for key in ("title_header", "subtitle", "info_box", "char_tables", "staff_tables"):
        assert key in _XPATHS


# =============================================================================
# _extract_episode_from_html — fixture-grounded
# =============================================================================


def test_extract_from_fixtures(
    mal_episode_html, mal_episode_filler_html, mal_episode_recap_html, mal_episode_no_synopsis_html
) -> None:
    ep1 = _extract_episode_from_html(mal_episode_html)
    assert ep1 is not None
    assert ep1["title_header"].startswith("#1")
    assert len(ep1["characters"]) == 10
    assert len(ep1["staff"]) == 13
    # 4 English dub entries have no role text — must be included with role=None
    assert len([s for s in ep1["staff"] if not s.get("role")]) == 4

    ep50 = _extract_episode_from_html(mal_episode_filler_html)
    assert ep50 is not None and "Filler" in ep50["title_header"]
    assert ep50["characters"] == [] and ep50["staff"] == []

    ep279 = _extract_episode_from_html(mal_episode_recap_html)
    assert ep279 is not None and "Recap" in ep279["title_header"]

    ep1152 = _extract_episode_from_html(mal_episode_no_synopsis_html)
    assert ep1152 is not None and ep1152.get("duration_raw") is None


def test_extract_invalid_html_returns_none() -> None:
    assert _extract_episode_from_html("") is None
    assert _extract_episode_from_html("<html><body><p>no h2 here</p></body></html>") is None


def test_extract_skips_tables_without_fw_b_link() -> None:
    html = """<html><body>
    <h2 class="fs18">#1 - Title</h2>
    <h2>Characters</h2>
    <table class="fl-l"><tr><td>no link</td></tr></table>
    <h2>Staff</h2>
    <table class="fl-l"><tr><td>no link</td></tr></table>
    </body></html>"""
    raw = _extract_episode_from_html(html)
    assert raw is not None
    assert raw["characters"] == [] and raw["staff"] == []


# =============================================================================
# _parse_title_info — fixture-grounded
# =============================================================================


def test_parse_title_from_fixtures(
    mal_episode_extracted, mal_episode_filler_extracted,
    mal_episode_recap_extracted, mal_episode_no_synopsis_extracted
) -> None:
    title, jp, romaji, filler, recap = _parse_title_info(
        mal_episode_extracted["title_header"], mal_episode_extracted["subtitle_raw"], 1
    )
    assert title == "I'm Luffy! The Man Who's Gonna Be King of the Pirates!"
    assert jp == "俺はルフィ！海賊王になる男だ！"
    assert romaji == "Ore wa Luffy! Kaizoku Ou ni Naru Otoko Da!"
    assert filler is False and recap is False

    title50, jp50, romaji50, filler50, recap50 = _parse_title_info(
        mal_episode_filler_extracted["title_header"], mal_episode_filler_extracted["subtitle_raw"], 50
    )
    assert title50 == "Usopp vs. Daddy the Parent! Showdown at High!"
    assert filler50 is True and recap50 is False
    assert jp50 == "ウソップＶＳ子連れのダディ真昼の決闘"
    assert romaji50 == "Usopp vs Kozure no Dadi Mahiru no Kettou"

    title279, jp279, _, filler279, recap279 = _parse_title_info(
        mal_episode_recap_extracted["title_header"], mal_episode_recap_extracted["subtitle_raw"], 279
    )
    assert title279 == "Jump Towards the Falls! Luffy's Feelings!"
    assert recap279 is True and filler279 is False
    assert jp279 == "滝に向かって飛べ！ルフィの想い！！"

    title1152, jp1152, _, filler1152, recap1152 = _parse_title_info(
        mal_episode_no_synopsis_extracted["title_header"],
        mal_episode_no_synopsis_extracted["subtitle_raw"], 1152
    )
    assert title1152 == "Her Father and Mother's Legacy! Bonney's Nika Punch"
    assert jp1152 == "父と母の想い! ボニーの解放の拳[ニカパンチ]"
    assert filler1152 is False and recap1152 is False


def test_parse_title_edge_cases() -> None:
    # Whitespace collapse handles newlines injected by HTML structure
    title, *_, filler, _ = _parse_title_info("#50 - Title\n  Filler", None, 50)
    assert filler is True and "Filler" not in title

    title, *_, _, recap = _parse_title_info("#279 - Title\n  Recap", None, 279)
    assert recap is True and "Recap" not in title

    # Case-insensitive badge matching
    title, *_, filler, _ = _parse_title_info("#1 - Title FILLER", None, 1)
    assert filler is True and "FILLER" not in title

    # No header falls back to episode number
    title, *_, filler, recap = _parse_title_info(None, None, 42)
    assert title == "Episode 42" and filler is False and recap is False

    # Subtitle without kanji
    _, jp, romaji, *_ = _parse_title_info("#1 - Test", "Romaji Title Only", 1)
    assert jp is None and romaji == "Romaji Title Only"


# =============================================================================
# _parse_episode_characters — fixture-grounded
# =============================================================================


def test_parse_episode_characters_from_fixture(mal_episode_extracted, mal_episode_filler_extracted) -> None:
    result = _parse_episode_characters(mal_episode_extracted["characters"])
    assert len(result) == 10
    luffy = result[0]
    assert luffy.name == "Monkey D., Luffy" and luffy.mal_id == 40 and luffy.role == "Main"
    assert len(luffy.voice_actors) == 4
    ja_va = next(v for v in luffy.voice_actors if v.language == "Japanese")
    assert ja_va.person_id == 75 and ja_va.name == "Tanaka, Mayumi"

    assert _parse_episode_characters(mal_episode_filler_extracted["characters"]) == []


def test_parse_episode_characters_edge_cases() -> None:
    assert _parse_episode_characters(None) == []
    assert _parse_episode_characters([]) == []
    assert _parse_episode_characters([{"char_name": "Luffy", "char_url": ""}]) == []
    assert _parse_episode_characters(
        [{"char_name": "Luffy", "char_url": "https://myanimelist.net/anime/21"}]
    ) == []
    # Missing role defaults to Supporting
    items = [{"char_name": "X", "char_url": "https://myanimelist.net/character/999", "role": None, "voice_actors_html": ""}]
    assert _parse_episode_characters(items)[0].role == "Supporting"


def test_parse_episode_characters_multiple_vas() -> None:
    va_html = (
        '<a class="fw-b" href="https://myanimelist.net/people/70/X">Tanaka, Mayumi</a> (Japanese)<br>'
        '<a class="fw-b" href="https://myanimelist.net/people/81/Y">Clinkenbeard, Colleen</a> (English)<br>'
    )
    items = [{"char_name": "Luffy", "char_url": "https://myanimelist.net/character/40", "role": "Main", "voice_actors_html": va_html}]
    vas = _parse_episode_characters(items)[0].voice_actors
    assert vas[0].person_id == 70 and vas[0].language == "Japanese"
    assert vas[1].person_id == 81 and vas[1].language == "English"


# =============================================================================
# _parse_episode_staff — fixture-grounded
# =============================================================================


def test_parse_episode_staff_from_fixture(mal_episode_extracted, mal_episode_filler_extracted) -> None:
    result = _parse_episode_staff(mal_episode_extracted["staff"])
    assert len(result) == 13
    named = [s for s in result if s.role is not None]
    assert named[0].name == "Takegami, Junki" and named[0].person_id == 5163 and named[0].role == "Script"
    # 4 English dub entries have no role — included with role=None
    assert len([s for s in result if s.role is None]) == 4

    assert _parse_episode_staff(mal_episode_filler_extracted["staff"]) == []


def test_parse_episode_staff_edge_cases() -> None:
    assert _parse_episode_staff(None) == []
    assert _parse_episode_staff([]) == []

    # Empty name → skipped; /character/ URL → skipped (not /people/)
    assert _parse_episode_staff([{"name": "", "person_url": "https://myanimelist.net/people/1", "role": "X"}]) == []
    assert _parse_episode_staff([{"name": "X", "person_url": "https://myanimelist.net/character/1", "role": "X"}]) == []

    # Empty role → included with role=None
    result = _parse_episode_staff([{"name": "X", "person_url": "https://myanimelist.net/people/999", "role": ""}])
    assert len(result) == 1 and result[0].role is None

    # Full valid item
    result = _parse_episode_staff([{"name": "Takegami, Junki", "person_url": "https://myanimelist.net/people/999/X", "role": "Script"}])
    assert result[0].person_id == 999 and result[0].role == "Script"


# =============================================================================
# _build_episode_from_raw — fixture-grounded
# =============================================================================


def test_build_from_fixtures(
    mal_episode_extracted, mal_episode_filler_extracted,
    mal_episode_recap_extracted, mal_episode_no_synopsis_extracted
) -> None:
    ep1 = _build(mal_episode_extracted, 1, _EP1_URL)
    assert ep1.title == "I'm Luffy! The Man Who's Gonna Be King of the Pirates!"
    assert ep1.title_japanese == "俺はルフィ！海賊王になる男だ！"
    assert ep1.aired == "1999-10-20" and ep1.duration == 1477
    assert ep1.synopsis is not None and "Alvida" in ep1.synopsis
    assert ep1.filler is False and ep1.recap is False
    assert len(ep1.characters) == 10 and len(ep1.staff) == 13

    ep50 = _build(mal_episode_filler_extracted, 50, _EP50_URL)
    assert ep50.title == "Usopp vs. Daddy the Parent! Showdown at High!"
    assert ep50.filler is True and ep50.recap is False
    assert ep50.aired == "2000-11-29" and ep50.characters == [] and ep50.staff == []
    assert "Filler" not in ep50.title

    ep279 = _build(mal_episode_recap_extracted, 279, _EP279_URL)
    assert ep279.title == "Jump Towards the Falls! Luffy's Feelings!"
    assert ep279.recap is True and ep279.filler is False
    assert ep279.aired == "2006-10-01" and ep279.duration == 1440
    assert "Recap" not in ep279.title

    ep1152 = _build(mal_episode_no_synopsis_extracted, 1152, _EP1152_URL)
    assert ep1152.title == "Her Father and Mother's Legacy! Bonney's Nika Punch"
    assert ep1152.synopsis is None and ep1152.duration is None
    assert ep1152.aired == "2025-12-07"


def test_build_edge_cases(mal_episode_extracted) -> None:
    # No title falls back to episode number
    ep = _build({}, 42, "https://myanimelist.net/anime/21/One_Piece/episode/42")
    assert ep.title == "Episode 42"

    # _url in raw takes precedence over argument url
    raw = {**mal_episode_extracted, "_url": _EP1_URL}
    ep = _build(raw, 1, "https://other.url/episode/1")
    assert ep.source == _EP1_URL


# =============================================================================
# _fetch_episode_html — zendriver mock
# =============================================================================


async def test_fetch_episode_html(mal_episode_html) -> None:
    page_mock = AsyncMock()
    page_mock.wait_for = AsyncMock()
    page_mock.get_content = AsyncMock(return_value=mal_episode_html)
    page_mock.url = _EP1_URL
    browser_mock = AsyncMock()
    browser_mock.get = AsyncMock(return_value=page_mock)

    result = await _fetch_episode_html(browser_mock, _EP1_URL)
    assert result is not None
    html, url = result
    assert html == mal_episode_html and url == _EP1_URL
    page_mock.wait_for.assert_awaited_once()

    # Navigation failure returns None
    page_mock.wait_for = AsyncMock(side_effect=Exception("timeout"))
    assert await _fetch_episode_html(browser_mock, _EP1_URL) is None


# =============================================================================
# _fetch_mal_episode_data — single-URL cache+fetch
# =============================================================================


async def test_fetch_data_success(mocker, mal_episode_html) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    page_mock = AsyncMock()
    page_mock.wait_for = AsyncMock()
    page_mock.get_content = AsyncMock(return_value=mal_episode_html)
    page_mock.url = _EP1_URL
    browser_mock = AsyncMock()
    browser_mock.get = AsyncMock(return_value=page_mock)
    browser_mock.stop = AsyncMock()

    with pytest.MonkeyPatch.context() as mp:
        import zendriver as zd
        mp.setattr(zd, "start", AsyncMock(return_value=browser_mock))
        result = await _fetch_mal_episode_data(_EP1_URL)

    assert result is not None and result["title_header"].startswith("#1")


async def test_fetch_data_failures_return_none(mocker) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    browser_mock = AsyncMock()
    browser_mock.stop = AsyncMock()

    with pytest.MonkeyPatch.context() as mp:
        import zendriver as zd
        mp.setattr(zd, "start", AsyncMock(return_value=browser_mock))

        mocker.patch(
            "enrichment.sources.mal.mal_episode_crawler._fetch_episode_html",
            new=AsyncMock(return_value=None),
        )
        assert await _fetch_mal_episode_data(_EP1_URL) is None

        mocker.patch(
            "enrichment.sources.mal.mal_episode_crawler._fetch_episode_html",
            new=AsyncMock(return_value=("<html><body><p>no h2</p></body></html>", _EP1_URL)),
        )
        assert await _fetch_mal_episode_data(_EP1_URL) is None

        # browser.stop exception is swallowed silently
        browser_mock.stop = AsyncMock(side_effect=Exception("stop failed"))
        mocker.patch(
            "enrichment.sources.mal.mal_episode_crawler._fetch_episode_html",
            new=AsyncMock(return_value=None),
        )
        assert await _fetch_mal_episode_data(_EP1_URL) is None


# =============================================================================
# MalEpisodeCrawler
# =============================================================================


async def test_mal_episode_crawler_class(mocker, mal_episode_extracted) -> None:
    crawler = MalEpisodeCrawler.__new__(MalEpisodeCrawler)
    assert "xpaths" in crawler.get_extraction_schema()
    assert crawler.normalize_identifier("foo") == "foo"

    mocker.patch(
        "enrichment.sources.mal.mal_episode_crawler._fetch_mal_episode_data",
        new=AsyncMock(return_value=mal_episode_extracted),
    )
    raw = await crawler.fetch_raw_data(_EP1_URL)
    assert raw is not None

    ep = crawler.build_source_model(dict(mal_episode_extracted), _EP1_URL)
    assert ep.episode_number == 1

    canonical = crawler.map_to_canonical(ep)
    assert canonical["title"] == "I'm Luffy! The Man Who's Gonna Be King of the Pirates!"


# =============================================================================
# fetch_mal_episode
# =============================================================================


async def test_fetch_mal_episode(mocker, mal_episode_extracted) -> None:
    mocker.patch(
        "enrichment.sources.mal.mal_episode_crawler._fetch_mal_episode_data",
        new=AsyncMock(return_value={**mal_episode_extracted, "_url": _EP1_URL}),
    )
    result = await fetch_mal_episode(_EP1_URL)
    assert result is not None
    assert result["title"] == "I'm Luffy! The Man Who's Gonna Be King of the Pirates!"


# =============================================================================
# fetch_mal_episodes — shared browser session
# =============================================================================


async def test_returns_empty_for_no_urls() -> None:
    assert await fetch_mal_episodes([]) == []


async def test_all_cached_no_browser_started(mocker, mal_episode_extracted, mal_episode_filler_extracted) -> None:
    raw1 = {**mal_episode_extracted, "_url": _EP1_URL}
    raw2 = {**mal_episode_filler_extracted, "_url": _EP50_URL}
    mocker.patch.object(
        _fetch_mal_episode_data, "cache_batch_get",
        new=AsyncMock(return_value=([raw1, raw2], [])),
    )
    zd_start = mocker.patch("zendriver.start", new_callable=AsyncMock)

    result = await fetch_mal_episodes([_EP1_URL, _EP50_URL])
    assert len(result) == 2
    assert result[0]["title"] == "I'm Luffy! The Man Who's Gonna Be King of the Pirates!"
    assert result[1]["filler"] is True
    zd_start.assert_not_awaited()


async def test_misses_fetched_with_browser(
    mocker, mal_episode_extracted, mal_episode_filler_html
) -> None:
    raw1 = {**mal_episode_extracted, "_url": _EP1_URL}
    mocker.patch.object(
        _fetch_mal_episode_data, "cache_batch_get",
        new=AsyncMock(return_value=([raw1, None], [1])),
    )
    cache_set = AsyncMock()
    mocker.patch.object(_fetch_mal_episode_data, "cache_batch_set", new=cache_set)
    browser_mock = AsyncMock()
    browser_mock.stop = AsyncMock()
    mocker.patch("zendriver.start", new_callable=AsyncMock, return_value=browser_mock)
    mocker.patch(
        "enrichment.sources.mal.mal_episode_crawler._fetch_episode_html",
        new=AsyncMock(return_value=(mal_episode_filler_html, _EP50_URL)),
    )

    result = await fetch_mal_episodes([_EP1_URL, _EP50_URL])
    assert len(result) == 2
    assert result[0]["title"] == "I'm Luffy! The Man Who's Gonna Be King of the Pirates!"
    assert result[1]["filler"] is True
    cache_set.assert_awaited_once()


async def test_inter_request_delay_between_misses(
    mocker, mal_episode_extracted, mal_episode_filler_extracted
) -> None:
    mocker.patch.object(
        _fetch_mal_episode_data, "cache_batch_get",
        new=AsyncMock(return_value=([None, None], [0, 1])),
    )
    mocker.patch.object(_fetch_mal_episode_data, "cache_batch_set", new=AsyncMock())
    browser_mock = AsyncMock()
    browser_mock.stop = AsyncMock()
    mocker.patch("zendriver.start", new_callable=AsyncMock, return_value=browser_mock)
    html1 = _ep_html_with_url(_EP1_URL)
    html2 = _ep_html_with_url(_EP50_URL)
    mocker.patch(
        "enrichment.sources.mal.mal_episode_crawler._fetch_episode_html",
        new=AsyncMock(side_effect=[(html1, _EP1_URL), (html2, _EP50_URL)]),
    )
    sleep_mock = mocker.patch("enrichment.sources.mal.mal_episode_crawler.asyncio.sleep", new=AsyncMock())

    await fetch_mal_episodes([_EP1_URL, _EP50_URL])
    sleep_mock.assert_awaited_once()


async def test_browser_stop_exception_in_fetch_mal_episodes(mocker, mal_episode_filler_html) -> None:
    mocker.patch.object(
        _fetch_mal_episode_data, "cache_batch_get",
        new=AsyncMock(return_value=([None], [0])),
    )
    mocker.patch.object(_fetch_mal_episode_data, "cache_batch_set", new=AsyncMock())
    browser_mock = AsyncMock()
    browser_mock.stop = AsyncMock(side_effect=Exception("stop failed"))
    mocker.patch("zendriver.start", new_callable=AsyncMock, return_value=browser_mock)
    mocker.patch(
        "enrichment.sources.mal.mal_episode_crawler._fetch_episode_html",
        new=AsyncMock(return_value=(mal_episode_filler_html, _EP50_URL)),
    )
    result = await fetch_mal_episodes([_EP50_URL])
    assert result[0] is not None


async def test_bad_cached_url_triggers_refetch(mocker, mal_episode_filler_html) -> None:
    bad_cached = {"_url": "https://myanimelist.net/anime/21/redirect", "title_header": "#1 - X"}
    mocker.patch.object(
        _fetch_mal_episode_data, "cache_batch_get",
        new=AsyncMock(return_value=([bad_cached], [])),
    )
    mocker.patch.object(_fetch_mal_episode_data, "cache_batch_set", new=AsyncMock())
    browser_mock = AsyncMock()
    browser_mock.stop = AsyncMock()
    mocker.patch("zendriver.start", new_callable=AsyncMock, return_value=browser_mock)
    mocker.patch(
        "enrichment.sources.mal.mal_episode_crawler._fetch_episode_html",
        new=AsyncMock(return_value=(mal_episode_filler_html, _EP50_URL)),
    )
    result = await fetch_mal_episodes([_EP50_URL])
    assert result[0] is not None


async def test_extraction_failure_in_miss_yields_none(mocker) -> None:
    mocker.patch.object(
        _fetch_mal_episode_data, "cache_batch_get",
        new=AsyncMock(return_value=([None], [0])),
    )
    mocker.patch.object(_fetch_mal_episode_data, "cache_batch_set", new=AsyncMock())
    browser_mock = AsyncMock()
    browser_mock.stop = AsyncMock()
    mocker.patch("zendriver.start", new_callable=AsyncMock, return_value=browser_mock)
    mocker.patch(
        "enrichment.sources.mal.mal_episode_crawler._fetch_episode_html",
        new=AsyncMock(return_value=("<html><body><p>no h2</p></body></html>", _EP1_URL)),
    )
    result = await fetch_mal_episodes([_EP1_URL])
    assert result[0] is None


async def test_canonical_url_without_episode_number_yields_none(mocker, mal_episode_filler_html) -> None:
    mocker.patch.object(
        _fetch_mal_episode_data, "cache_batch_get",
        new=AsyncMock(return_value=([None], [0])),
    )
    mocker.patch.object(_fetch_mal_episode_data, "cache_batch_set", new=AsyncMock())
    browser_mock = AsyncMock()
    browser_mock.stop = AsyncMock()
    mocker.patch("zendriver.start", new_callable=AsyncMock, return_value=browser_mock)
    mocker.patch(
        "enrichment.sources.mal.mal_episode_crawler._fetch_episode_html",
        new=AsyncMock(return_value=(mal_episode_filler_html, "https://myanimelist.net/anime/21/redirect")),
    )
    result = await fetch_mal_episodes([_EP50_URL])
    assert result[0] is None


async def test_navigation_failure_yields_none(mocker) -> None:
    mocker.patch.object(
        _fetch_mal_episode_data, "cache_batch_get",
        new=AsyncMock(return_value=([None], [0])),
    )
    mocker.patch.object(_fetch_mal_episode_data, "cache_batch_set", new=AsyncMock())
    browser_mock = AsyncMock()
    browser_mock.stop = AsyncMock()
    mocker.patch("zendriver.start", new_callable=AsyncMock, return_value=browser_mock)
    mocker.patch(
        "enrichment.sources.mal.mal_episode_crawler._fetch_episode_html",
        new=AsyncMock(return_value=None),
    )

    result = await fetch_mal_episodes([_EP1_URL])
    assert result[0] is None


async def test_no_synopsis_ep_yields_none_synopsis(mocker, mal_episode_no_synopsis_html) -> None:
    mocker.patch.object(
        _fetch_mal_episode_data, "cache_batch_get",
        new=AsyncMock(return_value=([None], [0])),
    )
    mocker.patch.object(_fetch_mal_episode_data, "cache_batch_set", new=AsyncMock())
    browser_mock = AsyncMock()
    browser_mock.stop = AsyncMock()
    mocker.patch("zendriver.start", new_callable=AsyncMock, return_value=browser_mock)
    mocker.patch(
        "enrichment.sources.mal.mal_episode_crawler._fetch_episode_html",
        new=AsyncMock(return_value=(mal_episode_no_synopsis_html, _EP1152_URL)),
    )

    result = await fetch_mal_episodes([_EP1152_URL])
    assert result[0] is not None and result[0].get("synopsis") is None


# =============================================================================
# main() — CLI entry point
# =============================================================================


async def test_main_success(mocker, mal_episode_extracted) -> None:
    mocker.patch("sys.argv", ["mal_episode_crawler", _EP1_URL, "--output", "out.json"])
    mocker.patch(
        "enrichment.sources.mal.mal_episode_crawler.fetch_mal_episode",
        new=AsyncMock(return_value={"title": "Test"}),
    )
    assert await main() == 0


async def test_main_failure(mocker) -> None:
    mocker.patch("sys.argv", ["mal_episode_crawler", _EP1_URL, "--output", "out.json"])
    mocker.patch(
        "enrichment.sources.mal.mal_episode_crawler.fetch_mal_episode",
        new=AsyncMock(return_value=None),
    )
    assert await main() == 1


# =============================================================================
# helpers
# =============================================================================


def _ep_html_with_url(url: str) -> str:
    num = url.rstrip("/").split("/")[-1]
    return f"""<html><body>
    <h2 class="fs18">#{num} - Title</h2>
    <div class="di-tc ar"><p>Aired: Oct 20, 1999</p></div>
    </body></html>"""
