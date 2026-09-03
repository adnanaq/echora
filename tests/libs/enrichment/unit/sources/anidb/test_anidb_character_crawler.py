"""Unit tests for anidb_character_crawler.py — async functions.

Covers: _fetch_page_html, _solve_cf, fetch_anidb_characters,
        fetch_anidb_character, main().
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from enrichment.sources.anidb.anidb_character_crawler import (
    _fetch_page_html,
    _solve_cf,
    fetch_anidb_character,
    fetch_anidb_characters,
)
from enrichment.sources.anidb.anidb_models import AniDBCharacterPage

_CHAR_HTML = '<html><body><div id="tab_1_pane"><span itemprop="name">Luffy</span></div></body></html>'
_CF_HTML = "<html><body>Just a moment...</body></html>"
_EMPTY_HTML = "<html><body><p>Not found</p></body></html>"


async def _collect(gen):
    return [(cid, page) async for cid, page in gen]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fetch_mocks(mocker):
    """Patches cache, zendriver.start, and asyncio.sleep for fetch tests."""
    cache = mocker.MagicMock()
    cache.cache_batch_get = mocker.AsyncMock(return_value=([None], [0]))
    cache.cache_batch_set = mocker.AsyncMock()
    mocker.patch(
        "enrichment.sources.anidb.anidb_character_crawler._anidb_character_cache",
        cache,
    )

    browser = mocker.AsyncMock()
    browser.stop = mocker.AsyncMock()
    mock_start = mocker.patch(
        "zendriver.start", new_callable=mocker.AsyncMock, return_value=browser
    )
    mocker.patch("asyncio.sleep", new_callable=mocker.AsyncMock)

    return SimpleNamespace(cache=cache, start=mock_start, browser=browser)


# =============================================================================
# _fetch_page_html
# =============================================================================


@pytest.mark.asyncio
async def test_fetch_page_html_success() -> None:
    page_mock = AsyncMock()
    page_mock.get_content = AsyncMock(return_value=_CHAR_HTML)
    browser = AsyncMock()
    browser.get = AsyncMock(return_value=page_mock)

    with patch("asyncio.sleep", new_callable=AsyncMock):
        html, crashed, page = await _fetch_page_html(
            browser, "https://anidb.net/character/474"
        )

    assert html == _CHAR_HTML
    assert crashed is False
    assert page is page_mock


@pytest.mark.asyncio
async def test_fetch_page_html_runtime_error() -> None:
    browser = AsyncMock()
    browser.get = AsyncMock(side_effect=RuntimeError("tab died"))

    with patch("asyncio.sleep", new_callable=AsyncMock):
        html, crashed, page = await _fetch_page_html(
            browser, "https://anidb.net/character/474"
        )

    assert html is None
    assert crashed is True
    assert page is None


@pytest.mark.asyncio
async def test_fetch_page_html_stop_iteration() -> None:
    browser = AsyncMock()
    browser.get = AsyncMock(side_effect=StopIteration)

    with patch("asyncio.sleep", new_callable=AsyncMock):
        _, crashed, _ = await _fetch_page_html(
            browser, "https://anidb.net/character/474"
        )

    assert crashed is True


# =============================================================================
# _solve_cf
# =============================================================================


def _cf_patches(cf_present: bool, verify_raises: bool = False, find_returns=None):
    """Return patch context managers for zendriver.core.cloudflare."""
    cf_mod = {
        "zendriver.core.cloudflare.cf_is_interactive_challenge_present": AsyncMock(
            return_value=cf_present
        ),
        "zendriver.core.cloudflare.verify_cf": AsyncMock(
            side_effect=Exception("err") if verify_raises else None
        ),
    }
    return cf_mod


@pytest.mark.asyncio
async def test_solve_cf_no_challenge_clears_quickly() -> None:
    page = AsyncMock()
    page.get_content = AsyncMock(return_value=_CHAR_HTML)

    with patch(
        "zendriver.core.cloudflare.cf_is_interactive_challenge_present",
        AsyncMock(return_value=False),
    ):
        with patch("zendriver.core.cloudflare.verify_cf", AsyncMock()):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with patch("time.monotonic", side_effect=[0.0, 1.0]):
                    result = await _solve_cf(page)

    assert result is True


@pytest.mark.asyncio
async def test_solve_cf_challenge_present_btn_clicked() -> None:
    btn = AsyncMock()
    page = AsyncMock()
    page.find = AsyncMock(return_value=btn)
    page.get_content = AsyncMock(return_value=_CHAR_HTML)

    with patch(
        "zendriver.core.cloudflare.cf_is_interactive_challenge_present",
        AsyncMock(return_value=True),
    ):
        with patch("zendriver.core.cloudflare.verify_cf", AsyncMock()):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with patch("time.monotonic", side_effect=[0.0, 1.0]):
                    result = await _solve_cf(page)

    assert result is True
    btn.click.assert_awaited_once()


@pytest.mark.asyncio
async def test_solve_cf_challenge_present_btn_none() -> None:
    page = AsyncMock()
    page.find = AsyncMock(return_value=None)
    page.get_content = AsyncMock(return_value=_CHAR_HTML)

    with patch(
        "zendriver.core.cloudflare.cf_is_interactive_challenge_present",
        AsyncMock(return_value=True),
    ):
        with patch("zendriver.core.cloudflare.verify_cf", AsyncMock()):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with patch("time.monotonic", side_effect=[0.0, 1.0]):
                    result = await _solve_cf(page)

    assert result is True


@pytest.mark.asyncio
async def test_solve_cf_verify_raises_and_find_raises() -> None:
    page = AsyncMock()
    page.find = AsyncMock(side_effect=Exception("no btn"))
    page.get_content = AsyncMock(return_value=_CHAR_HTML)

    with patch(
        "zendriver.core.cloudflare.cf_is_interactive_challenge_present",
        AsyncMock(return_value=True),
    ):
        with patch(
            "zendriver.core.cloudflare.verify_cf",
            AsyncMock(side_effect=Exception("err")),
        ):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with patch("time.monotonic", side_effect=[0.0, 1.0]):
                    result = await _solve_cf(page)

    assert result is True


@pytest.mark.asyncio
async def test_solve_cf_timeout_returns_false() -> None:
    page = AsyncMock()
    page.get_content = AsyncMock(return_value=_CF_HTML)

    with patch(
        "zendriver.core.cloudflare.cf_is_interactive_challenge_present",
        AsyncMock(return_value=False),
    ):
        with patch("zendriver.core.cloudflare.verify_cf", AsyncMock()):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with patch("time.monotonic", side_effect=[0.0, 31.0]):
                    result = await _solve_cf(page)

    assert result is False


@pytest.mark.asyncio
async def test_solve_cf_get_content_raises_then_clears() -> None:
    page = AsyncMock()
    page.get_content = AsyncMock(side_effect=[Exception("dead"), _CHAR_HTML])

    with patch(
        "zendriver.core.cloudflare.cf_is_interactive_challenge_present",
        AsyncMock(return_value=False),
    ):
        with patch("zendriver.core.cloudflare.verify_cf", AsyncMock()):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with patch("time.monotonic", side_effect=[0.0, 1.0, 2.0]):
                    result = await _solve_cf(page)

    assert result is True


# =============================================================================
# fetch_anidb_characters
# =============================================================================


@pytest.mark.asyncio
async def test_fetch_characters_empty_ids() -> None:
    results = await _collect(fetch_anidb_characters([]))
    assert results == []


@pytest.mark.asyncio
async def test_fetch_characters_all_cache_hits(mocker) -> None:
    page_dict = AniDBCharacterPage(name_main="Luffy").model_dump(mode="json")
    cache = mocker.MagicMock()
    cache.cache_batch_get = mocker.AsyncMock(return_value=([page_dict, page_dict], []))
    cache.cache_batch_set = mocker.AsyncMock()
    mocker.patch(
        "enrichment.sources.anidb.anidb_character_crawler._anidb_character_cache", cache
    )

    results = await _collect(fetch_anidb_characters([474, 475]))

    assert len(results) == 2
    assert results[0][0] == 474
    assert results[1][0] == 475
    cache.cache_batch_set.assert_not_called()


@pytest.mark.asyncio
async def test_fetch_characters_cache_hit_none_value(mocker) -> None:
    cache = mocker.MagicMock()
    cache.cache_batch_get = mocker.AsyncMock(return_value=([None], []))
    cache.cache_batch_set = mocker.AsyncMock()
    mocker.patch(
        "enrichment.sources.anidb.anidb_character_crawler._anidb_character_cache", cache
    )

    results = await _collect(fetch_anidb_characters([474]))

    assert results == [(474, None)]


@pytest.mark.asyncio
async def test_fetch_characters_cache_miss_success(fetch_mocks, mocker) -> None:
    page_obj = AsyncMock()
    mocker.patch(
        "enrichment.sources.anidb.anidb_character_crawler._fetch_page_html",
        new_callable=AsyncMock,
        return_value=(_CHAR_HTML, False, page_obj),
    )

    results = await _collect(fetch_anidb_characters([474]))

    assert len(results) == 1
    char_id, page = results[0]
    assert char_id == 474
    assert page is not None
    fetch_mocks.cache.cache_batch_set.assert_awaited_once()


@pytest.mark.asyncio
async def test_fetch_characters_no_character_data(fetch_mocks, mocker) -> None:
    mocker.patch(
        "enrichment.sources.anidb.anidb_character_crawler._fetch_page_html",
        new_callable=AsyncMock,
        return_value=(_EMPTY_HTML, False, AsyncMock()),
    )

    results = await _collect(fetch_anidb_characters([474]))

    assert results == [(474, None)]
    fetch_mocks.cache.cache_batch_set.assert_not_called()


@pytest.mark.asyncio
async def test_fetch_characters_cf_blocked_solved(fetch_mocks, mocker) -> None:
    page_obj = AsyncMock()
    page_obj.get_content = AsyncMock(return_value=_CHAR_HTML)
    mocker.patch(
        "enrichment.sources.anidb.anidb_character_crawler._fetch_page_html",
        new_callable=AsyncMock,
        return_value=(_CF_HTML, False, page_obj),
    )
    mocker.patch(
        "enrichment.sources.anidb.anidb_character_crawler._solve_cf",
        new_callable=AsyncMock,
        return_value=True,
    )

    results = await _collect(fetch_anidb_characters([474]))

    assert results[0][0] == 474
    assert results[0][1] is not None


@pytest.mark.asyncio
async def test_fetch_characters_cf_solve_fails(fetch_mocks, mocker) -> None:
    mocker.patch(
        "enrichment.sources.anidb.anidb_character_crawler._fetch_page_html",
        new_callable=AsyncMock,
        return_value=(_CF_HTML, False, AsyncMock()),
    )
    mocker.patch(
        "enrichment.sources.anidb.anidb_character_crawler._solve_cf",
        new_callable=AsyncMock,
        return_value=False,
    )

    results = await _collect(fetch_anidb_characters([474]))

    assert results == [(474, None)]


@pytest.mark.asyncio
async def test_fetch_characters_cf_solved_get_content_raises(
    fetch_mocks, mocker
) -> None:
    page_obj = AsyncMock()
    page_obj.get_content = AsyncMock(side_effect=Exception("gone"))
    mocker.patch(
        "enrichment.sources.anidb.anidb_character_crawler._fetch_page_html",
        new_callable=AsyncMock,
        return_value=(_CF_HTML, False, page_obj),
    )
    mocker.patch(
        "enrichment.sources.anidb.anidb_character_crawler._solve_cf",
        new_callable=AsyncMock,
        return_value=True,
    )

    results = await _collect(fetch_anidb_characters([474]))

    assert results == [(474, None)]


@pytest.mark.asyncio
async def test_fetch_characters_browser_crash_recovers(fetch_mocks, mocker) -> None:
    page_obj = AsyncMock()
    mocker.patch(
        "enrichment.sources.anidb.anidb_character_crawler._fetch_page_html",
        new_callable=AsyncMock,
        side_effect=[(None, True, None), (_CHAR_HTML, False, page_obj)],
    )

    results = await _collect(fetch_anidb_characters([474]))

    assert results[0][1] is not None
    assert fetch_mocks.start.await_count == 2


@pytest.mark.asyncio
async def test_fetch_characters_browser_crash_twice_skips(fetch_mocks, mocker) -> None:
    mocker.patch(
        "enrichment.sources.anidb.anidb_character_crawler._fetch_page_html",
        new_callable=AsyncMock,
        return_value=(None, True, None),
    )

    results = await _collect(fetch_anidb_characters([474]))

    assert results == [(474, None)]


@pytest.mark.asyncio
async def test_fetch_characters_crash_stop_raises(fetch_mocks, mocker) -> None:
    fetch_mocks.browser.stop = AsyncMock(side_effect=Exception("stop error"))
    browser2 = AsyncMock()
    browser2.stop = AsyncMock()
    fetch_mocks.start.side_effect = [fetch_mocks.browser, browser2]
    page_obj = AsyncMock()
    mocker.patch(
        "enrichment.sources.anidb.anidb_character_crawler._fetch_page_html",
        new_callable=AsyncMock,
        side_effect=[(None, True, None), (_CHAR_HTML, False, page_obj)],
    )

    results = await _collect(fetch_anidb_characters([474]))

    assert results[0][1] is not None


@pytest.mark.asyncio
async def test_fetch_characters_inter_request_delay(fetch_mocks, mocker) -> None:
    fetch_mocks.cache.cache_batch_get = AsyncMock(return_value=([None, None], [0, 1]))
    mock_sleep = mocker.patch("asyncio.sleep", new_callable=AsyncMock)
    mocker.patch(
        "enrichment.sources.anidb.anidb_character_crawler._fetch_page_html",
        new_callable=AsyncMock,
        return_value=(_CHAR_HTML, False, AsyncMock()),
    )

    results = await _collect(fetch_anidb_characters([474, 475]))

    assert len(results) == 2
    mock_sleep.assert_awaited()


@pytest.mark.asyncio
async def test_fetch_characters_no_delay_after_last_miss(fetch_mocks, mocker) -> None:
    mock_sleep = mocker.patch("asyncio.sleep", new_callable=AsyncMock)
    mocker.patch(
        "enrichment.sources.anidb.anidb_character_crawler._fetch_page_html",
        new_callable=AsyncMock,
        return_value=(_CHAR_HTML, False, AsyncMock()),
    )

    await _collect(fetch_anidb_characters([474]))

    mock_sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_fetch_characters_finally_stop_raises(fetch_mocks, mocker) -> None:
    fetch_mocks.browser.stop = AsyncMock(side_effect=Exception("stop failed"))
    mocker.patch(
        "enrichment.sources.anidb.anidb_character_crawler._fetch_page_html",
        new_callable=AsyncMock,
        return_value=(_CHAR_HTML, False, AsyncMock()),
    )

    results = await _collect(fetch_anidb_characters([474]))

    assert len(results) == 1


# =============================================================================
# fetch_anidb_character
# =============================================================================


@pytest.mark.asyncio
async def test_fetch_character_returns_page(mocker) -> None:
    page = AniDBCharacterPage(name_main="Luffy")
    page_dict = page.model_dump(mode="json")
    cache = mocker.MagicMock()
    cache.cache_batch_get = mocker.AsyncMock(return_value=([page_dict], []))
    cache.cache_batch_set = mocker.AsyncMock()
    mocker.patch(
        "enrichment.sources.anidb.anidb_character_crawler._anidb_character_cache", cache
    )

    result = await fetch_anidb_character(474)

    assert result == AniDBCharacterPage.model_validate(page_dict)


@pytest.mark.asyncio
async def test_fetch_character_returns_none(mocker) -> None:
    cache = mocker.MagicMock()
    cache.cache_batch_get = mocker.AsyncMock(return_value=([None], []))
    cache.cache_batch_set = mocker.AsyncMock()
    mocker.patch(
        "enrichment.sources.anidb.anidb_character_crawler._anidb_character_cache", cache
    )

    assert await fetch_anidb_character(474) is None


# =============================================================================
# main() CLI
# =============================================================================


@pytest.mark.asyncio
async def test_main_success(tmp_path, mocker) -> None:
    from enrichment.sources.anidb.anidb_character_crawler import main

    out = str(tmp_path / "out.json")
    mocker.patch("sys.argv", ["prog", "474", "--output", out])
    mocker.patch(
        "enrichment.sources.anidb.anidb_character_crawler.fetch_anidb_character",
        new_callable=AsyncMock,
        return_value=AniDBCharacterPage(name_main="Luffy"),
    )

    code = await main()

    assert code == 0
    data = json.loads(Path(out).read_text())
    assert data.get("name") is not None


@pytest.mark.asyncio
async def test_main_no_data_returns_1(mocker) -> None:
    from enrichment.sources.anidb.anidb_character_crawler import main

    mocker.patch("sys.argv", ["prog", "474"])
    mocker.patch(
        "enrichment.sources.anidb.anidb_character_crawler.fetch_anidb_character",
        new_callable=AsyncMock,
        return_value=None,
    )

    assert await main() == 1
