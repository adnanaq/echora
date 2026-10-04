"""Unit tests for anidb_character_crawler.py — async functions.

Covers: _fetch_page_html, _solve_antileech, fetch_anidb_characters,
        fetch_anidb_character, main().
"""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, create_autospec, patch

import pytest
import zendriver
from enrichment.sources.anidb import anidb_character_crawler as crawler_module
from enrichment.sources.anidb.anidb_character_crawler import (
    _fetch_page_html,
    _solve_antileech,
    fetch_anidb_character,
    fetch_anidb_characters,
)
from enrichment.sources.anidb.anidb_models import AniDBCharacterPage
from enrichment.sources.base import browser as browser_module
from enrichment.sources.base import cloudflare_challenge
from enrichment.sources.base.cloudflare_challenge import PageRead

_CHAR_HTML = '<html><body><div id="tab_1_pane"><span itemprop="name">Luffy</span></div></body></html>'
_CF_HTML = "<html><body>Just a moment...</body></html>"
_ANTILEECH_HTML = (
    "<html><head><title>AniDB AntiLeech</title></head><body></body></html>"
)
_INTERSTITIAL = (
    Path(__file__).parent / "fixtures" / "anidb_cloudflare_interstitial.html"
).read_text()
_URL = "https://anidb.net/character/474"
_EMPTY_HTML = "<html><body><p>Not found</p></body></html>"
_LUFFY_HTML = (
    '<html><body><table><tr class="mainname"><td><span itemprop="name">'
    'Monkey D. Luffy</span></td></tr></table><div id="tab_1_pane"></div></body></html>'
)


class _TabMock(AsyncMock):
    """A zendriver Tab is awaitable — awaiting it waits for the page to settle.

    Plain AsyncMock is not, so the block-clearing path that does ``await page``
    raises TypeError against it.
    """

    def __await__(self):
        async def _settled() -> _TabMock:
            return self

        return _settled().__await__()


class ScriptedTab(zendriver.Tab):
    def __init__(self, pages: list[str]) -> None:
        self.pages = list(pages)
        self.html = ""

    def __await__(self):
        async def settled() -> ScriptedTab:
            return self

        return settled().__await__()

    async def get_content(self, **_: object) -> str:
        self.html = self.pages.pop(0) if len(self.pages) > 1 else self.pages[0]
        return self.html

    async def query_selector(self, selector: str, **_: object) -> zendriver.Tab | None:
        return self if 'id="tab_1_pane"' in self.html else None


class UnreadableTab(ScriptedTab):
    async def get_content(self, **_: object) -> str:
        raise ConnectionError("connection closed")


def _ready(html: str) -> PageRead:
    return PageRead(html=html, ready=True, challenged=False)


def _not_ready(html: str) -> PageRead:
    return PageRead(html=html, ready=False, challenged=False)


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
    load = mocker.patch.object(browser_module, "load_clearance", autospec=True)
    save = mocker.patch.object(crawler_module, "save_clearance", autospec=True)

    return SimpleNamespace(
        cache=cache, start=mock_start, browser=browser, load=load, save=save
    )


@pytest.fixture
def short_page_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(crawler_module, "_PAGE_TIMEOUT_SECONDS", 0.05)


@pytest.fixture
def short_challenge_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cloudflare_challenge, "CHALLENGE_WAIT_SECONDS", 0.05)


# =============================================================================
# _fetch_page_html
# =============================================================================


@pytest.mark.asyncio
async def test_fetch_page_html_character_page_returns_ready_read_and_tab() -> None:
    tab = ScriptedTab([_CHAR_HTML])
    browser = create_autospec(zendriver.Browser, instance=True)
    browser.get.return_value = tab

    read, page = await _fetch_page_html(browser, _URL)

    assert read == PageRead(html=_CHAR_HTML, ready=True, challenged=False)
    assert page is tab


@pytest.mark.asyncio
@pytest.mark.parametrize("crash", [RuntimeError("tab died"), StopIteration()])
async def test_fetch_page_html_navigation_crashes_returns_no_read(
    crash: BaseException,
) -> None:
    browser = create_autospec(zendriver.Browser, instance=True)
    browser.get.side_effect = crash

    assert await _fetch_page_html(browser, _URL) == (None, None)


@pytest.mark.asyncio
@pytest.mark.usefixtures("short_page_timeout")
async def test_fetch_page_html_unreadable_page_returns_read_without_html() -> None:
    browser = create_autospec(zendriver.Browser, instance=True)
    browser.get.return_value = UnreadableTab([])

    read, _ = await _fetch_page_html(browser, _URL)

    assert read == PageRead(html=None, ready=False, challenged=False)


@pytest.mark.asyncio
@pytest.mark.usefixtures("short_page_timeout")
async def test_fetch_page_html_page_without_character_returns_not_ready_read() -> None:
    browser = create_autospec(zendriver.Browser, instance=True)
    browser.get.return_value = ScriptedTab([_EMPTY_HTML])

    read, _ = await _fetch_page_html(browser, _URL)

    assert read == PageRead(html=_EMPTY_HTML, ready=False, challenged=False)


# =============================================================================
# _solve_antileech
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
async def test_solve_antileech_no_challenge_clears_quickly() -> None:
    page = AsyncMock()
    page.get_content = AsyncMock(return_value=_CHAR_HTML)

    with patch(
        "zendriver.core.cloudflare.cf_is_interactive_challenge_present",
        AsyncMock(return_value=False),
    ):
        with patch("zendriver.core.cloudflare.verify_cf", AsyncMock()):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with patch("time.monotonic", side_effect=[0.0, 1.0]):
                    result = await _solve_antileech(page)

    assert result is True


@pytest.mark.asyncio
async def test_solve_antileech_challenge_present_btn_clicked() -> None:
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
                    result = await _solve_antileech(page)

    assert result is True
    btn.click.assert_awaited_once()


@pytest.mark.asyncio
async def test_solve_antileech_challenge_present_btn_none() -> None:
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
                    result = await _solve_antileech(page)

    assert result is True


@pytest.mark.asyncio
async def test_solve_antileech_verify_raises_and_find_raises() -> None:
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
                    result = await _solve_antileech(page)

    assert result is True


@pytest.mark.asyncio
async def test_solve_antileech_timeout_returns_false() -> None:
    page = AsyncMock()
    page.get_content = AsyncMock(return_value=_CF_HTML)

    with patch(
        "zendriver.core.cloudflare.cf_is_interactive_challenge_present",
        AsyncMock(return_value=False),
    ):
        with patch("zendriver.core.cloudflare.verify_cf", AsyncMock()):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with patch("time.monotonic", side_effect=[0.0, 31.0]):
                    result = await _solve_antileech(page)

    assert result is False


@pytest.mark.asyncio
async def test_solve_antileech_get_content_raises_then_clears() -> None:
    page = AsyncMock()
    page.get_content = AsyncMock(side_effect=[Exception("dead"), _CHAR_HTML])

    with patch(
        "zendriver.core.cloudflare.cf_is_interactive_challenge_present",
        AsyncMock(return_value=False),
    ):
        with patch("zendriver.core.cloudflare.verify_cf", AsyncMock()):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with patch("time.monotonic", side_effect=[0.0, 1.0, 2.0]):
                    result = await _solve_antileech(page)

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
        autospec=True,
        return_value=(_ready(_CHAR_HTML), page_obj),
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
        autospec=True,
        return_value=(_not_ready(_EMPTY_HTML), AsyncMock()),
    )

    results = await _collect(fetch_anidb_characters([474]))

    assert results == [(474, None)]
    fetch_mocks.cache.cache_batch_set.assert_not_called()


@pytest.mark.asyncio
async def test_fetch_characters_antileech_solved_returns_page(
    fetch_mocks, mocker
) -> None:
    page_obj = _TabMock()
    page_obj.get_content = AsyncMock(return_value=_CHAR_HTML)
    mocker.patch(
        "enrichment.sources.anidb.anidb_character_crawler._fetch_page_html",
        autospec=True,
        return_value=(_ready(_ANTILEECH_HTML), page_obj),
    )
    mocker.patch(
        "enrichment.sources.anidb.anidb_character_crawler._solve_antileech",
        autospec=True,
        return_value=True,
    )

    results = await _collect(fetch_anidb_characters([474]))

    assert results[0][0] == 474
    assert results[0][1] is not None


@pytest.mark.asyncio
async def test_fetch_characters_antileech_not_cleared_returns_none(
    fetch_mocks, mocker
) -> None:
    mocker.patch(
        "enrichment.sources.anidb.anidb_character_crawler._fetch_page_html",
        autospec=True,
        return_value=(_ready(_ANTILEECH_HTML), AsyncMock()),
    )
    mocker.patch(
        "enrichment.sources.anidb.anidb_character_crawler._solve_antileech",
        autospec=True,
        return_value=False,
    )

    results = await _collect(fetch_anidb_characters([474]))

    assert results == [(474, None)]


@pytest.mark.asyncio
async def test_fetch_characters_antileech_solved_page_unreadable_returns_none(
    fetch_mocks, mocker
) -> None:
    page_obj = _TabMock()
    page_obj.get_content = AsyncMock(side_effect=Exception("gone"))
    mocker.patch(
        "enrichment.sources.anidb.anidb_character_crawler._fetch_page_html",
        autospec=True,
        return_value=(_ready(_ANTILEECH_HTML), page_obj),
    )
    mocker.patch(
        "enrichment.sources.anidb.anidb_character_crawler._solve_antileech",
        autospec=True,
        return_value=True,
    )

    results = await _collect(fetch_anidb_characters([474]))

    assert results == [(474, None)]


@pytest.mark.asyncio
async def test_fetch_characters_browser_crash_recovers(fetch_mocks, mocker) -> None:
    page_obj = AsyncMock()
    mocker.patch(
        "enrichment.sources.anidb.anidb_character_crawler._fetch_page_html",
        autospec=True,
        side_effect=[(None, None), (_ready(_CHAR_HTML), page_obj)],
    )

    results = await _collect(fetch_anidb_characters([474]))

    assert results[0][1] is not None
    assert fetch_mocks.start.await_count == 2


@pytest.mark.asyncio
async def test_fetch_characters_browser_crash_twice_skips(fetch_mocks, mocker) -> None:
    mocker.patch(
        "enrichment.sources.anidb.anidb_character_crawler._fetch_page_html",
        autospec=True,
        return_value=(None, None),
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
        autospec=True,
        side_effect=[(None, None), (_ready(_CHAR_HTML), page_obj)],
    )

    results = await _collect(fetch_anidb_characters([474]))

    assert results[0][1] is not None


@pytest.mark.asyncio
async def test_fetch_characters_inter_request_delay(fetch_mocks, mocker) -> None:
    fetch_mocks.cache.cache_batch_get = AsyncMock(return_value=([None, None], [0, 1]))
    mock_sleep = mocker.patch("asyncio.sleep", new_callable=AsyncMock)
    mocker.patch(
        "enrichment.sources.anidb.anidb_character_crawler._fetch_page_html",
        autospec=True,
        return_value=(_ready(_CHAR_HTML), AsyncMock()),
    )

    results = await _collect(fetch_anidb_characters([474, 475]))

    assert len(results) == 2
    mock_sleep.assert_awaited()


@pytest.mark.asyncio
async def test_fetch_characters_no_delay_after_last_miss(fetch_mocks, mocker) -> None:
    mock_sleep = mocker.patch("asyncio.sleep", new_callable=AsyncMock)
    mocker.patch(
        "enrichment.sources.anidb.anidb_character_crawler._fetch_page_html",
        autospec=True,
        return_value=(_ready(_CHAR_HTML), AsyncMock()),
    )

    await _collect(fetch_anidb_characters([474]))

    mock_sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_fetch_characters_finally_stop_raises(fetch_mocks, mocker) -> None:
    fetch_mocks.browser.stop = AsyncMock(side_effect=Exception("stop failed"))
    mocker.patch(
        "enrichment.sources.anidb.anidb_character_crawler._fetch_page_html",
        autospec=True,
        return_value=(_ready(_CHAR_HTML), AsyncMock()),
    )

    results = await _collect(fetch_anidb_characters([474]))

    assert len(results) == 1


@pytest.mark.asyncio
async def test_fetch_characters_interstitial_clears_by_itself_runs_no_solver(
    fetch_mocks, mocker
) -> None:
    fetch_mocks.browser.get = AsyncMock(
        return_value=ScriptedTab([_INTERSTITIAL, _INTERSTITIAL, _LUFFY_HTML])
    )
    verify_cf = mocker.patch.object(cloudflare_challenge, "verify_cf", autospec=True)
    solve_antileech = mocker.patch.object(
        crawler_module, "_solve_antileech", autospec=True
    )

    results = await _collect(fetch_anidb_characters([474]))

    assert results[0][1].name_main == "Monkey D. Luffy"
    verify_cf.assert_not_awaited()
    solve_antileech.assert_not_awaited()
    fetch_mocks.save.assert_awaited_once_with(fetch_mocks.browser, "anidb.net")


@pytest.mark.asyncio
@pytest.mark.usefixtures("short_challenge_wait")
async def test_fetch_characters_interstitial_persists_solves_once_and_replaces_clearance(
    fetch_mocks, mocker
) -> None:
    tab = ScriptedTab([_INTERSTITIAL])

    async def solve(page: ScriptedTab, **_: object) -> None:
        page.pages = [_LUFFY_HTML]

    fetch_mocks.browser.get = AsyncMock(return_value=tab)
    verify_cf = mocker.patch.object(
        cloudflare_challenge, "verify_cf", autospec=True, side_effect=solve
    )

    results = await _collect(fetch_anidb_characters([474]))

    assert results[0][1].name_main == "Monkey D. Luffy"
    verify_cf.assert_awaited_once()
    fetch_mocks.save.assert_awaited_once_with(fetch_mocks.browser, "anidb.net")
    fetch_mocks.cache.cache_batch_set.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.usefixtures("short_challenge_wait")
async def test_fetch_characters_interstitial_never_clears_returns_none_and_stores_nothing(
    fetch_mocks, mocker, caplog
) -> None:
    fetch_mocks.browser.get = AsyncMock(return_value=ScriptedTab([_INTERSTITIAL]))
    verify_cf = mocker.patch.object(cloudflare_challenge, "verify_cf", autospec=True)

    async with asyncio.timeout(5):
        results = await _collect(fetch_anidb_characters([474]))

    assert results == [(474, None)]
    verify_cf.assert_awaited_once()
    fetch_mocks.cache.cache_batch_set.assert_not_called()
    fetch_mocks.save.assert_not_awaited()
    assert _URL in caplog.text


@pytest.mark.asyncio
async def test_fetch_characters_antileech_page_runs_unban_flow(
    fetch_mocks, mocker
) -> None:
    tab = ScriptedTab([_ANTILEECH_HTML])

    async def unban(page: ScriptedTab) -> bool:
        page.pages = [_LUFFY_HTML]
        return True

    fetch_mocks.browser.get = AsyncMock(return_value=tab)
    verify_cf = mocker.patch.object(cloudflare_challenge, "verify_cf", autospec=True)
    solve_antileech = mocker.patch.object(
        crawler_module, "_solve_antileech", autospec=True, side_effect=unban
    )

    results = await _collect(fetch_anidb_characters([474]))

    assert results[0][1].name_main == "Monkey D. Luffy"
    solve_antileech.assert_awaited_once_with(tab)
    verify_cf.assert_not_awaited()


@pytest.mark.asyncio
async def test_fetch_characters_no_challenge_keeps_stored_clearance(
    fetch_mocks,
) -> None:
    fetch_mocks.browser.get = AsyncMock(return_value=ScriptedTab([_LUFFY_HTML]))

    results = await _collect(fetch_anidb_characters([474]))

    assert results[0][1].name_main == "Monkey D. Luffy"
    fetch_mocks.load.assert_awaited_once_with(fetch_mocks.browser, "anidb.net")
    fetch_mocks.save.assert_not_awaited()


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


@pytest.mark.asyncio
async def test_fetch_character_stops_browser_before_returning(
    fetch_mocks, mocker
) -> None:
    mocker.patch(
        "enrichment.sources.anidb.anidb_character_crawler._fetch_page_html",
        autospec=True,
        return_value=(_ready(_CHAR_HTML), AsyncMock()),
    )

    page = await fetch_anidb_character(474)

    assert page is not None
    fetch_mocks.browser.stop.assert_awaited_once()


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
