import ast
import asyncio
from pathlib import Path
from unittest.mock import create_autospec

import pytest
import zendriver
from enrichment.sources.base import page_readiness
from enrichment.sources.base.page_readiness import (
    PAGE_ELEMENT_TIMEOUT_SECONDS,
    PageElementTimeoutError,
    wait_for_page,
)

_URL = "https://www.anisearch.com/anime/2227,one-piece"
SOURCES_DIR = Path(page_readiness.__file__).resolve().parents[1]
CRAWLER_DIRS = ("anime_planet", "anisearch", "mal")


class LoadingTab(zendriver.Tab):
    def __init__(self, loading_reads: int) -> None:
        self.loading_reads = loading_reads

    async def wait_for(self, selector: str | None = None, **_: object) -> None:
        return None

    async def evaluate(self, expression: str, **_: object) -> str:
        if self.loading_reads:
            self.loading_reads -= 1
            return "loading"
        return "interactive"


class UnresponsiveTab(LoadingTab):
    async def evaluate(self, expression: str, **_: object) -> str:
        await asyncio.Event().wait()
        return "complete"


def _crawler_calls() -> list[tuple[str, ast.Call]]:
    return [
        (f"{path.relative_to(SOURCES_DIR)}:{node.lineno}", node)
        for directory in CRAWLER_DIRS
        for path in sorted((SOURCES_DIR / directory).glob("*.py"))
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]


async def test_wait_for_page_document_still_loading_returns_once_finished() -> None:
    tab = LoadingTab(loading_reads=3)

    await wait_for_page(tab, "#htitle", _URL)

    assert tab.loading_reads == 0


async def test_wait_for_page_document_never_finishes_logs_warning_at_deadline(
    caplog: pytest.LogCaptureFixture,
) -> None:
    tab = LoadingTab(loading_reads=1_000_000)

    async with asyncio.timeout(2):
        await wait_for_page(tab, "#htitle", _URL, document_timeout=0.1)

    assert f"document still loading after 0.1s, reading anyway: {_URL}" in caplog.text


async def test_wait_for_page_browser_never_answers_logs_warning_at_deadline(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async with asyncio.timeout(2):
        await wait_for_page(UnresponsiveTab(0), "#htitle", _URL, document_timeout=0.2)

    assert "document still loading after 0.2s" in caplog.text


async def test_wait_for_page_element_missing_raises_timeout_error() -> None:
    tab = create_autospec(zendriver.Tab, instance=True)
    tab.wait_for.side_effect = asyncio.TimeoutError

    with pytest.raises(asyncio.TimeoutError):
        await wait_for_page(tab, "#htitle", _URL)


async def test_wait_for_page_default_element_deadline_is_thirty_seconds() -> None:
    tab = create_autospec(zendriver.Tab, instance=True)
    tab.evaluate.return_value = "complete"

    await wait_for_page(tab, "#htitle", _URL)

    tab.wait_for.assert_awaited_once_with(selector="#htitle", timeout=30.0)
    assert PAGE_ELEMENT_TIMEOUT_SECONDS == 30.0


async def test_wait_for_page_cloudflare_site_element_missing_raises_page_element_timeout_error() -> (
    None
):
    tab = create_autospec(zendriver.Tab, instance=True)
    tab.get_content.return_value = "<html><body>Not found</body></html>"
    tab.query_selector.return_value = None

    with pytest.raises(PageElementTimeoutError, match="#htitle"):
        async with asyncio.timeout(2):
            await wait_for_page(
                tab,
                "#htitle",
                _URL,
                element_timeout=0.1,
                cloudflare_site="Anime-Planet",
            )


async def test_wait_for_page_cloudflare_site_element_present_waits_for_document() -> (
    None
):
    tab = create_autospec(zendriver.Tab, instance=True)
    tab.get_content.return_value = "<html><body><h1>One Piece</h1></body></html>"
    tab.query_selector.return_value = create_autospec(zendriver.Element, instance=True)
    tab.evaluate.return_value = "complete"

    await wait_for_page(tab, "h1", _URL, cloudflare_site="Anime-Planet")

    tab.wait_for.assert_not_awaited()
    tab.evaluate.assert_awaited_with("document.readyState")


def test_crawler_sources_sleep_only_for_politeness_delay() -> None:
    offenders = [
        location
        for location, call in _crawler_calls()
        if call.func.attr == "sleep"
        and [ast.unparse(argument) for argument in call.args]
        != ["_INTER_REQUEST_DELAY"]
    ]

    assert offenders == []


def test_crawler_sources_never_scroll_or_wait_for_element_directly() -> None:
    offenders = [
        location
        for location, call in _crawler_calls()
        if call.func.attr in {"scroll_down", "scroll_up", "wait_for"}
        or "scroll" in ast.unparse(call).lower()
    ]

    assert offenders == []
