from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from types import ModuleType
from unittest.mock import create_autospec, patch

import pytest
import zendriver
from enrichment.sources.base.browser import BrowserSession
from http_cache import result_cache
from http_cache.config import CacheConfig
from lxml import html as lxml_html


def _page_has(page_html: str, selector: str | None) -> bool:
    if not page_html or not selector or not page_html.strip():
        return False
    document = lxml_html.fromstring(page_html)
    for part in selector.split(","):
        tag, _, css_class = part.strip().partition(".")
        class_test = (
            f"[contains(concat(' ', normalize-space(@class), ' '), ' {css_class} ')]"
            if css_class
            else ""
        )
        if document.xpath(f"//{tag or '*'}{class_test}"):
            return True
    return False


def _build_tab(page_html: str, url: str) -> zendriver.Tab:
    tab = create_autospec(zendriver.Tab, instance=True)

    def wait_for(selector=None, text=None, timeout=10):
        if not _page_has(page_html, selector):
            raise TimeoutError(f"{selector} not on page")

    def query_selector(selector, _node=None):
        if _page_has(page_html, selector):
            return create_autospec(zendriver.Element, instance=True)
        return None

    tab.wait_for.side_effect = wait_for
    tab.query_selector.side_effect = query_selector
    tab.evaluate.return_value = "complete"
    tab.get_content.return_value = page_html
    tab.url = url
    return tab


@pytest.fixture
def tab_showing() -> Callable[[str, str], zendriver.Tab]:
    return _build_tab


@pytest.fixture
def browser_serving() -> Callable[[dict[str, str]], zendriver.Browser]:
    def build(pages: dict[str, str]) -> zendriver.Browser:
        browser = create_autospec(zendriver.Browser, instance=True)

        def get(url="about:blank", new_tab=False, new_window=False):
            return _build_tab(pages.get(url, ""), url)

        browser.get.side_effect = get
        return browser

    return build


@pytest.fixture
def open_browser():
    def install(module: ModuleType, browser: zendriver.Browser):
        @asynccontextmanager
        async def browser_session(**settings) -> AsyncIterator[BrowserSession]:
            yield BrowserSession(
                headless=settings["headless"],
                allowed_site=settings.get("allowed_site"),
                clearance_site=settings.get("clearance_site"),
                block_unused_resources=settings.get("block_unused_resources", True),
                browser=browser,
            )

        return patch.object(module, "browser_session", browser_session)

    with patch.object(
        result_cache,
        "get_cache_config",
        autospec=True,
        return_value=CacheConfig(cache_enabled=False),
    ):
        yield install
