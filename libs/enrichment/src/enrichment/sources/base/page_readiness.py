"""Wait until a crawled page holds everything the crawlers read from it.

Every field the crawlers extract arrives with the page's own HTML, not through
scrolling or a later request. Some of it comes after the element a crawler waits
for: AniSearch anime, episode, appearance list and character pages lost fields
when read as soon as that element appeared. Once ``document.readyState`` leaves
``"loading"`` the page matched one read 8 seconds later on every page type
crawled, typically 0.05-0.2 s after the element.
"""

import asyncio
import logging

import zendriver
from enrichment.sources.base.cloudflare_challenge import wait_through_challenge

logger = logging.getLogger(__name__)

PAGE_ELEMENT_TIMEOUT_SECONDS = 30.0
DOCUMENT_TIMEOUT_SECONDS = 10.0

_READY_STATE_READ_TIMEOUT_SECONDS = 5.0
_READY_STATE_POLL_SECONDS = 0.02


class PageElementTimeoutError(TimeoutError):
    """Raised when the element proving the right page loaded never appears."""

    def __init__(self, selector: str, timeout: float, url: str) -> None:
        super().__init__(f"{selector} did not appear within {timeout}s: {url}")


async def wait_for_page(
    page: zendriver.Tab,
    selector: str,
    url: str,
    *,
    element_timeout: float = PAGE_ELEMENT_TIMEOUT_SECONDS,
    document_timeout: float = DOCUMENT_TIMEOUT_SECONDS,
    cloudflare_site: str | None = None,
) -> None:
    """Wait for the element that proves the right page loaded, then its HTML.

    The element deadline is generous because a site's response time swings: a
    One Piece cast list took 14 s on MAL and Anime-Planet. The wait still ends as
    soon as the element appears.

    Args:
        page: The tab being read.
        selector: CSS selector of an element only the expected page has.
        url: The page's URL, for the log line if its HTML never finishes.
        element_timeout: Seconds to wait for ``selector``.
        document_timeout: Seconds to wait for the HTML after the element
            appears; once they pass the page is read as it is, with a warning.
        cloudflare_site: For a site behind Cloudflare, its name: a challenge
            shown instead of the page is waited out, and solved only if it
            stays; see ``cloudflare_challenge``.

    Raises:
        TimeoutError: If ``selector`` does not appear in time.
        CloudflareChallengeError: If a challenge on ``cloudflare_site`` does
            not clear.
    """
    if cloudflare_site is None:
        await page.wait_for(selector=selector, timeout=element_timeout)
    else:
        read = await wait_through_challenge(
            page, url, site=cloudflare_site, selector=selector, timeout=element_timeout
        )
        if not read.ready:
            raise PageElementTimeoutError(selector, element_timeout, url)
    await _wait_for_document(page, url, document_timeout)


async def _wait_for_document(page: zendriver.Tab, url: str, timeout: float) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while (remaining := deadline - loop.time()) > 0:
        try:
            ready_state = await asyncio.wait_for(
                page.evaluate("document.readyState"),
                min(_READY_STATE_READ_TIMEOUT_SECONDS, remaining),
            )
        except TimeoutError:
            continue
        if ready_state != "loading":
            return
        await asyncio.sleep(_READY_STATE_POLL_SECONDS)
    logger.warning(f"document still loading after {timeout}s, reading anyway: {url}")
