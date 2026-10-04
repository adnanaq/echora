"""Wait out Cloudflare's "Just a moment..." page, solving it only if it stays.

AniDB's interstitial clears by itself in a headed browser: on 2026-10-04 six
fresh browsers saw it disappear 1.3-1.5 s after loading, with no clicks.
Solving it at first sight was slower, not faster: ``verify_cf`` waits 3 s after
each click, and today's crawler took 20-23 s per fresh browser. So the page is
polled for its content, and ``verify_cf`` runs once, only when the interstitial
is still there after ``CHALLENGE_WAIT_SECONDS``.

zendriver's ``cf_is_interactive_challenge_present`` cannot make that decision:
it reports ``True`` for the interstitial that clears by itself.
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass

import zendriver
from zendriver.core.cloudflare import verify_cf

logger = logging.getLogger(__name__)

# Text of Cloudflare's interstitial and block pages. Anime-Planet embeds the
# invisible Turnstile script on every normal page, so "challenges.cloudflare.com"
# is deliberately not a marker.
CHALLENGE_MARKERS = (
    "Just a moment",
    "cf-browser-verification",
    "cf-challenge",
    "Attention Required",
)
CHALLENGE_WAIT_SECONDS = 10.0

_POLL_SECONDS = 0.25
_SOLVE_TIMEOUT_SECONDS = 15.0
_SOLVE_CLICK_DELAY_SECONDS = 3.0


class CloudflareChallengeError(Exception):
    """Raised when a challenge is still there after waiting and one solve."""

    def __init__(self, site: str, url: str) -> None:
        super().__init__(f"{site} Cloudflare challenge did not clear: {url}")
        self.site = site
        self.url = url


@dataclass(frozen=True)
class PageRead:
    """What a page held when the wait ended.

    Attributes:
        html: The last HTML read, or ``None`` when no read succeeded.
        ready: Whether the expected content (or a ``stop_on`` page) arrived.
        challenged: Whether a Cloudflare challenge was seen on the way.
    """

    html: str | None
    ready: bool
    challenged: bool


def is_cloudflare_challenge(html: str) -> bool:
    """Return whether ``html`` is a Cloudflare interstitial or block page."""
    return any(marker in html for marker in CHALLENGE_MARKERS)


async def wait_through_challenge(
    page: zendriver.Tab,
    url: str,
    *,
    site: str,
    selector: str,
    stop_on: Callable[[str], bool] | None = None,
    timeout: float = 30.0,
) -> PageRead:
    """Poll a page until its content arrives, waiting out any challenge.

    A challenge pauses the ``timeout``: it gets ``CHALLENGE_WAIT_SECONDS`` to
    clear by itself, then one ``verify_cf``, then the same wait again. The total
    is bounded by twice that wait plus the solve timeout.

    Args:
        page: The tab that was navigated to ``url``.
        url: The page's URL, for log lines.
        site: Site name for log lines and the error, such as ``"AniDB"``.
        selector: CSS selector of an element only the expected page has.
        stop_on: Also stop when it returns ``True`` for the HTML, for pages the
            caller handles itself, such as AniDB's AntiLeech page.
        timeout: Seconds to wait for the content when no challenge is showing.

    Returns:
        The last read; ``ready`` is ``False`` when ``timeout`` passed first.

    Raises:
        CloudflareChallengeError: If the challenge is still there after the
            solve and the second wait.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    challenge_since: float | None = None
    solve_tried = False
    challenged = False
    html: str | None = None
    while True:
        try:
            html = await page.get_content()
        except Exception as error:
            logger.debug(f"{site}: page not readable yet ({error}): {url}")
        else:
            if is_cloudflare_challenge(html):
                challenged = True
                challenge_since = challenge_since or loop.time()
                if loop.time() - challenge_since >= CHALLENGE_WAIT_SECONDS:
                    if solve_tried:
                        logger.warning(
                            f"{site}: Cloudflare challenge did not clear after "
                            f"waiting and one solve, giving up: {url}"
                        )
                        raise CloudflareChallengeError(site, url)
                    logger.info(
                        f"{site}: Cloudflare challenge persists, solving: {url}"
                    )
                    await _solve(page, site, url)
                    solve_tried = True
                    challenge_since = loop.time()
                    continue
            elif (stop_on is not None and stop_on(html)) or await _has_element(
                page, selector
            ):
                return PageRead(html=html, ready=True, challenged=challenged)
            else:
                challenge_since = None
        if challenge_since is None and loop.time() >= deadline:
            return PageRead(html=html, ready=False, challenged=challenged)
        await asyncio.sleep(_POLL_SECONDS)


async def _solve(page: zendriver.Tab, site: str, url: str) -> None:
    try:
        await verify_cf(
            page,
            click_delay=_SOLVE_CLICK_DELAY_SECONDS,
            timeout=_SOLVE_TIMEOUT_SECONDS,
        )
    except Exception as error:
        logger.info(f"{site}: verify_cf did not finish ({error}): {url}")


async def _has_element(page: zendriver.Tab, selector: str) -> bool:
    try:
        return await page.query_selector(selector) is not None
    except Exception:
        return False
