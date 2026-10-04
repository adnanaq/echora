"""Fetch AniSearch pages over plain HTTP instead of a browser.

AniSearch is not behind Cloudflare and its pages need no JavaScript: on
2026-10-04 every page type the crawlers read (anime, relations, episodes, cast
list, character, appearance lists) extracted the same data over plain HTTP as
through a browser, for three anime and three characters, and 60 more character
pages at one request every 3 s drew no block.

AniSearch bans by User-Agent: a tool's default agent got HTTP 423, and a second
one got every connection from that IP refused, browser traffic included. So
every request carries the headers Chrome itself sends (captured from the
crawlers' own Chrome 153 against a local server), requests start at least
``_INTER_REQUEST_DELAY`` seconds apart across the process, and the first block
stops all AniSearch traffic; see ``PoliteHttpClient``. ``Accept-Encoding`` omits
Chrome's ``br`` and ``zstd`` because aiohttp decodes only gzip and deflate here.
"""

from enrichment.sources.base.polite_http import FetchedPage, PoliteHttpClient

_INTER_REQUEST_DELAY = 3.0

BROWSER_HEADERS = {
    "sec-ch-ua": '"Google Chrome";v="153", "Not_A Brand";v="8", "Chromium";v="153"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Linux"',
    "Upgrade-Insecure-Requests": "1",
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,"
        "image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7"
    ),
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-User": "?1",
    "Sec-Fetch-Dest": "document",
    "Accept-Encoding": "gzip, deflate",
    "Accept-Language": "en-US,en;q=0.9",
}

ANISEARCH_CLIENT = PoliteHttpClient(
    "anisearch", BROWSER_HEADERS, min_interval=_INTER_REQUEST_DELAY
)


async def fetch_anisearch_page(url: str) -> FetchedPage | None:
    """Fetch one AniSearch page with Chrome's headers, spaced out from all others.

    Args:
        url: Full AniSearch page URL.

    Returns:
        The page and its URL after redirects, or ``None`` if it could not be read.

    Raises:
        ServiceBlockedError: If AniSearch blocked this request or an earlier one.
    """
    return await ANISEARCH_CLIENT.fetch(url)
