"""Keep a site's Cloudflare clearance so the next browser is not challenged.

A browser that passes Cloudflare gets a ``cf_clearance`` cookie, and that cookie
alone skips the challenge in another browser with the same User-Agent: on
2026-10-04 a fresh AniDB browser given only it loaded a character in 0.67 s with
no challenge, against 1.9 s through the interstitial. A different User-Agent
makes the challenge unsolvable, so clearances are keyed by site and User-Agent.

The cookie is stored in Redis as JSON, never as a pickle (zendriver's
``CookieJar.save`` pickles), and its value is never logged.
"""

import hashlib
import json
import logging
import time

import zendriver
from http_cache.config import get_cache_config
from http_cache.result_cache import get_result_cache_redis_client
from zendriver import cdp

logger = logging.getLogger(__name__)


class BrowserNotStartedError(RuntimeError):
    """Raised when a clearance is read for a browser with no connection."""

    def __init__(self) -> None:
        super().__init__("the browser has no connection yet")


CLEARANCE_COOKIE_NAME = "cf_clearance"
_KEY_PREFIX = "cloudflare_clearance"
_STORED_FIELDS = (
    "name",
    "value",
    "domain",
    "path",
    "secure",
    "httpOnly",
    "sameSite",
    "expires",
)


async def load_clearance(browser: zendriver.Browser, site: str) -> bool:
    """Give the browser the clearance stored for ``site`` and its User-Agent.

    Call before the browser's first navigation. Never raises: a failure is
    logged and the browser is simply challenged as usual.

    Args:
        browser: A started browser that has not navigated yet.
        site: The cookie's registrable domain, such as ``"anidb.net"``.

    Returns:
        Whether a stored clearance was given to the browser.
    """
    try:
        stored = await (await get_result_cache_redis_client()).get(
            await _clearance_key(browser, site)
        )
        if stored is None:
            return False
        await browser.cookies.set_all(
            [cdp.network.CookieParam.from_json(json.loads(stored))]
        )
    except Exception as error:
        logger.warning(
            f"could not load the stored {site} clearance: {type(error).__name__}"
        )
        return False
    logger.info(f"gave the browser the stored {site} Cloudflare clearance")
    return True


async def save_clearance(browser: zendriver.Browser, site: str) -> bool:
    """Store the browser's clearance for ``site``, replacing any stored one.

    Call after the browser passed a challenge on ``site``. Only the
    ``cf_clearance`` cookie for that domain is kept, until the cookie expires or
    ``cloudflare_clearance_max_ttl`` passes, whichever is sooner. Never raises.

    Args:
        browser: The browser that passed the challenge.
        site: The cookie's registrable domain, such as ``"anidb.net"``.

    Returns:
        Whether a clearance was stored.
    """
    try:
        cookie = next(
            (
                cookie
                for cookie in await browser.cookies.get_all()
                if cookie.name == CLEARANCE_COOKIE_NAME
                and cookie.domain.lstrip(".") == site
            ),
            None,
        )
        if cookie is None:
            return False
        ttl = get_cache_config().cloudflare_clearance_max_ttl
        if cookie.expires is not None and not cookie.session:
            ttl = min(ttl, int(cookie.expires - time.time()))
        if ttl <= 0:
            return False
        cookie_json = cookie.to_json()
        stored = {
            field: cookie_json[field]
            for field in _STORED_FIELDS
            if field in cookie_json
        }
        await (await get_result_cache_redis_client()).set(
            await _clearance_key(browser, site), json.dumps(stored), ex=ttl
        )
    except Exception as error:
        logger.warning(f"could not store the {site} clearance: {type(error).__name__}")
        return False
    logger.info(f"stored the {site} Cloudflare clearance for {ttl}s")
    return True


async def _clearance_key(browser: zendriver.Browser, site: str) -> str:
    if browser.connection is None:
        raise BrowserNotStartedError
    version = await browser.connection.send(cdp.browser.get_version())
    user_agent = version[3]
    return (
        f"{_KEY_PREFIX}:{site}:{hashlib.sha256(user_agent.encode()).hexdigest()[:16]}"
    )
