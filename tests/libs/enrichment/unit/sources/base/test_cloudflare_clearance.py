import hashlib
import json
import time
from collections.abc import Iterator
from unittest.mock import create_autospec, patch

import pytest
import zendriver
from enrichment.sources.base import cloudflare_clearance
from enrichment.sources.base.cloudflare_clearance import load_clearance, save_clearance
from redis.asyncio import Redis
from zendriver import cdp
from zendriver.core.browser import CookieJar
from zendriver.core.connection import Connection

HEADED_AGENT = "Mozilla/5.0 (X11; Linux x86_64) Chrome/141.0.0.0 Safari/537.36"
HEADLESS_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) HeadlessChrome/141.0.0.0 Safari/537.36"
)
CLEARANCE_VALUE = "clearance-value-that-must-stay-secret"


class InMemoryRedis(Redis):
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.expiry: dict[str, int | None] = {}

    def __del__(self) -> None:
        return None

    async def get(self, name: str) -> str | None:
        return self.values.get(name)

    async def set(
        self, name: str, value: str, ex: int | None = None, **_: object
    ) -> bool:
        self.values[name] = value
        self.expiry[name] = ex
        return True


class FailingRedis(InMemoryRedis):
    async def set(
        self, name: str, value: str, ex: int | None = None, **_: object
    ) -> bool:
        raise ConnectionError(f"cannot store {value}")


def _cookie(name: str, domain: str, expires_in: float = 3600) -> cdp.network.Cookie:
    return cdp.network.Cookie.from_json(
        {
            "name": name,
            "value": CLEARANCE_VALUE if name == "cf_clearance" else "other",
            "domain": domain,
            "path": "/",
            "size": 40,
            "httpOnly": True,
            "secure": True,
            "session": False,
            "priority": "Medium",
            "sourceScheme": "Secure",
            "sourcePort": 443,
            "expires": time.time() + expires_in,
            "sameSite": "None",
        }
    )


def _browser(user_agent: str, cookies: list[cdp.network.Cookie]) -> zendriver.Browser:
    browser = create_autospec(zendriver.Browser, instance=True)
    browser.connection = create_autospec(Connection, instance=True)
    browser.connection.send.return_value = (
        "1.3",
        "Chrome/141",
        "rev",
        user_agent,
        "13.0",
    )
    browser.cookies = create_autospec(CookieJar, instance=True)
    browser.cookies.get_all.return_value = cookies
    return browser


@pytest.fixture
def redis() -> Iterator[InMemoryRedis]:
    store = InMemoryRedis()
    with patch.object(
        cloudflare_clearance,
        "get_result_cache_redis_client",
        autospec=True,
        return_value=store,
    ):
        yield store


async def test_save_clearance_stores_only_site_cf_clearance_keyed_by_user_agent(
    redis: InMemoryRedis,
) -> None:
    browser = _browser(
        HEADED_AGENT,
        [
            _cookie("cf_clearance", ".anidb.net"),
            _cookie("__cf_ob", ".anidb.net"),
            _cookie("cf_clearance", ".example.com"),
        ],
    )

    assert await save_clearance(browser, "anidb.net")

    agent_hash = hashlib.sha256(HEADED_AGENT.encode()).hexdigest()[:16]
    key = f"cloudflare_clearance:anidb.net:{agent_hash}"
    assert list(redis.values) == [key]
    stored = json.loads(redis.values[key])
    assert (stored["name"], stored["value"], stored["domain"]) == (
        "cf_clearance",
        CLEARANCE_VALUE,
        ".anidb.net",
    )
    assert 3590 <= redis.expiry[key] <= 3600


async def test_save_clearance_long_lived_cookie_kept_for_max_ttl(
    redis: InMemoryRedis,
) -> None:
    browser = _browser(
        HEADED_AGENT, [_cookie("cf_clearance", ".anidb.net", 365 * 86400)]
    )

    await save_clearance(browser, "anidb.net")

    assert list(redis.expiry.values()) == [86400]


async def test_save_clearance_without_clearance_cookie_stores_nothing(
    redis: InMemoryRedis,
) -> None:
    browser = _browser(HEADED_AGENT, [_cookie("__cf_ob", ".anidb.net")])

    assert not await save_clearance(browser, "anidb.net")
    assert redis.values == {}


async def test_load_clearance_same_user_agent_gives_stored_cookie(
    redis: InMemoryRedis,
) -> None:
    await save_clearance(
        _browser(HEADED_AGENT, [_cookie("cf_clearance", ".anidb.net")]), "anidb.net"
    )
    fresh = _browser(HEADED_AGENT, [])

    assert await load_clearance(fresh, "anidb.net")

    [given] = fresh.cookies.set_all.await_args.args[0]
    assert (given.name, given.value, given.domain) == (
        "cf_clearance",
        CLEARANCE_VALUE,
        ".anidb.net",
    )


async def test_load_clearance_different_user_agent_gives_nothing(
    redis: InMemoryRedis,
) -> None:
    await save_clearance(
        _browser(HEADED_AGENT, [_cookie("cf_clearance", ".anidb.net")]), "anidb.net"
    )
    headless = _browser(HEADLESS_AGENT, [])

    assert not await load_clearance(headless, "anidb.net")
    headless.cookies.set_all.assert_not_awaited()


async def test_load_clearance_redis_unavailable_returns_false() -> None:
    with patch.object(
        cloudflare_clearance,
        "get_result_cache_redis_client",
        autospec=True,
        side_effect=ConnectionError("refused"),
    ):
        assert not await load_clearance(_browser(HEADED_AGENT, []), "anidb.net")


async def test_save_clearance_value_never_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("DEBUG")
    browser = _browser(HEADED_AGENT, [_cookie("cf_clearance", ".anidb.net")])

    for store in (InMemoryRedis(), FailingRedis()):
        with patch.object(
            cloudflare_clearance,
            "get_result_cache_redis_client",
            autospec=True,
            return_value=store,
        ):
            await save_clearance(browser, "anidb.net")
            await load_clearance(_browser(HEADED_AGENT, []), "anidb.net")

    assert "clearance" in caplog.text
    assert CLEARANCE_VALUE not in caplog.text
