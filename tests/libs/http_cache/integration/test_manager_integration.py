import asyncio
from collections import Counter
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from aiohttp import web
from common.config import CacheConfig, RedisConfig
from hishel import AsyncRedisStorage, Headers, Request, Response
from http_cache import HTTPCacheManager, close_cache, configure_cache
from http_cache.aiohttp_adapter import CachedAiohttpSession

from tests.conftest import TEST_REDIS_URL

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def server() -> AsyncIterator[tuple[str, Counter[str]]]:
    hits: Counter[str] = Counter()

    async def respond(request: web.Request) -> web.Response:
        hits[request.path_qs] += 1
        match request.path:
            case "/json":
                return web.json_response({"id": request.query["id"]})
            case "/graphql":
                body = await request.text()
                hits[body] += 1
                return web.json_response({"echo": body})
            case "/error":
                return web.Response(status=500, text="boom")
            case "/banned":
                return web.Response(text='<error code="500">banned</error>')
            case "/large":
                return web.Response(body=b"x" * 12 * 1024 * 1024)
            case "/medium":
                return web.Response(body=bytes(range(256)) * 11 * 1024)
        return web.Response(status=404)

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", respond)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    host, port = runner.addresses[0][:2]
    yield f"http://{host}:{port}", hits
    await runner.cleanup()


@pytest_asyncio.fixture
async def cache() -> AsyncIterator[HTTPCacheManager]:
    manager = configure_cache(
        CacheConfig(cache_enabled=True, ttl_kitsu=2),
        RedisConfig(redis_url=TEST_REDIS_URL),
    )
    redis_client = manager.get_result_client()
    assert redis_client is not None
    await redis_client.flushdb()
    yield manager
    await redis_client.flushdb()
    await close_cache()


async def _get(
    cache: HTTPCacheManager,
    url: str,
    service: str = "anilist",
    *,
    json: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[bool, bytes]:
    session = cache.get_aiohttp_session(service)
    assert isinstance(session, CachedAiohttpSession)
    async with session:
        if json is None:
            request = session.get(url, headers=headers)
        else:
            request = session.post(url, json=json, headers=headers)
        async with request as response:
            return response.from_cache, await response.read()


async def test_get_aiohttp_session_repeat_get_served_from_cache(
    cache: HTTPCacheManager, server: tuple[str, Counter[str]]
) -> None:
    base, hits = server

    first = await _get(cache, f"{base}/json?id=1")
    second = await _get(cache, f"{base}/json?id=1")

    assert (first[0], second[0]) == (False, True)
    assert second[1] == first[1]
    assert hits["/json?id=1"] == 1


async def test_get_aiohttp_session_post_body_key_caches_each_body(
    cache: HTTPCacheManager, server: tuple[str, Counter[str]]
) -> None:
    base, hits = server
    headers = {"X-Hishel-Body-Key": "true"}

    first = await _get(cache, f"{base}/graphql", json={"q": "a"}, headers=headers)
    other = await _get(cache, f"{base}/graphql", json={"q": "b"}, headers=headers)
    repeat = await _get(cache, f"{base}/graphql", json={"q": "a"}, headers=headers)

    assert (first[0], other[0], repeat[0]) == (False, False, True)
    assert repeat[1] == first[1] != other[1]
    assert hits["/graphql"] == 2


@pytest.mark.parametrize("path", ["/error", "/banned"])
async def test_get_aiohttp_session_error_response_not_cached(
    cache: HTTPCacheManager, server: tuple[str, Counter[str]], path: str
) -> None:
    base, hits = server

    await _get(cache, f"{base}{path}")
    second = await _get(cache, f"{base}{path}")

    assert second[0] is False
    assert hits[path] == 2


async def test_get_aiohttp_session_response_over_10_mb_returned_but_not_cached(
    cache: HTTPCacheManager, server: tuple[str, Counter[str]]
) -> None:
    base, hits = server

    first = await _get(cache, f"{base}/large")
    second = await _get(cache, f"{base}/large")

    assert len(first[1]) == len(second[1]) == 12 * 1024 * 1024
    assert second[0] is False
    assert hits["/large"] == 2


async def test_get_aiohttp_session_expires_after_service_ttl_from_fetch(
    cache: HTTPCacheManager, server: tuple[str, Counter[str]]
) -> None:
    base, hits = server

    await _get(cache, f"{base}/json?id=ttl", "kitsu")
    await asyncio.sleep(1.2)
    before_expiry = await _get(cache, f"{base}/json?id=ttl", "kitsu")
    await asyncio.sleep(1.2)
    after_expiry = await _get(cache, f"{base}/json?id=ttl", "kitsu")

    assert before_expiry[0] is True
    assert after_expiry[0] is False
    assert hits["/json?id=ttl"] == 2


async def test_get_aiohttp_session_closed_session_leaves_redis_client_open(
    cache: HTTPCacheManager, server: tuple[str, Counter[str]]
) -> None:
    base, _ = server

    await _get(cache, f"{base}/json?id=open")

    assert (await _get(cache, f"{base}/json?id=open"))[0] is True


async def test_async_redis_storage_broken_stream_not_served(
    cache: HTTPCacheManager,
) -> None:
    async def cut_off() -> AsyncIterator[bytes]:
        yield b"partial"
        raise ConnectionError("cut")

    session = cache.get_aiohttp_session("anilist")
    assert isinstance(session, CachedAiohttpSession)
    storage = session.storage
    assert isinstance(storage, AsyncRedisStorage)
    entry = await storage.create_entry(
        Request(method="GET", url="https://example.com/broken", headers=Headers({})),
        Response(status_code=200, headers=Headers({}), stream=cut_off()),
        "broken",
    )
    with pytest.raises(ConnectionError):
        await entry.response.aread()
    await session.close()

    assert await storage.get_entries("broken") == []


async def _stored_entry(cache: HTTPCacheManager, key: str) -> AsyncRedisStorage:
    async def body() -> AsyncIterator[bytes]:
        yield b"stored body"

    session = cache.get_aiohttp_session("anilist")
    assert isinstance(session, CachedAiohttpSession)
    storage = session.storage
    assert isinstance(storage, AsyncRedisStorage)
    entry = await storage.create_entry(
        Request(method="GET", url=f"https://example.com/{key}", headers=Headers({})),
        Response(status_code=200, headers=Headers({}), stream=body()),
        key,
    )
    await entry.response.aread()
    await session.close()
    return storage


async def test_get_aiohttp_session_request_ttl_header_overrides_service_ttl(
    cache: HTTPCacheManager, server: tuple[str, Counter[str]]
) -> None:
    base, hits = server
    headers = {"X-Hishel-Ttl": "1"}

    await _get(cache, f"{base}/json?id=short", headers=headers)
    await asyncio.sleep(1.4)
    after_expiry = await _get(cache, f"{base}/json?id=short", headers=headers)

    assert after_expiry[0] is False
    assert hits["/json?id=short"] == 2


async def test_get_aiohttp_session_body_larger_than_storage_chunk_served_intact(
    cache: HTTPCacheManager, server: tuple[str, Counter[str]]
) -> None:
    base, hits = server

    first = await _get(cache, f"{base}/medium")
    second = await _get(cache, f"{base}/medium")

    assert second == (True, first[1])
    assert len(first[1]) == 256 * 11 * 1024
    assert hits["/medium"] == 1


async def test_get_aiohttp_session_expired_response_fetched_then_served_again(
    cache: HTTPCacheManager, server: tuple[str, Counter[str]]
) -> None:
    base, hits = server

    await _get(cache, f"{base}/json?id=refetch", "kitsu")
    await asyncio.sleep(2.4)
    refetched = await _get(cache, f"{base}/json?id=refetch", "kitsu")
    served_again = await _get(cache, f"{base}/json?id=refetch", "kitsu")

    assert (refetched[0], served_again[0]) == (False, True)
    assert hits["/json?id=refetch"] == 2


async def test_get_aiohttp_session_every_stored_key_has_ttl(
    cache: HTTPCacheManager, server: tuple[str, Counter[str]]
) -> None:
    base, _ = server
    headers = {"X-Hishel-Body-Key": "true"}

    await _get(cache, f"{base}/json?id=keys")
    await _get(cache, f"{base}/graphql", json={"q": "keys"}, headers=headers)
    redis_client = cache.get_result_client()
    assert redis_client is not None
    keys = await redis_client.keys("hishel_cache:*")

    assert keys
    assert all([await redis_client.ttl(key) > 0 for key in keys])


async def test_async_redis_storage_update_entry_keeps_remaining_ttl(
    cache: HTTPCacheManager,
) -> None:
    storage = await _stored_entry(cache, "update")
    redis_client = cache.get_result_client()
    assert redis_client is not None
    [entry] = await storage.get_entries("update")
    entry_key = f"hishel_cache:entry:{entry.id.hex}"
    ttl_before = await redis_client.pttl(entry_key)

    updated = await storage.update_entry(entry.id, lambda current: current)

    assert updated is not None
    assert 0 <= ttl_before - await redis_client.pttl(entry_key) < 1000


async def test_async_redis_storage_removed_entry_not_served(
    cache: HTTPCacheManager,
) -> None:
    storage = await _stored_entry(cache, "remove")
    [entry] = await storage.get_entries("remove")

    await storage.remove_entry(entry.id)

    assert await storage.get_entries("remove") == []


async def test_get_aiohttp_session_concurrent_calls_share_one_redis_client(
    cache: HTTPCacheManager,
) -> None:
    async def open_storage(task_id: int) -> AsyncRedisStorage:
        session = cache.get_aiohttp_session(f"service_{task_id}")
        assert isinstance(session, CachedAiohttpSession)
        await session.close()
        assert isinstance(session.storage, AsyncRedisStorage)
        return session.storage

    storages = await asyncio.gather(*(open_storage(task) for task in range(10)))

    assert len({id(storage._client) for storage in storages}) == 1
