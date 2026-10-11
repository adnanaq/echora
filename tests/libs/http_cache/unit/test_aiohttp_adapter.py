from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from unittest.mock import AsyncMock, create_autospec, patch

import aiohttp
import pytest
from aiohttp.client import _RequestContextManager
from hishel import Headers, Request, Response
from http_cache.aiohttp_adapter import CachedAiohttpSession, _CachedResponse
from multidict import CIMultiDict, CIMultiDictProxy  # pants: no-infer-dep
from yarl import URL  # pants: no-infer-dep

_API_URL = "https://example.com/api"


async def _no_result() -> None:
    return None


def _origin_response(
    body: bytes = b"{}",
    *,
    status: int = 200,
    cache_control: str = "public, max-age=86400",
    headers: dict | CIMultiDict | None = None,
) -> aiohttp.ClientResponse:
    response = create_autospec(aiohttp.ClientResponse, instance=True)
    response.status = status
    response.headers = (
        headers
        if headers is not None
        else {"Content-Type": "application/json", "Cache-Control": cache_control}
    )
    response.read.return_value = body
    response.release.side_effect = _no_result
    response.__aenter__.return_value = response
    return response


def _origin_session(
    *responses: aiohttp.ClientResponse, default_headers: dict | None = None
) -> aiohttp.ClientSession:
    session = create_autospec(aiohttp.ClientSession, instance=True)
    session.headers = default_headers or {}
    queue = list(responses)

    async def respond() -> aiohttp.ClientResponse:
        return queue.pop(0) if len(queue) > 1 else queue[0]

    session.request.side_effect = lambda *args, **kwargs: _RequestContextManager(
        respond()
    )
    return session


def _cached_session(
    storage: AsyncMock, origin: aiohttp.ClientSession, **settings: object
) -> CachedAiohttpSession:
    return CachedAiohttpSession(storage=storage, session=origin, **settings)


async def _body(chunk: bytes = b"{}") -> AsyncIterator[bytes]:
    yield chunk


def _hishel_request(
    headers: dict | None = None, metadata: dict | None = None, method: str = "GET"
) -> Request:
    return Request(
        method=method,
        url=_API_URL,
        headers=Headers(headers or {}),
        stream=_body(),
        metadata=metadata or {},
    )


def _handled_by(cached: CachedAiohttpSession, handle: Callable[[Request], Response]):
    return patch.object(
        cached._proxy, "handle_request", autospec=True, side_effect=handle
    )


def _fresh_response(request: Request, *, from_cache: bool = False) -> Response:
    return Response(
        status_code=200, stream=_body(), metadata={"hishel_from_cache": from_cache}
    )


def _stored_response(storage: AsyncMock) -> Response:
    _request, response, _key = storage.create_entry.call_args.args
    return response


def _cached_response(status: int, body: bytes = b"ok") -> _CachedResponse:
    return _CachedResponse(
        status=status,
        headers={"Content-Type": "application/json"},
        body=body,
        url=_API_URL,
        method="GET",
        request_headers={"Accept": "application/json"},
        from_cache=True,
    )


async def test_cached_response_body_reads_as_bytes_json_and_text() -> None:
    response = _cached_response(200, b'{"ok": true}')

    assert (response.status, response.url, response.from_cache) == (
        200,
        URL(_API_URL),
        True,
    )
    assert isinstance(response.headers, CIMultiDictProxy)
    assert await response.read() == b'{"ok": true}'
    assert await response.json() == {"ok": True}
    assert await response.text() == '{"ok": true}'


def test_cached_response_raise_for_status_client_error_raises() -> None:
    with pytest.raises(aiohttp.ClientResponseError) as raised:
        _cached_response(404, b"not found").raise_for_status()

    assert raised.value.status == 404


def test_cached_response_raise_for_status_ok_returns_none() -> None:
    assert _cached_response(200).raise_for_status() is None


async def test_cached_response_context_manager_yields_itself() -> None:
    response = _cached_response(200)

    async with response as entered:
        body = await entered.read()

    assert entered is response
    assert body == b"ok"


async def test_cached_aiohttp_session_request_moves_hishel_headers_to_metadata(
    storage: AsyncMock,
) -> None:
    origin = _origin_session(
        _origin_response(),
        default_headers={"X-Hishel-Body-Key": "true", "User-Agent": "ua"},
    )
    cached = _cached_session(storage, origin)

    with _handled_by(cached, _fresh_response) as handle_request:
        response = await cached._request(
            "POST",
            "https://example.com/graphql",
            json={"query": "q"},
            headers={"Content-Type": "application/json"},
        )

    [sent] = handle_request.call_args.args
    assert sent.headers.get("User-Agent") == "ua"
    assert "X-Hishel-Body-Key" not in dict(sent.headers)
    assert sent.metadata.get("hishel_body_key") is True
    assert isinstance(response, _CachedResponse)
    assert response.from_cache is False


async def test_cached_aiohttp_session_request_params_become_part_of_url(
    storage: AsyncMock,
) -> None:
    cached = _cached_session(storage, _origin_session(_origin_response()))

    with _handled_by(
        cached, lambda request: _fresh_response(request, from_cache=True)
    ) as handle_request:
        response = await cached._request(
            "GET", _API_URL, params={"q": "one piece", "page": 2}
        )

    [sent] = handle_request.call_args.args
    assert str(sent.url) == f"{_API_URL}?q=one+piece&page=2"
    assert response.url == URL(f"{_API_URL}?q=one+piece&page=2")
    assert response.from_cache is True


async def test_cached_aiohttp_session_origin_request_omits_hishel_headers(
    storage: AsyncMock,
) -> None:
    origin = _origin_session(_origin_response(cache_control="no-store"))
    cached = _cached_session(storage, origin)

    response = await cached._proxy.handle_request(
        _hishel_request(
            {"X-Hishel-Body-Key": "true", "Content-Type": "application/json"},
            {"hishel_body_key": True},
            method="POST",
        )
    )

    sent_headers = origin.request.call_args.kwargs["headers"]
    assert response.status_code == 200
    assert sent_headers["content-type"] == "application/json"
    assert not any(name.lower().startswith("x-hishel-") for name in sent_headers)


@pytest.mark.parametrize(
    ("directive", "expected_status", "origin_called"),
    [
        ("only-if-cached", 504, False),
        ("max-age=0, Only-If-Cached", 504, False),
        ("x-only-if-cached", 200, True),
        ("no-cache", 200, True),
        (None, 200, True),
    ],
)
async def test_cached_aiohttp_session_only_if_cached_miss_returns_504_without_origin(
    storage: AsyncMock, directive: str | None, expected_status: int, origin_called: bool
) -> None:
    origin = _origin_session(_origin_response())
    cached = _cached_session(storage, origin)

    response = await cached._proxy.handle_request(
        _hishel_request({"Cache-Control": directive} if directive else {})
    )

    assert response.status_code == expected_status
    assert origin.request.called is origin_called


@pytest.mark.parametrize("default_headers", [42, None])
def test_cached_aiohttp_session_unreadable_session_headers_uses_no_default_headers(
    storage: AsyncMock, default_headers: object
) -> None:
    origin = create_autospec(aiohttp.ClientSession, instance=True)
    origin.headers = default_headers

    assert _cached_session(storage, origin)._session_default_headers == {}


@pytest.mark.parametrize("data", ["hello", b"raw"])
async def test_cached_aiohttp_session_post_text_or_bytes_body_reaches_origin(
    storage: AsyncMock, data: str | bytes
) -> None:
    origin = _origin_session(_origin_response(b'{"ok":true}'))
    cached = _cached_session(storage, origin)

    async with cached.post(_API_URL, data=data) as response:
        assert response.status == 200

    assert origin.request.call_args.kwargs["data"] == (
        data.encode() if isinstance(data, str) else data
    )


@pytest.mark.parametrize(
    ("directive", "expected_status", "origin_called"),
    [
        (None, 200, True),
        ("only-if-cached", 504, False),
        ("x-only-if-cached", 200, True),
    ],
)
async def test_cached_aiohttp_session_post_form_body_bypasses_cache(
    storage: AsyncMock, directive: str | None, expected_status: int, origin_called: bool
) -> None:
    origin = _origin_session(_origin_response(b"direct"))
    cached = _cached_session(storage, origin)
    headers = {"Cache-Control": directive} if directive else {}

    async with cached.post(
        _API_URL, data=aiohttp.FormData({"field": "value"}), headers=headers
    ) as response:
        assert response.status == expected_status

    assert origin.request.called is origin_called
    assert storage.create_entry.call_count == 0


@pytest.mark.parametrize(
    ("caller_directive", "sent_directive"),
    [
        (None, "no-cache"),
        ("only-if-cached", "only-if-cached"),
        ("x-only-if-cached", "no-cache"),
    ],
)
async def test_cached_aiohttp_session_always_revalidate_sends_no_cache(
    storage: AsyncMock, caller_directive: str | None, sent_directive: str
) -> None:
    cached = _cached_session(
        storage, _origin_session(_origin_response()), always_revalidate=True
    )
    headers = {"Cache-Control": caller_directive} if caller_directive else {}

    with _handled_by(cached, _fresh_response) as handle_request:
        await cached._request("GET", _API_URL, headers=headers)

    [sent] = handle_request.call_args.args
    assert sent.headers.get("Cache-Control") == sent_directive


async def test_cached_aiohttp_session_headers_none_sends_session_default_headers(
    storage: AsyncMock,
) -> None:
    cached = _cached_session(
        storage,
        _origin_session(_origin_response(), default_headers={"User-Agent": "ua"}),
    )

    with _handled_by(cached, _fresh_response) as handle_request:
        response = await cached._request("GET", "https://example.com/a", headers=None)

    [sent] = handle_request.call_args.args
    assert sent.headers.get("User-Agent") == "ua"
    assert response.status == 200


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"X-Hishel-Ttl": "3600"}, {"hishel_ttl": 3600.0}),
        ({"X-Hishel-Ttl": "not-a-number"}, {}),
        (
            {"X-Hishel-Refresh-Ttl-On-Access": "true"},
            {"hishel_refresh_ttl_on_access": True},
        ),
        ({"X-Hishel-Spec-Ignore": "1"}, {"hishel_spec_ignore": True}),
        ({"X-Hishel-Body-Key": "false"}, {}),
    ],
)
def test_extract_hishel_metadata_and_strip_headers_hishel_header_returns_metadata(
    headers: dict, expected: dict
) -> None:
    metadata = CachedAiohttpSession._extract_hishel_metadata_and_strip_headers(headers)

    assert metadata == expected
    assert headers == {}


def test_extract_hishel_metadata_and_strip_headers_other_headers_kept() -> None:
    headers = {"Accept": "application/json", "Authorization": "Bearer token"}

    metadata = CachedAiohttpSession._extract_hishel_metadata_and_strip_headers(headers)

    assert metadata == {}
    assert headers == {"Accept": "application/json", "Authorization": "Bearer token"}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        *((value, True) for value in ["1", "true", "yes", "on", "TRUE", "  True  "]),
        *((value, False) for value in ["0", "false", "no", "off", "FALSE"]),
        *((value, None) for value in ["maybe", "2", "", "yep"]),
    ],
)
def test_parse_hishel_bool_text_returns_flag(value: str, expected: bool | None) -> None:
    assert CachedAiohttpSession._parse_hishel_bool(value) is expected


@pytest.mark.parametrize(
    ("settings", "metadata", "expected"),
    [
        ({}, {"hishel_ttl": 1200}, "public, max-age=1200"),
        ({"default_ttl": 7200}, {}, "public, max-age=7200"),
        ({"default_ttl": 0}, {}, "public, max-age=0"),
        ({}, {}, "public, max-age=86400"),
    ],
)
async def test_cached_aiohttp_session_force_cache_stores_max_age(
    storage: AsyncMock, settings: dict, metadata: dict, expected: str
) -> None:
    cached = _cached_session(
        storage,
        _origin_session(_origin_response(cache_control="no-cache")),
        force_cache=True,
        **settings,
    )

    response = await cached._proxy.handle_request(_hishel_request(metadata=metadata))

    assert response.status_code == 200
    assert _stored_response(storage).headers.get("Cache-Control") == expected


async def test_cached_aiohttp_session_force_cache_error_response_stores_no_store(
    storage: AsyncMock,
) -> None:
    cached = _cached_session(
        storage,
        _origin_session(
            _origin_response(b"rate limited", status=429, cache_control="no-cache")
        ),
        force_cache=True,
    )

    response = await cached._proxy.handle_request(_hishel_request())

    assert response.status_code == 429
    assert "no-store" in (_stored_response(storage).headers.get("Cache-Control") or "")


async def test_cached_aiohttp_session_multi_value_header_stores_joined_values(
    storage: AsyncMock,
) -> None:
    headers = CIMultiDict([("X-Custom", "val1"), ("X-Custom", "val2")])
    cached = _cached_session(
        storage, _origin_session(_origin_response(headers=headers))
    )

    response = await cached._proxy.handle_request(_hishel_request())

    assert response.status_code == 200
    assert _stored_response(storage).headers.get("X-Custom") == "val1, val2"


async def test_cached_aiohttp_session_close_closes_session_only(
    storage: AsyncMock,
) -> None:
    origin = _origin_session(_origin_response())

    await _cached_session(storage, origin).close()

    assert origin.close.await_count == 1
    assert storage.close.call_count == 0


async def test_cached_aiohttp_session_context_manager_closes_session_only(
    storage: AsyncMock,
) -> None:
    origin = _origin_session(_origin_response())

    async with _cached_session(storage, origin) as cached:
        assert isinstance(cached, CachedAiohttpSession)

    assert origin.close.await_count == 1
    assert storage.close.call_count == 0


async def test_cached_aiohttp_session_get_different_urls_store_different_keys(
    storage: AsyncMock,
) -> None:
    cached = _cached_session(
        storage,
        _origin_session(
            _origin_response(b'{"url": "a"}'), _origin_response(b'{"url": "b"}')
        ),
    )

    for url in (f"{_API_URL}/a", f"{_API_URL}/b"):
        async with cached.get(url) as response:
            await response.read()

    keys = [call.args[2] for call in storage.create_entry.call_args_list]
    assert len(set(keys)) == 2


async def test_cached_aiohttp_session_post_different_bodies_store_different_keys(
    storage: AsyncMock,
) -> None:
    cached = _cached_session(
        storage,
        _origin_session(
            _origin_response(b'{"data": {"anime": {"id": 1}}}'),
            _origin_response(b'{"data": {"anime": {"id": 2}}}'),
            default_headers={"X-Hishel-Body-Key": "true"},
        ),
    )

    for query in ("{ anime(id: 1) { title } }", "{ anime(id: 2) { title } }"):
        async with cached.post("https://graphql.anilist.co", json={"query": query}):
            pass

    keys = [call.args[2] for call in storage.create_entry.call_args_list]
    assert len(set(keys)) == 2
