"""
Unit tests for http_cache.aiohttp_adapter.

Focus:
- _CachedResponse interface (status, headers, body, from_cache, raise_for_status)
- ReusableBodyStream async iteration
- Request shaping: header merging, X-Hishel-* stripping, params encoding
- _extract_hishel_metadata_and_strip_headers: TTL, refresh, spec-ignore, body-key
- _parse_hishel_bool: truthy/falsy/invalid strings
- request_sender: force_cache TTL resolution, error suppression, getall headers
- Unsupported body fallback (FormData / arbitrary data objects)
- always_revalidate injects Cache-Control: no-cache
- Upstream isolation: X-Hishel-* headers must not reach the origin server
- Cache key isolation: GET URLs use URL hash, POST bodies use body hash
- Session lifecycle: close(), async context manager
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hishel import Request, Response
from http_cache.aiohttp_adapter import CachedAiohttpSession, _CachedResponse
from multidict import CIMultiDictProxy  # pants: no-infer-dep
from yarl import URL  # pants: no-infer-dep


# =============================================================================
# Helpers
# =============================================================================


def _make_origin_resp(
    body: bytes = b"{}",
    *,
    status: int = 200,
    cache_control: str = "public, max-age=86400",
    extra_headers: dict | None = None,
) -> MagicMock:
    """Return a minimal mock aiohttp response for request_sender tests."""
    resp = MagicMock()
    resp.status = status
    headers = {"Content-Type": "application/json", "Cache-Control": cache_control}
    if extra_headers:
        headers.update(extra_headers)
    resp.headers = headers
    resp.read = AsyncMock(return_value=body)
    resp.release = AsyncMock()
    return resp


# =============================================================================
# Test Class: _CachedResponse
# =============================================================================


class TestCachedResponse:
    @pytest.mark.asyncio
    async def test_minimal_interface(self) -> None:
        resp = _CachedResponse(
            status=200,
            headers={"Content-Type": "application/json"},
            body=b'{"ok": true}',
            url="https://example.com/api",
            method="GET",
            request_headers={"Accept": "application/json"},
            from_cache=True,
        )
        assert resp.status == 200
        assert resp.url == URL("https://example.com/api")
        assert resp.from_cache is True
        assert isinstance(resp.headers, CIMultiDictProxy)
        assert await resp.read() == b'{"ok": true}'
        assert await resp.json() == {"ok": True}
        assert await resp.text() == '{"ok": true}'

    @pytest.mark.asyncio
    async def test_raise_for_status_on_4xx(self) -> None:
        import aiohttp

        resp = _CachedResponse(
            status=404,
            headers={},
            body=b"not found",
            url="https://example.com/api",
            method="GET",
            request_headers={},
        )
        with pytest.raises(aiohttp.ClientResponseError) as exc_info:
            resp.raise_for_status()
        assert exc_info.value.status == 404

    def test_raise_for_status_ok_does_not_raise(self) -> None:
        resp = _CachedResponse(
            status=200,
            headers={},
            body=b"ok",
            url="https://example.com/api",
            method="GET",
            request_headers={},
        )
        resp.raise_for_status()  # must not raise

    @pytest.mark.asyncio
    async def test_async_context_manager(self) -> None:
        resp = _CachedResponse(
            status=200,
            headers={},
            body=b"ok",
            url="https://example.com/api",
            method="GET",
            request_headers={},
        )
        async with resp as r:
            assert r is resp
            data = await r.read()
        assert data == b"ok"


# =============================================================================
# Test Class: CachedAiohttpSession — request shaping and upstream isolation
# =============================================================================


class TestCachedAiohttpSessionRequestBuilding:
    @pytest.mark.asyncio
    async def test_merges_session_default_headers_and_strips_x_hishel(
        self, mock_storage: AsyncMock
    ) -> None:
        """Session default headers are forwarded; X-Hishel-* headers are moved
        to request metadata and removed from the outgoing headers."""
        mock_session = AsyncMock()
        mock_session.headers = {"X-Hishel-Body-Key": "true", "User-Agent": "ua"}

        cached = CachedAiohttpSession(storage=mock_storage, session=mock_session)

        def handle(req: Request) -> Response:
            assert req.headers.get("User-Agent") == "ua"
            assert "X-Hishel-Body-Key" not in dict(req.headers)
            assert req.metadata.get("hishel_body_key") is True

            async def stream():
                yield b"{}"

            return Response(
                status_code=200, stream=stream(), metadata={"hishel_from_cache": False}
            )

        cached._proxy.handle_request = AsyncMock(side_effect=handle)

        resp = await cached._request(
            "POST",
            "https://example.com/graphql",
            json={"query": "q"},
            headers={"Content-Type": "application/json"},
        )
        assert isinstance(resp, _CachedResponse)
        assert resp.from_cache is False

    @pytest.mark.asyncio
    async def test_includes_params_in_url(self, mock_storage: AsyncMock) -> None:
        """Query params are encoded into the URL so they become part of the cache key."""
        mock_session = AsyncMock()
        mock_session.headers = {}
        cached = CachedAiohttpSession(storage=mock_storage, session=mock_session)

        def handle(req: Request) -> Response:
            assert str(req.url) == "https://example.com/api?q=one+piece&page=2"

            async def stream():
                yield b"ok"

            return Response(
                status_code=200, stream=stream(), metadata={"hishel_from_cache": True}
            )

        cached._proxy.handle_request = AsyncMock(side_effect=handle)

        resp = await cached._request(
            "GET",
            "https://example.com/api",
            params={"q": "one piece", "page": 2},
        )
        assert resp.url == URL("https://example.com/api?q=one+piece&page=2")
        assert resp.from_cache is True

    @pytest.mark.asyncio
    async def test_does_not_forward_x_hishel_headers_upstream(
        self, mock_storage: AsyncMock
    ) -> None:
        """X-Hishel-* control headers must not be sent to the origin server."""
        mock_session = AsyncMock()
        mock_session.headers = {}
        cached = CachedAiohttpSession(storage=mock_storage, session=mock_session)
        mock_session.request = AsyncMock(
            return_value=_make_origin_resp(cache_control="no-store")
        )

        async def body_stream():
            yield b"{}"

        req = Request(
            method="POST",
            url="https://example.com/graphql",
            headers={"X-Hishel-Body-Key": "true", "Content-Type": "application/json"},
            stream=body_stream(),
            metadata={"hishel_body_key": True},
        )
        hishel_resp = await cached._proxy.handle_request(req)
        assert hishel_resp.status_code == 200

        _, call_kwargs = mock_session.request.call_args
        assert call_kwargs["headers"]["Content-Type"] == "application/json"
        assert "X-Hishel-Body-Key" not in call_kwargs["headers"]

    def test_session_headers_type_error_falls_back_to_empty(
        self, mock_storage: AsyncMock
    ) -> None:
        """If session.headers raises TypeError (e.g. AsyncMock), defaults to {}."""
        mock_session = AsyncMock()
        mock_session.headers = MagicMock(side_effect=TypeError("not iterable"))
        cached = CachedAiohttpSession(storage=mock_storage, session=mock_session)
        assert cached._session_default_headers == {}

    @pytest.mark.asyncio
    async def test_data_str_body_encoded_as_utf8(self, mock_storage: AsyncMock) -> None:
        """data=str is encoded to bytes for the cache key."""
        mock_session = AsyncMock()
        mock_session.headers = {}
        cached = CachedAiohttpSession(storage=mock_storage, session=mock_session)
        mock_session.request = AsyncMock(return_value=_make_origin_resp(b'{"ok":true}'))

        async with cached.post("https://example.com/api", data="hello") as resp:
            assert resp.status == 200

    @pytest.mark.asyncio
    async def test_data_bytes_body_used_directly(self, mock_storage: AsyncMock) -> None:
        """data=bytes is used directly as the body."""
        mock_session = AsyncMock()
        mock_session.headers = {}
        cached = CachedAiohttpSession(storage=mock_storage, session=mock_session)
        mock_session.request = AsyncMock(return_value=_make_origin_resp(b'{"ok":true}'))

        async with cached.post("https://example.com/api", data=b"raw") as resp:
            assert resp.status == 200

    @pytest.mark.asyncio
    async def test_unsupported_body_bypasses_cache(self, mock_storage: AsyncMock) -> None:
        """data=FormData (unsupported type) must bypass the cache entirely."""
        mock_session = AsyncMock()
        mock_session.headers = {}
        cached = CachedAiohttpSession(storage=mock_storage, session=mock_session)

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=_make_origin_resp(b"direct"))
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_session.request = MagicMock(return_value=mock_cm)

        # Pass an object that is not str/bytes/bytearray (e.g. a mock FormData)
        form_data = MagicMock()
        async with cached.post("https://example.com/api", data=form_data) as resp:
            assert resp.status == 200
        # Cache must not have been written
        mock_storage.create_entry.assert_not_called()

    @pytest.mark.asyncio
    async def test_always_revalidate_injects_no_cache(
        self, mock_storage: AsyncMock
    ) -> None:
        """always_revalidate=True must add Cache-Control: no-cache to the request."""
        mock_session = AsyncMock()
        mock_session.headers = {}
        cached = CachedAiohttpSession(
            storage=mock_storage, session=mock_session, always_revalidate=True
        )

        captured: list[Request] = []

        def handle(req: Request) -> Response:
            captured.append(req)

            async def stream():
                yield b"{}"

            return Response(
                status_code=200, stream=stream(), metadata={"hishel_from_cache": False}
            )

        cached._proxy.handle_request = AsyncMock(side_effect=handle)
        await cached._request("GET", "https://example.com/api")

        assert captured
        assert captured[0].headers.get("Cache-Control") == "no-cache"


# =============================================================================
# Test Class: CachedAiohttpSession — _extract_hishel_metadata_and_strip_headers
# =============================================================================


class TestExtractHishelMetadata:
    def test_extracts_ttl(self) -> None:
        headers = {"X-Hishel-Ttl": "3600", "Accept": "application/json"}
        meta = CachedAiohttpSession._extract_hishel_metadata_and_strip_headers(headers)
        assert meta["hishel_ttl"] == 3600.0
        assert "X-Hishel-Ttl" not in headers

    def test_invalid_ttl_ignored(self) -> None:
        headers = {"X-Hishel-Ttl": "not-a-number"}
        meta = CachedAiohttpSession._extract_hishel_metadata_and_strip_headers(headers)
        assert "hishel_ttl" not in meta

    def test_extracts_refresh_ttl_on_access(self) -> None:
        headers = {"X-Hishel-Refresh-Ttl-On-Access": "true"}
        meta = CachedAiohttpSession._extract_hishel_metadata_and_strip_headers(headers)
        assert meta["hishel_refresh_ttl_on_access"] is True

    def test_extracts_spec_ignore(self) -> None:
        headers = {"X-Hishel-Spec-Ignore": "1"}
        meta = CachedAiohttpSession._extract_hishel_metadata_and_strip_headers(headers)
        assert meta["hishel_spec_ignore"] is True

    def test_falsy_body_key_not_set_in_metadata(self) -> None:
        headers = {"X-Hishel-Body-Key": "false"}
        meta = CachedAiohttpSession._extract_hishel_metadata_and_strip_headers(headers)
        assert "hishel_body_key" not in meta

    def test_no_hishel_headers_returns_empty_metadata(self) -> None:
        headers = {"Accept": "application/json", "Authorization": "Bearer token"}
        meta = CachedAiohttpSession._extract_hishel_metadata_and_strip_headers(headers)
        assert meta == {}
        assert "Accept" in headers  # non-hishel headers untouched


# =============================================================================
# Test Class: CachedAiohttpSession — _parse_hishel_bool
# =============================================================================


class TestParseHishelBool:
    @pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE", "  True  "])
    def test_truthy_values(self, value: str) -> None:
        assert CachedAiohttpSession._parse_hishel_bool(value) is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "FALSE"])
    def test_falsy_values(self, value: str) -> None:
        assert CachedAiohttpSession._parse_hishel_bool(value) is False

    @pytest.mark.parametrize("value", ["maybe", "2", "", "yep"])
    def test_invalid_values_return_none(self, value: str) -> None:
        assert CachedAiohttpSession._parse_hishel_bool(value) is None


# =============================================================================
# Test Class: CachedAiohttpSession — force_cache and request_sender behaviour
# =============================================================================


class TestRequestSenderForceCacheBehaviour:
    @pytest.mark.asyncio
    async def test_force_cache_writes_max_age_from_metadata_ttl(
        self, mock_storage: AsyncMock
    ) -> None:
        """force_cache=True uses hishel_ttl from request metadata when present."""
        mock_session = AsyncMock()
        mock_session.headers = {}
        cached = CachedAiohttpSession(
            storage=mock_storage, session=mock_session, force_cache=True
        )
        mock_session.request = AsyncMock(
            return_value=_make_origin_resp(cache_control="no-cache")
        )

        async def body_stream():
            yield b"{}"

        req = Request(
            method="GET",
            url="https://example.com/api",
            headers={},
            stream=body_stream(),
            metadata={"hishel_ttl": 1200},
        )
        hishel_resp = await cached._proxy.handle_request(req)
        assert hishel_resp.status_code == 200

        _req, stored_response, _key = mock_storage.create_entry.call_args.args
        cc = stored_response.headers.get("Cache-Control") or ""
        assert "max-age=1200" in cc

    @pytest.mark.asyncio
    async def test_force_cache_falls_back_to_storage_default_ttl(
        self, mock_storage: AsyncMock
    ) -> None:
        """force_cache=True falls back to storage.default_ttl when no metadata TTL."""
        mock_session = AsyncMock()
        mock_session.headers = {}
        mock_storage.default_ttl = 7200.0
        cached = CachedAiohttpSession(
            storage=mock_storage, session=mock_session, force_cache=True
        )
        mock_session.request = AsyncMock(
            return_value=_make_origin_resp(cache_control="no-cache")
        )

        async def body_stream():
            yield b"{}"

        req = Request(
            method="GET",
            url="https://example.com/api",
            headers={},
            stream=body_stream(),
            metadata={},
        )
        await cached._proxy.handle_request(req)

        _req, stored_response, _key = mock_storage.create_entry.call_args.args
        cc = stored_response.headers.get("Cache-Control") or ""
        assert "max-age=7200" in cc

    @pytest.mark.asyncio
    async def test_force_cache_falls_back_to_default_86400(
        self, mock_storage: AsyncMock
    ) -> None:
        """force_cache=True uses 86400s when no TTL available anywhere."""
        mock_session = AsyncMock()
        mock_session.headers = {}
        # storage has no default_ttl attribute
        del mock_storage.default_ttl
        cached = CachedAiohttpSession(
            storage=mock_storage, session=mock_session, force_cache=True
        )
        mock_session.request = AsyncMock(
            return_value=_make_origin_resp(cache_control="no-cache")
        )

        async def body_stream():
            yield b"{}"

        req = Request(
            method="GET",
            url="https://example.com/api",
            headers={},
            stream=body_stream(),
            metadata={},
        )
        await cached._proxy.handle_request(req)

        _req, stored_response, _key = mock_storage.create_entry.call_args.args
        cc = stored_response.headers.get("Cache-Control") or ""
        assert "max-age=86400" in cc

    @pytest.mark.asyncio
    async def test_force_cache_error_response_gets_no_store(
        self, mock_storage: AsyncMock
    ) -> None:
        """4xx/5xx responses must get Cache-Control: no-store even with force_cache."""
        mock_session = AsyncMock()
        mock_session.headers = {}
        cached = CachedAiohttpSession(
            storage=mock_storage, session=mock_session, force_cache=True
        )
        mock_session.request = AsyncMock(
            return_value=_make_origin_resp(b"rate limited", status=429, cache_control="no-cache")
        )

        async def body_stream():
            yield b"{}"

        req = Request(
            method="GET",
            url="https://example.com/api",
            headers={},
            stream=body_stream(),
            metadata={},
        )
        hishel_resp = await cached._proxy.handle_request(req)
        assert hishel_resp.status_code == 429

        _req, stored_response, _key = mock_storage.create_entry.call_args.args
        cc = stored_response.headers.get("Cache-Control") or ""
        assert "no-store" in cc

    @pytest.mark.asyncio
    async def test_request_sender_uses_getall_for_multi_value_headers(
        self, mock_storage: AsyncMock
    ) -> None:
        """When response.headers has getall(), multi-values are joined with ', '."""
        mock_session = AsyncMock()
        mock_session.headers = {}
        cached = CachedAiohttpSession(storage=mock_storage, session=mock_session)

        origin_resp = MagicMock()
        origin_resp.status = 200
        # Simulate a multi-value header dict (e.g. CIMultiDict)
        origin_resp.headers = MagicMock()
        origin_resp.headers.keys.return_value = ["X-Custom"]
        origin_resp.headers.getall = MagicMock(return_value=["val1", "val2"])
        origin_resp.read = AsyncMock(return_value=b"{}")
        origin_resp.release = AsyncMock()
        mock_session.request = AsyncMock(return_value=origin_resp)

        async def body_stream():
            yield b"{}"

        req = Request(
            method="GET",
            url="https://example.com/api",
            headers={},
            stream=body_stream(),
            metadata={},
        )
        hishel_resp = await cached._proxy.handle_request(req)
        assert hishel_resp.status_code == 200

        _req, stored_response, _key = mock_storage.create_entry.call_args.args
        assert stored_response.headers.get("X-Custom") == "val1, val2"


# =============================================================================
# Test Class: CachedAiohttpSession — lifecycle (close, async context)
# =============================================================================


class TestCachedAiohttpSessionLifecycle:
    @pytest.mark.asyncio
    async def test_close_closes_session_and_storage(
        self, mock_storage: AsyncMock
    ) -> None:
        mock_session = AsyncMock()
        mock_session.headers = {}
        cached = CachedAiohttpSession(storage=mock_storage, session=mock_session)
        await cached.close()
        mock_session.close.assert_called_once()
        mock_storage.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_storage: AsyncMock) -> None:
        mock_session = AsyncMock()
        mock_session.headers = {}
        async with CachedAiohttpSession(
            storage=mock_storage, session=mock_session
        ) as cached:
            assert isinstance(cached, CachedAiohttpSession)
        mock_session.close.assert_called_once()
        mock_storage.close.assert_called_once()


# =============================================================================
# Test Class: CachedAiohttpSession — cache key isolation
# =============================================================================


class TestCachedAiohttpSessionCacheKeyIsolation:
    """Each URL and each unique POST body must produce a distinct cache entry.

    Regression: setting policy.use_body_key=True globally causes all GET
    requests (no body) to hash to SHA256(b"") — the same key — collapsing
    every URL into one cache slot so the cache never hits.
    """

    @pytest.mark.asyncio
    async def test_get_requests_to_different_urls_use_different_keys(
        self, mock_storage: AsyncMock
    ) -> None:
        """Two distinct GET URLs must be stored under different cache keys."""
        mock_session = AsyncMock()
        mock_session.headers = {}
        cached = CachedAiohttpSession(storage=mock_storage, session=mock_session)

        mock_session.request = AsyncMock(
            side_effect=[
                _make_origin_resp(b'{"url": "a"}'),
                _make_origin_resp(b'{"url": "b"}'),
            ]
        )

        async with cached.get("https://example.com/api/a") as resp_a:
            await resp_a.read()
        async with cached.get("https://example.com/api/b") as resp_b:
            await resp_b.read()

        assert mock_storage.create_entry.call_count == 2, (
            "Both URLs must be stored as separate cache entries"
        )
        key_a = mock_storage.create_entry.call_args_list[0].args[2]
        key_b = mock_storage.create_entry.call_args_list[1].args[2]
        assert key_a != key_b, (
            f"Different URLs must produce different cache keys, got same: {key_a!r}"
        )

    @pytest.mark.asyncio
    async def test_post_requests_with_different_bodies_use_different_keys(
        self, mock_storage: AsyncMock
    ) -> None:
        """Different GraphQL query bodies sent to the same POST URL must produce
        separate cache entries via the per-request X-Hishel-Body-Key header."""
        mock_session = AsyncMock()
        mock_session.headers = {"X-Hishel-Body-Key": "true"}
        cached = CachedAiohttpSession(storage=mock_storage, session=mock_session)

        mock_session.request = AsyncMock(
            side_effect=[
                _make_origin_resp(b'{"data": {"anime": {"id": 1}}}'),
                _make_origin_resp(b'{"data": {"anime": {"id": 2}}}'),
            ]
        )

        url = "https://graphql.anilist.co"
        async with cached.post(url, json={"query": "{ anime(id: 1) { title } }"}) as _:
            pass
        async with cached.post(url, json={"query": "{ anime(id: 2) { title } }"}) as _:
            pass

        assert mock_storage.create_entry.call_count == 2, (
            "Different POST bodies to same URL must be stored as separate entries"
        )
        key_1 = mock_storage.create_entry.call_args_list[0].args[2]
        key_2 = mock_storage.create_entry.call_args_list[1].args[2]
        assert key_1 != key_2, "Different POST bodies must produce different cache keys"


# =============================================================================
# Test Class: ReusableBodyStream — __anext__ no-op stub
# =============================================================================


class TestReusableBodyStream:
    @pytest.mark.asyncio
    async def test_anext_raises_stop_async_iteration(self) -> None:
        """__anext__ is a protocol stub — calling it directly raises StopAsyncIteration."""
        from http_cache.aiohttp_adapter import ReusableBodyStream

        stream = ReusableBodyStream(b"hello")
        with pytest.raises(StopAsyncIteration):
            await stream.__anext__()

    @pytest.mark.asyncio
    async def test_aiter_yields_body_on_each_iteration(self) -> None:
        """__aiter__ returns a fresh generator each time — stream can be read twice."""
        from http_cache.aiohttp_adapter import ReusableBodyStream

        stream = ReusableBodyStream(b"hello")
        chunks_1 = [chunk async for chunk in stream]
        chunks_2 = [chunk async for chunk in stream]
        assert chunks_1 == [b"hello"]
        assert chunks_2 == [b"hello"]


# =============================================================================
# Remaining coverage: text() encoding, headers TypeError, sync stream path
# =============================================================================


class TestCachedResponseTextEncoding:
    @pytest.mark.asyncio
    async def test_text_with_custom_encoding(self) -> None:
        """text(encoding=...) decodes body with the specified codec."""
        body = "héllo".encode("latin-1")
        resp = _CachedResponse(
            status=200,
            headers={},
            body=body,
            url="https://example.com/",
            method="GET",
            request_headers={},
        )
        assert await resp.text(encoding="latin-1") == "héllo"


class TestSessionHeadersAttributeErrorFallback:
    def test_attribute_error_in_session_headers_falls_back_to_empty(
        self, mock_storage: AsyncMock
    ) -> None:
        """dict(session.headers) raising AttributeError defaults to {}."""
        mock_session = MagicMock()
        # Make dict() on the headers object raise AttributeError
        bad_headers = MagicMock()
        bad_headers.keys = MagicMock(side_effect=AttributeError("no keys"))
        mock_session.headers = bad_headers
        cached = CachedAiohttpSession(storage=mock_storage, session=mock_session)
        assert cached._session_default_headers == {}


class TestSyncStreamFallback:
    @pytest.mark.asyncio
    async def test_sync_iterable_stream_is_consumed(self, mock_storage: AsyncMock) -> None:
        """When hishel_response.stream has no __aiter__, fall back to list()."""
        mock_session = AsyncMock()
        mock_session.headers = {}
        cached = CachedAiohttpSession(storage=mock_storage, session=mock_session)

        # iter() produces a sync iterator — no __aiter__ attribute
        sync_stream = iter([b"chunk1", b"chunk2"])
        assert not hasattr(sync_stream, "__aiter__")

        hishel_resp = MagicMock()
        hishel_resp.status_code = 200
        hishel_resp.headers = {}
        hishel_resp.stream = sync_stream
        hishel_resp.metadata = {"hishel_from_cache": False}

        cached._proxy.handle_request = AsyncMock(return_value=hishel_resp)

        resp = await cached._request("GET", "https://example.com/api")
        assert resp.status == 200
        assert resp._body == b"chunk1chunk2"
