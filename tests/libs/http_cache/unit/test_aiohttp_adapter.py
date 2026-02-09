"""
Unit tests for http_cache.aiohttp_adapter.

Focus:
- Request/header shaping for Hishel integration
- Session default header merging (for GraphQL body-key)
- URL query param inclusion (cache key + origin request parity)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from hishel._core.models import Request, Response
from http_cache.aiohttp_adapter import CachedAiohttpSession, _CachedResponse
from multidict import CIMultiDictProxy  # pants: no-infer-dep
from yarl import URL  # pants: no-infer-dep


class TestCachedResponse:
    @pytest.mark.asyncio
    async def test_cached_response_minimal_interface(self) -> None:
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


class TestCachedAiohttpSessionRequestBuilding:
    @pytest.mark.asyncio
    async def test_request_merges_session_default_headers_and_strips_x_hishel(
        self, mock_storage: AsyncMock
    ) -> None:
        # Session has default body-key header; per-request provides content headers.
        mock_session = AsyncMock()
        mock_session.headers = {"X-Hishel-Body-Key": "true", "User-Agent": "ua"}

        cached = CachedAiohttpSession(storage=mock_storage, session=mock_session)

        # Intercept what we send to Hishel.
        def handle(req: Request) -> Response:
            assert req.headers.get("User-Agent") == "ua"
            # Hishel control header should be stripped from req.headers and moved to metadata.
            assert "X-Hishel-Body-Key" not in dict(req.headers)
            assert req.metadata.get("hishel_body_key") is True

            async def stream():
                yield b"{}"

            return Response(status_code=200, stream=stream(), metadata={"hishel_from_cache": False})

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
    async def test_request_includes_params_in_url(self, mock_storage: AsyncMock) -> None:
        mock_session = AsyncMock()
        mock_session.headers = {}
        cached = CachedAiohttpSession(storage=mock_storage, session=mock_session)

        def handle(req: Request) -> Response:
            assert str(req.url) == "https://example.com/api?q=one+piece&page=2"

            async def stream():
                yield b"ok"

            return Response(status_code=200, stream=stream(), metadata={"hishel_from_cache": True})

        cached._proxy.handle_request = AsyncMock(side_effect=handle)

        resp = await cached._request(
            "GET",
            "https://example.com/api",
            params={"q": "one piece", "page": 2},
        )
        assert resp.url == URL("https://example.com/api?q=one+piece&page=2")
        assert resp.from_cache is True


class TestCachedAiohttpSessionRequestSender:
    @pytest.mark.asyncio
    async def test_request_sender_does_not_forward_x_hishel_headers_upstream(
        self, mock_storage: AsyncMock
    ) -> None:
        """
        The internal request_sender closure should not leak X-Hishel-* headers to origin.
        """
        mock_session = AsyncMock()
        mock_session.headers = {}

        cached = CachedAiohttpSession(storage=mock_storage, session=mock_session)

        # Make aiohttp request return a simple response.
        origin_resp = MagicMock()
        origin_resp.status = 200
        origin_resp.headers = {"Content-Type": "text/plain"}
        origin_resp.read = AsyncMock(return_value=b"hello")
        origin_resp.close = MagicMock()
        mock_session.request = AsyncMock(return_value=origin_resp)

        async def body_stream():
            yield b"{}"

        req = Request(
            method="POST",
            url="https://example.com/graphql",
            headers={"X-Hishel-Body-Key": "true", "Content-Type": "application/json"},
            stream=body_stream(),
            metadata={"hishel_body_key": True},
        )

        hishel_resp = await cached._proxy.send_request(req)
        assert hishel_resp.status_code == 200

        # Verify upstream request did not receive X-Hishel-* header.
        _, call_kwargs = mock_session.request.call_args
        assert call_kwargs["headers"]["Content-Type"] == "application/json"
        assert "X-Hishel-Body-Key" not in call_kwargs["headers"]
