import asyncio
import http.server
import socket
import threading
import time
from collections.abc import Iterator

import pytest
from enrichment.sources.base.exceptions import ServiceBlockedError
from enrichment.sources.base.polite_http import FetchedPage, PoliteHttpClient

BROWSER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) Chrome/153.0.0.0 Safari/537.36"


class ScriptedSite(http.server.BaseHTTPRequestHandler):
    requests: list[tuple[str, str | None, float]] = []

    def do_GET(self) -> None:
        ScriptedSite.requests.append(
            (self.path, self.headers.get("User-Agent"), time.monotonic())
        )
        if self.path == "/old":
            self.send_response(301)
            self.send_header("Location", "/anime/1,cowboy-bebop")
            self.end_headers()
            return
        if self.path == "/slow":
            time.sleep(1)
        if self.path == "/dropped":
            self.close_connection = True
            return
        status = {
            "/missing": 404,
            "/locked": 423,
            "/forbidden": 403,
            "/limited": 429,
        }.get(self.path, 200)
        body = f"<html><body>{self.path}</body></html>".encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


@pytest.fixture
def site() -> Iterator[str]:
    ScriptedSite.requests = []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), ScriptedSite)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()
    thread.join()


@pytest.fixture
def closed_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _client(min_interval: float = 0.0, timeout: float = 5.0) -> PoliteHttpClient:
    return PoliteHttpClient(
        "TestSite",
        {"User-Agent": BROWSER_AGENT},
        min_interval=min_interval,
        timeout=timeout,
    )


async def test_fetch_redirected_page_returns_html_and_final_url(site: str) -> None:
    page = await _client().fetch(f"{site}/old")

    assert page == FetchedPage(
        url=f"{site}/anime/1,cowboy-bebop",
        html="<html><body>/anime/1,cowboy-bebop</body></html>",
    )


async def test_fetch_sends_configured_user_agent(site: str) -> None:
    await _client().fetch(f"{site}/page")

    assert [agent for _, agent, _ in ScriptedSite.requests] == [BROWSER_AGENT]


async def test_fetch_missing_page_returns_none_and_keeps_fetching(site: str) -> None:
    client = _client()

    assert await client.fetch(f"{site}/missing") is None
    assert await client.fetch(f"{site}/page") is not None
    assert not client.blocked


@pytest.mark.parametrize("path", ["/locked", "/forbidden", "/limited"])
async def test_fetch_block_status_raises_and_sends_nothing_afterwards(
    site: str, path: str, caplog: pytest.LogCaptureFixture
) -> None:
    client = _client()

    with pytest.raises(ServiceBlockedError, match=path):
        await client.fetch(f"{site}{path}")
    with pytest.raises(ServiceBlockedError):
        await client.fetch(f"{site}/page")

    assert client.blocked
    assert [requested for requested, _, _ in ScriptedSite.requests] == [path]
    assert "TestSite blocked this client" in caplog.text


async def test_fetch_connection_refused_raises_and_blocks(closed_port: int) -> None:
    client = _client()

    with pytest.raises(ServiceBlockedError, match="connection refused"):
        await client.fetch(f"http://127.0.0.1:{closed_port}/page")

    assert client.blocked


async def test_fetch_connection_dropped_returns_none_without_blocking(
    site: str,
) -> None:
    client = _client()

    assert await client.fetch(f"{site}/dropped") is None
    assert not client.blocked


async def test_fetch_no_answer_within_timeout_returns_none_without_blocking(
    site: str,
) -> None:
    client = _client(timeout=0.2)

    assert await client.fetch(f"{site}/slow") is None
    assert not client.blocked


async def test_fetch_concurrent_requests_start_min_interval_apart(site: str) -> None:
    client = _client(min_interval=0.2)

    await asyncio.gather(*(client.fetch(f"{site}/page{number}") for number in range(4)))

    starts = sorted(started for _, _, started in ScriptedSite.requests)
    assert len(starts) == 4
    assert min(later - earlier for earlier, later in zip(starts, starts[1:])) >= 0.19
