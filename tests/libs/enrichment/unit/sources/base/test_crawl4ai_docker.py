"""Unit tests for crawl4ai_docker.py — HTTP transport layer for the crawl4ai Docker REST server."""

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
import enrichment.sources.base.crawl4ai_docker as _docker_mod
from enrichment.sources.base.crawl4ai_docker import (
    _align_results,
    _bypass_waf_with_zendriver,
    _extract_transient_failed_urls,
    _extract_waf_blocked_urls,
    _get_base_url,
    _inject_cookies,
    _poll_job,
    _probe_waf_recovery,
    _retry_failed_urls,
    _submit_job,
    _wait_for_waf_unblock,
    crawl_batch_urls,
    crawl_single_url,
)

_BC = {"type": "BrowserConfig", "params": {}}
_CC = {"type": "CrawlerRunConfig", "params": {}}
URL = "https://myanimelist.net/anime/21/One_Piece/episode/1"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_cf_state() -> None:
    _docker_mod._CF_COOKIE_CACHE.clear()
    _docker_mod._CF_PASSIVE_DOMAINS.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cm(resp: AsyncMock) -> MagicMock:
    """Wrap a response mock in an async context manager."""
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=resp)
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm


def _post_session(
    status: int, json_data: dict | None = None, text_data: str = ""
) -> AsyncMock:
    """Build a mock session whose .post() context manager yields a response.

    .post must be a MagicMock (not AsyncMock): aiohttp calls it synchronously
    and uses the return value as an async context manager, not as a coroutine.
    """
    mock_resp = AsyncMock()
    mock_resp.status = status
    mock_resp.json.return_value = json_data or {}
    mock_resp.text.return_value = text_data
    session = AsyncMock()
    session.post = MagicMock(return_value=_cm(mock_resp))
    return session


def _get_session(*responses: tuple[int, dict]) -> AsyncMock:
    """Build a mock session whose .get() returns each response in sequence."""
    session = AsyncMock()
    resps = []
    for status, data in responses:
        r = AsyncMock()
        r.status = status
        r.json.return_value = data
        resps.append(_cm(r))
    session.get = MagicMock(side_effect=resps)
    return session


def _job_resp(url: str = URL) -> dict:
    """Minimal successful job response for a single URL."""
    return {"result": {"results": [{"url": url, "success": True, "status_code": 200}]}}


# ---------------------------------------------------------------------------
# _get_base_url
# ---------------------------------------------------------------------------


def test_get_base_url_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CRAWL4AI_DOCKER_URL", raising=False)
    assert _get_base_url() == "http://localhost:11235"


def test_get_base_url_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRAWL4AI_DOCKER_URL", "http://myserver:9000/")
    assert _get_base_url() == "http://myserver:9000"


# ---------------------------------------------------------------------------
# _submit_job
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status,task_id", [(200, "abc123"), (202, "abc202")])
async def test_submit_job_success(status: int, task_id: str) -> None:
    session = _post_session(status, {"task_id": task_id})
    assert await _submit_job(session, "http://x", [URL], _BC, _CC) == task_id


async def test_submit_job_4xx() -> None:
    session = _post_session(400, text_data="bad request")
    assert await _submit_job(session, "http://x", [URL], _BC, _CC) is None


async def test_submit_job_5xx_raises() -> None:
    session = _post_session(503, text_data="server error")
    with pytest.raises(RuntimeError, match="crawl4ai server error"):
        await _submit_job(session, "http://x", [URL], _BC, _CC)


async def test_submit_job_no_task_id() -> None:
    session = _post_session(200, {"task_id": None})
    assert await _submit_job(session, "http://x", [URL], _BC, _CC) is None


async def test_submit_job_client_error() -> None:
    session = AsyncMock()
    session.post = MagicMock(side_effect=aiohttp.ClientError("unreachable"))
    assert await _submit_job(session, "http://x", [URL], _BC, _CC) is None


# ---------------------------------------------------------------------------
# _poll_job
# ---------------------------------------------------------------------------


async def test_poll_job_completed() -> None:
    data = {"status": "completed", "result": {}}
    session = _get_session((200, data))
    assert (
        await _poll_job(session, "http://x", "t1", timeout=10.0, poll_interval=0)
        == data
    )


async def test_poll_job_failed() -> None:
    session = _get_session((200, {"status": "failed", "error": "oops"}))
    assert (
        await _poll_job(session, "http://x", "t1", timeout=10.0, poll_interval=0)
        is None
    )


async def test_poll_job_non_200() -> None:
    session = _get_session((404, {}))
    assert (
        await _poll_job(session, "http://x", "t1", timeout=10.0, poll_interval=0)
        is None
    )


async def test_poll_job_client_error() -> None:
    session = AsyncMock()
    session.get = MagicMock(side_effect=aiohttp.ClientError("fail"))
    assert (
        await _poll_job(session, "http://x", "t1", timeout=10.0, poll_interval=0)
        is None
    )


async def test_poll_job_timeout() -> None:
    assert (
        await _poll_job(AsyncMock(), "http://x", "t1", timeout=0, poll_interval=0)
        is None
    )


async def test_poll_job_pending_then_completed() -> None:
    completed = {"status": "completed", "result": {"ok": True}}
    session = _get_session(
        (200, {"status": "pending"}),
        (200, completed),
    )
    assert (
        await _poll_job(session, "http://x", "t1", timeout=10.0, poll_interval=0)
        == completed
    )


# ---------------------------------------------------------------------------
# _align_results
# ---------------------------------------------------------------------------


def test_align_results_success() -> None:
    entry = {"url": URL, "success": True, "status_code": 200}
    assert _align_results([URL], [entry]) == [entry]


def test_align_results_missing_url() -> None:
    assert _align_results([URL], []) == [None]


@pytest.mark.parametrize("entry", [
    {"url": URL, "success": False, "error_message": "boom"},
    {"url": URL, "success": False, "error": "boom"},
])
def test_align_results_failure(entry: dict) -> None:
    assert _align_results([URL], [entry]) == [None]


@pytest.mark.parametrize("code", [404, 405])
def test_align_results_waf_code(code: int) -> None:
    assert _align_results([URL], [{"url": URL, "success": True, "status_code": code}]) == [None]


def test_align_results_reordered() -> None:
    url_a, url_b = "https://a.com", "https://b.com"
    raw = [
        {"url": url_b, "success": True, "status_code": 200},
        {"url": url_a, "success": True, "status_code": 200},
    ]
    aligned = _align_results([url_a, url_b], raw)
    assert aligned[0]["url"] == url_a
    assert aligned[1]["url"] == url_b


def test_align_unicode_percent() -> None:
    """Playwright percent-encodes URLs; submitted Unicode must still match."""
    unicode_url = "https://myanimelist.net/character/270864/Broyé_Charlotte"
    encoded_url = "https://myanimelist.net/character/270864/Broy%C3%A9_Charlotte"
    entry = {"url": encoded_url, "success": True, "status_code": 200}
    result = _align_results([unicode_url], [entry])
    assert result == [entry]


def test_align_percent_unicode() -> None:
    """Symmetric: percent-encoded submitted URL matches Unicode result URL."""
    unicode_url = "https://myanimelist.net/character/152902/Brûlée_Charlotte"
    encoded_url = "https://myanimelist.net/character/152902/Br%C3%BBl%C3%A9e_Charlotte"
    entry = {"url": unicode_url, "success": True, "status_code": 200}
    result = _align_results([encoded_url], [entry])
    assert result == [entry]


# ---------------------------------------------------------------------------
# _extract_waf_blocked_urls
# ---------------------------------------------------------------------------


def test_extract_waf_blocked_urls() -> None:
    raw = [
        {"url": "https://a.com", "status_code": 405},
        {"url": "https://b.com", "status_code": 200},
        {"status_code": 405},  # no url — excluded
    ]
    assert _extract_waf_blocked_urls(raw) == ["https://a.com"]


# ---------------------------------------------------------------------------
# _extract_transient_failed_urls
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("error_msg,field", [
    ("ERR_NAME_NOT_RESOLVED at https://...", "error_message"),
    ("Target page, context or browser has been closed", "error_message"),
    ("Failed on navigating ACS-GOTO:\nPage.goto: Timeout 90000ms exceeded.", "error_message"),
    ("ERR_NAME_NOT_RESOLVED", "error"),
])
def test_extract_transient_recognized(error_msg: str, field: str) -> None:
    raw = [{"url": URL, "success": False, field: error_msg}]
    assert _extract_transient_failed_urls(raw) == [URL]


@pytest.mark.parametrize("entry", [
    {"url": URL, "success": True, "error_message": "ERR_NAME_NOT_RESOLVED"},
    {"success": False, "error_message": "ERR_NAME_NOT_RESOLVED"},
    {"url": URL, "success": False, "error_message": "some unknown error"},
])
def test_extract_transient_ignored(entry: dict) -> None:
    assert _extract_transient_failed_urls([entry]) == []


# ---------------------------------------------------------------------------
# _retry_failed_urls
# ---------------------------------------------------------------------------


async def test_retry_submit_fails() -> None:
    with patch(
        "enrichment.sources.base.crawl4ai_docker._submit_job",
        new_callable=AsyncMock,
        return_value=None,
    ):
        aligned, waf_blocked = await _retry_failed_urls(
            AsyncMock(), "http://x", [URL], [None], [URL], _BC, _CC, 10.0, 0.1
        )
    assert aligned == [None]
    assert waf_blocked == []


async def test_retry_poll_fails() -> None:
    with (
        patch(
            "enrichment.sources.base.crawl4ai_docker._submit_job",
            new_callable=AsyncMock,
            return_value="tid",
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker._poll_job",
            new_callable=AsyncMock,
            return_value=None,
        ),
    ):
        aligned, waf_blocked = await _retry_failed_urls(
            AsyncMock(), "http://x", [URL], [None], [URL], _BC, _CC, 10.0, 0.1
        )
    assert aligned == [None]
    assert waf_blocked == []


async def test_retry_patches_aligned() -> None:
    entry = {"url": URL, "success": True, "status_code": 200}
    with (
        patch(
            "enrichment.sources.base.crawl4ai_docker._submit_job",
            new_callable=AsyncMock,
            return_value="tid",
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker._poll_job",
            new_callable=AsyncMock,
            return_value={"result": {"results": [entry]}},
        ),
    ):
        aligned, waf_blocked = await _retry_failed_urls(
            AsyncMock(), "http://x", [URL], [None], [URL], _BC, _CC, 10.0, 0.1
        )
    assert aligned == [entry]
    assert waf_blocked == []


async def test_retry_returns_waf_blocked() -> None:
    """When a retry gets a 405, it must be returned in waf_blocked — not silently dropped."""
    waf_entry = {"url": URL, "success": True, "status_code": 405}
    with (
        patch(
            "enrichment.sources.base.crawl4ai_docker._submit_job",
            new_callable=AsyncMock,
            return_value="tid",
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker._poll_job",
            new_callable=AsyncMock,
            return_value={"result": {"results": [waf_entry]}},
        ),
    ):
        aligned, waf_blocked = await _retry_failed_urls(
            AsyncMock(), "http://x", [URL], [None], [URL], _BC, _CC, 10.0, 0.1
        )
    assert aligned == [None]
    assert waf_blocked == [URL]


# ---------------------------------------------------------------------------
# _probe_waf_recovery
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("submit_rv,poll_rv", [
    (None, None),
    ("tid", None),
    ("tid", {"result": {"results": []}}),
])
async def test_probe_waf_recovery_false(submit_rv: str | None, poll_rv: dict | None) -> None:
    with (
        patch("enrichment.sources.base.crawl4ai_docker._submit_job", new_callable=AsyncMock, return_value=submit_rv),
        patch("enrichment.sources.base.crawl4ai_docker._poll_job", new_callable=AsyncMock, return_value=poll_rv),
    ):
        assert await _probe_waf_recovery(AsyncMock(), "http://x", URL, _BC, _CC) is False


async def test_probe_waf_recovery_405() -> None:
    with (
        patch(
            "enrichment.sources.base.crawl4ai_docker._submit_job",
            new_callable=AsyncMock,
            return_value="tid",
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker._poll_job",
            new_callable=AsyncMock,
            return_value={"result": {"results": [{"status_code": 405}]}},
        ),
    ):
        assert (
            await _probe_waf_recovery(AsyncMock(), "http://x", URL, _BC, _CC) is False
        )


async def test_probe_waf_recovery_200() -> None:
    with (
        patch(
            "enrichment.sources.base.crawl4ai_docker._submit_job",
            new_callable=AsyncMock,
            return_value="tid",
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker._poll_job",
            new_callable=AsyncMock,
            return_value={"result": {"results": [{"status_code": 200}]}},
        ),
    ):
        assert await _probe_waf_recovery(AsyncMock(), "http://x", URL, _BC, _CC) is True


# ---------------------------------------------------------------------------
# _wait_for_waf_unblock
# ---------------------------------------------------------------------------


async def test_waf_unblock_timeout() -> None:
    with patch("enrichment.sources.base.crawl4ai_docker._WAF_MAX_WAIT", -1.0):
        assert (
            await _wait_for_waf_unblock(AsyncMock(), "http://x", URL, _BC, _CC) is False
        )


async def test_waf_unblock_first_probe() -> None:
    with (
        patch("enrichment.sources.base.crawl4ai_docker._WAF_PROBE_INTERVAL", 0.0),
        patch(
            "enrichment.sources.base.crawl4ai_docker._probe_waf_recovery",
            new_callable=AsyncMock,
            return_value=True,
        ),
    ):
        assert (
            await _wait_for_waf_unblock(AsyncMock(), "http://x", URL, _BC, _CC) is True
        )


async def test_waf_unblock_second_probe() -> None:
    with (
        patch("enrichment.sources.base.crawl4ai_docker._WAF_PROBE_INTERVAL", 0.0),
        patch(
            "enrichment.sources.base.crawl4ai_docker._probe_waf_recovery",
            new_callable=AsyncMock,
            side_effect=[False, True],
        ),
    ):
        assert (
            await _wait_for_waf_unblock(AsyncMock(), "http://x", URL, _BC, _CC) is True
        )


# ---------------------------------------------------------------------------
# _inject_cookies
# ---------------------------------------------------------------------------


def test_inject_cookies_empty_params() -> None:
    bc = {"type": "BrowserConfig", "params": {}}
    cookies = [{"name": "cf_clearance", "value": "abc", "domain": ".example.com", "path": "/"}]
    result = _inject_cookies(bc, cookies)
    assert result["params"]["cookies"] == cookies


def test_inject_cookies_existing() -> None:
    existing = [{"name": "session", "value": "xyz"}]
    bc = {"type": "BrowserConfig", "params": {"cookies": existing}}
    new_cookies = [{"name": "cf_clearance", "value": "abc", "domain": ".example.com", "path": "/"}]
    result = _inject_cookies(bc, new_cookies)
    assert result["params"]["cookies"] == existing + new_cookies


def test_inject_cookies_no_mutate() -> None:
    bc = {"type": "BrowserConfig", "params": {"headless": True}}
    _inject_cookies(bc, [{"name": "cf_clearance", "value": "x"}])
    assert "cookies" not in bc["params"]


# ---------------------------------------------------------------------------
# _bypass_waf_with_zendriver
# ---------------------------------------------------------------------------


async def test_zendriver_import_error() -> None:
    import sys
    with patch.dict(sys.modules, {"zendriver": None}):
        result = await _bypass_waf_with_zendriver(URL)
    assert result is None


async def test_zendriver_exception() -> None:
    with patch("zendriver.start", side_effect=RuntimeError("browser crash")):
        result = await _bypass_waf_with_zendriver(URL)
    assert result is None


async def test_zendriver_no_cf_cookie() -> None:
    mock_page = AsyncMock()
    # CF marker on first call → enters challenge path; clean on second → exits loop
    mock_page.get_content.side_effect = ["Just a moment...", "<html>clean</html>"]
    mock_page.send = AsyncMock(return_value=[])  # no cf_clearance in cookies

    mock_browser = AsyncMock()
    mock_browser.get.return_value = mock_page
    mock_browser.stop = AsyncMock(side_effect=RuntimeError("stop failed"))  # covers finally except

    with (
        patch("zendriver.start", new_callable=AsyncMock, return_value=mock_browser),
        patch("enrichment.sources.base.crawl4ai_docker.asyncio.sleep", new_callable=AsyncMock),
        patch(
            "zendriver.core.cloudflare.cf_is_interactive_challenge_present",
            new_callable=AsyncMock,
            return_value=False,
        ),
    ):
        result = await _bypass_waf_with_zendriver(URL)
    assert result is None


async def test_zendriver_returns_cf_clearance() -> None:
    mock_cookie = MagicMock()
    mock_cookie.name = "cf_clearance"
    mock_cookie.value = "token123"
    mock_cookie.domain = ".myanimelist.net"
    mock_cookie.path = "/"

    other_cookie = MagicMock()
    other_cookie.name = "session"
    other_cookie.value = "sess456"

    mock_page = AsyncMock()
    # CF marker → interactive challenge present → verify_cf called (line 345)
    # wait loop: exception (lines 352-354), still blocked/sleep (line 357), then clean → break
    mock_page.get_content.side_effect = [
        "Just a moment...",          # initial check: CF present
        Exception("get_content err"),# loop iter 1: exception path
        "Just a moment...",          # loop iter 2: still blocked → sleep (line 357)
        "<html>clean</html>",        # loop iter 3: clean → break
    ]
    mock_page.send = AsyncMock(return_value=[mock_cookie, other_cookie])

    mock_browser = AsyncMock()
    mock_browser.get.return_value = mock_page

    with (
        patch("zendriver.start", new_callable=AsyncMock, return_value=mock_browser),
        patch("enrichment.sources.base.crawl4ai_docker.asyncio.sleep", new_callable=AsyncMock),
        patch(
            "zendriver.core.cloudflare.cf_is_interactive_challenge_present",
            new_callable=AsyncMock,
            return_value=True,  # interactive challenge → verify_cf called (line 345)
        ),
        patch("zendriver.core.cloudflare.verify_cf", new_callable=AsyncMock),
    ):
        result = await _bypass_waf_with_zendriver(URL)

    assert result == [{"name": "cf_clearance", "value": "token123", "domain": ".myanimelist.net", "path": "/"}]


async def test_zendriver_no_challenge_passive() -> None:
    """No CF markers on page → marked as passive domain, returns None."""
    mock_page = AsyncMock()
    mock_page.get_content.return_value = "<html>clean</html>"

    mock_browser = AsyncMock()
    mock_browser.get.return_value = mock_page

    with (
        patch("zendriver.start", new_callable=AsyncMock, return_value=mock_browser),
        patch("enrichment.sources.base.crawl4ai_docker.asyncio.sleep", new_callable=AsyncMock),
    ):
        result = await _bypass_waf_with_zendriver(URL)

    assert result is None
    assert "myanimelist.net" in _docker_mod._CF_PASSIVE_DOMAINS


async def test_zendriver_still_blocked() -> None:
    mock_page = AsyncMock()
    mock_page.get_content.return_value = "Just a moment..."

    mock_browser = AsyncMock()
    mock_browser.get.return_value = mock_page
    mock_browser.__aenter__ = AsyncMock(return_value=mock_browser)
    mock_browser.__aexit__ = AsyncMock(return_value=None)

    # cf_is_interactive_challenge_present is imported inside the function body,
    # so patch its attribute on the source module — not on crawl4ai_docker.
    with (
        patch("zendriver.start", new_callable=AsyncMock, return_value=mock_browser),
        patch("enrichment.sources.base.crawl4ai_docker.asyncio.sleep", new_callable=AsyncMock),
        patch(
            "zendriver.core.cloudflare.cf_is_interactive_challenge_present",
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch("enrichment.sources.base.crawl4ai_docker._ZENDRIVER_UNBLOCK_WAIT", -1.0),
    ):
        result = await _bypass_waf_with_zendriver(URL)
    assert result is None


# ---------------------------------------------------------------------------
# crawl_single_url
# ---------------------------------------------------------------------------


async def test_single_submit_fails() -> None:
    with (
        patch("enrichment.sources.base.crawl4ai_docker.aiohttp.ClientSession"),
        patch(
            "enrichment.sources.base.crawl4ai_docker._submit_job",
            new_callable=AsyncMock,
            return_value=None,
        ),
    ):
        assert await crawl_single_url(URL, _BC, _CC) is None


async def test_single_poll_fails() -> None:
    with (
        patch("enrichment.sources.base.crawl4ai_docker.aiohttp.ClientSession"),
        patch(
            "enrichment.sources.base.crawl4ai_docker._submit_job",
            new_callable=AsyncMock,
            return_value="tid",
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker._poll_job",
            new_callable=AsyncMock,
            return_value=None,
        ),
    ):
        assert await crawl_single_url(URL, _BC, _CC) is None


async def test_single_success() -> None:
    entry = {"url": URL, "success": True, "status_code": 200}
    with (
        patch("enrichment.sources.base.crawl4ai_docker.aiohttp.ClientSession"),
        patch(
            "enrichment.sources.base.crawl4ai_docker._submit_job",
            new_callable=AsyncMock,
            return_value="tid",
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker._poll_job",
            new_callable=AsyncMock,
            return_value={"result": {"results": [entry]}},
        ),
    ):
        assert await crawl_single_url(URL, _BC, _CC) == entry


# ---------------------------------------------------------------------------
# crawl_batch_urls
# ---------------------------------------------------------------------------


async def test_batch_empty() -> None:
    assert await crawl_batch_urls([], _BC, _CC) == []


async def test_batch_submit_fails() -> None:
    with (
        patch("enrichment.sources.base.crawl4ai_docker.aiohttp.ClientSession"),
        patch(
            "enrichment.sources.base.crawl4ai_docker._submit_job",
            new_callable=AsyncMock,
            return_value=None,
        ),
    ):
        assert await crawl_batch_urls([URL], _BC, _CC) == [None]


async def test_batch_poll_fails() -> None:
    with (
        patch("enrichment.sources.base.crawl4ai_docker.aiohttp.ClientSession"),
        patch(
            "enrichment.sources.base.crawl4ai_docker._submit_job",
            new_callable=AsyncMock,
            return_value="tid",
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker._poll_job",
            new_callable=AsyncMock,
            return_value=None,
        ),
    ):
        assert await crawl_batch_urls([URL], _BC, _CC) == [None]


async def test_batch_success() -> None:
    entry = {"url": URL, "success": True, "status_code": 200}
    with (
        patch("enrichment.sources.base.crawl4ai_docker.aiohttp.ClientSession"),
        patch(
            "enrichment.sources.base.crawl4ai_docker._submit_job",
            new_callable=AsyncMock,
            return_value="tid",
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker._poll_job",
            new_callable=AsyncMock,
            return_value={"result": {"results": [entry]}},
        ),
    ):
        assert await crawl_batch_urls([URL], _BC, _CC) == [entry]


async def test_batch_transient_retry() -> None:
    transient = {"url": URL, "success": False, "error_message": "ERR_NAME_NOT_RESOLVED"}
    recovered = {"url": URL, "success": True, "status_code": 200}
    with (
        patch("enrichment.sources.base.crawl4ai_docker.aiohttp.ClientSession"),
        patch(
            "enrichment.sources.base.crawl4ai_docker._submit_job",
            new_callable=AsyncMock,
            return_value="tid",
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker._poll_job",
            new_callable=AsyncMock,
            side_effect=[
                {"result": {"results": [transient]}},
                {"result": {"results": [recovered]}},
            ],
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker.asyncio.sleep",
            new_callable=AsyncMock,
        ),
    ):
        assert await crawl_batch_urls([URL], _BC, _CC) == [recovered]


async def test_batch_transient_third_attempt() -> None:
    transient = {"url": URL, "success": False, "error_message": "ERR_NAME_NOT_RESOLVED"}
    recovered = {"url": URL, "success": True, "status_code": 200}
    with (
        patch("enrichment.sources.base.crawl4ai_docker.aiohttp.ClientSession"),
        patch(
            "enrichment.sources.base.crawl4ai_docker._submit_job",
            new_callable=AsyncMock,
            return_value="tid",
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker._poll_job",
            new_callable=AsyncMock,
            side_effect=[
                {"result": {"results": [transient]}},  # original batch
                {"result": {"results": [transient]}},  # retry 1 — still failing
                {"result": {"results": [transient]}},  # retry 2 — still failing
                {"result": {"results": [recovered]}},  # retry 3 — recovered
            ],
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker.asyncio.sleep",
            new_callable=AsyncMock,
        ),
    ):
        assert await crawl_batch_urls([URL], _BC, _CC) == [recovered]


async def test_batch_transient_exhausted() -> None:
    transient = {"url": URL, "success": False, "error_message": "ERR_NAME_NOT_RESOLVED"}
    with (
        patch("enrichment.sources.base.crawl4ai_docker.aiohttp.ClientSession"),
        patch(
            "enrichment.sources.base.crawl4ai_docker._submit_job",
            new_callable=AsyncMock,
            return_value="tid",
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker._poll_job",
            new_callable=AsyncMock,
            side_effect=[
                {"result": {"results": [transient]}},  # original batch
                {"result": {"results": [transient]}},  # retry 1
                {"result": {"results": [transient]}},  # retry 2
                {"result": {"results": [transient]}},  # retry 3
            ],
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker.asyncio.sleep",
            new_callable=AsyncMock,
        ),
    ):
        assert await crawl_batch_urls([URL], _BC, _CC) == [None]


async def test_batch_waf_recovered() -> None:
    waf = {"url": URL, "success": True, "status_code": 405}
    recovered = {"url": URL, "success": True, "status_code": 200}
    with (
        patch("enrichment.sources.base.crawl4ai_docker.aiohttp.ClientSession"),
        patch(
            "enrichment.sources.base.crawl4ai_docker._submit_job",
            new_callable=AsyncMock,
            return_value="tid",
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker._poll_job",
            new_callable=AsyncMock,
            side_effect=[
                {"result": {"results": [waf]}},
                {"result": {"results": [recovered]}},
            ],
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker._bypass_waf_with_zendriver",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker._wait_for_waf_unblock",
            new_callable=AsyncMock,
            return_value=True,
        ),
    ):
        assert await crawl_batch_urls([URL], _BC, _CC) == [recovered]


async def test_batch_waf_not_recovered() -> None:
    # Pre-populate passive domain → skips zendriver entirely (covers line 561-562)
    _docker_mod._CF_PASSIVE_DOMAINS.add("myanimelist.net")
    waf = {"url": URL, "success": True, "status_code": 405}
    with (
        patch("enrichment.sources.base.crawl4ai_docker.aiohttp.ClientSession"),
        patch(
            "enrichment.sources.base.crawl4ai_docker._submit_job",
            new_callable=AsyncMock,
            return_value="tid",
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker._poll_job",
            new_callable=AsyncMock,
            return_value={"result": {"results": [waf]}},
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker._wait_for_waf_unblock",
            new_callable=AsyncMock,
            return_value=False,
        ),
    ):
        assert await crawl_batch_urls([URL], _BC, _CC) == [None]


async def test_batch_zendriver_skips_passive() -> None:
    """Cached cf_clearance reused (covers _CF_COOKIE_CACHE hit); passive probe never called."""
    waf = {"url": URL, "success": True, "status_code": 403}
    recovered = {"url": URL, "success": True, "status_code": 200}
    cf_cookies = [{"name": "cf_clearance", "value": "tok", "domain": ".myanimelist.net", "path": "/"}]
    # Pre-populate cache → hits line 564-565 instead of calling _bypass_waf_with_zendriver
    _docker_mod._CF_COOKIE_CACHE["myanimelist.net"] = cf_cookies
    with (
        patch("enrichment.sources.base.crawl4ai_docker.aiohttp.ClientSession"),
        patch(
            "enrichment.sources.base.crawl4ai_docker._submit_job",
            new_callable=AsyncMock,
            return_value="tid",
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker._poll_job",
            new_callable=AsyncMock,
            side_effect=[
                {"result": {"results": [waf]}},
                {"result": {"results": [recovered]}},
            ],
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker._wait_for_waf_unblock",
            new_callable=AsyncMock,
        ) as mock_passive,
        patch("enrichment.sources.base.crawl4ai_docker.asyncio.sleep", new_callable=AsyncMock),
    ):
        result = await crawl_batch_urls([URL], _BC, _CC)

    assert result == [recovered]
    mock_passive.assert_not_called()


async def test_batch_zendriver_fallback_recovered() -> None:
    """zendriver returns None → falls back to passive probe → URL recovered."""
    waf = {"url": URL, "success": True, "status_code": 403}
    recovered = {"url": URL, "success": True, "status_code": 200}
    with (
        patch("enrichment.sources.base.crawl4ai_docker.aiohttp.ClientSession"),
        patch(
            "enrichment.sources.base.crawl4ai_docker._submit_job",
            new_callable=AsyncMock,
            return_value="tid",
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker._poll_job",
            new_callable=AsyncMock,
            side_effect=[
                {"result": {"results": [waf]}},
                {"result": {"results": [recovered]}},
            ],
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker._bypass_waf_with_zendriver",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker._wait_for_waf_unblock",
            new_callable=AsyncMock,
            return_value=True,
        ),
    ):
        assert await crawl_batch_urls([URL], _BC, _CC) == [recovered]


async def test_batch_zendriver_fallback_failed() -> None:
    """zendriver returns None → passive probe also fails → None."""
    waf = {"url": URL, "success": True, "status_code": 403}
    with (
        patch("enrichment.sources.base.crawl4ai_docker.aiohttp.ClientSession"),
        patch(
            "enrichment.sources.base.crawl4ai_docker._submit_job",
            new_callable=AsyncMock,
            return_value="tid",
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker._poll_job",
            new_callable=AsyncMock,
            return_value={"result": {"results": [waf]}},
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker._bypass_waf_with_zendriver",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker._wait_for_waf_unblock",
            new_callable=AsyncMock,
            return_value=False,
        ),
    ):
        assert await crawl_batch_urls([URL], _BC, _CC) == [None]


async def test_batch_cookie_partial_passive() -> None:
    """Cookie retries recover some URLs; remaining go to passive probe and are recovered."""
    URL2 = "https://www.anime-planet.com/anime/one-piece/characters/2"
    waf1 = {"url": URL, "success": True, "status_code": 403}
    waf2 = {"url": URL2, "success": True, "status_code": 403}
    ok1 = {"url": URL, "success": True, "status_code": 200}
    ok2 = {"url": URL2, "success": True, "status_code": 200}
    cf_cookies = [{"name": "cf_clearance", "value": "tok", "domain": ".ap.net", "path": "/"}]
    with (
        patch("enrichment.sources.base.crawl4ai_docker.aiohttp.ClientSession"),
        patch(
            "enrichment.sources.base.crawl4ai_docker._submit_job",
            new_callable=AsyncMock,
            return_value="tid",
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker._poll_job",
            new_callable=AsyncMock,
            side_effect=[
                {"result": {"results": [waf1, waf2]}},  # initial batch: both blocked
                {"result": {"results": [ok1]}},          # cookie retry URL1: recovered
                {"result": {"results": [waf2]}},          # cookie retry URL2: still blocked
                {"result": {"results": [ok2]}},           # passive probe fallback: URL2 recovered
            ],
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker._bypass_waf_with_zendriver",
            new_callable=AsyncMock,
            return_value=cf_cookies,
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker._wait_for_waf_unblock",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_passive,
        patch(
            "enrichment.sources.base.crawl4ai_docker.asyncio.sleep",
            new_callable=AsyncMock,
        ),
    ):
        result = await crawl_batch_urls([URL, URL2], _BC, _CC)

    assert result == [ok1, ok2]
    mock_passive.assert_called_once()


async def test_batch_cookie_zero_evicts() -> None:
    """Cookie recovers 0 URLs across all retries → cache evicted → passive probe fallback."""
    waf = {"url": URL, "success": True, "status_code": 403}
    recovered = {"url": URL, "success": True, "status_code": 200}
    netloc = "myanimelist.net"
    cf_cookies = [{"name": "cf_clearance", "value": "tok", "domain": f".{netloc}", "path": "/"}]
    with (
        patch("enrichment.sources.base.crawl4ai_docker.aiohttp.ClientSession"),
        patch(
            "enrichment.sources.base.crawl4ai_docker._submit_job",
            new_callable=AsyncMock,
            return_value="tid",
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker._poll_job",
            new_callable=AsyncMock,
            side_effect=[
                {"result": {"results": [waf]}},       # initial batch
                {"result": {"results": [waf]}},       # cookie retry: 0 recovered, cache evicted
                {"result": {"results": [recovered]}}, # passive probe fallback: recovered
            ],
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker._bypass_waf_with_zendriver",
            new_callable=AsyncMock,
            return_value=cf_cookies,
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker._wait_for_waf_unblock",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_passive,
        patch(
            "enrichment.sources.base.crawl4ai_docker.asyncio.sleep",
            new_callable=AsyncMock,
        ),
    ):
        result = await crawl_batch_urls([URL], _BC, _CC)

    assert result == [recovered]
    assert netloc not in _docker_mod._CF_COOKIE_CACHE  # stale cookie was evicted
    mock_passive.assert_called_once()


async def test_batch_transient_hits_waf() -> None:
    """Gap scenario: transient failure → retry returns 405 → WAF recovery → URL recovered."""
    transient = {
        "url": URL,
        "success": False,
        "error_message": "Target page, context or browser has been closed",
    }
    recovered = {"url": URL, "success": True, "status_code": 200}
    with (
        patch("enrichment.sources.base.crawl4ai_docker.aiohttp.ClientSession"),
        patch(
            "enrichment.sources.base.crawl4ai_docker._submit_job",
            new_callable=AsyncMock,
            return_value="tid",
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker._poll_job",
            new_callable=AsyncMock,
            # original batch → transient failure; WAF retry → recovered
            side_effect=[
                {"result": {"results": [transient]}},
                {"result": {"results": [recovered]}},
            ],
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker._bypass_waf_with_zendriver",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker._wait_for_waf_unblock",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker.asyncio.sleep",
            new_callable=AsyncMock,
        ),
    ):
        assert await crawl_batch_urls([URL], _BC, _CC) == [recovered]


async def test_batch_waf_second_pass() -> None:
    """URL re-blocked during first sequential retry → second probe succeeds → recovered."""
    waf = {"url": URL, "success": True, "status_code": 403}
    reblocked = {"url": URL, "success": True, "status_code": 307}
    recovered = {"url": URL, "success": True, "status_code": 200}
    with (
        patch("enrichment.sources.base.crawl4ai_docker.aiohttp.ClientSession"),
        patch(
            "enrichment.sources.base.crawl4ai_docker._submit_job",
            new_callable=AsyncMock,
            return_value="tid",
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker._poll_job",
            new_callable=AsyncMock,
            side_effect=[
                {"result": {"results": [waf]}},        # initial batch: WAF blocked
                {"result": {"results": [reblocked]}},  # first sequential retry: re-blocked
                {"result": {"results": [recovered]}},  # second-pass retry: recovered
            ],
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker._bypass_waf_with_zendriver",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker._wait_for_waf_unblock",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker.asyncio.sleep",
            new_callable=AsyncMock,
        ),
    ):
        assert await crawl_batch_urls([URL], _BC, _CC) == [recovered]


async def test_batch_second_pass_fails() -> None:
    """URL re-blocked during first sequential retry → second probe fails → URL dropped."""
    waf = {"url": URL, "success": True, "status_code": 403}
    reblocked = {"url": URL, "success": True, "status_code": 307}
    with (
        patch("enrichment.sources.base.crawl4ai_docker.aiohttp.ClientSession"),
        patch(
            "enrichment.sources.base.crawl4ai_docker._submit_job",
            new_callable=AsyncMock,
            return_value="tid",
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker._poll_job",
            new_callable=AsyncMock,
            side_effect=[
                {"result": {"results": [waf]}},        # initial batch: WAF blocked
                {"result": {"results": [reblocked]}},  # first sequential retry: re-blocked
            ],
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker._bypass_waf_with_zendriver",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker._wait_for_waf_unblock",
            new_callable=AsyncMock,
            side_effect=[True, False],  # first probe clears, second probe fails
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker.asyncio.sleep",
            new_callable=AsyncMock,
        ),
    ):
        assert await crawl_batch_urls([URL], _BC, _CC) == [None]


async def test_batch_transient_waf_fails() -> None:
    """Gap scenario: transient failure → retry returns 405 → WAF recovery times out → None."""
    transient = {
        "url": URL,
        "success": False,
        "error_message": "Target page, context or browser has been closed",
    }
    waf = {"url": URL, "success": True, "status_code": 405}
    with (
        patch("enrichment.sources.base.crawl4ai_docker.aiohttp.ClientSession"),
        patch(
            "enrichment.sources.base.crawl4ai_docker._submit_job",
            new_callable=AsyncMock,
            return_value="tid",
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker._poll_job",
            new_callable=AsyncMock,
            side_effect=[
                {"result": {"results": [transient]}},
                {"result": {"results": [waf]}},
            ],
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker._bypass_waf_with_zendriver",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker._wait_for_waf_unblock",
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker.asyncio.sleep",
            new_callable=AsyncMock,
        ),
    ):
        assert await crawl_batch_urls([URL], _BC, _CC) == [None]
