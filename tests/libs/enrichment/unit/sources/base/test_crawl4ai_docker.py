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
def clear_cf_cookie_cache() -> None:
    _docker_mod._CF_COOKIE_CACHE.clear()


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


async def test_submit_job_200_returns_task_id() -> None:
    session = _post_session(200, {"task_id": "abc123"})
    assert await _submit_job(session, "http://x", [URL], _BC, _CC) == "abc123"


async def test_submit_job_202_returns_task_id() -> None:
    session = _post_session(202, {"task_id": "abc202"})
    assert await _submit_job(session, "http://x", [URL], _BC, _CC) == "abc202"


async def test_submit_job_4xx_returns_none() -> None:
    session = _post_session(400, text_data="bad request")
    assert await _submit_job(session, "http://x", [URL], _BC, _CC) is None


async def test_submit_job_5xx_raises() -> None:
    session = _post_session(503, text_data="server error")
    with pytest.raises(RuntimeError, match="crawl4ai server error"):
        await _submit_job(session, "http://x", [URL], _BC, _CC)


async def test_submit_job_missing_task_id_returns_none() -> None:
    session = _post_session(200, {"task_id": None})
    assert await _submit_job(session, "http://x", [URL], _BC, _CC) is None


async def test_submit_job_client_error_returns_none() -> None:
    session = AsyncMock()
    session.post = MagicMock(side_effect=aiohttp.ClientError("unreachable"))
    assert await _submit_job(session, "http://x", [URL], _BC, _CC) is None


# ---------------------------------------------------------------------------
# _poll_job
# ---------------------------------------------------------------------------


async def test_poll_job_completed_returns_data() -> None:
    data = {"status": "completed", "result": {}}
    session = _get_session((200, data))
    assert (
        await _poll_job(session, "http://x", "t1", timeout=10.0, poll_interval=0)
        == data
    )


async def test_poll_job_failed_returns_none() -> None:
    session = _get_session((200, {"status": "failed", "error": "oops"}))
    assert (
        await _poll_job(session, "http://x", "t1", timeout=10.0, poll_interval=0)
        is None
    )


async def test_poll_job_non_200_returns_none() -> None:
    session = _get_session((404, {}))
    assert (
        await _poll_job(session, "http://x", "t1", timeout=10.0, poll_interval=0)
        is None
    )


async def test_poll_job_client_error_returns_none() -> None:
    session = AsyncMock()
    session.get = MagicMock(side_effect=aiohttp.ClientError("fail"))
    assert (
        await _poll_job(session, "http://x", "t1", timeout=10.0, poll_interval=0)
        is None
    )


async def test_poll_job_timeout_returns_none() -> None:
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


def test_align_results_missing_url_returns_none() -> None:
    assert _align_results([URL], []) == [None]


def test_align_results_failure_error_message_returns_none() -> None:
    assert _align_results(
        [URL], [{"url": URL, "success": False, "error_message": "boom"}]
    ) == [None]


def test_align_results_failure_error_field_returns_none() -> None:
    assert _align_results([URL], [{"url": URL, "success": False, "error": "boom"}]) == [
        None
    ]


def test_align_results_404_returns_none() -> None:
    assert _align_results(
        [URL], [{"url": URL, "success": True, "status_code": 404}]
    ) == [None]


def test_align_results_405_returns_none() -> None:
    assert _align_results(
        [URL], [{"url": URL, "success": True, "status_code": 405}]
    ) == [None]


def test_align_results_reordered() -> None:
    url_a, url_b = "https://a.com", "https://b.com"
    raw = [
        {"url": url_b, "success": True, "status_code": 200},
        {"url": url_a, "success": True, "status_code": 200},
    ]
    aligned = _align_results([url_a, url_b], raw)
    assert aligned[0]["url"] == url_a
    assert aligned[1]["url"] == url_b


def test_align_results_unicode_matches_percent_encoded_result() -> None:
    """Playwright percent-encodes URLs; submitted Unicode must still match."""
    unicode_url = "https://myanimelist.net/character/270864/Broyé_Charlotte"
    encoded_url = "https://myanimelist.net/character/270864/Broy%C3%A9_Charlotte"
    entry = {"url": encoded_url, "success": True, "status_code": 200}
    result = _align_results([unicode_url], [entry])
    assert result == [entry]


def test_align_results_percent_encoded_matches_unicode_result() -> None:
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


def test_extract_transient_dns_error() -> None:
    raw = [
        {
            "url": URL,
            "success": False,
            "error_message": "ERR_NAME_NOT_RESOLVED at https://...",
        }
    ]
    assert _extract_transient_failed_urls(raw) == [URL]


def test_extract_transient_browser_closed() -> None:
    raw = [
        {
            "url": URL,
            "success": False,
            "error_message": "Target page, context or browser has been closed",
        }
    ]
    assert _extract_transient_failed_urls(raw) == [URL]


def test_extract_transient_error_field_fallback() -> None:
    raw = [{"url": URL, "success": False, "error": "ERR_NAME_NOT_RESOLVED"}]
    assert _extract_transient_failed_urls(raw) == [URL]


def test_extract_transient_ignores_success() -> None:
    raw = [{"url": URL, "success": True, "error_message": "ERR_NAME_NOT_RESOLVED"}]
    assert _extract_transient_failed_urls(raw) == []


def test_extract_transient_ignores_missing_url() -> None:
    raw = [{"success": False, "error_message": "ERR_NAME_NOT_RESOLVED"}]
    assert _extract_transient_failed_urls(raw) == []


def test_extract_transient_page_timeout() -> None:
    raw = [
        {
            "url": URL,
            "success": False,
            "error_message": "Failed on navigating ACS-GOTO:\nPage.goto: Timeout 90000ms exceeded.",
        }
    ]
    assert _extract_transient_failed_urls(raw) == [URL]


def test_extract_transient_ignores_unknown_error() -> None:
    raw = [{"url": URL, "success": False, "error_message": "some unknown error"}]
    assert _extract_transient_failed_urls(raw) == []


# ---------------------------------------------------------------------------
# _retry_failed_urls
# ---------------------------------------------------------------------------


async def test_retry_failed_urls_submit_fails_returns_aligned() -> None:
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


async def test_retry_failed_urls_poll_fails_returns_aligned() -> None:
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


async def test_retry_failed_urls_patches_aligned() -> None:
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


async def test_retry_failed_urls_returns_waf_blocked_from_retry() -> None:
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


async def test_probe_waf_recovery_no_task_id() -> None:
    with patch(
        "enrichment.sources.base.crawl4ai_docker._submit_job",
        new_callable=AsyncMock,
        return_value=None,
    ):
        assert (
            await _probe_waf_recovery(AsyncMock(), "http://x", URL, _BC, _CC) is False
        )


async def test_probe_waf_recovery_no_response() -> None:
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
        assert (
            await _probe_waf_recovery(AsyncMock(), "http://x", URL, _BC, _CC) is False
        )


async def test_probe_waf_recovery_empty_results() -> None:
    with (
        patch(
            "enrichment.sources.base.crawl4ai_docker._submit_job",
            new_callable=AsyncMock,
            return_value="tid",
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker._poll_job",
            new_callable=AsyncMock,
            return_value={"result": {"results": []}},
        ),
    ):
        assert (
            await _probe_waf_recovery(AsyncMock(), "http://x", URL, _BC, _CC) is False
        )


async def test_probe_waf_recovery_405_returns_false() -> None:
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


async def test_probe_waf_recovery_200_returns_true() -> None:
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


async def test_wait_for_waf_unblock_timeout_returns_false() -> None:
    with patch("enrichment.sources.base.crawl4ai_docker._WAF_MAX_WAIT", -1.0):
        assert (
            await _wait_for_waf_unblock(AsyncMock(), "http://x", URL, _BC, _CC) is False
        )


async def test_wait_for_waf_unblock_first_probe_succeeds() -> None:
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


async def test_wait_for_waf_unblock_second_probe_succeeds() -> None:
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


def test_inject_cookies_merges_into_empty_params() -> None:
    bc = {"type": "BrowserConfig", "params": {}}
    cookies = [{"name": "cf_clearance", "value": "abc", "domain": ".example.com", "path": "/"}]
    result = _inject_cookies(bc, cookies)
    assert result["params"]["cookies"] == cookies


def test_inject_cookies_merges_with_existing_cookies() -> None:
    existing = [{"name": "session", "value": "xyz"}]
    bc = {"type": "BrowserConfig", "params": {"cookies": existing}}
    new_cookies = [{"name": "cf_clearance", "value": "abc", "domain": ".example.com", "path": "/"}]
    result = _inject_cookies(bc, new_cookies)
    assert result["params"]["cookies"] == existing + new_cookies


def test_inject_cookies_does_not_mutate_original() -> None:
    bc = {"type": "BrowserConfig", "params": {"headless": True}}
    _inject_cookies(bc, [{"name": "cf_clearance", "value": "x"}])
    assert "cookies" not in bc["params"]


# ---------------------------------------------------------------------------
# _bypass_waf_with_zendriver
# ---------------------------------------------------------------------------


async def test_bypass_waf_with_zendriver_import_error_returns_none() -> None:
    import sys
    with patch.dict(sys.modules, {"zendriver": None}):
        result = await _bypass_waf_with_zendriver(URL)
    assert result is None


async def test_bypass_waf_with_zendriver_exception_returns_none() -> None:
    with patch("zendriver.start", side_effect=RuntimeError("browser crash")):
        result = await _bypass_waf_with_zendriver(URL)
    assert result is None


async def test_bypass_waf_with_zendriver_no_cf_cookie_returns_none() -> None:
    mock_page = AsyncMock()
    mock_page.get_content.return_value = "<html>clean</html>"
    mock_page.send = AsyncMock(return_value=[])

    mock_browser = AsyncMock()
    mock_browser.get.return_value = mock_page
    mock_browser.__aenter__ = AsyncMock(return_value=mock_browser)
    mock_browser.__aexit__ = AsyncMock(return_value=None)

    with (
        patch("zendriver.start", new_callable=AsyncMock, return_value=mock_browser),
        patch("enrichment.sources.base.crawl4ai_docker.asyncio.sleep", new_callable=AsyncMock),
    ):
        result = await _bypass_waf_with_zendriver(URL)
    assert result is None


async def test_bypass_waf_with_zendriver_returns_cf_clearance_cookie() -> None:
    mock_cookie = MagicMock()
    mock_cookie.name = "cf_clearance"
    mock_cookie.value = "token123"
    mock_cookie.domain = ".myanimelist.net"
    mock_cookie.path = "/"

    other_cookie = MagicMock()
    other_cookie.name = "session"
    other_cookie.value = "sess456"

    mock_page = AsyncMock()
    mock_page.get_content.return_value = "<html>clean</html>"
    mock_page.send = AsyncMock(return_value=[mock_cookie, other_cookie])

    mock_browser = AsyncMock()
    mock_browser.get.return_value = mock_page
    mock_browser.__aenter__ = AsyncMock(return_value=mock_browser)
    mock_browser.__aexit__ = AsyncMock(return_value=None)

    with (
        patch("zendriver.start", new_callable=AsyncMock, return_value=mock_browser),
        patch("enrichment.sources.base.crawl4ai_docker.asyncio.sleep", new_callable=AsyncMock),
    ):
        result = await _bypass_waf_with_zendriver(URL)

    assert result == [{"name": "cf_clearance", "value": "token123", "domain": ".myanimelist.net", "path": "/"}]


async def test_bypass_waf_with_zendriver_still_blocked_returns_none() -> None:
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


async def test_crawl_single_url_submit_fails_returns_none() -> None:
    with (
        patch("enrichment.sources.base.crawl4ai_docker.aiohttp.ClientSession"),
        patch(
            "enrichment.sources.base.crawl4ai_docker._submit_job",
            new_callable=AsyncMock,
            return_value=None,
        ),
    ):
        assert await crawl_single_url(URL, _BC, _CC) is None


async def test_crawl_single_url_poll_fails_returns_none() -> None:
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


async def test_crawl_single_url_success() -> None:
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


async def test_crawl_batch_urls_empty_returns_empty() -> None:
    assert await crawl_batch_urls([], _BC, _CC) == []


async def test_crawl_batch_urls_submit_fails_returns_nones() -> None:
    with (
        patch("enrichment.sources.base.crawl4ai_docker.aiohttp.ClientSession"),
        patch(
            "enrichment.sources.base.crawl4ai_docker._submit_job",
            new_callable=AsyncMock,
            return_value=None,
        ),
    ):
        assert await crawl_batch_urls([URL], _BC, _CC) == [None]


async def test_crawl_batch_urls_poll_fails_returns_nones() -> None:
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


async def test_crawl_batch_urls_clean_success() -> None:
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


async def test_crawl_batch_urls_transient_retry_succeeds() -> None:
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


async def test_crawl_batch_transient_retry_succeeds_on_third_attempt() -> None:
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


async def test_crawl_batch_urls_transient_all_retries_exhausted() -> None:
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


async def test_crawl_batch_urls_waf_blocked_recovered() -> None:
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


async def test_crawl_batch_urls_waf_blocked_not_recovered() -> None:
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


async def test_crawl_batch_urls_waf_zendriver_succeeds_skips_passive() -> None:
    """Active CF bypass succeeds → retry with injected cookie; passive probe never called."""
    waf = {"url": URL, "success": True, "status_code": 403}
    recovered = {"url": URL, "success": True, "status_code": 200}
    cf_cookies = [{"name": "cf_clearance", "value": "tok", "domain": ".myanimelist.net", "path": "/"}]
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
            return_value=cf_cookies,
        ),
        patch(
            "enrichment.sources.base.crawl4ai_docker._wait_for_waf_unblock",
            new_callable=AsyncMock,
        ) as mock_passive,
    ):
        result = await crawl_batch_urls([URL], _BC, _CC)

    assert result == [recovered]
    mock_passive.assert_not_called()


async def test_crawl_batch_urls_waf_zendriver_fails_falls_back_recovered() -> None:
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


async def test_crawl_batch_urls_waf_zendriver_fails_falls_back_not_recovered() -> None:
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


async def test_crawl_batch_urls_waf_cookie_partial_recovery_then_passive_recovered() -> None:
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
                {"result": {"results": [ok1, waf2]}},    # cookie retry 1: URL1 recovered
                {"result": {"results": [waf2]}},          # cookie retry 2: URL2 still blocked
                {"result": {"results": [waf2]}},          # cookie retry 3: URL2 still blocked
                {"result": {"results": [ok2]}},           # passive retry: URL2 recovered
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


async def test_crawl_batch_urls_waf_cookie_zero_recovery_evicts_cache_then_passive() -> None:
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
                {"result": {"results": [waf]}},       # cookie retry 1: 0 recovered
                {"result": {"results": [waf]}},       # cookie retry 2: 0 recovered
                {"result": {"results": [waf]}},       # cookie retry 3: 0 recovered
                {"result": {"results": [recovered]}}, # passive retry: recovered
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


async def test_crawl_batch_transient_retry_hits_waf_triggers_recovery() -> None:
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


async def test_crawl_batch_transient_retry_hits_waf_recovery_fails() -> None:
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
