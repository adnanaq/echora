"""Reusable HTTP transport layer for the crawl4ai Docker REST server.

All enrichment crawlers that have been migrated away from in-process Playwright
use this module instead of instantiating ``AsyncWebCrawler`` directly.  The
Docker server runs a shared browser pool which eliminates per-crawler process
overhead and provides a single rate-limiting point via ``MAX_CONCURRENT_TASKS``.

Public API
----------
``crawl_single_url`` — submit one URL, poll until done, return result dict.
``crawl_batch_urls``  — submit N URLs as one job, return list aligned to input.
                        Automatically retries WAF-blocked (403/405) URLs after
                        polling until the block lifts.

Both functions accept plain dicts for ``browser_config`` / ``crawler_config``
so callers never import crawl4ai Python types through this boundary.

Environment
-----------
``CRAWL4AI_DOCKER_URL`` — base URL of the Docker server (default: http://localhost:11235).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any
from urllib.parse import quote, unquote

import aiohttp

logger = logging.getLogger(__name__)

_DEFAULT_DOCKER_URL = "http://localhost:11235"

# How long to wait between WAF recovery probes, and the max total wait time.
_WAF_PROBE_INTERVAL = 60.0  # seconds between probe attempts
_WAF_MAX_WAIT = 600.0  # give up after 10 minutes

# HTTP status codes treated as soft WAF blocks (pause + retry).
# 403 = Cloudflare (Anime-Planet), 405 = AWS WAF (MAL).
_WAF_BLOCKED_CODES = frozenset({403, 405})

# Transient error retry
_TRANSIENT_RETRY_DELAY = 10.0  # seconds before retrying transient failures
_TRANSIENT_MAX_RETRIES = 3
_TRANSIENT_ERROR_FRAGMENTS = (
    "ERR_NAME_NOT_RESOLVED",
    "ERR_CONNECTION_REFUSED",
    "Target page, context or browser has been closed",
    "Timeout 90000ms exceeded",
)


def _get_base_url() -> str:
    return os.environ.get("CRAWL4AI_DOCKER_URL", _DEFAULT_DOCKER_URL).rstrip("/")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


async def _submit_job(
    session: aiohttp.ClientSession,
    base_url: str,
    urls: list[str],
    browser_config: dict[str, Any],
    crawler_config: dict[str, Any],
) -> str | None:
    """POST /crawl/job and return the task_id, or None on failure."""
    payload: dict[str, Any] = {
        "urls": urls,
        "browser_config": browser_config,
        "crawler_config": crawler_config,
    }
    try:
        async with session.post(f"{base_url}/crawl/job", json=payload) as resp:
            if resp.status not in (200, 202):
                text = await resp.text()
                if resp.status >= 500:
                    raise RuntimeError(
                        f"crawl4ai server error: HTTP {resp.status} — {text[:200]}"
                    )
                logger.error(
                    f"crawl4ai job submission failed: HTTP {resp.status} — {text[:200]}"
                )
                return None
            data: dict[str, Any] = await resp.json()
            task_id: str | None = data.get("task_id")
            if not task_id:
                logger.error(f"crawl4ai job response missing task_id: {data}")
                return None
            return task_id
    except aiohttp.ClientError as exc:
        logger.warning(f"crawl4ai Docker not reachable: {exc}")
        return None


async def _poll_job(
    session: aiohttp.ClientSession,
    base_url: str,
    task_id: str,
    timeout: float,
    poll_interval: float,
) -> dict[str, Any] | None:
    """Poll GET /crawl/job/{task_id} until completed or timeout. Returns raw response or None."""
    deadline = asyncio.get_running_loop().time() + timeout
    url = f"{base_url}/crawl/job/{task_id}"

    while asyncio.get_running_loop().time() < deadline:
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    logger.error(
                        f"crawl4ai poll error: HTTP {resp.status} for task {task_id}"
                    )
                    return None
                data: dict[str, Any] = await resp.json()
        except aiohttp.ClientError as exc:
            logger.warning(f"crawl4ai poll request failed: {exc}")
            return None

        status = data.get("status")
        if status == "completed":
            return data
        if status == "failed":
            logger.error(
                f"crawl4ai job {task_id} failed: {data.get('error', 'unknown')}"
            )
            return None

        await asyncio.sleep(poll_interval)

    logger.error(f"crawl4ai job {task_id} timed out after {timeout:.0f}s")
    return None


def _normalize_url(url: str) -> str:
    """Normalize a URL to a consistent percent-encoded form for matching.

    Decodes any existing percent-encoding then re-encodes non-ASCII characters,
    so that raw Unicode URLs and percent-encoded URLs compare equal.
    """
    return quote(unquote(url), safe=":/@?=#&%+!$'()*+,;")


def _align_results(
    urls: list[str], raw_results: list[dict[str, Any]]
) -> list[dict[str, Any] | None]:
    """Align job results back to the input URL list.

    The Docker server may reorder or drop results; we match by the ``url``
    field returned in each result entry. Both exact-match and normalized
    (percent-encoded) forms are indexed to handle cases where Playwright
    encodes non-ASCII characters in the URL differently from the submitted form
    (e.g. raw Unicode slug submitted, percent-encoded slug returned).
    """
    index: dict[str, dict[str, Any]] = {}
    for entry in raw_results:
        entry_url = entry.get("url") or ""
        if entry_url:
            index[entry_url] = entry
            normalized = _normalize_url(entry_url)
            if normalized != entry_url:
                index[normalized] = entry

    aligned: list[dict[str, Any] | None] = []
    for url in urls:
        entry = index.get(url) or index.get(_normalize_url(url))
        if entry is None:
            logger.warning(f"crawl4ai returned no result for URL: {url}")
            aligned.append(None)
            continue

        if not entry.get("success", True):
            err = entry.get("error_message") or entry.get("error", "")
            logger.warning(f"crawl4ai result failure for {url}: {err}")
            aligned.append(None)
            continue

        status_code = entry.get("status_code")
        if status_code == 404:
            logger.warning(f"crawl4ai 404 for {url}")
            aligned.append(None)
            continue
        if status_code in _WAF_BLOCKED_CODES:
            logger.error(f"crawl4ai {status_code} (WAF block) for {url}")
            aligned.append(None)
            continue

        aligned.append(entry)

    return aligned


def _extract_waf_blocked_urls(raw_results: list[dict[str, Any]]) -> list[str]:
    """Return URLs soft-blocked by a WAF (403 Cloudflare or 405 AWS WAF)."""
    return [
        entry["url"]
        for entry in raw_results
        if entry.get("status_code") in _WAF_BLOCKED_CODES and entry.get("url")
    ]


def _extract_transient_failed_urls(raw_results: list[dict[str, Any]]) -> list[str]:
    """Return URLs that failed with a known transient error (DNS or browser closed)."""
    return [
        entry["url"]
        for entry in raw_results
        if not entry.get("success", True)
        and entry.get("url")
        and any(
            frag in (entry.get("error_message") or entry.get("error", ""))
            for frag in _TRANSIENT_ERROR_FRAGMENTS
        )
    ]


async def _retry_failed_urls(
    session: aiohttp.ClientSession,
    base_url: str,
    failed_urls: list[str],
    aligned: list[dict[str, Any] | None],
    all_urls: list[str],
    browser_config: dict[str, Any],
    crawler_config: dict[str, Any],
    timeout: float,
    poll_interval: float,
) -> tuple[list[dict[str, Any] | None], list[str]]:
    """Submit failed_urls as a new job and patch successes back into aligned.

    Returns (aligned, newly_waf_blocked) so callers can fold WAF blocks from
    the retry into the main WAF recovery pass instead of silently dropping them.
    """
    task_id = await _submit_job(
        session, base_url, failed_urls, browser_config, crawler_config
    )
    if not task_id:
        return aligned, []
    response = await _poll_job(session, base_url, task_id, timeout, poll_interval)
    if not response:
        return aligned, []
    raw: list[dict[str, Any]] = response.get("result", {}).get("results") or []
    newly_waf_blocked = _extract_waf_blocked_urls(raw)
    retry_by_url = dict(zip(failed_urls, _align_results(failed_urls, raw)))
    for i, url in enumerate(all_urls):
        if aligned[i] is None and retry_by_url.get(url) is not None:
            aligned[i] = retry_by_url[url]
    return aligned, newly_waf_blocked


async def _probe_waf_recovery(
    session: aiohttp.ClientSession,
    base_url: str,
    probe_url: str,
    browser_config: dict[str, Any],
    crawler_config: dict[str, Any],
) -> bool:
    """Submit a single URL and return True if it comes back without a WAF block."""
    task_id = await _submit_job(
        session, base_url, [probe_url], browser_config, crawler_config
    )
    if not task_id:
        return False
    response = await _poll_job(
        session, base_url, task_id, timeout=90.0, poll_interval=3.0
    )
    if not response:
        return False
    results: list[dict[str, Any]] = response.get("result", {}).get("results") or []
    if not results:
        return False
    return results[0].get("status_code") not in _WAF_BLOCKED_CODES


async def _wait_for_waf_unblock(
    session: aiohttp.ClientSession,
    base_url: str,
    probe_url: str,
    browser_config: dict[str, Any],
    crawler_config: dict[str, Any],
) -> bool:
    """Poll until the WAF block lifts or the max wait time is exceeded.

    Probes ``probe_url`` every ``_WAF_PROBE_INTERVAL`` seconds.
    Returns True when unblocked, False on timeout.
    """
    deadline = asyncio.get_running_loop().time() + _WAF_MAX_WAIT
    attempt = 0
    while asyncio.get_running_loop().time() < deadline:
        attempt += 1
        logger.info(
            f"WAF recovery: waiting {_WAF_PROBE_INTERVAL:.0f}s before probe #{attempt}..."
        )
        await asyncio.sleep(_WAF_PROBE_INTERVAL)
        if await _probe_waf_recovery(
            session, base_url, probe_url, browser_config, crawler_config
        ):
            logger.info(f"WAF unblocked after {attempt} probe(s) — resuming crawl")
            return True
        logger.warning(f"WAF probe #{attempt}: still blocked")
    logger.error(
        f"WAF did not unblock within {_WAF_MAX_WAIT:.0f}s — giving up on blocked URLs"
    )
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def crawl_single_url(
    url: str,
    browser_config: dict[str, Any],
    crawler_config: dict[str, Any],
    timeout: float = 200.0,
    poll_interval: float = 2.0,
) -> dict[str, Any] | None:
    """Submit one URL to the crawl4ai Docker server and return its result dict.

    Args:
        url: The URL to crawl.
        browser_config: Browser configuration dict (e.g. headless, stealth).
        crawler_config: Crawler configuration dict (extraction strategy, delays).
        timeout: Maximum seconds to poll for job completion.
        poll_interval: Seconds between poll requests.

    Returns:
        The result dict for the URL, or None on any failure.
    """
    base_url = _get_base_url()
    async with aiohttp.ClientSession() as session:
        task_id = await _submit_job(
            session, base_url, [url], browser_config, crawler_config
        )
        if not task_id:
            return None

        job_response = await _poll_job(
            session, base_url, task_id, timeout, poll_interval
        )
        if not job_response:
            return None

        raw_results: list[dict[str, Any]] = (
            job_response.get("result", {}).get("results") or []
        )
        aligned = _align_results([url], raw_results)
        return aligned[0] if aligned else None


async def crawl_batch_urls(
    urls: list[str],
    browser_config: dict[str, Any],
    crawler_config: dict[str, Any],
    timeout: float = 600.0,
    poll_interval: float = 5.0,
) -> list[dict[str, Any] | None]:
    """Submit N URLs as a single crawl4ai Docker job and return aligned results.

    Efficient for bulk fetching: one HTTP round-trip regardless of N.
    The Docker server processes URLs at ``MAX_CONCURRENT_TASKS`` concurrency,
    which acts as the rate limiter — no Python-level throttling needed.

    If a WAF soft-blocks any URLs (403 Cloudflare or 405 AWS WAF), the function pauses, polls until
    the block lifts, then retries only the affected URLs and merges them back
    into the original result list — so callers always get a full-length list
    aligned to the input.

    Args:
        urls: List of URLs to crawl.
        browser_config: Browser configuration dict.
        crawler_config: Crawler configuration dict.
        timeout: Maximum seconds to poll for job completion.
        poll_interval: Seconds between poll requests.

    Returns:
        List aligned to ``urls`` — None for any failed or missing result.
    """
    if not urls:
        return []

    base_url = _get_base_url()
    async with aiohttp.ClientSession() as session:
        task_id = await _submit_job(
            session, base_url, urls, browser_config, crawler_config
        )
        if not task_id:
            return [None] * len(urls)

        job_response = await _poll_job(
            session, base_url, task_id, timeout, poll_interval
        )
        if not job_response:
            return [None] * len(urls)

        raw_results: list[dict[str, Any]] = (
            job_response.get("result", {}).get("results") or []
        )

        waf_blocked = _extract_waf_blocked_urls(raw_results)
        transient_failed = _extract_transient_failed_urls(raw_results)
        aligned = _align_results(urls, raw_results)

        still_failing = transient_failed
        for attempt in range(1, _TRANSIENT_MAX_RETRIES + 1):
            if not still_failing:
                break
            logger.warning(
                f"{len(still_failing)} URL(s) hit transient errors — retry {attempt}/{_TRANSIENT_MAX_RETRIES} in {_TRANSIENT_RETRY_DELAY:.0f}s"
            )
            await asyncio.sleep(_TRANSIENT_RETRY_DELAY)
            aligned, retry_waf_blocked = await _retry_failed_urls(
                session,
                base_url,
                still_failing,
                aligned,
                urls,
                browser_config,
                crawler_config,
                timeout,
                poll_interval,
            )
            waf_blocked = list(dict.fromkeys(waf_blocked + retry_waf_blocked))
            still_failing = [
                url
                for url in still_failing
                if aligned[urls.index(url)] is None and url not in waf_blocked
            ]

        if waf_blocked:
            blocked_codes = sorted(
                {
                    entry.get("status_code")
                    for entry in raw_results
                    if entry.get("url") in set(waf_blocked)
                }
                - {None}
            )
            codes_str = "/".join(map(str, blocked_codes))
            logger.warning(
                f"WAF blocked {len(waf_blocked)} URL(s) with {codes_str} — pausing until unblocked"
            )
            recovered = await _wait_for_waf_unblock(
                session, base_url, waf_blocked[0], browser_config, crawler_config
            )
            if recovered:
                aligned, _ = await _retry_failed_urls(
                    session,
                    base_url,
                    waf_blocked,
                    aligned,
                    urls,
                    browser_config,
                    crawler_config,
                    timeout,
                    poll_interval,
                )

        return aligned
