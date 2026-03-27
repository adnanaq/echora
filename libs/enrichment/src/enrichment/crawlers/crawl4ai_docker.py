"""Reusable HTTP transport layer for the crawl4ai Docker REST server.

All enrichment crawlers that have been migrated away from in-process Playwright
use this module instead of instantiating ``AsyncWebCrawler`` directly.  The
Docker server runs a shared browser pool which eliminates per-crawler process
overhead and provides a single rate-limiting point via ``MAX_CONCURRENT_TASKS``.

Public API
----------
``crawl_single_url`` — submit one URL, poll until done, return result dict.
``crawl_batch_urls``  — submit N URLs as one job, return list aligned to input.

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

import aiohttp

logger = logging.getLogger(__name__)

_DEFAULT_DOCKER_URL = "http://localhost:11235"


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
    deadline = asyncio.get_event_loop().time() + timeout
    url = f"{base_url}/crawl/job/{task_id}"

    while asyncio.get_event_loop().time() < deadline:
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    logger.error(f"crawl4ai poll error: HTTP {resp.status} for task {task_id}")
                    return None
                data: dict[str, Any] = await resp.json()
        except aiohttp.ClientError as exc:
            logger.warning(f"crawl4ai poll request failed: {exc}")
            return None

        status = data.get("status")
        if status == "completed":
            return data
        if status == "failed":
            logger.error(f"crawl4ai job {task_id} failed: {data.get('error', 'unknown')}")
            return None

        await asyncio.sleep(poll_interval)

    logger.error(f"crawl4ai job {task_id} timed out after {timeout:.0f}s")
    return None


def _align_results(
    urls: list[str], raw_results: list[dict[str, Any]]
) -> list[dict[str, Any] | None]:
    """Align job results back to the input URL list.

    The Docker server may reorder or drop results; we match by the ``url``
    field returned in each result entry.
    """
    index: dict[str, dict[str, Any]] = {}
    for entry in raw_results:
        entry_url = entry.get("url") or ""
        if entry_url:
            index[entry_url] = entry

    aligned: list[dict[str, Any] | None] = []
    for url in urls:
        entry = index.get(url)
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
        if status_code == 405:
            logger.error(f"crawl4ai 405 (AWS WAF soft block) for {url}")
            aligned.append(None)
            continue

        aligned.append(entry)

    return aligned


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def crawl_single_url(
    url: str,
    browser_config: dict[str, Any],
    crawler_config: dict[str, Any],
    timeout: float = 90.0,
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
        task_id = await _submit_job(session, base_url, [url], browser_config, crawler_config)
        if not task_id:
            return None

        job_response = await _poll_job(session, base_url, task_id, timeout, poll_interval)
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
    poll_interval: float = 3.0,
) -> list[dict[str, Any] | None]:
    """Submit N URLs as a single crawl4ai Docker job and return aligned results.

    Efficient for bulk fetching: one HTTP round-trip regardless of N.
    The Docker server processes URLs at ``MAX_CONCURRENT_TASKS`` concurrency,
    which acts as the rate limiter — no Python-level throttling needed.

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
        task_id = await _submit_job(session, base_url, urls, browser_config, crawler_config)
        if not task_id:
            return [None] * len(urls)

        job_response = await _poll_job(session, base_url, task_id, timeout, poll_interval)
        if not job_response:
            return [None] * len(urls)

        raw_results: list[dict[str, Any]] = (
            job_response.get("result", {}).get("results") or []
        )
        return _align_results(urls, raw_results)
