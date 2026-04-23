"""Concrete ``ITransport`` implementation backed by the crawl4ai Docker service.

``DockerTransport`` delegates all network I/O to ``crawl_single_url`` and
``crawl_batch_urls`` from the ``crawl4ai_docker`` module, which communicate
with a running crawl4ai container over its REST API.
"""

from typing import Any

from enrichment.sources.base.crawl4ai_docker import crawl_batch_urls, crawl_single_url
from enrichment.sources.base.framework.interfaces import ITransport


class DockerTransport(ITransport):
    """Network transport that uses the crawl4ai Docker REST API.

    This is the production transport.  It requires a crawl4ai container to be
    reachable at the URL configured in the crawl4ai settings.  For unit tests,
    inject a mock or stub that implements ``ITransport`` instead.
    """

    async def fetch_single(
        self,
        url: str,
        browser_config: dict[str, Any],
        crawler_config: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """Fetch a single URL via the crawl4ai Docker service.

        Args:
            url: The page URL to fetch.
            browser_config: Browser-level settings forwarded to crawl4ai
                (e.g. headless mode, viewport size).
            crawler_config: Crawl-level settings forwarded to crawl4ai
                (e.g. CSS/XPath extraction schemas, wait conditions).
            **kwargs: Additional keyword arguments forwarded to
                ``crawl_single_url``.

        Returns:
            Raw extraction result dict from crawl4ai, or ``None`` on failure.
        """
        return await crawl_single_url(
            url=url,
            browser_config=browser_config,
            crawler_config=crawler_config,
            **kwargs,
        )

    async def fetch_batch(
        self,
        urls: list[str],
        browser_config: dict[str, Any],
        crawler_config: dict[str, Any],
        **kwargs: Any,
    ) -> list[dict[str, Any] | None]:
        """Fetch multiple URLs via the crawl4ai Docker service.

        Args:
            urls: Ordered list of page URLs to fetch.
            browser_config: Browser-level settings forwarded to crawl4ai.
            crawler_config: Crawl-level settings forwarded to crawl4ai.
            **kwargs: Additional keyword arguments forwarded to
                ``crawl_batch_urls``.

        Returns:
            List of raw extraction result dicts (or ``None``) aligned with
            ``urls``.
        """
        return await crawl_batch_urls(
            urls=urls,
            browser_config=browser_config,
            crawler_config=crawler_config,
            **kwargs,
        )
