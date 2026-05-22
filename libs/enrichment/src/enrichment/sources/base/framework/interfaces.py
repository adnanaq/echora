"""Abstract interfaces for the crawler framework.

Defines the two pluggable boundaries that ``BaseCrawler`` depends on:

- ``ITransport`` — network layer (fetch HTML / JSON from a URL).
- ``IRepository`` — persistence layer (save canonical output).

Both are thin ``ABC``s so concrete implementations can be swapped without
touching crawler logic (e.g. ``DockerTransport`` vs a local-browser transport,
``FileRepository`` vs a database repository).
"""

from abc import ABC, abstractmethod
from typing import Any


class ITransport(ABC):
    """Abstract network transport for fetching pages.

    Concrete implementations delegate to a real browser automation backend
    (e.g. crawl4ai via Docker) or a test double.
    """

    @abstractmethod
    async def fetch_single(
        self,
        url: str,
        browser_config: dict[str, Any],
        crawler_config: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """Fetch a single URL and return the raw extraction result.

        Args:
            url: The page URL to fetch.
            browser_config: Browser-level settings forwarded to the backend
                (e.g. headless mode, viewport size).
            crawler_config: Crawl-level settings forwarded to the backend
                (e.g. CSS/XPath extraction schemas, wait conditions).
            **kwargs: Additional keyword arguments passed through to the
                underlying fetch implementation.

        Returns:
            A dict containing the raw extraction result (e.g. ``status_code``,
            ``extracted_content``, ``html``), or ``None`` if the fetch failed
            before any result could be produced.
        """

    @abstractmethod
    async def fetch_batch(
        self,
        urls: list[str],
        browser_config: dict[str, Any],
        crawler_config: dict[str, Any],
        **kwargs: Any,
    ) -> list[dict[str, Any] | None]:
        """Fetch multiple URLs and return aligned raw extraction results.

        The returned list is positionally aligned with ``urls``: index *i* of
        the result corresponds to ``urls[i]``.  A ``None`` entry means the
        fetch for that URL failed.

        Args:
            urls: Ordered list of page URLs to fetch.
            browser_config: Browser-level settings forwarded to the backend.
            crawler_config: Crawl-level settings forwarded to the backend.
            **kwargs: Additional keyword arguments passed through to the
                underlying fetch implementation.

        Returns:
            List of raw extraction dicts (or ``None``) aligned with ``urls``.
        """


class IRepository(ABC):
    """Abstract persistence layer for saving crawled output.

    Concrete implementations write to different backends (local file,
    database, in-memory store) without requiring changes in crawler code.
    """

    @abstractmethod
    def save(self, data: Any) -> None:
        """Persist data to the underlying storage backend.

        Args:
            data: The canonical data to persist.  Typically a ``dict`` but
                the interface accepts ``Any`` to remain generic.
        """
