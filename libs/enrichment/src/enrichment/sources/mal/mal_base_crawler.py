from typing import Any, Dict, Optional, TypeVar
from pydantic import BaseModel

from enrichment.sources.base.framework.crawler import BaseCrawler, T_Canonical, T_Source
from enrichment.sources.mal.mal_base import get_mal_scraping_limiter

_limiter = get_mal_scraping_limiter()


class MalCrawlerBase(BaseCrawler[T_Source, T_Canonical]):
    """Common base for all MyAnimeList crawlers."""

    async def fetch_raw_data(self, url: str) -> Optional[Dict[str, Any]]:
        """Fetch MAL page with rate limiting."""
        await _limiter.acquire()
        # Subclasses will define the browser/crawler config and call self.transport.fetch_single
        return await self._do_fetch(url)

    async def _do_fetch(self, url: str) -> Optional[Dict[str, Any]]:
        """Actual fetch call to be implemented by concrete MAL crawlers."""
        raise NotImplementedError("Subclasses must implement _do_fetch")
