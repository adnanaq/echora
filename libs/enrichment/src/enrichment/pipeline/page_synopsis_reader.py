"""Read a provider page's synopsis through the enrichment helpers.

Answer suggestions send Jev each page's synopsis, so this reader fetches the
page the way seed cleaning does and keeps only the synopsis. Each item gives
the rule, then the reason for it.

- The page is fetched through its provider's helper with characters and
  episodes off, so a cached page costs no request and a missing one is fetched
  at the helper's pace.
- A page whose provider has no helper, a failed fetch and a page without a
  record give no synopsis. A suggestion then rests on the other details.
- The provider models already turn a "no synopsis" placeholder into none.
"""

import logging

from enrichment.pipeline.api_fetcher import ApiFetcher
from enrichment.pipeline.page_requests import build_page_request
from work_matching import ProviderPage

logger = logging.getLogger(__name__)


class PageSynopsisReader:
    """Read the synopsis of provider pages, one page at a time."""

    def __init__(self, api_fetcher: ApiFetcher) -> None:
        """Hold the fetcher the pages are read through.

        Args:
            api_fetcher: The fetcher, which keeps each provider's pacing.
        """
        self._api_fetcher = api_fetcher

    async def read_synopsis(self, page: ProviderPage) -> str | None:
        """Read one page's synopsis.

        Args:
            page: The provider page.

        Returns:
            The page's synopsis, or None when it has none or cannot be read.
        """
        title = page.titles[0] if page.titles else ""
        page_request = build_page_request(page.key, title, page.address)
        if page_request is None:
            return None
        records_by_service = await self._api_fetcher.fetch_all_data(
            {page_request.id_field: page_request.address},
            {"title": title, "sources": [page_request.address]},
            only_services=[page_request.service],
            fetch_characters=False,
            fetch_episodes=False,
        )
        if page_request.service in self._api_fetcher.api_errors:
            logger.warning(
                f"No synopsis for {page.key}: "
                f"{self._api_fetcher.api_errors[page_request.service]}"
            )
            return None
        anime_record = (records_by_service.get(page_request.service) or {}).get("anime")
        return anime_record.get("synopsis") if anime_record else None
