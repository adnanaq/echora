"""AniSearch raw page fetcher implementing the PageFetcher interface."""

from datetime import timedelta
from typing import ClassVar

from enrichment.sources.anisearch.anisearch_http import fetch_anisearch_page
from enrichment.sources.base.page_fetcher import (
    FetchedPage,
    PageFetcher,
    build_html_response,
    validate_page_key,
)
from shared_proto.v1.anime_pb2 import EntityType
from shared_proto.v1.merge_pb2 import PageKey, PageResponse

_ANISEARCH_BASE = "https://www.anisearch.com"


class AniSearchFetcher(PageFetcher):
    """Fetches raw AniSearch HTML responses over PoliteHttpClient."""

    provider: ClassVar[str] = "anisearch"
    supported_entity_types: ClassVar[frozenset[EntityType]] = frozenset(
        {EntityType.ENTITY_TYPE_ANIME, EntityType.ENTITY_TYPE_CHARACTER}
    )

    async def fetch_page(
        self,
        page_key: PageKey,
        *,
        max_age: timedelta | None = None,
    ) -> FetchedPage:
        """Fetch raw HTML responses for an AniSearch anime or character page key.

        Args:
            page_key: AniSearch PageKey with a numeric address.
            max_age: Optional cache freshness bound (unused by PoliteHttpClient).

        Returns:
            FetchedPage holding raw HTML PageResponse objects or not-found status.
        """
        del max_age
        validate_page_key(
            page_key,
            self.provider,
            self.supported_entity_types,
            numeric_address=True,
        )
        if page_key.entity_type == EntityType.ENTITY_TYPE_ANIME:
            return await self._fetch_anime_page(page_key)
        return await self._fetch_character_page(page_key)

    async def _fetch_sub_pages(
        self,
        base_url: str,
        sub_paths: tuple[tuple[str, str], ...],
    ) -> list[PageResponse]:
        """Fetch optional sub-pages relative to a resolved canonical AniSearch URL.

        Args:
            base_url: Resolved canonical AniSearch page URL.
            sub_paths: Sequence of ``(request_name, suffix)`` pairs.

        Returns:
            List of PageResponse objects for every sub-page that returned HTML.
        """
        responses: list[PageResponse] = []
        clean_base = base_url.rstrip("/")
        for request_name, suffix in sub_paths:
            fetched = await fetch_anisearch_page(f"{clean_base}{suffix}")
            if fetched is not None and fetched.html:
                responses.append(build_html_response(request_name, fetched.html))
        return responses

    async def _fetch_anime_page(self, page_key: PageKey) -> FetchedPage:
        """Fetch raw AniSearch anime main, relations, episodes, and characters HTML.

        Args:
            page_key: AniSearch anime PageKey.

        Returns:
            FetchedPage holding raw HTML responses or not-found status.
        """
        main_page = await fetch_anisearch_page(
            f"{_ANISEARCH_BASE}/anime/{page_key.address}"
        )
        if main_page is None or not main_page.html:
            return FetchedPage(page_key=page_key, responses=(), is_not_found=True)

        sub_responses = await self._fetch_sub_pages(
            main_page.url,
            (
                ("relations", "/relations?show=overall"),
                ("episodes", "/episodes"),
                ("characters", "/characters"),
            ),
        )
        responses = [build_html_response("main", main_page.html), *sub_responses]
        return FetchedPage(page_key=page_key, responses=tuple(responses))

    async def _fetch_character_page(self, page_key: PageKey) -> FetchedPage:
        """Fetch raw AniSearch character main, /anime, and /manga HTML pages.

        Args:
            page_key: AniSearch character PageKey.

        Returns:
            FetchedPage holding raw HTML responses or not-found status.
        """
        main_page = await fetch_anisearch_page(
            f"{_ANISEARCH_BASE}/character/{page_key.address}"
        )
        if main_page is None or not main_page.html:
            return FetchedPage(page_key=page_key, responses=(), is_not_found=True)

        sub_responses = await self._fetch_sub_pages(
            main_page.url,
            (("anime", "/anime"), ("manga", "/manga")),
        )
        responses = [build_html_response("main", main_page.html), *sub_responses]
        return FetchedPage(page_key=page_key, responses=tuple(responses))
