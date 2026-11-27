"""Turn resolved page keys into stored page keys and choose each linked page's URL and title."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from shared_proto.v1 import anime_pb2
from shared_proto.v1.merge_pb2 import PageKey, PageSource
from work_matching import (
    read_page_id,
    read_page_kind,
    read_page_provider,
    resolve_page_key,
)

_ENTITY_TYPE_BY_PAGE_KIND: dict[str, anime_pb2.EntityType] = {
    "anime": anime_pb2.ENTITY_TYPE_ANIME,
    "character": anime_pb2.ENTITY_TYPE_CHARACTER,
    "episode": anime_pb2.ENTITY_TYPE_EPISODE,
}


def build_stored_page_key(page_key: str) -> PageKey | None:
    """Build the stored ``PageKey`` a resolved page key names.

    Args:
        page_key: A page key, as ``resolve_page_key`` gives it.

    Returns:
        The stored page key, or ``None`` when the key is not a known provider's anime,
        character or episode page.

    Examples:
        >>> build_stored_page_key("anime_planet:anime:one-piece").provider
        'anime_planet'
        >>> build_stored_page_key("anilist:manga:30013") is None
        True
    """
    provider = read_page_provider(page_key)
    entity_type = _ENTITY_TYPE_BY_PAGE_KIND.get(read_page_kind(page_key) or "")
    if provider is None or entity_type is None:
        return None
    return PageKey(
        provider=provider,
        entity_type=entity_type,
        address=read_page_id(page_key),
    )


def build_page_sources(
    page_keys: Sequence[PageKey],
    records_by_provider: Mapping[str, Mapping[str, Any]],
    build_template_url: Callable[[PageKey], str],
    title_field: str,
) -> list[PageSource]:
    """Choose the URL and title of each linked page from the mapped provider records.

    Each page gets one URL, the first of:

    - the longest URL any record reports that names the page, so a URL with the
      provider's slug wins over the bare one;
    - the longest URL the page's own provider record reports for a page of its own
      provider and kind, for a page whose key its URL does not hold, such as a Kitsu
      episode keyed by its id while its URL holds its work's slug and number;
    - the URL ``build_template_url`` builds from the page key.

    A page with none of these gets no ``PageSource``. The page's title is the
    ``title_field`` of its own provider's record, or empty when that record has none.

    Args:
        page_keys: Linked pages, in the order their sources are returned.
        records_by_provider: Each provider's mapped record, whose ``sources`` holds the
            URLs it reports.
        build_template_url: Builds a page's URL from its key, or returns an empty
            string when the provider has no URL template.
        title_field: The record field holding the title, such as ``"title"`` or
            ``"name"``.

    Returns:
        One ``PageSource`` per page with a URL, in page order.
    """
    reported_urls_by_provider = {
        provider: [str(url) for url in record.get("sources") or [] if url]
        for provider, record in records_by_provider.items()
    }
    named_urls = [
        (named_page, url)
        for urls in reported_urls_by_provider.values()
        for url in urls
        if (named_page := build_stored_page_key(resolve_page_key(url))) is not None
    ]
    page_sources: list[PageSource] = []
    for page_key in page_keys:
        own_urls = set(reported_urls_by_provider.get(page_key.provider, ()))
        exact_urls = [url for named_page, url in named_urls if named_page == page_key]
        own_kind_urls = [
            url
            for named_page, url in named_urls
            if url in own_urls
            and named_page.provider == page_key.provider
            and named_page.entity_type == page_key.entity_type
        ]
        page_url = max(
            exact_urls or own_kind_urls or [""], key=len
        ) or build_template_url(page_key)
        if page_url:
            own_record = records_by_provider.get(page_key.provider) or {}
            page_sources.append(
                PageSource(
                    page_key=page_key,
                    source=page_url,
                    title=str(own_record.get(title_field) or ""),
                )
            )
    return page_sources
