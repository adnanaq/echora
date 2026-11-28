"""Turn resolved page keys into stored page keys and choose each linked page's URL and title."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from enrichment.pipeline.relationship_merger import PROVIDER_PRIORITY
from shared_proto.v1 import anime_pb2
from shared_proto.v1.merge_pb2 import PageKey, PageSource, StoredPage
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


def is_mappable_stored_page(stored_page: StoredPage) -> bool:
    """Check whether a stored provider page can contribute data to a merge.

    A page marked ``is_not_found`` keeps its previously fetched responses and
    still participates in merges. Only a dead page or a page without stored
    responses is skipped.

    Args:
        stored_page: Stored provider page passed into a merge request.

    Returns:
        ``True`` when the page is live and holds at least one stored response.
    """
    return not stored_page.is_dead and bool(stored_page.responses)


def validate_one_page_per_provider(providers: Iterable[str], source_name: str) -> None:
    """Refuse merge input that holds more than one page or list entry per provider.

    An entity links at most one page of each provider, which the database enforces.
    A merge keys its records by provider, so a repeated provider would silently replace
    the earlier record. Input that breaks this rule comes from a faulty caller and is
    refused instead.

    Args:
        providers: The provider of each page or list entry, in request order.
        source_name: The request field the providers come from, used in the error.

    Raises:
        ValueError: When a provider appears more than once.
    """
    repeated_providers = sorted(
        provider for provider, count in Counter(providers).items() if count > 1
    )
    if repeated_providers:
        error_message = (
            f"{source_name} holds more than one entry for provider(s) "
            f"{', '.join(repeated_providers)}"
        )
        raise ValueError(error_message)


def collect_live_page_keys(stored_pages: Sequence[StoredPage]) -> list[PageKey]:
    """Collect page keys of live stored pages, excluding dead pages.

    Args:
        stored_pages: Stored provider pages linked to an entity.

    Returns:
        List of ``PageKey`` messages for pages whose ``is_dead`` flag is false.
    """
    return [
        stored_page.page_key for stored_page in stored_pages if not stored_page.is_dead
    ]


def rank_provider_priority(provider: str) -> int:
    """Return the priority rank of a provider slug for merge ordering.

    Args:
        provider: Provider slug such as ``"mal"`` or ``"anilist"``.

    Returns:
        Zero-based priority index in ``PROVIDER_PRIORITY``, or
        ``len(PROVIDER_PRIORITY)`` for unknown providers.
    """
    if provider in PROVIDER_PRIORITY:
        return PROVIDER_PRIORITY.index(provider)
    return len(PROVIDER_PRIORITY)


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

    A page with none of these keeps an empty URL, which the save reads as unknown. The
    page's title is the ``title_field`` of its own provider's record, or empty when that
    record has none. A page with neither a URL nor a title gets no ``PageSource``.

    Args:
        page_keys: Linked pages, in the order their sources are returned.
        records_by_provider: Each provider's mapped record, whose ``sources`` holds the
            URLs it reports.
        build_template_url: Builds a page's URL from its key, or returns an empty
            string when the provider has no URL template.
        title_field: The record field holding the title, such as ``"title"`` or
            ``"name"``.

    Returns:
        One ``PageSource`` per page with a URL or a title, in page order.
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
        own_record = records_by_provider.get(page_key.provider) or {}
        page_title = str(own_record.get(title_field) or "")
        if page_url or page_title:
            page_sources.append(
                PageSource(page_key=page_key, source=page_url, title=page_title)
            )
    return page_sources


def read_page_urls(page_sources: Sequence[PageSource]) -> list[str]:
    """Read the sorted, distinct URLs of the given page sources.

    A merged record's ``sources`` lists one URL per linked page, the one
    ``build_page_sources`` chose, so a page reported both with and without its slug is
    listed once. A page whose URL is unknown is left out.

    Args:
        page_sources: The linked pages with their chosen URLs.

    Returns:
        The non-empty URLs, sorted and each listed once.
    """
    return sorted(
        {page_source.source for page_source in page_sources if page_source.source}
    )
