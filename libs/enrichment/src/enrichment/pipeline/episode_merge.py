"""Merge an episode from its stored pages, list entries, and field corrections."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from enrichment.pipeline.anime_field_corrections import apply_field_corrections
from enrichment.pipeline.anime_proto_mapper import to_proto_timestamp
from enrichment.pipeline.cast_proto_mapper import (
    merge_episode_characters,
    merge_episode_staff,
)
from enrichment.pipeline.page_sources import (
    build_page_sources,
    collect_live_page_keys,
    is_mappable_stored_page,
    rank_provider_priority,
    read_page_urls,
    validate_one_page_per_provider,
)
from enrichment.sources.kitsu.kitsu_mapper import map_kitsu_episode_responses
from enrichment.sources.mal.mal_mapper import map_mal_episode_responses
from enrichment_proto.v1.enrichment_service_pb2 import (
    MergeEpisodeRequest,
    MergeEpisodeResponse,
)
from shared_proto.v1 import anime_pb2
from shared_proto.v1.merge_pb2 import (
    EpisodeListEntry,
    PageKey,
    PageResponse,
    PageSource,
    StoredPage,
)

# A MAL or Kitsu episode's URL needs its work's slug, which its page key does not hold,
# so neither has a template and their URL comes from the fetched episode.
_EPISODE_URL_TEMPLATES: dict[str, str] = {
    "anidb": "https://anidb.net/episode/{address}",
}

_EPISODE_PAGE_MAPPERS: dict[str, Callable[[Sequence[PageResponse]], dict[str, Any]]] = {
    "mal": map_mal_episode_responses,
    "kitsu": map_kitsu_episode_responses,
}


def _build_episode_url(page_key: PageKey) -> str:
    """Build a canonical episode URL from an episode ``PageKey``.

    Args:
        page_key: Episode ``PageKey`` message.

    Returns:
        Canonical URL string, or empty string for a provider without a template.
    """
    template = _EPISODE_URL_TEMPLATES.get(page_key.provider)
    return template.format(address=page_key.address) if template else ""


def _map_stored_episode_page(stored_page: StoredPage) -> dict[str, Any] | None:
    """Map one stored episode ``StoredPage`` into a canonical Episode dict.

    Args:
        stored_page: Stored provider episode page with raw responses.

    Returns:
        Canonical Episode dict, or ``None`` when the page has no mappable responses.
    """
    if not is_mappable_stored_page(stored_page):
        return None
    mapper = _EPISODE_PAGE_MAPPERS.get(stored_page.page_key.provider)
    return mapper(stored_page.responses) if mapper is not None else None


def _convert_list_entry(list_entry: EpisodeListEntry) -> dict[str, Any]:
    """Convert an ``EpisodeListEntry`` proto into a partial canonical Episode dict.

    Args:
        list_entry: Provider episode list entry from a work's page.

    Returns:
        Dictionary of populated episode fields.
    """
    episode_record: dict[str, Any] = {}
    if list_entry.HasField("title") and list_entry.title:
        episode_record["title"] = list_entry.title
    if list_entry.HasField("aired_at"):
        episode_record["aired_at"] = list_entry.aired_at.ToDatetime()
    if list_entry.HasField("duration_seconds"):
        episode_record["duration_seconds"] = list_entry.duration_seconds
    if list_entry.HasField("is_filler"):
        episode_record["is_filler"] = list_entry.is_filler
    if list_entry.HasField("is_recap"):
        episode_record["is_recap"] = list_entry.is_recap
    return episode_record


def _combine_provider_sources(
    stored_pages: Sequence[StoredPage],
    list_entries: Sequence[EpisodeListEntry],
) -> list[tuple[str, dict[str, Any]]]:
    """Combine mapped episode pages and list entries in provider priority order.

    Args:
        stored_pages: Stored episode pages passed to ``MergeEpisode``.
        list_entries: Episode list entries from work pages.

    Returns:
        List of ``(provider, record)`` pairs ordered by provider priority.

    Raises:
        ValueError: When a provider has more than one stored page or more than one
            list entry.
    """
    validate_one_page_per_provider(
        (stored_page.page_key.provider for stored_page in stored_pages), "pages"
    )
    validate_one_page_per_provider(
        (list_entry.provider for list_entry in list_entries), "list_entries"
    )
    records_by_provider: dict[str, dict[str, Any]] = {}
    for list_entry in list_entries:
        records_by_provider[list_entry.provider] = _convert_list_entry(list_entry)
    for stored_page in stored_pages:
        mapped_record = _map_stored_episode_page(stored_page)
        if mapped_record is not None:
            provider = stored_page.page_key.provider
            records_by_provider[provider] = {
                **records_by_provider.get(provider, {}),
                **mapped_record,
            }
    return sorted(
        records_by_provider.items(),
        key=lambda provider_entry: rank_provider_priority(provider_entry[0]),
    )


def _find_first_present(
    ranked_records: Sequence[tuple[str, Mapping[str, Any]]],
    field_name: str,
) -> Any:
    """Find the first non-None, non-empty scalar value across ranked providers.

    Args:
        ranked_records: Provider records in priority order.
        field_name: Field name to resolve.

    Returns:
        First concrete value, or ``None`` if no provider supplied the field.
    """
    for _, provider_record in ranked_records:
        field_value = provider_record.get(field_name)
        if field_value not in (None, "", [], {}):
            return field_value
    return None


def _resolve_episode_flag(
    ranked_records: Sequence[tuple[str, Mapping[str, Any]]],
    flag_name: str,
) -> bool:
    """Resolve an episode boolean flag from the highest-priority provider reporting it.

    Args:
        ranked_records: Provider records in priority order.
        flag_name: Boolean flag key such as ``"is_filler"`` or ``"is_recap"``.

    Returns:
        First explicitly reported boolean value, or ``False`` when unreported.
    """
    return bool(
        next(
            (
                provider_record[flag_name]
                for _, provider_record in ranked_records
                if flag_name in provider_record
            ),
            False,
        )
    )


def _populate_episode_maps(
    episode_message: anime_pb2.Episode,
    ranked_records: Sequence[tuple[str, Mapping[str, Any]]],
) -> None:
    """Populate ``titles`` and ``streaming`` maps on ``Episode`` by priority.

    Args:
        episode_message: Target ``anime_pb2.Episode`` message mutated in place.
        ranked_records: Combined provider records in priority order.
    """
    for _, provider_record in ranked_records:
        for language, title_value in (provider_record.get("titles") or {}).items():
            if language and title_value and language not in episode_message.titles:
                episode_message.titles[str(language)] = str(title_value)
        for platform, streaming_url in (provider_record.get("streaming") or {}).items():
            if platform and streaming_url and platform not in episode_message.streaming:
                episode_message.streaming[str(platform)] = str(streaming_url)


def _build_episode_page_sources(
    request: MergeEpisodeRequest,
    ranked_records: Sequence[tuple[str, Mapping[str, Any]]],
) -> list[PageSource]:
    """Build the URL and title chosen for each live linked episode page.

    ``build_page_sources`` chooses each URL from every URL the mapped provider
    records report, falling back to the provider's canonical episode page URL, and
    takes each page's title from its own provider record's ``title``.

    Args:
        request: Incoming ``MergeEpisodeRequest`` message.
        ranked_records: Combined provider records in priority order.

    Returns:
        One ``PageSource`` per live linked page with a URL or title, in page order.
    """
    return build_page_sources(
        collect_live_page_keys(request.pages),
        dict(ranked_records),
        _build_episode_url,
        "title",
    )


def _build_episode_proto(
    request: MergeEpisodeRequest,
    ranked_records: Sequence[tuple[str, Mapping[str, Any]]],
    page_sources: Sequence[PageSource],
) -> anime_pb2.Episode:
    """Assemble and deterministically sort a ``shared_proto.v1.Episode`` message.

    Args:
        request: Incoming ``MergeEpisodeRequest`` message.
        ranked_records: Combined provider records in priority order.
        page_sources: URL of each linked episode page.

    Returns:
        Populated and sorted ``anime_pb2.Episode`` message.
    """
    image_urls = sorted(
        {
            str(image_url)
            for _, provider_record in ranked_records
            for image_url in (provider_record.get("images") or [])
            if image_url
        }
    )
    episode_message = anime_pb2.Episode(
        id=request.episode_id,
        anime_id=request.anime_id,
        entity_type=anime_pb2.ENTITY_TYPE_EPISODE,
        episode_number=request.episode_number,
        title=str(_find_first_present(ranked_records, "title") or ""),
        is_filler=_resolve_episode_flag(ranked_records, "is_filler"),
        is_recap=_resolve_episode_flag(ranked_records, "is_recap"),
        images=image_urls,
        sources=read_page_urls(page_sources),
        characters=merge_episode_characters(ranked_records),
        staff=merge_episode_staff(ranked_records),
    )
    for string_field in ("description", "synopsis", "title_japanese", "title_romaji"):
        if (
            field_value := _find_first_present(ranked_records, string_field)
        ) is not None:
            setattr(episode_message, string_field, str(field_value))
    for integer_field in ("duration_seconds", "season_number"):
        if (
            field_value := _find_first_present(ranked_records, integer_field)
        ) is not None:
            setattr(episode_message, integer_field, int(field_value))
    if (score := _find_first_present(ranked_records, "score")) is not None:
        episode_message.score = float(score)
    if (
        aired_timestamp := to_proto_timestamp(
            _find_first_present(ranked_records, "aired_at")
        )
    ) is not None:
        episode_message.aired_at.CopyFrom(aired_timestamp)
    _populate_episode_maps(episode_message, ranked_records)
    return episode_message


def merge_episode_pages(request: MergeEpisodeRequest) -> MergeEpisodeResponse:
    """Merge an episode's stored pages, list entries, and corrections into ``Episode``.

    Args:
        request: Incoming ``MergeEpisodeRequest`` protobuf message.

    Returns:
        ``MergeEpisodeResponse`` holding the merged ``Episode`` message and the URL of
        each linked page.
    """
    ranked_records = _combine_provider_sources(request.pages, request.list_entries)
    page_sources = _build_episode_page_sources(request, ranked_records)
    episode_proto = _build_episode_proto(request, ranked_records, page_sources)
    apply_field_corrections(episode_proto, request.corrections)
    episode_proto.images[:] = sorted(episode_proto.images)
    episode_proto.sources[:] = sorted(episode_proto.sources)
    return MergeEpisodeResponse(episode=episode_proto, page_sources=page_sources)
