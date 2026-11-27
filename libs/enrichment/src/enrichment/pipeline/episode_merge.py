"""Merge an episode from its stored pages, list entries, and field corrections."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from enrichment.pipeline.anime_field_corrections import apply_field_corrections
from enrichment.pipeline.anime_proto_mapper import to_proto_timestamp
from enrichment.pipeline.relationship_merger import PROVIDER_PRIORITY
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

_EPISODE_URL_TEMPLATES: dict[str, str] = {
    "mal": "https://myanimelist.net/anime/{address}",
    "kitsu": "https://kitsu.app/episodes/{address}",
    "anidb": "https://anidb.net/episode/{address}",
}


def _page_key_to_episode_url(page_key: PageKey) -> str:
    """Build a canonical episode URL from an episode ``PageKey``.

    Args:
        page_key: Episode ``PageKey`` message.

    Returns:
        Canonical URL string, or empty string for an unknown provider.
    """
    if page_key.provider == "mal":
        if ":" in page_key.address:
            anime_id, _, ep_num = page_key.address.partition(":")
            return f"https://myanimelist.net/anime/{anime_id}/episode/{ep_num}"
        return ""
    template = _EPISODE_URL_TEMPLATES.get(page_key.provider)
    return template.format(address=page_key.address) if template else ""


def _map_stored_episode_page(page: StoredPage) -> dict[str, Any] | None:
    """Map one stored episode ``StoredPage`` into a canonical Episode dict.

    Args:
        page: Stored provider episode page with raw responses.

    Returns:
        Canonical Episode dict, or ``None`` when the page has no mappable responses.
    """
    if page.is_dead or page.is_not_found or not page.responses:
        return None
    provider = page.page_key.provider
    fallback_url = _page_key_to_episode_url(page.page_key)
    responses: Sequence[PageResponse] = page.responses
    mappers: dict[str, Callable[[], dict[str, Any]]] = {
        "mal": lambda: map_mal_episode_responses(responses, source_url=fallback_url),
        "kitsu": lambda: map_kitsu_episode_responses(responses),
    }
    mapper = mappers.get(provider)
    return mapper() if mapper is not None else None


def _list_entry_to_record(entry: EpisodeListEntry) -> dict[str, Any]:
    """Convert an ``EpisodeListEntry`` proto into a partial canonical Episode dict.

    Args:
        entry: Provider episode list entry from a work's page.

    Returns:
        Dictionary of populated episode fields.
    """
    record: dict[str, Any] = {}
    if entry.HasField("title") and entry.title:
        record["title"] = entry.title
    if entry.HasField("aired_at"):
        record["aired_at"] = entry.aired_at.ToDatetime()
    if entry.HasField("duration_seconds"):
        record["duration_seconds"] = entry.duration_seconds
    if entry.HasField("is_filler"):
        record["is_filler"] = entry.is_filler
    if entry.HasField("is_recap"):
        record["is_recap"] = entry.is_recap
    return record


def _combine_provider_sources(
    pages: Sequence[StoredPage],
    list_entries: Sequence[EpisodeListEntry],
) -> list[tuple[str, dict[str, Any]]]:
    """Combine mapped episode pages and list entries in ``PROVIDER_PRIORITY`` order.

    Args:
        pages: Stored episode pages passed to ``MergeEpisode``.
        list_entries: Episode list entries from work pages.

    Returns:
        List of ``(provider, record)`` pairs ordered by provider priority.
    """
    by_provider: dict[str, dict[str, Any]] = {}
    for entry in list_entries:
        by_provider[entry.provider] = _list_entry_to_record(entry)
    for page in pages:
        mapped = _map_stored_episode_page(page)
        if mapped is not None:
            merged_entry = {**by_provider.get(page.page_key.provider, {}), **mapped}
            by_provider[page.page_key.provider] = merged_entry

    return sorted(
        by_provider.items(),
        key=lambda item: (
            PROVIDER_PRIORITY.index(item[0])
            if item[0] in PROVIDER_PRIORITY
            else len(PROVIDER_PRIORITY)
        ),
    )


def _first_present(ranked: list[tuple[str, dict[str, Any]]], field: str) -> Any:
    """Return the first non-None, non-empty scalar value across ranked providers.

    Args:
        ranked: Provider records in priority order.
        field: Field name to resolve.

    Returns:
        First concrete value, or ``None`` if no provider supplied it.
    """
    for _, record in ranked:
        val = record.get(field)
        if val not in (None, "", [], {}):
            return val
    return None


def _populate_episode_maps(
    ep: anime_pb2.Episode,
    ranked: list[tuple[str, dict[str, Any]]],
) -> None:
    """Populate ``titles`` and ``streaming`` maps on ``Episode`` by priority.

    Args:
        ep: Target ``anime_pb2.Episode`` message mutated in place.
        ranked: Combined provider records in priority order.
    """
    for _, rec in ranked:
        for lang, title_val in (rec.get("titles") or {}).items():
            if lang and title_val and lang not in ep.titles:
                ep.titles[str(lang)] = str(title_val)
        for platform, url in (rec.get("streaming") or {}).items():
            if platform and url and platform not in ep.streaming:
                ep.streaming[str(platform)] = str(url)


def _build_episode_page_sources(request: MergeEpisodeRequest) -> list[PageSource]:
    """Build the canonical URL of each linked episode page.

    A page whose provider has no episode URL, or a MAL page without an episode number in
    its address, gets none.

    Args:
        request: Incoming ``MergeEpisodeRequest`` message.

    Returns:
        One ``PageSource`` per linked page with a URL, in page order.
    """
    return [
        PageSource(page_key=page.page_key, source=url)
        for page in request.pages
        if (url := _page_key_to_episode_url(page.page_key))
    ]


def _collect_episode_sources(
    page_sources: Sequence[PageSource],
    ranked: list[tuple[str, dict[str, Any]]],
) -> list[str]:
    """Collect and sort all canonical episode source URLs.

    Args:
        page_sources: URL of each linked episode page.
        ranked: Combined provider records in priority order.

    Returns:
        Sorted list of unique canonical episode URLs.
    """
    return sorted(
        {
            *(page_source.source for page_source in page_sources),
            *(
                str(src)
                for _, rec in ranked
                for src in (rec.get("sources") or [])
                if src
            ),
        }
    )


def _build_episode_proto(
    request: MergeEpisodeRequest,
    ranked: list[tuple[str, dict[str, Any]]],
    page_sources: Sequence[PageSource],
) -> anime_pb2.Episode:
    """Assemble and deterministically sort a ``shared_proto.v1.Episode`` message.

    Args:
        request: Incoming ``MergeEpisodeRequest`` message.
        ranked: Combined provider records in priority order.
        page_sources: URL of each linked episode page.

    Returns:
        Populated and sorted ``anime_pb2.Episode`` message.
    """
    sources = _collect_episode_sources(page_sources, ranked)
    images = sorted(
        {str(img) for _, rec in ranked for img in (rec.get("images") or []) if img}
    )
    ep = anime_pb2.Episode(
        id=request.episode_id,
        anime_id=request.anime_id,
        entity_type=anime_pb2.ENTITY_TYPE_EPISODE,
        episode_number=request.episode_number,
        title=str(_first_present(ranked, "title") or ""),
        is_filler=bool(
            next(
                (rec["is_filler"] for _, rec in ranked if "is_filler" in rec),
                False,
            )
        ),
        is_recap=bool(
            next(
                (rec["is_recap"] for _, rec in ranked if "is_recap" in rec),
                False,
            )
        ),
        images=images,
        sources=sources,
    )
    for str_field in ("description", "synopsis", "title_japanese", "title_romaji"):
        if (val := _first_present(ranked, str_field)) is not None:
            setattr(ep, str_field, str(val))
    for int_field in ("duration_seconds", "season_number"):
        if (val := _first_present(ranked, int_field)) is not None:
            setattr(ep, int_field, int(val))
    if (score := _first_present(ranked, "score")) is not None:
        ep.score = float(score)
    if (aired_ts := to_proto_timestamp(_first_present(ranked, "aired_at"))) is not None:
        ep.aired_at.CopyFrom(aired_ts)
    _populate_episode_maps(ep, ranked)
    return ep


def merge_episode_pages(request: MergeEpisodeRequest) -> MergeEpisodeResponse:
    """Merge an episode's stored pages, list entries, and corrections into ``Episode``.

    Args:
        request: Incoming ``MergeEpisodeRequest`` protobuf message.

    Returns:
        ``MergeEpisodeResponse`` holding the merged ``Episode`` message and the URL of
        each linked page.
    """
    ranked = _combine_provider_sources(request.pages, request.list_entries)
    page_sources = _build_episode_page_sources(request)
    episode_proto = _build_episode_proto(request, ranked, page_sources)
    apply_field_corrections(episode_proto, request.corrections)
    episode_proto.images[:] = sorted(episode_proto.images)
    episode_proto.sources[:] = sorted(episode_proto.sources)
    return MergeEpisodeResponse(episode=episode_proto, page_sources=page_sources)
