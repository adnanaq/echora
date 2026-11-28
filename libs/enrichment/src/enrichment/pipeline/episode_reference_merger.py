"""Extract and match episode references from stored anime pages."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple

from common.utils.id_generation import generate_entity_id
from enrichment.pipeline.anime_proto_mapper import to_proto_timestamp
from enrichment.pipeline.page_sources import (
    build_stored_page_key,
    is_mappable_stored_page,
)
from enrichment.pipeline.relationship_merger import PROVIDER_PRIORITY
from enrichment.sources.anidb.anidb_mapper import map_anidb_anime_episode_responses
from enrichment.sources.anisearch.anisearch_mapper import (
    map_anisearch_anime_episode_responses,
)
from enrichment.sources.kitsu.kitsu_mapper import map_kitsu_anime_episode_responses
from enrichment.sources.mal.mal_mapper import map_mal_anime_episode_responses
from shared_proto.v1.anime_pb2 import ENTITY_TYPE_EPISODE
from shared_proto.v1.merge_pb2 import (
    EpisodeListEntry,
    EpisodeReference,
    PageKey,
    PageSource,
    StoredPage,
)
from work_matching import resolve_page_key


class _ListedEpisode(NamedTuple):
    """One episode a work's page lists, with the pages that page links it by."""

    page_sources: list[PageSource]
    episode: dict[str, Any]


def _find_episode_page_sources(episode: dict[str, Any]) -> list[PageSource]:
    """Find the episode pages a listed episode's reported URLs name, with URL and title.

    Args:
        episode: Canonical Episode dict from a provider's episode list.

    Returns:
        The stored key of each reported URL that names an episode page, with the URL.
    """
    return [
        PageSource(
            page_key=page_key, source=str(url), title=str(episode.get("title") or "")
        )
        for url in episode.get("sources") or []
        if (page_key := build_stored_page_key(resolve_page_key(str(url)))) is not None
        and page_key.entity_type == ENTITY_TYPE_EPISODE
    ]


def _list_provider_episodes(page: StoredPage, page_url: str) -> list[_ListedEpisode]:
    """List the episodes a single stored anime ``StoredPage`` names.

    A Kitsu episode is keyed by its Kitsu episode ID, which its URL does not hold, and
    keeps the URL its entry reports. Every other provider's episode is keyed by the
    episode page URLs its entry reports.

    Args:
        page: Stored anime page with raw responses.
        page_url: Canonical URL of the anime page.

    Returns:
        The listed episodes with their pages and URLs, in list order.
    """
    provider = page.page_key.provider
    if provider == "kitsu":
        return [
            _ListedEpisode(
                page_sources=[
                    PageSource(
                        page_key=PageKey(
                            provider="kitsu",
                            entity_type=ENTITY_TYPE_EPISODE,
                            address=kitsu_episode_id,
                        ),
                        source=next(iter(episode.get("sources") or []), ""),
                        title=str(episode.get("title") or ""),
                    )
                ],
                episode=episode,
            )
            for kitsu_episode_id, episode in map_kitsu_anime_episode_responses(
                page.responses
            )
        ]
    if provider == "anidb":
        episodes = map_anidb_anime_episode_responses(page.responses)
    elif provider == "mal":
        episodes = map_mal_anime_episode_responses(page.responses)
    elif provider == "anisearch":
        episodes = map_anisearch_anime_episode_responses(
            page.responses,
            source_url=f"{page_url}/episodes" if page_url else "",
        )
    else:
        return []
    return [
        _ListedEpisode(
            page_sources=_find_episode_page_sources(episode), episode=episode
        )
        for episode in episodes
    ]


def _to_episode_list_entry(
    provider: str,
    episode_dict: dict[str, Any],
) -> EpisodeListEntry:
    """Convert one provider's canonical episode dict into an ``EpisodeListEntry``.

    Args:
        provider: Provider slug.
        episode_dict: Canonical episode dictionary from the provider mapper.

    Returns:
        Populated ``EpisodeListEntry`` protobuf message.
    """
    entry = EpisodeListEntry(provider=provider)
    if episode_dict.get("title"):
        entry.title = str(episode_dict["title"])
    if (aired_ts := to_proto_timestamp(episode_dict.get("aired_at"))) is not None:
        entry.aired_at.CopyFrom(aired_ts)
    if episode_dict.get("duration_seconds") is not None:
        entry.duration_seconds = int(episode_dict["duration_seconds"])
    if episode_dict.get("is_filler") is not None:
        entry.is_filler = bool(episode_dict["is_filler"])
    if episode_dict.get("is_recap") is not None:
        entry.is_recap = bool(episode_dict["is_recap"])
    return entry


def _pick_episode_title(
    list_entries: Sequence[EpisodeListEntry],
    known_match: EpisodeReference | None,
) -> str:
    """Select the primary display title for an ``EpisodeReference``.

    Args:
        list_entries: Provider list entries for this episode.
        known_match: Existing ``EpisodeReference`` if previously known.

    Returns:
        Selected episode title, or empty string if none is available.
    """
    by_provider = {
        entry.provider: entry.title
        for entry in list_entries
        if entry.HasField("title") and entry.title
    }
    for provider in PROVIDER_PRIORITY:
        if provider in by_provider:
            return by_provider[provider]
    for entry in list_entries:
        if entry.HasField("title") and entry.title:
            return entry.title
    return known_match.title if known_match is not None else ""


def _match_known_episodes(
    sorted_numbers: Sequence[int],
    grouped_sources: dict[int, dict[tuple[str, int, str], PageSource]],
    by_page_key: dict[tuple[str, int, str], EpisodeReference],
    by_number: dict[int, EpisodeReference],
) -> dict[int, EpisodeReference]:
    """Match listed episode numbers to known episodes without reusing an ID.

    Pass one matches every listed episode by provider page key and claims each
    matched ``episode_id``. Pass two falls back to matching remaining episodes
    by episode number only when that known ``episode_id`` has not already been
    claimed by a page-key match, preventing duplicate IDs after a renumber.

    Args:
        sorted_numbers: Ascending listed episode numbers.
        grouped_sources: Episode ``PageSource`` map per episode number.
        by_page_key: Index of known episodes by ``(provider, entity_type, address)``.
        by_number: Index of known episodes by ``episode_number``.

    Returns:
        Mapping from episode number to its matched ``EpisodeReference``.
    """
    matched_by_number: dict[int, EpisodeReference] = {}
    claimed_episode_ids: set[str] = set()

    for episode_number in sorted_numbers:
        source_map = grouped_sources.get(episode_number, {})
        for identity in sorted(source_map):
            known = by_page_key.get(identity)
            if known is not None and known.episode_id not in claimed_episode_ids:
                matched_by_number[episode_number] = known
                claimed_episode_ids.add(known.episode_id)
                break

    for episode_number in sorted_numbers:
        if episode_number in matched_by_number:
            continue
        known = by_number.get(episode_number)
        if known is not None and known.episode_id not in claimed_episode_ids:
            matched_by_number[episode_number] = known
            claimed_episode_ids.add(known.episode_id)

    return matched_by_number


def _index_known_episodes(
    known_episodes: Sequence[EpisodeReference],
) -> tuple[dict[tuple[str, int, str], EpisodeReference], dict[int, EpisodeReference]]:
    """Index known episodes by ``PageKey`` tuple and by ``episode_number``.

    Args:
        known_episodes: Previously stored ``EpisodeReference`` messages.

    Returns:
        Tuple of ``(by_page_key, by_number)`` lookup dictionaries.
    """
    by_page_key: dict[tuple[str, int, str], EpisodeReference] = {}
    by_number: dict[int, EpisodeReference] = {}
    for known in known_episodes:
        by_number.setdefault(known.episode_number, known)
        for page_source in known.page_sources:
            key = page_source.page_key
            by_page_key[(key.provider, key.entity_type, key.address)] = known
    return by_page_key, by_number


def _collect_provider_episodes(
    source_pages: Sequence[StoredPage],
    page_urls: dict[tuple[str, str], str],
) -> tuple[
    dict[int, dict[str, EpisodeListEntry]],
    dict[int, dict[tuple[str, int, str], PageSource]],
]:
    """Collect episode list entries and pages with their URLs grouped by episode number.

    A page several lists name keeps the longest URL, so a URL with a slug wins.

    Args:
        source_pages: Stored anime pages for the work.
        page_urls: Mapping from ``(provider, address)`` to canonical anime URL.

    Returns:
        Tuple of ``(grouped_entries, grouped_sources)`` keyed by episode number.
    """
    grouped_entries: dict[int, dict[str, EpisodeListEntry]] = {}
    grouped_sources: dict[int, dict[tuple[str, int, str], PageSource]] = {}
    for page in source_pages:
        if not is_mappable_stored_page(page):
            continue
        provider = page.page_key.provider
        page_url = page_urls.get((provider, page.page_key.address), "")
        for listed_episode in _list_provider_episodes(page, page_url):
            raw_number = listed_episode.episode.get("episode_number")
            if raw_number is None:
                continue
            ep_number = int(raw_number)
            grouped_entries.setdefault(ep_number, {})[provider] = (
                _to_episode_list_entry(provider, listed_episode.episode)
            )
            source_map = grouped_sources.setdefault(ep_number, {})
            for page_source in listed_episode.page_sources:
                key = page_source.page_key
                identity = (key.provider, key.entity_type, key.address)
                listed_source = source_map.get(identity)
                if listed_source is None or len(page_source.source) > len(
                    listed_source.source
                ):
                    source_map[identity] = page_source
    return grouped_entries, grouped_sources


def merge_episode_references(
    source_pages: Sequence[StoredPage],
    page_urls: dict[tuple[str, str], str],
    known_episodes: Sequence[EpisodeReference],
) -> list[EpisodeReference]:
    """Build sorted ``EpisodeReference`` messages from stored anime pages.

    Args:
        source_pages: Stored anime pages for the work.
        page_urls: Mapping from ``(provider, address)`` to canonical anime URL.
        known_episodes: Previously known episodes for ID preservation.

    Returns:
        Sorted list of ``EpisodeReference`` messages with deterministic ordering.
    """
    by_page_key, by_number = _index_known_episodes(known_episodes)
    grouped_entries, grouped_sources = _collect_provider_episodes(
        source_pages, page_urls
    )

    sorted_numbers = sorted(grouped_entries)
    matched_by_number = _match_known_episodes(
        sorted_numbers, grouped_sources, by_page_key, by_number
    )

    references: list[EpisodeReference] = []
    for episode_number in sorted_numbers:
        sorted_entries = sorted(
            grouped_entries[episode_number].values(),
            key=lambda entry: entry.provider,
        )
        sorted_sources = [
            grouped_sources[episode_number][identity]
            for identity in sorted(grouped_sources.get(episode_number, {}))
        ]
        known_match = matched_by_number.get(episode_number)
        episode_id = (
            known_match.episode_id if known_match is not None else generate_entity_id()
        )
        references.append(
            EpisodeReference(
                episode_id=episode_id,
                episode_number=episode_number,
                title=_pick_episode_title(sorted_entries, known_match),
                page_sources=sorted_sources,
                list_entries=sorted_entries,
            )
        )

    references.sort(key=lambda episode_ref: episode_ref.episode_id)
    return references
