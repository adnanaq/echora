"""Extract and match episode references from stored anime pages."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from urllib.parse import urlparse

from common.utils.id_generation import generate_entity_id
from enrichment.pipeline.anime_proto_mapper import to_proto_timestamp
from enrichment.pipeline.relationship_merger import PROVIDER_PRIORITY
from enrichment.sources.anidb.anidb_mapper import map_anidb_anime_episode_responses
from enrichment.sources.anisearch.anisearch_mapper import (
    map_anisearch_anime_episode_responses,
)
from enrichment.sources.kitsu.kitsu_mapper import map_kitsu_anime_episode_responses
from shared_proto.v1.anime_pb2 import ENTITY_TYPE_EPISODE
from shared_proto.v1.merge_pb2 import (
    EpisodeListEntry,
    EpisodeReference,
    PageKey,
    StoredPage,
)


def _parse_episode_page_key(url: str) -> PageKey | None:
    """Parse an episode URL into a ``PageKey`` with ``ENTITY_TYPE_EPISODE``.

    Args:
        url: Canonical provider episode URL.

    Returns:
        Parsed ``PageKey``, or ``None`` when the URL is not an episode URL.
    """
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower().removeprefix("www.")
    segments = [seg for seg in parsed.path.split("/") if seg]
    if host == "anidb.net" and len(segments) >= 2 and segments[0] == "episode":
        return PageKey(
            provider="anidb",
            entity_type=ENTITY_TYPE_EPISODE,
            address=segments[1],
        )
    if host in {"kitsu.app", "kitsu.io"} and "episodes" in segments:
        idx = segments.index("episodes")
        if idx + 1 < len(segments):
            return PageKey(
                provider="kitsu",
                entity_type=ENTITY_TYPE_EPISODE,
                address=segments[idx + 1],
            )
    return None


def _extract_provider_episodes(
    page: StoredPage,
    page_url: str,
) -> list[dict[str, Any]]:
    """Map episode entries listed on a single stored anime ``StoredPage``.

    Args:
        page: Stored anime page with raw responses.
        page_url: Canonical URL of the anime page.

    Returns:
        List of canonical Episode dicts extracted from the page.
    """
    provider = page.page_key.provider
    if provider == "anidb":
        return map_anidb_anime_episode_responses(page.responses)
    if provider == "kitsu":
        return map_kitsu_anime_episode_responses(page.responses)
    if provider == "anisearch":
        return map_anisearch_anime_episode_responses(
            page.responses,
            source_url=f"{page_url}/episodes" if page_url else "",
        )
    return []


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


def _match_known_episode(
    page_keys: Sequence[PageKey],
    episode_number: int,
    by_page_key: dict[tuple[str, int, str], EpisodeReference],
    by_number: dict[int, EpisodeReference],
) -> EpisodeReference | None:
    """Find a matching known episode by page key first, then episode number.

    Args:
        page_keys: Episode ``PageKey``s extracted from stored pages.
        episode_number: Episode number.
        by_page_key: Index of known episodes by ``(provider, entity_type, address)``.
        by_number: Index of known episodes by ``episode_number``.

    Returns:
        Matching ``EpisodeReference`` if found, otherwise ``None``.
    """
    for key in page_keys:
        if (
            match := by_page_key.get((key.provider, key.entity_type, key.address))
        ) is not None:
            return match
    return by_number.get(episode_number)


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
        for key in known.page_keys:
            by_page_key[(key.provider, key.entity_type, key.address)] = known
    return by_page_key, by_number


def _collect_provider_episodes(
    source_pages: Sequence[StoredPage],
    page_urls: dict[tuple[str, str], str],
) -> tuple[
    dict[int, dict[str, EpisodeListEntry]],
    dict[int, dict[tuple[str, int, str], PageKey]],
]:
    """Collect episode list entries and page keys grouped by episode number.

    Args:
        source_pages: Stored anime pages for the work.
        page_urls: Mapping from ``(provider, address)`` to canonical anime URL.

    Returns:
        Tuple of ``(grouped_entries, grouped_keys)`` keyed by episode number.
    """
    grouped_entries: dict[int, dict[str, EpisodeListEntry]] = {}
    grouped_keys: dict[int, dict[tuple[str, int, str], PageKey]] = {}
    for page in source_pages:
        if page.is_dead or page.is_not_found or not page.responses:
            continue
        provider = page.page_key.provider
        page_url = page_urls.get((provider, page.page_key.address), "")
        for ep_dict in _extract_provider_episodes(page, page_url):
            raw_number = ep_dict.get("episode_number")
            if raw_number is None:
                continue
            ep_number = int(raw_number)
            grouped_entries.setdefault(ep_number, {})[provider] = (
                _to_episode_list_entry(provider, ep_dict)
            )
            key_map = grouped_keys.setdefault(ep_number, {})
            for url in ep_dict.get("sources") or []:
                if (parsed_key := _parse_episode_page_key(str(url))) is not None:
                    key_map[
                        (
                            parsed_key.provider,
                            parsed_key.entity_type,
                            parsed_key.address,
                        )
                    ] = parsed_key
    return grouped_entries, grouped_keys


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
    grouped_entries, grouped_keys = _collect_provider_episodes(source_pages, page_urls)

    references: list[EpisodeReference] = []
    for ep_number in sorted(grouped_entries):
        sorted_entries = sorted(
            grouped_entries[ep_number].values(),
            key=lambda entry: entry.provider,
        )
        sorted_keys = [
            grouped_keys[ep_number][k] for k in sorted(grouped_keys.get(ep_number, {}))
        ]
        known_match = _match_known_episode(
            sorted_keys, ep_number, by_page_key, by_number
        )
        episode_id = (
            known_match.episode_id if known_match is not None else generate_entity_id()
        )
        references.append(
            EpisodeReference(
                episode_id=episode_id,
                episode_number=ep_number,
                title=_pick_episode_title(sorted_entries, known_match),
                page_keys=sorted_keys,
                list_entries=sorted_entries,
            )
        )

    references.sort(key=lambda ep: ep.episode_id)
    return references
