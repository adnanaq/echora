"""Merge an anime work from its stored provider pages, overrides, and generated text."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Sequence
from typing import Any

from enrichment.pipeline.anime_field_corrections import apply_field_corrections
from enrichment.pipeline.anime_proto_mapper import (
    ANIME_RELATION_TYPE_MAP,
    build_anime_proto,
    sort_anime_proto_lists,
)
from enrichment.pipeline.episode_reference_merger import merge_episode_references
from enrichment.pipeline.metadata_merger import merge_provider_records
from enrichment.sources.anidb.anidb_mapper import map_anidb_anime_responses
from enrichment.sources.anilist.anilist_mapper import map_anilist_anime_responses
from enrichment.sources.anime_planet.animeplanet_mapper import (
    map_animeplanet_anime_responses,
)
from enrichment.sources.animeschedule.animeschedule_mapper import (
    map_animeschedule_anime_responses,
)
from enrichment.sources.anisearch.anisearch_mapper import map_anisearch_anime_responses
from enrichment.sources.kitsu.kitsu_mapper import map_kitsu_anime_responses
from enrichment.sources.mal.mal_mapper import map_mal_anime_responses
from enrichment_proto.v1.enrichment_service_pb2 import (
    MergeAnimeRequest,
    MergeAnimeResponse,
)
from shared_proto.v1 import anime_pb2
from shared_proto.v1.merge_pb2 import (
    GeneratedText,
    PageKey,
    PageResponse,
    RelationClaim,
    StoredPage,
)
from work_matching import resolve_page_key

logger = logging.getLogger(__name__)

_ANIME_URL_TEMPLATES: dict[str, str] = {
    "mal": "https://myanimelist.net/anime/{address}",
    "anilist": "https://anilist.co/anime/{address}",
    "kitsu": "https://kitsu.app/anime/{address}",
    "anidb": "https://anidb.net/anime/{address}",
    "anime_planet": "https://www.anime-planet.com/anime/{address}",
    "anisearch": "https://www.anisearch.com/anime/{address}",
    "animeschedule": "https://animeschedule.net/anime/{address}",
    "syoboi": "https://cal.syoboi.jp/tid/{address}",
    "livechart": "https://www.livechart.me/anime/{address}",
    "simkl": "https://simkl.com/anime/{address}",
    "animecountdown": "https://animecountdown.com/{address}",
    "ann": "https://www.animenewsnetwork.com/encyclopedia/anime.php?id={address}",
}


def page_key_to_anime_url(page_key: PageKey) -> str:
    """Build a canonical anime page URL from a ``PageKey``.

    Args:
        page_key: Provider page key identifying an anime page.

    Returns:
        Canonical URL string, or empty string for an unknown provider.
    """
    template = _ANIME_URL_TEMPLATES.get(page_key.provider)
    if template is None:
        return ""
    return template.format(address=page_key.address)


def _map_stored_anime_page(page: StoredPage) -> dict[str, Any] | None:
    """Map one stored anime ``StoredPage`` into a canonical Anime dict.

    Args:
        page: Stored provider page with raw responses.

    Returns:
        Canonical Anime dict, or ``None`` when the page has no mappable responses.
    """
    if page.is_dead or page.is_not_found or not page.responses:
        return None
    provider = page.page_key.provider
    fallback_url = page_key_to_anime_url(page.page_key)
    responses: Sequence[PageResponse] = page.responses

    mappers: dict[str, Callable[[], dict[str, Any]]] = {
        "anilist": lambda: map_anilist_anime_responses(responses),
        "kitsu": lambda: map_kitsu_anime_responses(responses),
        "mal": lambda: map_mal_anime_responses(responses, source_url=fallback_url),
        "anime_planet": lambda: map_animeplanet_anime_responses(
            responses, slug=page.page_key.address
        ),
        "anisearch": lambda: map_anisearch_anime_responses(
            responses, source_url=fallback_url
        ),
        "anidb": lambda: map_anidb_anime_responses(responses, anidb_url=fallback_url),
        "animeschedule": lambda: map_animeschedule_anime_responses(responses),
    }
    mapper = mappers.get(provider)
    if mapper is None:
        return None
    return mapper()


def compute_provider_synopses_hash(records: dict[str, dict[str, Any]]) -> str:
    """Compute a deterministic SHA-256 hash of all non-empty provider synopses.

    Args:
        records: Mapping of provider slug to mapped canonical Anime dict.

    Returns:
        Hex-encoded SHA-256 digest over sorted ``(provider, synopsis)`` pairs.
    """
    pairs = sorted(
        (provider, str(record["synopsis"]).strip())
        for provider, record in records.items()
        if record.get("synopsis") and str(record["synopsis"]).strip()
    )
    payload = json.dumps(pairs, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _extract_relation_claims(records: dict[str, dict[str, Any]]) -> list[RelationClaim]:
    """Extract and sort raw per-provider ``RelationClaim``s before merge resolution.

    Args:
        records: Mapping of provider slug to mapped canonical Anime dict.

    Returns:
        Deduplicated and deterministically sorted list of ``RelationClaim``s.
    """
    claims_by_key: dict[tuple[str, int, str, int, str], RelationClaim] = {}
    for provider, record in records.items():
        related_anime = record.get("related_anime") or {}
        for rel_type_raw, entries in related_anime.items():
            rel_key = (
                rel_type_raw.value
                if hasattr(rel_type_raw, "value")
                else str(rel_type_raw)
            ).upper()
            proto_rel_type = ANIME_RELATION_TYPE_MAP.get(
                rel_key, anime_pb2.ANIME_RELATION_TYPE_OTHER
            )
            for entry in entries or []:
                for url in entry.get("sources") or []:
                    resolved = resolve_page_key(str(url))
                    target_prov, _, rest = resolved.partition(":")
                    kind, _, target_addr = rest.partition(":")
                    if kind != "anime" or not target_prov or not target_addr:
                        continue
                    target_page = PageKey(
                        provider=target_prov,
                        entity_type=anime_pb2.ENTITY_TYPE_ANIME,
                        address=target_addr,
                    )
                    dedup_key = (
                        provider,
                        proto_rel_type,
                        target_prov,
                        anime_pb2.ENTITY_TYPE_ANIME,
                        target_addr,
                    )
                    claims_by_key.setdefault(
                        dedup_key,
                        RelationClaim(
                            provider=provider,
                            relation_type=proto_rel_type,
                            target_page=target_page,
                        ),
                    )
    return [claims_by_key[key] for key in sorted(claims_by_key)]


def _resolve_linked_urls(
    source_pages: Sequence[StoredPage],
    partial_source_pages: Sequence[PageKey],
    records: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str], dict[tuple[str, str], str]]:
    """Build ``sources``, ``partial_sources``, and per-page canonical URLs from links.

    Args:
        source_pages: Linked full source pages for the work.
        partial_source_pages: Linked partial source page keys for the work.
        records: Mapped canonical Anime dicts keyed by provider.

    Returns:
        Tuple of ``(sources, partial_sources, page_urls)``.
    """
    best_by_key: dict[str, str] = {}
    for record in records.values():
        for url in record.get("sources") or []:
            if not url:
                continue
            key = resolve_page_key(str(url))
            if len(str(url)) > len(best_by_key.get(key, "")):
                best_by_key[key] = str(url)

    page_urls: dict[tuple[str, str], str] = {}
    sources: set[str] = set()
    for page in source_pages:
        key_str = f"{page.page_key.provider}:anime:{page.page_key.address}"
        url = best_by_key.get(key_str) or page_key_to_anime_url(page.page_key)
        if url:
            sources.add(url)
            page_urls[(page.page_key.provider, page.page_key.address)] = url

    partial_sources: set[str] = set()
    for page_key in partial_source_pages:
        key_str = f"{page_key.provider}:anime:{page_key.address}"
        url = best_by_key.get(key_str) or page_key_to_anime_url(page_key)
        if url:
            partial_sources.add(url)

    return sorted(sources), sorted(partial_sources), page_urls


def _latest_generated_synopsis(generated_texts: Sequence[GeneratedText]) -> str | None:
    """Return the newest generated synopsis text by ``written_at`` timestamp.

    Args:
        generated_texts: Generated texts passed into the merge request.

    Returns:
        Latest synopsis string, or ``None`` when none is present.
    """
    candidates = [
        gt for gt in generated_texts if gt.field_path == "synopsis" and gt.text
    ]
    if not candidates:
        return None
    latest = max(
        candidates,
        key=lambda gt: (gt.written_at.seconds, gt.written_at.nanos),
    )
    return latest.text


def merge_anime_pages(request: MergeAnimeRequest) -> MergeAnimeResponse:
    """Map stored pages, merge into one ``Anime``, and apply overrides.

    Args:
        request: Incoming ``MergeAnimeRequest`` protobuf message.

    Returns:
        ``MergeAnimeResponse`` containing the merged ``Anime``, sorted
        ``EpisodeReference`` list, sorted ``RelationClaim`` list, and
        ``provider_synopses_hash``.
    """
    records: dict[str, dict[str, Any]] = {}
    for page in request.source_pages:
        mapped = _map_stored_anime_page(page)
        if mapped is not None:
            records[page.page_key.provider] = mapped

    provider_synopses_hash = compute_provider_synopses_hash(records)
    relation_claims = _extract_relation_claims(records)

    records_without_synopsis = {
        provider: {k: v for k, v in record.items() if k != "synopsis"}
        for provider, record in records.items()
    }
    merged = merge_provider_records(records_without_synopsis, offline_data=None)

    sources, partial_sources, page_urls = _resolve_linked_urls(
        request.source_pages,
        request.partial_source_pages,
        records,
    )
    merged["id"] = request.anime_id
    merged["sources"] = sources
    if partial_sources:
        merged["partial_sources"] = partial_sources
    else:
        merged.pop("partial_sources", None)

    generated_synopsis = _latest_generated_synopsis(request.generated_texts)
    if generated_synopsis is not None:
        merged["synopsis"] = generated_synopsis
    else:
        merged.pop("synopsis", None)

    anime_proto = build_anime_proto(merged)
    apply_field_corrections(anime_proto, request.corrections)
    sort_anime_proto_lists(anime_proto)

    episodes = merge_episode_references(
        request.source_pages,
        page_urls,
        request.known_episodes,
    )
    return MergeAnimeResponse(
        anime=anime_proto,
        episodes=episodes,
        relation_claims=relation_claims,
        provider_synopses_hash=provider_synopses_hash,
    )
