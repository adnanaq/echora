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
from enrichment.pipeline.page_sources import (
    build_page_sources,
    build_stored_page_key,
    collect_live_page_keys,
    is_mappable_stored_page,
    read_page_urls,
    validate_one_page_per_provider,
)
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
    PageSource,
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
    if not is_mappable_stored_page(page):
        return None
    provider = page.page_key.provider
    responses: Sequence[PageResponse] = page.responses

    mappers: dict[str, Callable[[], dict[str, Any]]] = {
        "anilist": lambda: map_anilist_anime_responses(responses),
        "kitsu": lambda: map_kitsu_anime_responses(responses),
        "mal": lambda: map_mal_anime_responses(responses),
        "anime_planet": lambda: map_animeplanet_anime_responses(
            responses, slug=page.page_key.address
        ),
        "anisearch": lambda: map_anisearch_anime_responses(responses),
        "anidb": lambda: map_anidb_anime_responses(responses),
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
                    target_page = build_stored_page_key(resolve_page_key(str(url)))
                    if (
                        target_page is None
                        or target_page.entity_type != anime_pb2.ENTITY_TYPE_ANIME
                    ):
                        continue
                    dedup_key = (
                        provider,
                        proto_rel_type,
                        target_page.provider,
                        target_page.entity_type,
                        target_page.address,
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


def _build_linked_page_sources(
    source_pages: Sequence[StoredPage],
    partial_source_pages: Sequence[PageKey],
    records: dict[str, dict[str, Any]],
) -> tuple[list[PageSource], list[PageSource]]:
    """Build the URL and title chosen for each linked source and partial-source page.

    ``build_page_sources`` chooses each URL from every URL the mapped provider
    records report, falling back to the provider's canonical anime page URL, and
    takes each page's title from its own provider record's ``title``.

    Args:
        source_pages: Linked full source pages for the work.
        partial_source_pages: Linked partial source page keys for the work.
        records: Mapped canonical Anime dicts keyed by provider.

    Returns:
        Tuple of ``(source_page_sources, partial_page_sources)``, each in page order.
    """
    source_page_sources = build_page_sources(
        collect_live_page_keys(source_pages),
        records,
        page_key_to_anime_url,
        "title",
    )
    partial_page_sources = build_page_sources(
        partial_source_pages, records, page_key_to_anime_url, "title"
    )
    return source_page_sources, partial_page_sources


def _latest_generated_synopsis(generated_texts: Sequence[GeneratedText]) -> str | None:
    """Return the newest generated synopsis text by ``written_at`` timestamp.

    An empty newest synopsis wins over older synopses and leaves synopsis unset.

    Args:
        generated_texts: Generated texts passed into the merge request.

    Returns:
        Latest non-empty synopsis string, or ``None`` when absent or cleared.
    """
    candidates = [
        generated_text
        for generated_text in generated_texts
        if generated_text.field_path == "synopsis"
    ]
    if not candidates:
        return None
    latest = max(
        candidates,
        key=lambda generated_text: (
            generated_text.written_at.seconds,
            generated_text.written_at.nanos,
        ),
    )
    return latest.text or None


def _map_source_anime_records(request: MergeAnimeRequest) -> dict[str, dict[str, Any]]:
    """Map every mappable source page on ``request`` into provider records.

    Args:
        request: Incoming ``MergeAnimeRequest`` protobuf message.

    Returns:
        Mapping from provider slug to canonical Anime dict.

    Raises:
        ValueError: When a provider has more than one source page, or no source page
            yields a mappable record.
    """
    validate_one_page_per_provider(
        (page.page_key.provider for page in request.source_pages), "source_pages"
    )
    records: dict[str, dict[str, Any]] = {}
    for page in request.source_pages:
        mapped = _map_stored_anime_page(page)
        if mapped is not None:
            records[page.page_key.provider] = mapped
    if not records:
        error_message = f"No mappable source pages for anime {request.anime_id!r}"
        raise ValueError(error_message)
    return records


def merge_anime_pages(request: MergeAnimeRequest) -> MergeAnimeResponse:
    """Map stored pages, merge into one ``Anime``, and apply overrides.

    Args:
        request: Incoming ``MergeAnimeRequest`` protobuf message.

    Returns:
        ``MergeAnimeResponse`` containing the merged ``Anime``, sorted
        ``EpisodeReference`` list, sorted ``RelationClaim`` list,
        ``provider_synopses_hash``, and the URL chosen for each linked source page.
    """
    records = _map_source_anime_records(request)
    provider_synopses_hash = compute_provider_synopses_hash(records)
    relation_claims = _extract_relation_claims(records)
    records_without_synopsis = {
        provider: {k: v for k, v in record.items() if k != "synopsis"}
        for provider, record in records.items()
    }
    merged = merge_provider_records(
        records_without_synopsis,
        offline_data=None,
        anime_id=request.anime_id,
    )
    source_page_sources, partial_page_sources = _build_linked_page_sources(
        request.source_pages,
        request.partial_source_pages,
        records,
    )
    partial_sources = read_page_urls(partial_page_sources)
    merged["id"] = request.anime_id
    merged["sources"] = read_page_urls(source_page_sources)
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
    page_urls = {
        (page.page_key.provider, page.page_key.address): page.source
        for page in source_page_sources
    }
    episodes = merge_episode_references(
        request.source_pages, page_urls, request.known_episodes
    )
    return MergeAnimeResponse(
        anime=anime_proto,
        episodes=episodes,
        relation_claims=relation_claims,
        provider_synopses_hash=provider_synopses_hash,
        page_sources=source_page_sources,
    )
