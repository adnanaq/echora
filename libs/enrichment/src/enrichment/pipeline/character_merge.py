"""Merge a character from its stored provider pages and field corrections."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from enrichment.pipeline.anime_field_corrections import apply_field_corrections
from enrichment.pipeline.relationship_merger import PROVIDER_PRIORITY, normalize_title
from enrichment.sources.anidb.anidb_mapper import map_anidb_character_responses
from enrichment.sources.anilist.anilist_mapper import (
    map_anilist_character_page_responses,
)
from enrichment.sources.anime_planet.animeplanet_mapper import (
    map_animeplanet_character_responses,
)
from enrichment.sources.anisearch.anisearch_mapper import (
    map_anisearch_character_responses,
)
from enrichment.sources.kitsu.kitsu_mapper import map_kitsu_character_page_responses
from enrichment.sources.mal.mal_mapper import map_mal_character_responses
from enrichment_proto.v1.enrichment_service_pb2 import (
    MergeCharacterRequest,
    MergeCharacterResponse,
)
from shared_proto.v1 import anime_pb2
from shared_proto.v1.merge_pb2 import PageKey, PageResponse, PageSource, StoredPage
from work_matching import resolve_page_key

_CHARACTER_URL_TEMPLATES: dict[str, str] = {
    "mal": "https://myanimelist.net/character/{address}",
    "anilist": "https://anilist.co/character/{address}",
    "kitsu": "https://kitsu.app/characters/{address}",
    "anidb": "https://anidb.net/character/{address}",
    "anime_planet": "https://www.anime-planet.com/characters/{address}",
    "anisearch": "https://www.anisearch.com/character/{address}",
}

_CHARACTER_ROLE_MAP: dict[str, anime_pb2.CharacterRole] = {
    "MAIN": anime_pb2.CHARACTER_ROLE_MAIN,
    "SUPPORTING": anime_pb2.CHARACTER_ROLE_SUPPORTING,
    "BACKGROUND": anime_pb2.CHARACTER_ROLE_BACKGROUND,
    "UNKNOWN": anime_pb2.CHARACTER_ROLE_UNKNOWN,
}


def _page_key_to_character_url(page_key: PageKey) -> str:
    """Build a canonical character URL from a character ``PageKey``.

    Args:
        page_key: Character ``PageKey`` message.

    Returns:
        Canonical URL string, or empty string for an unknown provider.
    """
    template = _CHARACTER_URL_TEMPLATES.get(page_key.provider)
    return template.format(address=page_key.address) if template else ""


def _map_stored_character_page(page: StoredPage) -> dict[str, Any] | None:
    """Map one stored character ``StoredPage`` into a canonical Character dict.

    Args:
        page: Stored provider character page with raw responses.

    Returns:
        Canonical Character dict, or ``None`` when the page has no mappable responses.
    """
    if page.is_dead or page.is_not_found or not page.responses:
        return None
    provider = page.page_key.provider
    fallback_url = _page_key_to_character_url(page.page_key)
    responses: Sequence[PageResponse] = page.responses
    mappers: dict[str, Callable[[], dict[str, Any]]] = {
        "mal": lambda: map_mal_character_responses(responses, source_url=fallback_url),
        "anilist": lambda: map_anilist_character_page_responses(responses),
        "kitsu": lambda: map_kitsu_character_page_responses(responses),
        "anidb": lambda: map_anidb_character_responses(
            responses, source_url=fallback_url
        ),
        "anime_planet": lambda: map_animeplanet_character_responses(
            responses, source_url=fallback_url
        ),
        "anisearch": lambda: map_anisearch_character_responses(
            responses, source_url=fallback_url
        ),
    }
    mapper = mappers.get(provider)
    return mapper() if mapper is not None else None


def _rank_records(
    pages: Sequence[StoredPage],
) -> list[tuple[str, dict[str, Any]]]:
    """Map stored character pages and sort by ``PROVIDER_PRIORITY``.

    Args:
        pages: Sequence of stored character pages.

    Returns:
        List of ``(provider, record)`` pairs ordered by provider trust.
    """
    records: dict[str, dict[str, Any]] = {}
    for page in pages:
        mapped = _map_stored_character_page(page)
        if mapped is not None:
            records[page.page_key.provider] = mapped
    return sorted(
        records.items(),
        key=lambda item: (
            PROVIDER_PRIORITY.index(item[0])
            if item[0] in PROVIDER_PRIORITY
            else len(PROVIDER_PRIORITY)
        ),
    )


def _union_strings(ranked: list[tuple[str, dict[str, Any]]], field: str) -> list[str]:
    """Union and sort unique non-empty strings across ranked character records.

    Args:
        ranked: Ranked ``(provider, record)`` pairs.
        field: List field name on the character dict.

    Returns:
        Sorted list of unique non-empty strings.
    """
    values = {
        str(item).strip()
        for _, rec in ranked
        for item in (rec.get(field) or [])
        if str(item).strip()
    }
    return sorted(values)


def _group_ography_entries(
    ranked: list[tuple[str, dict[str, Any]]],
    field: str,
) -> list[list[dict[str, Any]]]:
    """Group provider ography entries representing the same work.

    Args:
        ranked: Ranked ``(provider, record)`` pairs.
        field: ``"animeography"`` or ``"mangaography"``.

    Returns:
        List of grouped ography entry lists.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    key_by_title: dict[str, str] = {}
    unique_groups: list[list[dict[str, Any]]] = []
    for _, rec in ranked:
        for entry in rec.get(field) or []:
            title = str(entry.get("title") or "").strip()
            norm_title = normalize_title(title) if title else ""
            work_keys = [
                resolve_page_key(str(src))
                for src in (entry.get("sources") or [])
                if src
            ]
            group_key = next((k for k in work_keys if k in groups), None)
            if group_key is None and norm_title in key_by_title:
                group_key = key_by_title[norm_title]
            if group_key is None:
                group_key = work_keys[0] if work_keys else f"title:{norm_title}"
                groups[group_key] = []
                unique_groups.append(groups[group_key])
            if norm_title:
                key_by_title.setdefault(norm_title, group_key)
            for wk in work_keys:
                groups.setdefault(wk, groups[group_key])
            groups[group_key].append(entry)
    return unique_groups


def _merge_ography(
    ranked: list[tuple[str, dict[str, Any]]],
    field: str,
) -> list[anime_pb2.Ography]:
    """Merge ``animeography`` or ``mangaography`` across providers by work identity.

    Args:
        ranked: Ranked ``(provider, record)`` pairs.
        field: ``"animeography"`` or ``"mangaography"``.

    Returns:
        Sorted list of merged ``anime_pb2.Ography`` messages.
    """
    merged: list[anime_pb2.Ography] = []
    for members in _group_ography_entries(ranked, field):
        title = next((str(m["title"]).strip() for m in members if m.get("title")), "")
        concrete_roles = [
            str(m.get("role") or "").upper()
            for m in members
            if str(m.get("role") or "").upper() in {"MAIN", "SUPPORTING", "BACKGROUND"}
        ]
        resolved_role_str = concrete_roles[0] if concrete_roles else "UNKNOWN"
        sources = sorted(
            {str(src) for m in members for src in (m.get("sources") or []) if src}
        )
        merged.append(
            anime_pb2.Ography(
                title=title,
                role=_CHARACTER_ROLE_MAP.get(
                    resolved_role_str, anime_pb2.CHARACTER_ROLE_UNKNOWN
                ),
                sources=sources,
            )
        )
    merged.sort(key=lambda o: (o.title, o.role, tuple(o.sources)))
    return merged


def _build_character_page_sources(request: MergeCharacterRequest) -> list[PageSource]:
    """Build the canonical URL of each linked character page.

    Args:
        request: Incoming ``MergeCharacterRequest`` message.

    Returns:
        One ``PageSource`` per linked page whose provider has a character URL, in page
        order.
    """
    return [
        PageSource(page_key=page.page_key, source=url)
        for page in request.pages
        if (url := _page_key_to_character_url(page.page_key))
    ]


def _build_character_proto(
    request: MergeCharacterRequest,
    ranked: list[tuple[str, dict[str, Any]]],
    page_sources: Sequence[PageSource],
) -> anime_pb2.Character:
    """Assemble and deterministically sort a ``shared_proto.v1.Character`` message.

    Args:
        request: Incoming ``MergeCharacterRequest`` message.
        ranked: Ranked ``(provider, record)`` pairs.
        page_sources: URL of each linked character page.

    Returns:
        Populated ``anime_pb2.Character`` message.
    """
    name = next(
        (str(rec["name"]) for _, rec in ranked if rec.get("name")),
        "",
    )
    all_sources = sorted(
        {page_source.source for page_source in page_sources}
        | set(_union_strings(ranked, "sources"))
    )
    msg = anime_pb2.Character(
        id=request.character_id,
        entity_type=anime_pb2.ENTITY_TYPE_CHARACTER,
        name=name,
        traits=_union_strings(ranked, "traits"),
        images=_union_strings(ranked, "images"),
        name_variations=_union_strings(ranked, "name_variations"),
        nicknames=_union_strings(ranked, "nicknames"),
        sources=all_sources,
        animeography=_merge_ography(ranked, "animeography"),
        mangaography=_merge_ography(ranked, "mangaography"),
    )
    if desc := next(
        (rec["description"] for _, rec in ranked if rec.get("description")), None
    ):
        msg.description = str(desc)
    if native := next(
        (rec["name_native"] for _, rec in ranked if rec.get("name_native")), None
    ):
        msg.name_native = str(native)
    if (
        favs := next(
            (rec["favorites"] for _, rec in ranked if rec.get("favorites") is not None),
            None,
        )
    ) is not None:
        msg.favorites = int(favs)
    for _, rec in ranked:
        for key, val in (rec.get("attributes") or {}).items():
            if key and val and key not in msg.attributes:
                msg.attributes[str(key)] = str(val)
        for key, val in (rec.get("spoilers") or {}).items():
            if key and val and key not in msg.spoilers:
                msg.spoilers[str(key)] = str(val)
    return msg


def merge_character_pages(request: MergeCharacterRequest) -> MergeCharacterResponse:
    """Merge a character's stored pages and field corrections into ``Character``.

    Args:
        request: Incoming ``MergeCharacterRequest`` protobuf message.

    Returns:
        ``MergeCharacterResponse`` holding the merged ``Character`` message and the URL
        of each linked page.
    """
    ranked = _rank_records(request.pages)
    page_sources = _build_character_page_sources(request)
    character_proto = _build_character_proto(request, ranked, page_sources)
    apply_field_corrections(character_proto, request.corrections)
    for field_name in ("traits", "images", "name_variations", "nicknames", "sources"):
        sorted_vals = sorted(getattr(character_proto, field_name))
        del getattr(character_proto, field_name)[:]
        getattr(character_proto, field_name).extend(sorted_vals)
    return MergeCharacterResponse(character=character_proto, page_sources=page_sources)
