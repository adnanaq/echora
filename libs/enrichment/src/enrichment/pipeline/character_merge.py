"""Merge a character from its stored provider pages and field corrections."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from enrichment.pipeline.anime_field_corrections import apply_field_corrections
from enrichment.pipeline.cast_proto_mapper import collect_voice_actors
from enrichment.pipeline.page_sources import (
    build_page_sources,
    collect_live_page_keys,
    is_mappable_stored_page,
    rank_provider_priority,
    read_page_urls,
    validate_one_page_per_provider,
)
from enrichment.pipeline.proto_enum_maps import CHARACTER_ROLE_MAP
from enrichment.pipeline.relationship_merger import normalize_title
from enrichment.sources.anidb.anidb_mapper import map_anidb_character_responses
from enrichment.sources.anilist.anilist_character_page_mapper import (
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
from networkx.utils import UnionFind
from shared_proto.v1 import anime_pb2
from shared_proto.v1.merge_pb2 import PageKey, PageResponse, PageSource, StoredPage
from work_matching import read_page_provider, resolve_page_key

_CHARACTER_URL_TEMPLATES: dict[str, str] = {
    "mal": "https://myanimelist.net/character/{address}",
    "anilist": "https://anilist.co/character/{address}",
    "kitsu": "https://kitsu.app/characters/{address}",
    "anidb": "https://anidb.net/character/{address}",
    "anime_planet": "https://www.anime-planet.com/characters/{address}",
    "anisearch": "https://www.anisearch.com/character/{address}",
}

_CHARACTER_PAGE_MAPPERS: dict[
    str, Callable[[Sequence[PageResponse]], dict[str, Any]]
] = {
    "mal": map_mal_character_responses,
    "anilist": map_anilist_character_page_responses,
    "kitsu": map_kitsu_character_page_responses,
    "anidb": map_anidb_character_responses,
    "anime_planet": map_animeplanet_character_responses,
    "anisearch": map_anisearch_character_responses,
}

_ATTRIBUTE_KEY_ALIASES: dict[str, str] = {
    "birthdate": "date_of_birth",
    "birthday": "date_of_birth",
}

_CONCRETE_CHARACTER_ROLES: frozenset[str] = frozenset(
    {"MAIN", "SUPPORTING", "BACKGROUND"}
)


def _build_character_url(page_key: PageKey) -> str:
    """Build a canonical character URL from a character ``PageKey``.

    Args:
        page_key: Character ``PageKey`` message.

    Returns:
        Canonical URL string, or empty string for an unknown provider.
    """
    template = _CHARACTER_URL_TEMPLATES.get(page_key.provider)
    return template.format(address=page_key.address) if template else ""


def _normalize_attribute_key(raw_key: str) -> str:
    """Normalize a character attribute key to canonical snake_case.

    Args:
        raw_key: Raw provider attribute key such as ``"blood-type"`` or ``"birthdate"``.

    Returns:
        Canonical snake_case key string, or empty string when blank.
    """
    normalized = raw_key.strip().lower().replace("-", "_").replace(" ", "_")
    return _ATTRIBUTE_KEY_ALIASES.get(normalized, normalized)


def _map_stored_character_page(stored_page: StoredPage) -> dict[str, Any] | None:
    """Map one stored character ``StoredPage`` into a canonical Character dict.

    Args:
        stored_page: Stored provider character page with raw responses.

    Returns:
        Canonical Character dict, or ``None`` when the page has no mappable responses.
    """
    if not is_mappable_stored_page(stored_page):
        return None
    mapper = _CHARACTER_PAGE_MAPPERS.get(stored_page.page_key.provider)
    return mapper(stored_page.responses) if mapper is not None else None


def _rank_character_records(
    stored_pages: Sequence[StoredPage],
) -> list[tuple[str, dict[str, Any]]]:
    """Map stored character pages and sort records by provider priority.

    Args:
        stored_pages: Sequence of stored character pages.

    Returns:
        List of ``(provider, record)`` pairs ordered by provider priority.

    Raises:
        ValueError: When a provider has more than one stored page.
    """
    validate_one_page_per_provider(
        (stored_page.page_key.provider for stored_page in stored_pages), "pages"
    )
    records_by_provider: dict[str, dict[str, Any]] = {}
    for stored_page in stored_pages:
        mapped_record = _map_stored_character_page(stored_page)
        if mapped_record is not None:
            records_by_provider[stored_page.page_key.provider] = mapped_record
    return sorted(
        records_by_provider.items(),
        key=lambda provider_entry: rank_provider_priority(provider_entry[0]),
    )


def _collect_unique_strings(
    ranked_records: Sequence[tuple[str, Mapping[str, Any]]],
    field_name: str,
) -> list[str]:
    """Collect and sort unique non-empty strings across ranked character records.

    Args:
        ranked_records: Ranked ``(provider, record)`` pairs.
        field_name: List field name on the character dict.

    Returns:
        Sorted list of unique non-empty stripped strings.
    """
    unique_values = {
        cleaned
        for _, provider_record in ranked_records
        for raw_value in (provider_record.get(field_name) or [])
        if (cleaned := str(raw_value).strip())
    }
    return sorted(unique_values)


def _collect_group_provider_keys(
    entry_indices: Sequence[int],
    work_keys_by_index: Sequence[list[str]],
) -> dict[str, set[str]]:
    """Collect resolved work page keys grouped by provider for a cluster of entries.

    Args:
        entry_indices: Indices of entries belonging to one union-find cluster.
        work_keys_by_index: Resolved work page keys per entry index.

    Returns:
        Mapping from provider slug to the set of resolved work keys in the cluster.
    """
    provider_keys: dict[str, set[str]] = {}
    for entry_index in entry_indices:
        for work_key in work_keys_by_index[entry_index]:
            provider = read_page_provider(work_key)
            if provider is not None:
                provider_keys.setdefault(provider, set()).add(work_key)
    return provider_keys


def _has_same_provider_conflict(
    left_keys: Mapping[str, set[str]],
    right_keys: Mapping[str, set[str]],
) -> bool:
    """Check whether two work clusters hold distinct work keys for the same provider.

    Args:
        left_keys: Provider-to-work-key sets for the first cluster.
        right_keys: Provider-to-work-key sets for the second cluster.

    Returns:
        ``True`` when any shared provider maps to disjoint work keys.
    """
    return any(
        provider in right_keys and left_keys[provider].isdisjoint(right_keys[provider])
        for provider in left_keys
    )


def _union_title_matching_clusters(
    entries: Sequence[dict[str, Any]],
    work_keys_by_index: Sequence[list[str]],
    union_find: UnionFind,
) -> None:
    """Merge title-matching ography clusters that have no same-provider conflict.

    Args:
        entries: Flat sequence of raw provider ography entry dicts.
        work_keys_by_index: Resolved work page keys per entry index.
        union_find: Mutable union-find structure updated in place.
    """
    title_clusters: dict[str, list[int]] = {}
    for entry_index, entry in enumerate(entries):
        raw_title = str(entry.get("title") or "").strip()
        normalized_title = normalize_title(raw_title) if raw_title else ""
        if not normalized_title:
            continue
        current_root = union_find[entry_index]
        current_members = [
            idx for idx in range(len(entries)) if union_find[idx] == current_root
        ]
        current_keys = _collect_group_provider_keys(current_members, work_keys_by_index)
        candidate_roots = title_clusters.setdefault(normalized_title, [])
        for candidate_root in candidate_roots:
            resolved_candidate = union_find[candidate_root]
            if resolved_candidate == current_root:
                break
            candidate_members = [
                idx
                for idx in range(len(entries))
                if union_find[idx] == resolved_candidate
            ]
            candidate_keys = _collect_group_provider_keys(
                candidate_members, work_keys_by_index
            )
            if not _has_same_provider_conflict(current_keys, candidate_keys):
                union_find.union(resolved_candidate, current_root)
                break
        else:
            candidate_roots.append(current_root)


def _group_ography_entries(
    ranked_records: Sequence[tuple[str, Mapping[str, Any]]],
    field_name: str,
) -> list[list[dict[str, Any]]]:
    """Group provider ography entries representing the same work via union-find.

    Pass one merges entries sharing any resolved work page key, handling
    multi-hop bridging entries. Pass two merges title-matching clusters only
    when neither cluster holds a conflicting page key from the same provider.

    Args:
        ranked_records: Ranked ``(provider, record)`` pairs.
        field_name: ``"animeography"`` or ``"mangaography"``.

    Returns:
        List of grouped ography entry lists in first-seen priority order.
    """
    entries = [
        ography_entry
        for _, provider_record in ranked_records
        for ography_entry in (provider_record.get(field_name) or [])
    ]
    if not entries:
        return []

    work_keys_by_index = [
        [
            resolve_page_key(str(source_url))
            for source_url in (entry.get("sources") or [])
            if source_url
        ]
        for entry in entries
    ]
    union_find = UnionFind(range(len(entries)))

    first_index_by_work_key: dict[str, int] = {}
    for entry_index, work_keys in enumerate(work_keys_by_index):
        for work_key in work_keys:
            if work_key in first_index_by_work_key:
                union_find.union(first_index_by_work_key[work_key], entry_index)
            else:
                first_index_by_work_key[work_key] = entry_index

    _union_title_matching_clusters(entries, work_keys_by_index, union_find)

    grouped_by_root: dict[int, list[dict[str, Any]]] = {}
    for entry_index, entry in enumerate(entries):
        grouped_by_root.setdefault(union_find[entry_index], []).append(entry)
    return list(grouped_by_root.values())


def _merge_ography(
    ranked_records: Sequence[tuple[str, Mapping[str, Any]]],
    field_name: str,
) -> list[anime_pb2.Ography]:
    """Merge ``animeography`` or ``mangaography`` across providers by work identity.

    Role tie-breaking selects the first concrete role (``MAIN``, ``SUPPORTING``,
    or ``BACKGROUND``) seen in ``PROVIDER_PRIORITY`` order; ``UNKNOWN`` never
    overrides a concrete role and is emitted only when no provider reports one.

    Args:
        ranked_records: Ranked ``(provider, record)`` pairs.
        field_name: ``"animeography"`` or ``"mangaography"``.

    Returns:
        Sorted list of merged ``anime_pb2.Ography`` messages.
    """
    merged_entries: list[anime_pb2.Ography] = []
    for member_entries in _group_ography_entries(ranked_records, field_name):
        title = next(
            (
                str(member["title"]).strip()
                for member in member_entries
                if member.get("title")
            ),
            "",
        )
        concrete_roles = [
            role_text
            for member in member_entries
            if (role_text := str(member.get("role") or "").upper())
            in _CONCRETE_CHARACTER_ROLES
        ]
        resolved_role = concrete_roles[0] if concrete_roles else "UNKNOWN"
        source_urls = sorted(
            {
                str(source_url)
                for member in member_entries
                for source_url in (member.get("sources") or [])
                if source_url
            }
        )
        merged_entries.append(
            anime_pb2.Ography(
                title=title,
                role=CHARACTER_ROLE_MAP.get(
                    resolved_role, anime_pb2.CHARACTER_ROLE_UNKNOWN
                ),
                sources=source_urls,
            )
        )
    merged_entries.sort(
        key=lambda ography: (ography.title, ography.role, tuple(ography.sources))
    )
    return merged_entries


def _build_character_page_sources(
    request: MergeCharacterRequest,
    ranked_records: Sequence[tuple[str, Mapping[str, Any]]],
) -> list[PageSource]:
    """Build the URL and title chosen for each live linked character page.

    ``build_page_sources`` chooses each URL from every URL the mapped provider
    records report, falling back to the provider's canonical character page URL, and
    takes each page's title from its own provider record's ``name``.

    Args:
        request: Incoming ``MergeCharacterRequest`` message.
        ranked_records: Ranked ``(provider, record)`` pairs.

    Returns:
        One ``PageSource`` per live linked page with a URL or title, in page order.
    """
    return build_page_sources(
        collect_live_page_keys(request.pages),
        dict(ranked_records),
        _build_character_url,
        "name",
    )


def _populate_character_maps(
    character_message: anime_pb2.Character,
    ranked_records: Sequence[tuple[str, Mapping[str, Any]]],
) -> None:
    """Populate normalized ``attributes`` and ``spoilers`` maps by provider priority.

    Attribute keys are normalized to canonical snake_case with known synonyms
    collapsed, and entries with blank or whitespace-only keys or values are
    omitted.

    Args:
        character_message: Target ``anime_pb2.Character`` message mutated in place.
        ranked_records: Ranked ``(provider, record)`` pairs.
    """
    for _, provider_record in ranked_records:
        for raw_key, raw_value in (provider_record.get("attributes") or {}).items():
            normalized_key = _normalize_attribute_key(str(raw_key))
            cleaned_value = str(raw_value).strip() if raw_value is not None else ""
            if (
                normalized_key
                and cleaned_value
                and normalized_key not in character_message.attributes
            ):
                character_message.attributes[normalized_key] = cleaned_value
        for raw_key, raw_value in (provider_record.get("spoilers") or {}).items():
            cleaned_key = str(raw_key).strip()
            cleaned_value = str(raw_value).strip() if raw_value is not None else ""
            if (
                cleaned_key
                and cleaned_value
                and cleaned_key not in character_message.spoilers
            ):
                character_message.spoilers[cleaned_key] = cleaned_value


def _find_first_record_value(
    ranked_records: Sequence[tuple[str, Mapping[str, Any]]],
    field_name: str,
    *,
    allow_falsy: bool = False,
) -> Any:
    """Find the highest-priority provider record value for a scalar field.

    Args:
        ranked_records: Ranked ``(provider, record)`` pairs.
        field_name: Field name to inspect on each provider record.
        allow_falsy: When ``True``, accept falsy non-``None`` values such as ``0``.

    Returns:
        First matching field value in provider priority order, or ``None``.
    """
    for _, provider_record in ranked_records:
        candidate = provider_record.get(field_name)
        if candidate is not None and (allow_falsy or candidate):
            return candidate
    return None


def _build_character_proto(
    request: MergeCharacterRequest,
    ranked_records: Sequence[tuple[str, Mapping[str, Any]]],
    page_sources: Sequence[PageSource],
) -> anime_pb2.Character:
    """Assemble and deterministically sort a ``shared_proto.v1.Character`` message.

    Args:
        request: Incoming ``MergeCharacterRequest`` message.
        ranked_records: Ranked ``(provider, record)`` pairs.
        page_sources: URL of each linked character page.

    Returns:
        Populated ``anime_pb2.Character`` message.
    """
    character_name = _find_first_record_value(ranked_records, "name")
    character_message = anime_pb2.Character(
        id=request.character_id,
        entity_type=anime_pb2.ENTITY_TYPE_CHARACTER,
        name=str(character_name) if character_name is not None else "",
        traits=_collect_unique_strings(ranked_records, "traits"),
        images=_collect_unique_strings(ranked_records, "images"),
        name_variations=_collect_unique_strings(ranked_records, "name_variations"),
        nicknames=_collect_unique_strings(ranked_records, "nicknames"),
        sources=read_page_urls(page_sources),
        animeography=_merge_ography(ranked_records, "animeography"),
        mangaography=_merge_ography(ranked_records, "mangaography"),
        voice_actors=collect_voice_actors(ranked_records),
    )
    if (
        description := _find_first_record_value(ranked_records, "description")
    ) is not None:
        character_message.description = str(description)
    if (
        native_name := _find_first_record_value(ranked_records, "name_native")
    ) is not None:
        character_message.name_native = str(native_name)
    if (
        favorites := _find_first_record_value(
            ranked_records, "favorites", allow_falsy=True
        )
    ) is not None:
        character_message.favorites = int(favorites)
    _populate_character_maps(character_message, ranked_records)
    return character_message


def merge_character_pages(request: MergeCharacterRequest) -> MergeCharacterResponse:
    """Merge a character's stored pages and field corrections into ``Character``.

    Args:
        request: Incoming ``MergeCharacterRequest`` protobuf message.

    Returns:
        ``MergeCharacterResponse`` holding the merged ``Character`` message and the URL
        of each linked page.
    """
    ranked_records = _rank_character_records(request.pages)
    page_sources = _build_character_page_sources(request, ranked_records)
    character_proto = _build_character_proto(request, ranked_records, page_sources)
    apply_field_corrections(character_proto, request.corrections)
    for field_name in ("traits", "images", "name_variations", "nicknames", "sources"):
        sorted_values = sorted(getattr(character_proto, field_name))
        del getattr(character_proto, field_name)[:]
        getattr(character_proto, field_name).extend(sorted_values)
    return MergeCharacterResponse(character=character_proto, page_sources=page_sources)
