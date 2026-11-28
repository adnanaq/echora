"""Convert and merge episode cast, staff, and voice actor protobuf messages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from common.models.anime import CharacterRole
from enrichment.pipeline.proto_enum_maps import CHARACTER_ROLE_MAP
from enrichment.utils.text_utils import fold_for_comparison
from shared_proto.v1 import anime_pb2

_VOICE_ACTOR_OPTIONAL_FIELDS: tuple[str, ...] = (
    "native_name",
    "id",
    "language",
    "image",
    "biography",
    "birth_date",
    "blood_type",
)


@dataclass
class _VoiceActorAccumulator:
    """Mutable accumulator for merging one voice actor across providers."""

    name: str
    optional_fields: dict[str, str] = field(default_factory=dict)
    character_assignments: set[str] = field(default_factory=set)
    sources: set[str] = field(default_factory=set)


def resolve_character_role(raw_role: Any) -> anime_pb2.CharacterRole:
    """Map a canonical or provider character role value to ``CharacterRole``.

    Args:
        raw_role: Canonical ``CharacterRole`` member or raw role string.

    Returns:
        Matching ``anime_pb2.CharacterRole`` enum value, defaulting to
        ``CHARACTER_ROLE_UNKNOWN``.
    """
    role_text = (
        raw_role.value if isinstance(raw_role, CharacterRole) else str(raw_role or "")
    )
    normalized = role_text.strip().upper()
    return CHARACTER_ROLE_MAP.get(normalized, anime_pb2.CHARACTER_ROLE_UNKNOWN)


def _normalize_url_key(url: str) -> str:
    """Normalize a source URL for identity matching across providers.

    Args:
        url: Raw profile URL string.

    Returns:
        Trimmed lowercase URL without trailing slash.
    """
    return url.strip().rstrip("/").lower()


def _build_name_token_key(name: str) -> tuple[str, ...]:
    """Build an order-independent folded token key for a person name.

    Args:
        name: Raw person name in Western or family-first order.

    Returns:
        Sorted tuple of folded name tokens.
    """
    return tuple(sorted(fold_for_comparison(name).split()))


def _find_matching_voice_actor(
    accumulators: Sequence[_VoiceActorAccumulator],
    url_index: Mapping[str, int],
    name_index: Mapping[tuple[str, ...], list[int]],
    url_keys: Sequence[str],
    name_tokens: tuple[str, ...],
    language: str,
) -> int | None:
    """Locate an existing voice actor accumulator matching URL or folded name.

    Args:
        accumulators: Existing merged voice actor accumulators.
        url_index: Mapping from normalized source URL to accumulator index.
        name_index: Mapping from sorted folded name tokens to accumulator indices.
        url_keys: Normalized URLs on the incoming voice actor entry.
        name_tokens: Order-independent folded name tokens for the incoming entry.
        language: Normalized language code on the incoming entry, or empty string.

    Returns:
        Matching accumulator index, or ``None`` when no existing entry matches.
    """
    for url_key in url_keys:
        if url_key in url_index:
            return url_index[url_key]
    if not name_tokens:
        return None
    candidate_indices = name_index.get(name_tokens, ())
    for candidate_index in candidate_indices:
        existing_language = accumulators[candidate_index].optional_fields.get(
            "language", ""
        )
        if not language or not existing_language or language == existing_language:
            return candidate_index
    return None


def _absorb_voice_actor(
    accumulator: _VoiceActorAccumulator,
    raw_voice_actor: Mapping[str, Any],
) -> None:
    """Merge non-empty scalar and list fields into a voice actor accumulator.

    Args:
        accumulator: Existing voice actor accumulator updated in place.
        raw_voice_actor: Incoming raw voice actor mapping.
    """
    for field_name in _VOICE_ACTOR_OPTIONAL_FIELDS:
        raw_value = raw_voice_actor.get(field_name)
        cleaned = str(raw_value).strip() if raw_value is not None else ""
        if cleaned and field_name not in accumulator.optional_fields:
            accumulator.optional_fields[field_name] = cleaned
    for assignment in raw_voice_actor.get("character_assignments") or []:
        cleaned_assignment = str(assignment).strip()
        if cleaned_assignment:
            accumulator.character_assignments.add(cleaned_assignment)
    for source_url in raw_voice_actor.get("sources") or []:
        cleaned_url = str(source_url).strip()
        if cleaned_url:
            accumulator.sources.add(cleaned_url)


def merge_voice_actors(
    raw_voice_actors: Sequence[Mapping[str, Any]],
) -> list[anime_pb2.VoiceActor]:
    """Deduplicate and deterministically sort voice actors in priority order.

    Entries are matched by normalized source URL first, then by order-independent
    folded name tokens with compatible language. The first non-empty scalar value
    seen in input order wins, while ``sources`` and ``character_assignments`` are
    unioned and sorted.

    Args:
        raw_voice_actors: Priority-ordered sequence of raw voice actor dicts.

    Returns:
        Deterministically sorted list of ``anime_pb2.VoiceActor`` messages.
    """
    accumulators: list[_VoiceActorAccumulator] = []
    url_index: dict[str, int] = {}
    name_index: dict[tuple[str, ...], list[int]] = {}

    for raw_voice_actor in raw_voice_actors:
        name = str(raw_voice_actor.get("name") or "").strip()
        if not name:
            continue
        url_keys = [
            _normalize_url_key(str(url))
            for url in (raw_voice_actor.get("sources") or [])
            if str(url).strip()
        ]
        name_tokens = _build_name_token_key(name)
        language = str(raw_voice_actor.get("language") or "").strip()

        matched_index = _find_matching_voice_actor(
            accumulators, url_index, name_index, url_keys, name_tokens, language
        )
        if matched_index is None:
            matched_index = len(accumulators)
            accumulators.append(_VoiceActorAccumulator(name=name))
            if name_tokens:
                name_index.setdefault(name_tokens, []).append(matched_index)
        _absorb_voice_actor(accumulators[matched_index], raw_voice_actor)
        for url_key in url_keys:
            url_index.setdefault(url_key, matched_index)

    voice_actor_messages = [
        anime_pb2.VoiceActor(
            name=accumulator.name,
            **accumulator.optional_fields,
            character_assignments=sorted(accumulator.character_assignments),
            sources=sorted(accumulator.sources),
        )
        for accumulator in accumulators
    ]
    voice_actor_messages.sort(
        key=lambda actor: (actor.name, actor.language, tuple(actor.sources))
    )
    return voice_actor_messages


def collect_voice_actors(
    ranked_records: Sequence[tuple[str, Mapping[str, Any]]],
) -> list[anime_pb2.VoiceActor]:
    """Collect and merge voice actors across ranked provider character records.

    Args:
        ranked_records: ``(provider, record)`` pairs ordered by provider priority.

    Returns:
        Deterministically sorted list of merged ``anime_pb2.VoiceActor`` messages.
    """
    raw_voice_actors = [
        voice_actor
        for _, provider_record in ranked_records
        for voice_actor in (provider_record.get("voice_actors") or [])
        if isinstance(voice_actor, Mapping)
    ]
    return merge_voice_actors(raw_voice_actors)


def _absorb_episode_character(
    grouped: list[dict[str, Any]],
    index_by_key: dict[str, int],
    raw_character: Mapping[str, Any],
) -> None:
    """Insert or merge one raw episode character entry into the grouped list.

    Args:
        grouped: Mutable list of grouped episode character dicts.
        index_by_key: Lookup index from normalized URL or folded name to group index.
        raw_character: Incoming raw episode character dict from a provider record.
    """
    name = str(raw_character.get("name") or "").strip()
    if not name:
        return
    sources = [
        str(url).strip()
        for url in (raw_character.get("sources") or [])
        if str(url).strip()
    ]
    lookup_keys = [_normalize_url_key(url) for url in sources]
    if folded_name := fold_for_comparison(name):
        lookup_keys.append(f"name:{folded_name}")
    matched_index = next(
        (index_by_key[key] for key in lookup_keys if key in index_by_key), None
    )
    if matched_index is None:
        matched_index = len(grouped)
        grouped.append(
            {
                "name": name,
                "role": resolve_character_role(raw_character.get("role")),
                "sources": set(sources),
                "voice_actors": [],
            }
        )
    else:
        target = grouped[matched_index]
        if target["role"] == anime_pb2.CHARACTER_ROLE_UNKNOWN:
            target["role"] = resolve_character_role(raw_character.get("role"))
        target["sources"].update(sources)
    grouped[matched_index]["voice_actors"].extend(
        actor
        for actor in (raw_character.get("voice_actors") or [])
        if isinstance(actor, Mapping)
    )
    for key in lookup_keys:
        index_by_key.setdefault(key, matched_index)


def merge_episode_characters(
    ranked_records: Sequence[tuple[str, Mapping[str, Any]]],
) -> list[anime_pb2.EpisodeCharacter]:
    """Merge and sort episode character appearances across ranked episode records.

    Args:
        ranked_records: ``(provider, record)`` pairs ordered by provider priority.

    Returns:
        Deterministically sorted list of ``anime_pb2.EpisodeCharacter`` messages.
    """
    grouped: list[dict[str, Any]] = []
    index_by_key: dict[str, int] = {}
    for _, provider_record in ranked_records:
        for raw_character in provider_record.get("characters") or []:
            if isinstance(raw_character, Mapping):
                _absorb_episode_character(grouped, index_by_key, raw_character)

    episode_characters = [
        anime_pb2.EpisodeCharacter(
            name=entry["name"],
            role=entry["role"],
            sources=sorted(entry["sources"]),
            voice_actors=merge_voice_actors(entry["voice_actors"]),
        )
        for entry in grouped
    ]
    episode_characters.sort(
        key=lambda character: (character.name, character.role, tuple(character.sources))
    )
    return episode_characters


def merge_episode_staff(
    ranked_records: Sequence[tuple[str, Mapping[str, Any]]],
) -> list[anime_pb2.EpisodeStaff]:
    """Merge and sort episode staff credits across ranked episode records.

    Args:
        ranked_records: ``(provider, record)`` pairs ordered by provider priority.

    Returns:
        Deterministically sorted list of ``anime_pb2.EpisodeStaff`` messages.
    """
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for _, provider_record in ranked_records:
        for raw_staff in provider_record.get("staff") or []:
            if not isinstance(raw_staff, Mapping):
                continue
            name = str(raw_staff.get("name") or "").strip()
            if not name:
                continue
            role = str(raw_staff.get("role") or "").strip()
            dedup_key = (fold_for_comparison(name) or name.casefold(), role.casefold())
            sources = {
                str(url).strip()
                for url in (raw_staff.get("sources") or [])
                if str(url).strip()
            }
            if dedup_key not in grouped:
                grouped[dedup_key] = {"name": name, "role": role, "sources": sources}
            else:
                grouped[dedup_key]["sources"].update(sources)

    episode_staff = [
        anime_pb2.EpisodeStaff(
            name=entry["name"],
            role=entry["role"],
            sources=sorted(entry["sources"]),
        )
        for entry in grouped.values()
    ]
    episode_staff.sort(
        key=lambda staff_member: (
            staff_member.name,
            staff_member.role,
            tuple(staff_member.sources),
        )
    )
    return episode_staff
