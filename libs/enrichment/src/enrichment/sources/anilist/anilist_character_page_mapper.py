"""Map stored AniList character page GraphQL responses into canonical Character dicts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from common.models.anime import Character, CharacterRole, Ography, VoiceActor
from enrichment.sources.base.page_fetcher import (
    collect_paginated_responses,
    index_responses_by_name,
    parse_json_response,
    require_json_object,
)
from shared_proto.v1.merge_pb2 import PageResponse


def _collect_anilist_character_media_edges(
    char_data: dict[str, Any],
    responses: Sequence[PageResponse],
) -> list[dict[str, Any]]:
    """Collect all AniList character media edges from main and paginated responses.

    Args:
        char_data: Main GraphQL ``Character`` object.
        responses: Stored PageResponse sequence containing optional ``media_page_*``.

    Returns:
        Combined list of media edge objects across all stored pages.
    """
    media_obj = require_json_object(char_data.get("media"), "AniList character 'media'")
    edges: list[dict[str, Any]] = list(media_obj.get("edges") or [])
    for page_payload in collect_paginated_responses(responses, "media_page_"):
        page_data = require_json_object(
            page_payload.get("data"), "AniList media_page 'data'"
        )
        page_char = require_json_object(
            page_data.get("Character"), "AniList media_page 'Character'"
        )
        page_media = require_json_object(
            page_char.get("media"), "AniList media_page 'media'"
        )
        edges.extend(page_media.get("edges") or [])
    return edges


def _extract_anilist_character_media_ography(
    edges: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract animeography and mangaography from AniList character media edges.

    Args:
        edges: Combined GraphQL media edges for the character.

    Returns:
        Tuple of ``(animeography, mangaography)`` normalized dicts.
    """
    animeography: list[dict[str, Any]] = []
    mangaography: list[dict[str, Any]] = []
    for edge in edges:
        node = edge.get("node") or {}
        media_id = node.get("id")
        media_type = str(node.get("type") or "").upper()
        title_obj = node.get("title") or {}
        title = (
            title_obj.get("romaji")
            or title_obj.get("english")
            or title_obj.get("native")
            or ""
        )
        if not media_id or not title:
            continue
        role = CharacterRole(str(edge.get("characterRole") or "UNKNOWN"))
        kind_path = "manga" if media_type == "MANGA" else "anime"
        entry = Ography(
            title=title,
            role=role,
            sources=[f"https://anilist.co/{kind_path}/{media_id}"],
        ).model_dump(mode="json", exclude_none=True)
        (mangaography if media_type == "MANGA" else animeography).append(entry)
    return animeography, mangaography


def _extract_anilist_character_voice_actors(
    edges: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract deduplicated voice actors from AniList character media edges.

    Args:
        edges: Combined GraphQL media edges carrying ``voiceActorRoles``.

    Returns:
        List of canonical ``VoiceActor`` dicts.
    """
    voice_actors: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    for edge in edges:
        for role_entry in edge.get("voiceActorRoles") or []:
            actor = (
                role_entry.get("voiceActor") if isinstance(role_entry, dict) else None
            )
            name_obj = actor.get("name") if isinstance(actor, dict) else None
            if not isinstance(actor, dict) or not isinstance(name_obj, dict):
                continue
            full_name = str(name_obj.get("full") or "").strip()
            if not full_name:
                continue
            staff_id = actor.get("id")
            site_url = str(actor.get("siteUrl") or "").strip() or (
                f"https://anilist.co/staff/{staff_id}" if staff_id is not None else ""
            )
            image_obj = actor.get("image")
            va_model = VoiceActor(
                name=full_name,
                native_name=name_obj.get("native") or None,
                language=actor.get("languageV2") or None,
                image=image_obj.get("large") if isinstance(image_obj, dict) else None,
                sources=[site_url] if site_url else [],
            )
            dedup_key = (site_url or full_name.casefold(), str(va_model.language or ""))
            if dedup_key not in seen_keys:
                seen_keys.add(dedup_key)
                voice_actors.append(va_model.model_dump(mode="json", exclude_none=True))
    return voice_actors


def _extract_anilist_character_attributes(
    char_data: dict[str, Any],
) -> dict[str, str]:
    """Extract gender, age, blood type, and date of birth from AniList Character.

    Args:
        char_data: Main GraphQL ``Character`` object.

    Returns:
        Mapping of attribute names to string values.
    """
    attributes: dict[str, str] = {}
    for key, src_key in (
        ("gender", "gender"),
        ("age", "age"),
        ("blood_type", "bloodType"),
    ):
        if char_data.get(src_key):
            attributes[key] = str(char_data[src_key])
    dob = require_json_object(
        char_data.get("dateOfBirth"), "AniList character 'dateOfBirth'"
    )
    if any(dob.get(part) for part in ("year", "month", "day")):
        attributes["date_of_birth"] = "-".join(
            str(dob[part]) for part in ("year", "month", "day") if dob.get(part)
        )
    return attributes


def map_anilist_character_page_responses(
    responses: Sequence[PageResponse],
) -> dict[str, Any]:
    """Map stored AniList character page responses into a canonical Character dict.

    Args:
        responses: Stored PageResponse sequence containing ``main`` and optional
            ``media_page_*`` JSON responses.

    Returns:
        Dict of canonical Character field values.

    Raises:
        ValueError: If ``main`` is missing.
        TypeError: If ``Character`` payload is not a JSON object.
    """
    by_name = index_responses_by_name(responses)
    main_response = by_name.get("main")
    if main_response is None:
        error_message = "AniList character responses missing required 'main' response"
        raise ValueError(error_message)
    main_payload = parse_json_response(main_response)
    data_obj = require_json_object(main_payload.get("data"), "AniList character 'data'")
    char_data = data_obj.get("Character")
    if not isinstance(char_data, dict):
        error_message = "AniList character 'main' response missing 'Character' object"
        raise TypeError(error_message)

    name_obj = require_json_object(char_data.get("name"), "AniList character 'name'")
    media_edges = _collect_anilist_character_media_edges(char_data, responses)
    animeography, mangaography = _extract_anilist_character_media_ography(media_edges)
    voice_actors = _extract_anilist_character_voice_actors(media_edges)
    spoilers = {
        f"name_spoiler_{idx}": str(spoiler_name)
        for idx, spoiler_name in enumerate(name_obj.get("alternativeSpoiler") or [])
        if spoiler_name
    }
    image_obj = require_json_object(char_data.get("image"), "AniList character 'image'")
    images = [
        str(url) for url in (image_obj.get("large"), image_obj.get("medium")) if url
    ]
    char_model = Character(
        name=str(name_obj.get("full") or ""),
        name_native=name_obj.get("native"),
        name_variations=[str(v) for v in (name_obj.get("alternative") or []) if v],
        description=char_data.get("description"),
        favorites=char_data.get("favourites"),
        images=images[:1],
        sources=[str(char_data["siteUrl"])] if char_data.get("siteUrl") else [],
        attributes=_extract_anilist_character_attributes(char_data),
        spoilers=spoilers,
        animeography=animeography,
        mangaography=mangaography,
        voice_actors=voice_actors,
    )
    return char_model.model_dump(mode="json", exclude_none=True)
