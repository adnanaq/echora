"""AnimePlanet → canonical model mapper.

Pure value normalization — no I/O, no side effects.

Receives validated AnimePlanetAnime / AnimePlanetCharacter source models and
returns canonical Anime / Character dicts. The source models turn the page
text into field values; the mapper only lifts them into canonical types.

Key AP-specific details:
- Ratings: AP uses 0–5 scale → multiply × 2 for canonical 0–10
- Season: the model reads it from the seasons link; mapper lifts to AnimeSeason
- Type: the entry bar's type ("OVA", "Web", "Music Video")
- Source material: the "Based on a ..." / "Original Work" tag
- Related anime subtype mapping documented in docs/source_api_field_mappings.md
- Manga entries with subtype "Original Manga" → related_source_material (ADAPTATION)
- All other manga entries → related_source_material (OTHER)
"""

import re
from collections.abc import Sequence
from typing import Any

from common.models.anime import (
    AiredDates,
    Anime,
    AnimeImages,
    AnimeRelationType,
    AnimeSeason,
    AnimeType,
    Character,
    CharacterRole,
    CompanyEntry,
    Ography,
    RelatedAnime,
    RelatedSourceMaterial,
    SourceMaterialRelationType,
    SourceMaterialType,
    Statistics,
    VoiceActor,
)
from common.utils.datetime_utils import (
    determine_anime_season,
    determine_anime_status,
    determine_anime_year,
    normalize_to_utc,
)
from enrichment.sources.anime_planet.anime_planet_character_models import (
    AnimePlanetCharacter,
)
from enrichment.sources.anime_planet.anime_planet_models import AnimePlanetAnime
from enrichment.sources.base.companies import companies_from_roles
from enrichment.sources.base.page_fetcher import (
    extract_canonical_url,
    index_responses_by_name,
    parse_html_response,
)
from shared_proto.v1.merge_pb2 import PageResponse

_AP_BASE_URL = "https://www.anime-planet.com"
_ANIME_SLUG_PATTERN = re.compile(r"/anime/([^/?#]+)")


def _extract_anime_slug(url: str | None) -> str:
    """Extract the Anime-Planet anime slug segment from a URL.

    Args:
        url: Candidate Anime-Planet anime URL, or None.

    Returns:
        Extracted anime slug string, or an empty string when absent.
    """
    if not url:
        return ""
    match = _ANIME_SLUG_PATTERN.search(url)
    return match.group(1) if match else ""


def anime_from_animeplanet(anime: AnimePlanetAnime) -> dict[str, Any]:
    """Map an AnimePlanetAnime source model to canonical Anime field values.

    Args:
        anime: Validated AnimePlanetAnime scraped model.

    Returns:
        Dict of canonical field name → normalized value (exclude_none applied).
    """
    # ── Scalars ───────────────────────────────────────────────────────────
    anime_type = AnimeType(anime.type)
    source_material = (
        SourceMaterialType(anime.source_material) if anime.source_material else None
    )

    # Season: crawler supplies a parsed name ("fall"); fall back to start_date derivation
    season: AnimeSeason | None = AnimeSeason(anime.season) if anime.season else None
    if season is None and anime.start_date:
        season = determine_anime_season(anime.start_date)

    # The page data carries startDate only for a full date; the entry bar still shows the year.
    year = (
        determine_anime_year(anime.start_date) if anime.start_date else anime.start_year
    )
    status = determine_anime_status(anime.start_date, anime.end_date)

    # ── Aired dates ───────────────────────────────────────────────────────
    aired_dates = None
    if anime.start_date or anime.end_date:
        aired_dates = AiredDates(
            aired_from=normalize_to_utc(anime.start_date),
            aired_to=normalize_to_utc(anime.end_date),
        )

    # ── Statistics ────────────────────────────────────────────────────────
    statistics: dict[str, Statistics] = {}
    rank = anime.rank
    stats_data: dict[str, Any] = {}
    if anime.aggregate_rating:
        if anime.aggregate_rating.rating_value is not None:
            # AP uses 0–5 scale; canonical model uses 0–10
            stats_data["score"] = anime.aggregate_rating.rating_value * 2
        if anime.aggregate_rating.rating_count is not None:
            stats_data["scored_by"] = anime.aggregate_rating.rating_count
    if rank is not None:
        stats_data["rank"] = rank
    if stats_data:
        statistics["anime_planet"] = Statistics(**stats_data)

    # ── Tags: merge JSON-LD genres + XPath tags, deduplicate ─────────────
    tags: list[str] = []
    seen: set[str] = set()
    for t in anime.genres + anime.tags:
        if t and t not in seen:
            tags.append(t)
            seen.add(t)

    # ── Images ────────────────────────────────────────────────────────────
    images = AnimeImages(covers=[anime.cover] if anime.cover else [])

    # ── Sources / studios ─────────────────────────────────────────────────
    sources = [anime.url] if anime.url else []
    studios = [
        CompanyEntry(name=studio.name, sources=[studio.url] if studio.url else [])
        for studio in anime.studios
    ]

    # ── Related anime (same_franchise + other_franchise buckets) ──────────
    related_anime: dict[AnimeRelationType, list[RelatedAnime]] = {}
    for entry in anime.related_anime + anime.related_anime_other:
        rel_type = AnimeRelationType(entry.relation_subtype or "same franchise")
        if rel_type not in related_anime:
            related_anime[rel_type] = []
        full_url = (
            entry.url
            if entry.url.startswith("http")
            else f"https://www.anime-planet.com{entry.url}"
        )
        related_anime[rel_type].append(
            RelatedAnime(
                title=entry.title,
                type=AnimeType(entry.type),
                sources=[full_url],
                episode_count=entry.episode_count,
            )
        )

    # ── Related source material (manga entries) ───────────────────────────
    related_source_material: dict[
        SourceMaterialRelationType, list[RelatedSourceMaterial]
    ] = {}
    for entry in anime.related_manga:
        subtype = (entry.relation_subtype or "").lower()
        rel_type = (
            SourceMaterialRelationType.ADAPTATION
            if subtype == "original manga"
            else SourceMaterialRelationType.OTHER
        )
        if rel_type not in related_source_material:
            related_source_material[rel_type] = []
        full_url = (
            entry.url
            if entry.url.startswith("http")
            else f"https://www.anime-planet.com{entry.url}"
        )

        src_type = SourceMaterialType(entry.type)

        related_source_material[rel_type].append(
            RelatedSourceMaterial(
                title=entry.title,
                type=src_type,
                sources=[full_url],
                volumes=entry.volumes,
                chapters=entry.chapters,
            )
        )

    # ── Build canonical Anime ─────────────────────────────────────────────
    result = Anime(
        title=anime.name,
        synopsis=anime.description,
        title_japanese=anime.alt_title,
        type=anime_type,
        status=status,
        year=year,
        season=season,
        episode_count=anime.number_of_episodes or 0,
        sources=sources,
        companies=companies_from_roles(
            studios=studios,
        ),
        tags=tags,
        source_material=source_material,
        images=images,
        aired_dates=aired_dates,
        related_anime=related_anime,
        related_source_material=related_source_material,
        statistics=statistics,
    )

    return result.model_dump(mode="json", exclude_none=True)


def character_from_animeplanet(char: AnimePlanetCharacter) -> dict[str, Any]:
    """Map an AnimePlanetCharacter source model to canonical Character field values.

    Args:
        char: Validated AnimePlanetCharacter scraped model.

    Returns:
        Dict of canonical Character field name → normalized value (exclude_none applied).
    """
    result: dict[str, Any] = {
        "name": char.name,
        "sources": [char.url] if char.url else [],
    }

    if char.loved_count is not None:
        result["favorites"] = char.loved_count

    if char.description:
        result["description"] = char.description
    if char.tags:
        result["traits"] = char.tags
    if char.alt_names:
        result["nicknames"] = char.alt_names
    if char.image:
        result["images"] = [char.image]

    # ── Attributes (merge entryBar + EntryMetadata + ranks) ──────────────
    attributes: dict[str, str] = {
        k.lower().replace(" ", "_"): v for k, v in char.attributes.items()
    }
    if char.gender:
        attributes["gender"] = char.gender
    if char.hair_color:
        attributes["hair_color"] = char.hair_color
    if char.loved_rank is not None:
        attributes["loved_rank"] = str(char.loved_rank)
    if char.hated_rank is not None:
        attributes["hated_rank"] = str(char.hated_rank)
    if attributes:
        result["attributes"] = attributes

    # ── Roles (unique roles across all ography entries, in page order) ────
    roles_in_page_order = dict.fromkeys(
        CharacterRole(entry.role).value
        for entry in [*char.anime_roles, *char.manga_roles]
        if entry.role
    )
    if roles_in_page_order:
        result["roles"] = list(roles_in_page_order)

    # ── Animeography ─────────────────────────────────────────────────────
    if char.anime_roles:
        result["animeography"] = [
            Ography(
                title=entry.title,
                role=CharacterRole(entry.role or ""),
                sources=[f"{_AP_BASE_URL}{entry.url}"],
            )
            for entry in char.anime_roles
        ]

    # ── Mangaography ─────────────────────────────────────────────────────
    if char.manga_roles:
        result["mangaography"] = [
            Ography(
                title=entry.title,
                role=CharacterRole(entry.role or ""),
                sources=[f"{_AP_BASE_URL}{entry.url}"],
            )
            for entry in char.manga_roles
        ]

    # ── Voice actors (once each across all anime roles, by name and language) ─
    seen_voice_actor_keys: set[tuple[str, str]] = set()
    voice_actors: list[VoiceActor] = []
    for anime_role in char.anime_roles:
        for language, role_voice_actors in anime_role.voice_actors.items():
            for voice_actor in role_voice_actors:
                voice_actor_key = (voice_actor.name, language)
                if voice_actor_key not in seen_voice_actor_keys:
                    seen_voice_actor_keys.add(voice_actor_key)
                    voice_actors.append(
                        VoiceActor(
                            name=voice_actor.name,
                            language=language,
                            sources=[f"{_AP_BASE_URL}{voice_actor.url}"],
                        )
                    )
    if voice_actors:
        result["voice_actors"] = voice_actors

    character = Character.model_validate(result)
    return character.model_dump(mode="json", exclude_none=True)


def map_animeplanet_anime_responses(
    responses: Sequence[PageResponse],
    *,
    slug: str = "",
) -> dict[str, Any]:
    """Map stored Anime-Planet anime HTML responses into a canonical Anime dict.

    Args:
        responses: Stored PageResponse sequence containing a ``main`` HTML response.
        slug: Canonical Anime-Planet anime slug associated with the page.

    Returns:
        Dict of canonical Anime field values.

    Raises:
        ValueError: If the ``main`` response is missing or yields no anime data.
    """
    from enrichment.sources.anime_planet.anime_planet_anime_crawler import (
        _build_anime_from_raw,
        _extract_anime_from_html,
    )

    by_name = index_responses_by_name(responses)
    main_response = by_name.get("main")
    if main_response is None:
        error_message = "Anime-Planet anime responses missing required 'main' response"
        raise ValueError(error_message)
    html = parse_html_response(main_response)
    raw = _extract_anime_from_html(html)
    if not raw:
        error_message = "Anime-Planet 'main' HTML yielded no anime data"
        raise ValueError(error_message)
    raw_data = dict(raw)
    canonical_url = extract_canonical_url(html)
    json_ld_url = raw_data.get("url")
    resolved_slug = (
        slug
        or _extract_anime_slug(canonical_url)
        or _extract_anime_slug(json_ld_url if isinstance(json_ld_url, str) else None)
    )
    raw_data["slug"] = resolved_slug
    model = _build_anime_from_raw(raw_data)
    if not model.url:
        if canonical_url:
            model.url = canonical_url
        elif resolved_slug:
            model.url = f"{_AP_BASE_URL}/anime/{resolved_slug}"
    return anime_from_animeplanet(model)


def map_animeplanet_character_responses(
    responses: Sequence[PageResponse],
) -> dict[str, Any]:
    """Map stored Anime-Planet character HTML response into a canonical Character dict.

    Args:
        responses: Stored PageResponse sequence containing a ``main`` HTML response.

    Returns:
        Dict of canonical Character field values.

    Raises:
        ValueError: If the ``main`` response is missing or yields no character data.
    """
    from enrichment.sources.anime_planet.anime_planet_character_crawler import (
        _build_character_from_raw,
        _extract_character_from_html,
    )

    by_name = index_responses_by_name(responses)
    main_response = by_name.get("main")
    if main_response is None:
        error_message = (
            "Anime-Planet character responses missing required 'main' response"
        )
        raise ValueError(error_message)
    html = parse_html_response(main_response)
    raw = _extract_character_from_html(html)
    if raw is None:
        error_message = "Anime-Planet 'main' HTML yielded no character data"
        raise ValueError(error_message)
    resolved_url = extract_canonical_url(html) or ""
    return character_from_animeplanet(_build_character_from_raw(raw, resolved_url))
