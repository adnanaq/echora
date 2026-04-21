"""AniSearch → canonical model mapper.

Pure value normalization — no I/O, no side effects.

Receives a validated AniSearchAnime source model and returns a canonical
Anime dict.  All AniSearch-specific normalization (type strings, relation
type strings) is delegated to the enum _missing_ methods — no per-crawler
lookup tables.

Key AniSearch-specific details:
- Dates are DD.MM.YYYY; datetime_utils handles this format natively.
- statistics.trending is not in the canonical Statistics model — dropped.
- genres (main/subsidiary) and tags are already split by the crawler.
- studio is a single string; mapped to studios=[CompanyEntry(name=...)].
- relation type/details parsing: "TV-Series, 12 (2025)" → type="TV-Series".
"""

import re
from typing import Any

from common.models.anime import (
    AiredDates,
    Anime,
    AnimeImages,
    AnimeRelationType,
    AnimeType,
    Broadcast,
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

from enrichment.crawlers.anisearch.anisearch_anime_models import (
    AniSearchAnime,
    AniSearchCharacter,
    AniSearchRelatedEntry,
)

_ANISEARCH_BASE_URL = "https://www.anisearch.com/"
_DETAILS_TYPE_RE = re.compile(r"^([^,]+)")


def _type_from_details(details: str | None) -> str:
    """Extract the type token from an AniSearch details string.

    e.g. "TV-Series, 12 (2025)" → "TV-Series"
         "Manga, 40 (2005)"     → "Manga"
    """
    if not details:
        return ""
    m = _DETAILS_TYPE_RE.match(details.strip())
    return m.group(1).strip() if m else ""


def _full_url(path: str | None) -> str | None:
    if not path:
        return None
    if path.startswith("http"):
        return path
    return _ANISEARCH_BASE_URL + path.lstrip("/")


def _build_related_anime(
    entries: list[AniSearchRelatedEntry],
) -> dict[AnimeRelationType, list[RelatedAnime]]:
    related: dict[AnimeRelationType, list[RelatedAnime]] = {}
    for entry in entries:
        if not entry.title:
            continue
        rel_type = AnimeRelationType(entry.relation_type or "")
        if rel_type not in related:
            related[rel_type] = []
        url = _full_url(entry.url)
        related[rel_type].append(
            RelatedAnime(
                title=entry.title,
                type=AnimeType(_type_from_details(entry.details)),
                sources=[url] if url else [],
                images=[entry.image] if entry.image else [],
            )
        )
    return related


def _build_related_source_material(
    entries: list[AniSearchRelatedEntry],
) -> dict[SourceMaterialRelationType, list[RelatedSourceMaterial]]:
    related: dict[SourceMaterialRelationType, list[RelatedSourceMaterial]] = {}
    for entry in entries:
        if not entry.title:
            continue
        rel_type = SourceMaterialRelationType(entry.relation_type or "")
        if rel_type not in related:
            related[rel_type] = []
        url = _full_url(entry.url)
        related[rel_type].append(
            RelatedSourceMaterial(
                title=entry.title,
                type=SourceMaterialType(_type_from_details(entry.details)),
                sources=[url] if url else [],
                images=[entry.image] if entry.image else [],
            )
        )
    return related


def anime_from_anisearch(anime: AniSearchAnime) -> dict[str, Any]:
    """Map an AniSearchAnime source model to canonical Anime field values.

    Args:
        anime: Validated AniSearchAnime scraped model.

    Returns:
        Dict of canonical field name → normalized value (exclude_none applied).
    """
    # ── Scalars ───────────────────────────────────────────────────────────
    anime_type = AnimeType(anime.type or "")
    source_material = SourceMaterialType(anime.source_material or "") if anime.source_material else None
    status = determine_anime_status(anime.start_date, anime.end_date)
    year = determine_anime_year(anime.start_date) if anime.start_date else None
    season = determine_anime_season(anime.start_date) if anime.start_date else None

    # ── Aired dates ───────────────────────────────────────────────────────
    aired_dates = None
    if anime.start_date or anime.end_date:
        aired_dates = AiredDates(
            aired_from=normalize_to_utc(anime.start_date),
            aired_to=normalize_to_utc(anime.end_date),
        )

    # ── Statistics ────────────────────────────────────────────────────────
    statistics: dict[str, Statistics] = {}
    if anime.statistics:
        stats_data: dict[str, Any] = {}
        for field in ("score", "rank"):
            v = getattr(anime.statistics, field)
            if v is not None:
                stats_data[field] = v
        if stats_data:
            statistics["anisearch"] = Statistics(**stats_data)

    # ── Images ────────────────────────────────────────────────────────────
    images = AnimeImages(covers=[anime.cover_image] if anime.cover_image else [])

    # ── Broadcast ─────────────────────────────────────────────────────────
    broadcast = None
    if anime.broadcast_day or anime.broadcast_time or anime.broadcast_timezone:
        broadcast = Broadcast(
            day=anime.broadcast_day,
            time=anime.broadcast_time,
            timezone=anime.broadcast_timezone,
        )

    # ── Companies ─────────────────────────────────────────────────────────
    studios = (
        [CompanyEntry(name=anime.studio, sources=[anime.studio_url] if anime.studio_url else [])]
        if anime.studio else []
    )

    # ── Relations ─────────────────────────────────────────────────────────
    related_anime = _build_related_anime(anime.anime_relations)
    related_source_material = _build_related_source_material(anime.manga_relations)

    # ── External sources ──────────────────────────────────────────────────
    external_sources = {w["name"]: w["url"] for w in anime.websites if w.get("name") and w.get("url")}

    result = Anime(
        title=anime.title or anime.title_japanese or "",
        title_japanese=anime.title_japanese,
        synonyms=anime.synonyms,
        type=anime_type,
        source_material=source_material,
        status=status,
        year=year,
        season=season,
        synopsis=anime.synopsis,
        genres=anime.genres,
        tags=anime.tags,
        studios=studios,
        sources=[anime.url] if anime.url else [],
        images=images,
        aired_dates=aired_dates,
        broadcast=broadcast,
        external_sources=external_sources,
        related_anime=related_anime,
        related_source_material=related_source_material,
        statistics=statistics,
    )

    return result.model_dump(mode="json", exclude_none=True)


def character_from_anisearch(char: AniSearchCharacter) -> dict[str, Any]:
    """Map an AniSearchCharacter source model to canonical Character field values.

    Args:
        char: Validated AniSearchCharacter scraped model.

    Returns:
        Dict of canonical Character field name → normalized value (exclude_none applied).
    """
    result: dict[str, Any] = {
        "name": char.name or "",
        "sources": [char.source],
    }

    if char.name_native:
        result["name_native"] = char.name_native
    if char.description:
        result["description"] = char.description
    if char.favorites is not None:
        result["favorites"] = char.favorites
    all_images = (
        ([char.image] if char.image else [])
        + char.screenshot_images
        + char.picture_images
    )
    if all_images:
        result["images"] = all_images
    if char.tags:
        result["traits"] = char.tags

    # ── Roles ─────────────────────────────────────────────────────────────
    all_roles: set[CharacterRole] = set()
    if char.role:
        all_roles.add(CharacterRole(char.role))
    for entry in char.anime_roles:
        if entry.role:
            all_roles.add(CharacterRole(entry.role))
    if all_roles:
        result["roles"] = [r.value for r in all_roles]

    # ── Animeography ──────────────────────────────────────────────────────
    if char.anime_roles:
        result["animeography"] = [
            Ography(
                title=entry.title,
                role=CharacterRole(entry.role or ""),
                sources=[entry.url] if entry.url else [],
            )
            for entry in char.anime_roles
            if entry.title
        ]

    # ── Voice actors ──────────────────────────────────────────────────────
    if char.voice_actors:
        result["voice_actors"] = [
            VoiceActor(
                name=va.name,
                language=va.language,
                sources=[va.url] if va.url else [],
            )
            for va in char.voice_actors
            if va.name
        ]

    # ── Attributes ────────────────────────────────────────────────────────
    if char.attributes:
        result["attributes"] = char.attributes

    character = Character.model_validate(result)
    return character.model_dump(mode="json", exclude_none=True)
