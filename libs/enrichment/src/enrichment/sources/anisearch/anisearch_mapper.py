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
    Episode,
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
from enrichment.sources.anisearch.anisearch_anime_models import (
    AniSearchAnime,
    AniSearchCharacter,
    AniSearchEpisode,
    AniSearchRelatedEntry,
)
from enrichment.sources.base.companies import companies_from_roles
from enrichment.sources.base.external_links import external_link
from enrichment.utils.text_utils import normalize_score

_ANISEARCH_BASE_URL = "https://www.anisearch.com/"
_DETAILS_TYPE_RE = re.compile(r"^([^,]+)")
_ANISEARCH_ANIME_ID_RE = re.compile(
    r"^https?://(?:www\.)?anisearch\.com/anime/(\d+)", re.IGNORECASE
)


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


def _anisearch_anime_key(url: str | None) -> str | None:
    """Reduce an AniSearch anime URL to its numeric id.

    The same anime is written several ways - ``/anime/2227``,
    ``/anime/2227,one-piece`` and ``/anime/2227,one-piece/characters`` - so the
    id is the only part safe to compare. Matching on the id alone also avoids
    the prefix trap, where ``/anime/466`` would otherwise match ``/anime/4661``.

    Args:
        url: An AniSearch anime URL, or None.

    Returns:
        The id as a string, or None if absent or unrecognised.
    """
    if not url:
        return None
    m = _ANISEARCH_ANIME_ID_RE.match(url)
    return m.group(1) if m else None


def _role_for_entry(
    entry_url: str | None, current_anime: str | None, role: str | None
) -> str:
    """Return ``role`` when this ography entry is the anime it describes.

    Args:
        entry_url: URL of the animeography entry under construction.
        current_anime: Id of the anime whose characters page supplied ``role``.
        role: Role label from that page's section heading.

    Returns:
        The role label, or "" to leave the entry UNKNOWN.
    """
    if not (role and current_anime):
        return ""
    return role if _anisearch_anime_key(entry_url) == current_anime else ""


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
    source_material = (
        SourceMaterialType(anime.source_material or "")
        if anime.source_material
        else None
    )
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
        # AniSearch rates out of 5 stars; every other provider lands on 0–10.
        score = normalize_score(anime.statistics.score, source_max=5.0)
        if score is not None:
            stats_data["score"] = score
        if anime.statistics.scored_by is not None:
            stats_data["scored_by"] = anime.statistics.scored_by
        if anime.statistics.rank is not None:
            stats_data["rank"] = anime.statistics.rank
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
        [
            CompanyEntry(
                name=anime.studio,
                sources=[anime.studio_url] if anime.studio_url else [],
            )
        ]
        if anime.studio
        else []
    )

    # ── Relations ─────────────────────────────────────────────────────────
    related_anime = _build_related_anime(anime.anime_relations)
    related_source_material = _build_related_source_material(anime.manga_relations)

    # ── External sources ──────────────────────────────────────────────────
    external_sources = [
        link
        for w in anime.websites
        if (link := external_link(w.get("url"), label=w.get("name")))
    ]

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
        companies=companies_from_roles(
            studios=studios,
        ),
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
    if all_roles:
        result["roles"] = [r.value for r in all_roles]

    # ── Animeography (full list from /anime sub-page; fallback to detail page) ──
    # The sub-page lists titles without roles, so every entry would otherwise be
    # UNKNOWN. `char.role` - the section heading from the anime's own characters
    # page - is the one per-title role AniSearch publishes, and it belongs to
    # `char.anime_url`; attach it to that entry. Remaining entries stay UNKNOWN
    # until another source or a later run of that anime fills them in.
    current_anime = _anisearch_anime_key(char.anime_url)
    ography_source = char.anime_ography or char.anime_roles
    if ography_source:
        result["animeography"] = [
            Ography(
                title=entry.title,
                role=CharacterRole(
                    entry.role or _role_for_entry(entry.url, current_anime, char.role)
                ),
                sources=[entry.url] if entry.url else [],
            )
            for entry in ography_source
            if entry.title
        ]

    # ── Mangaography ──────────────────────────────────────────────────────
    if char.manga_ography:
        result["mangaography"] = [
            Ography(
                title=entry.title,
                role=CharacterRole(entry.role or ""),
                sources=[entry.url] if entry.url else [],
            )
            for entry in char.manga_ography
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


# =============================================================================
# EPISODE MAPPER
# =============================================================================


def episode_from_anisearch(ep: AniSearchEpisode) -> dict[str, Any]:
    """Map a pre-parsed AniSearchEpisode into canonical Episode field values.

    All parsing (runtime → seconds, date string → ISO, title_ja split) is done
    by the crawler before this function is called.  This function only maps
    already-clean fields onto the canonical Episode model.

    Args:
        ep: Validated AniSearch episode source model with pre-parsed fields.

    Returns:
        Dict of canonical Episode field name → value (exclude_none=True).
    """
    episode = Episode(
        aired=normalize_to_utc(ep.aired),
        duration=ep.duration,
        episode_number=ep.episode_number,
        filler=ep.is_filler,
        recap=ep.is_recap,
        title=ep.title or f"Episode {ep.episode_number}",
        title_japanese=ep.title_japanese,
        title_romaji=ep.title_romaji,
        titles=ep.titles,
        sources=[ep.source] if ep.source else [],
    )
    return episode.model_dump(mode="json", exclude_none=True)
