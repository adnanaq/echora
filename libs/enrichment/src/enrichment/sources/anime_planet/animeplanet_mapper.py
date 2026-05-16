"""AnimePlanet → canonical model mapper.

Pure value normalization — no I/O, no side effects.

Receives validated AnimePlanetAnime / AnimePlanetCharacter source models and
returns canonical Anime / Character dicts.  The crawler is responsible for all
raw-string parsing; the mapper only lifts primitives into canonical types.

Key AP-specific details:
- Ratings: AP uses 0–5 scale → multiply × 2 for canonical 0–10
- Season: crawler parses slug → plain string; mapper lifts to AnimeSeason enum
- Type:   JSON-LD @type ("TVSeries" → TV, "Movie" → MOVIE) — coarse but reliable
- Related anime subtype mapping documented in docs/source_api_field_mappings.md
- Manga entries with subtype "Original Manga" → related_source_material (ADAPTATION)
- All other manga entries → related_source_material (OTHER)
"""

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

_AP_BASE_URL = "https://www.anime-planet.com"

_FLAG_TO_LANGUAGE: dict[str, str] = {
    "jp": "Japanese",
    "us": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ko": "Korean",
}


def anime_from_animeplanet(anime: AnimePlanetAnime) -> dict[str, Any]:
    """Map an AnimePlanetAnime source model to canonical Anime field values.

    Args:
        anime: Validated AnimePlanetAnime scraped model.

    Returns:
        Dict of canonical field name → normalized value (exclude_none applied).
    """
    # ── Scalars ───────────────────────────────────────────────────────────
    # schema_type is the JSON-LD @type value ("TVSeries", "Movie", etc.)
    anime_type = AnimeType(anime.schema_type or "")

    # Season: crawler supplies a parsed name ("fall"); fall back to start_date derivation
    season: AnimeSeason | None = AnimeSeason(anime.season) if anime.season else None
    if season is None and anime.start_date:
        season = determine_anime_season(anime.start_date)

    year = determine_anime_year(anime.start_date) if anime.start_date else None
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

    # ── Sources / producers ───────────────────────────────────────────────
    sources = [anime.url] if anime.url else []
    producers = [CompanyEntry(name=s) for s in anime.studios]

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
                type=AnimeType(entry.type or ""),
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

        src_type = SourceMaterialType(entry.type or "")

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
        producers=producers,
        tags=tags,
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
        "sources": [char.url],
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

    # ── Roles (aggregate unique roles across all ography entries) ─────────
    all_roles: set[CharacterRole] = set()
    for entry in char.anime_roles:
        if entry.role:
            all_roles.add(CharacterRole(entry.role))
    for entry in char.manga_roles:
        if entry.role:
            all_roles.add(CharacterRole(entry.role))
    if all_roles:
        result["roles"] = [r.value for r in all_roles]

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

    # ── Voice actors (flat dedup across all anime roles, keyed by name+lang) ─
    seen_vas: set[tuple[str, str]] = set()
    vas: list[VoiceActor] = []
    for anime_role in char.anime_roles:
        for lang_code, actors in anime_role.voice_actors.items():
            language = _FLAG_TO_LANGUAGE.get(lang_code, lang_code)
            for va in actors:
                key = (va.name, language)
                if key not in seen_vas:
                    seen_vas.add(key)
                    vas.append(
                        VoiceActor(
                            name=va.name,
                            language=language,
                            sources=[f"{_AP_BASE_URL}{va.url}"],
                        )
                    )
    if vas:
        result["voice_actors"] = vas

    character = Character.model_validate(result)
    return character.model_dump(mode="json", exclude_none=True)
