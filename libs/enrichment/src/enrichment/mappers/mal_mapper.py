"""MAL → canonical model mapper.

Pure value normalization functions. No I/O, no side effects.

Since MalScrapedAnime/Character/Episode field names already match the canonical
models wherever possible, these functions only:
  1. Normalize string values ("Currently Airing" → "ONGOING")
  2. Parse raw strings into typed values ("24 min." → 1440)
  3. Split relations into anime vs source-material buckets
  4. Build nested structures (statistics dict, images dict, etc.)

Field names are NOT renamed here — that's already handled at the model level.
"""

import re
from typing import Any

from common.models.anime import (
    Anime,
    AnimeImages,
    AnimeRating,
    AnimeRelationType,
    AnimeSeason,
    AnimeStatus,
    AnimeType,
    Character,
    CharacterRole,
    CompanyEntry,
    Episode,
    EpisodeCharacter,
    EpisodeRange,
    EpisodeStaff,
    Ography,
    RelatedAnime,
    RelatedSourceMaterial,
    SourceMaterialRelationType,
    SourceMaterialType,
    Statistics,
    StreamingEntry,
    ThemeEntry,
    ThemeSong,
    TrailerEntry,
    VoiceActor,
)
from common.utils.datetime_utils import normalize_to_utc

from enrichment.crawlers.mal_crawler.mal_models import (
    MalScrapedAnime,
    MalScrapedCharacter,
    MalScrapedEpisode,
)
from enrichment.mappers.normalization import SOURCE_RELATION


def anime_from_mal(anime: MalScrapedAnime) -> dict[str, Any]:
    """Normalize a MalScrapedAnime into canonical Anime field values.

    Args:
        anime: Scraped MAL anime model.

    Returns:
        Dict of canonical field name → normalized value.
    """
    # ── Scalars ──────────────────────────────────────────────────────────
    background = anime.background
    if background and "no background information" in background.lower():
        background = None

    duration = anime.duration
    episode_count = anime.episode_count or 0
    rating = AnimeRating(anime.rating) if anime.rating else AnimeRating.UNKNOWN
    season = AnimeSeason(anime.season.upper()) if anime.season else None
    source_material = SourceMaterialType(anime.source_material) if anime.source_material else None
    status = AnimeStatus(anime.status or "")
    synopsis = anime.synopsis
    title = anime.title
    title_english = anime.title_english
    title_japanese = anime.title_japanese
    anime_type = AnimeType(anime.type or "")
    year = anime.year

    # ── Arrays ───────────────────────────────────────────────────────────
    demographics = anime.demographics
    genres = anime.genres
    licensors = [
        CompanyEntry(name=l.name, sources=[l.source]) for l in anime.licensors
    ]
    producers = [
        CompanyEntry(name=p.name, sources=[p.source]) for p in anime.producers
    ]
    sources = [anime.source] if anime.source else []
    studios = [
        CompanyEntry(name=s.name, sources=[s.source]) for s in anime.studios
    ]
    synonyms = anime.synonyms
    themes = [ThemeEntry(name=t) for t in anime.themes]
    trailers = (
        [TrailerEntry(source=anime.trailer.source, title=anime.trailer.title, thumbnail=anime.trailer.thumbnail)]
        if anime.trailer
        else []
    )

    # ── Objects / Dicts ──────────────────────────────────────────────────
    aired_dates = None
    if anime.aired_from or anime.aired_to:
        from common.models.anime import AiredDates

        aired_dates = AiredDates(
            aired_from=normalize_to_utc(anime.aired_from),
            aired_to=normalize_to_utc(anime.aired_to),
        )

    broadcast = None
    if any([anime.broadcast_day, anime.broadcast_time, anime.broadcast_timezone]):
        from common.models.anime import Broadcast

        broadcast = Broadcast(
            day=anime.broadcast_day,
            time=anime.broadcast_time,
            timezone=anime.broadcast_timezone,
        )

    images = AnimeImages(covers=anime.picture_urls)

    # Relations
    related_anime: dict[AnimeRelationType, list[RelatedAnime]] = {}
    related_source_material: dict[SourceMaterialRelationType, list[RelatedSourceMaterial]] = {}

    for entry in anime.related_entries:
        if entry.is_anime:
            rel_type = AnimeRelationType(entry.relation)
            if rel_type not in related_anime:
                related_anime[rel_type] = []
            related_anime[rel_type].append(
                RelatedAnime(
                    title=entry.title,
                    type=AnimeType(entry.entry_type or ""),
                    sources=[entry.source],
                )
            )
        else:
            rel_type = SourceMaterialRelationType(
                SOURCE_RELATION.get(entry.relation.lower(), "OTHER")
            )
            if rel_type not in related_source_material:
                related_source_material[rel_type] = []
            related_source_material[rel_type].append(
                RelatedSourceMaterial(
                    title=entry.title,
                    type=SourceMaterialType(entry.entry_type or ""),
                    sources=[entry.source],
                )
            )

    # Statistics
    stats_data: dict[str, Any] = {}
    for field in ("score", "scored_by", "rank", "members", "favorites", "popularity"):
        v = getattr(anime, field)
        if v is not None:
            stats_data[field] = v

    statistics: dict[str, Statistics] = {}
    if stats_data:
        statistics["mal"] = Statistics(**stats_data)

    # Theme Songs
    opening_themes = [
        ThemeSong(
            title=s.title,
            artist=s.artist,
            episodes=[EpisodeRange(start=r.start, end=r.end) for r in s.episodes],
        )
        for s in anime.opening_themes
    ]
    ending_themes = [
        ThemeSong(
            title=s.title,
            artist=s.artist,
            episodes=[EpisodeRange(start=r.start, end=r.end) for r in s.episodes],
        )
        for s in anime.ending_themes
    ]

    # Links
    external_sources = {link.name: link.source for link in anime.external_sources}
    streaming_sources = [
        StreamingEntry(platform=link.name, source=link.source) for link in anime.streaming
    ]

    # Build final canonical object
    result = Anime(
        background=background,
        duration=duration,
        episode_count=episode_count,
        rating=rating,
        season=season,
        source_material=source_material,
        status=status,
        synopsis=synopsis,
        title=title,
        title_english=title_english,
        title_japanese=title_japanese,
        type=anime_type,
        year=year,
        demographics=demographics,
        ending_themes=ending_themes,
        genres=genres,
        licensors=licensors,
        opening_themes=opening_themes,
        producers=producers,
        related_anime=related_anime,
        related_source_material=related_source_material,
        sources=sources,
        streaming_sources=streaming_sources,
        studios=studios,
        synonyms=synonyms,
        themes=themes,
        trailers=trailers,
        aired_dates=aired_dates,
        broadcast=broadcast,
        external_sources=external_sources,
        images=images,
        statistics=statistics,
    )

    return result.model_dump(mode="json", exclude_none=True)


def character_from_mal(char: MalScrapedCharacter) -> dict[str, Any]:
    """Normalize a MalScrapedCharacter into canonical Character field values.

    Args:
        char: Scraped MAL character model.

    Returns:
        Dict of canonical Character field name → value. Only non-empty fields
        included.
    """
    result: dict[str, Any] = {}

    # ── Direct pass-through ───────────────────────────────────────────────
    for field in ("name", "name_native", "description", "nicknames", "favorites"):
        value = getattr(char, field)
        if value is not None and value != [] and value != "":
            result[field] = value

    # ── Sources ───────────────────────────────────────────────────────────
    if char.source:
        result["sources"] = [char.source]

    # ── Attributes (bio key/value pairs) ─────────────────────────────────
    if char.character_info:
        result["attributes"] = {k: str(v) for k, v in char.character_info.items()}
    if char.spoilers:
        result["spoilers"] = char.spoilers

    # ── Images ────────────────────────────────────────────────────────────
    if char.images:
        result["images"] = char.images

    # ── Voice actors ──────────────────────────────────────────────────────
    if char.voice_actors:
        result["voice_actors"] = [
            VoiceActor(name=va.name, language=va.language, sources=va.sources)
            for va in char.voice_actors
        ]


    # ── Roles (derived from animeography) ────────────────────────────────
    # Role is contextual per appearance; aggregate unique roles for search/filtering.
    if char.animeography:
        roles = list({CharacterRole(e.role) for e in char.animeography if e.role})
        if roles:
            result["roles"] = [r.value for r in roles]

    # ── Ography ───────────────────────────────────────────────────────────
    # Validate through OgraphyEntry so role gets normalized to CharacterRole enum.
    if char.animeography:
        result["animeography"] = [
            Ography.model_validate(e.model_dump()).model_dump(
                mode="json", exclude_none=True
            )
            for e in char.animeography
        ]
    if char.mangaography:
        result["mangaography"] = [
            Ography.model_validate(e.model_dump()).model_dump(
                mode="json", exclude_none=True
            )
            for e in char.mangaography
        ]

    character = Character.model_validate(result)
    return character.model_dump(mode="json", exclude_none=True)


def episode_from_mal(
    ep: MalScrapedEpisode,
    *,
    anime_id: str | None = None,
) -> dict[str, Any]:
    """Normalize a MalScrapedEpisode into canonical Episode field values.

    Args:
        ep: Scraped MAL episode model.
        anime_id: Optional UUID of the parent anime to inject as episode.anime_id.

    Returns:
        Dict of canonical Episode field name → value.
    """
    # Build result in Episode model field order:
    # SCALAR: aired, anime_id, description, duration, episode_number, filler,
    #         id, entity_type, recap, score, season_number, synopsis,
    #         title, title_japanese, title_romaji
    # ARRAY:  characters, images, sources, staff
    # DICT:   streamina
    aired = None
    if ep.aired:
        aired_dt = normalize_to_utc(ep.aired)
        if aired_dt:
            aired = aired_dt

    characters = [
        EpisodeCharacter(
            name=c.name,
            role=CharacterRole(c.role),
            sources=[f"https://myanimelist.net/character/{c.mal_id}"],
            voice_actors=[
                VoiceActor(
                    name=va.name,
                    language=va.language,
                    sources=[f"https://myanimelist.net/people/{va.person_id}"],
                )
                for va in c.voice_actors
            ],
        )
        for c in ep.characters
    ]

    staff = [
        EpisodeStaff(
            name=s.name,
            role=s.role,
            sources=[f"https://myanimelist.net/people/{s.person_id}"],
        )
        for s in ep.staff
    ]

    episode = Episode(
        aired=aired,
        anime_id=anime_id,
        duration=ep.duration,
        episode_number=ep.episode_number,
        filler=ep.filler,
        recap=ep.recap,
        score=None,
        synopsis=ep.synopsis,
        title=ep.title,
        title_japanese=ep.title_japanese,
        title_romaji=ep.title_romaji,
        characters=characters,
        staff=staff,
        sources=[ep.source],
    )

    return episode.model_dump(mode="json", exclude_none=True)
