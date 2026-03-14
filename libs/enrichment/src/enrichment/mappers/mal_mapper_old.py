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

from typing import Any

from common.models.anime import (
    Anime,
    Character,
    Episode,
    EpisodeCharacterAppearance,
    EpisodeRange,
    EpisodeStaffCredit,
    OgraphyEntry,
    RelatedAnimeEntry,
    SimpleVoiceActor,
    SourceMaterial,
    ThemeSong,
)
from common.utils.datetime_utils import normalize_to_utc

from enrichment.crawlers.mal_crawler.mal_models import (
    MalScrapedAnime,
    MalScrapedCharacter,
    MalScrapedEpisode,
)
from enrichment.mappers.normalization import (
    ANIME_RELATION,
    ANIME_STATUS,
    ANIME_TYPE,
    SOURCE_MATERIAL,
    SOURCE_RELATION,
    parse_duration,
)


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

    duration = parse_duration(anime.duration) if anime.duration else None
    episode_count = anime.episode_count or 0
    rating = anime.rating  # Matches AnimeRating enum values
    season = anime.season.upper() if anime.season else None
    status = ANIME_STATUS.get(anime.status.lower() if anime.status else "", "UNKNOWN")
    synopsis = anime.synopsis
    title = anime.title
    title_english = anime.title_english
    title_japanese = anime.title_japanese
    anime_type = ANIME_TYPE.get(anime.type.lower() if anime.type else "", "UNKNOWN")
    year = anime.year

    # ── Arrays ───────────────────────────────────────────────────────────
    demographics = anime.demographics
    genres = anime.genres
    producers = [
        {"name": p.name, "sources": [p.source]} for p in anime.producers
    ]
    sources = [anime.url] if anime.url else []
    studios = [
        {"name": s.name, "sources": [s.source]} for s in anime.studios
    ]
    synonyms = anime.synonyms
    themes = [{"name": t} for t in anime.themes]
    trailers = [{"url": anime.trailer_url}] if anime.trailer_url else []

    # ── Objects / Dicts ──────────────────────────────────────────────────
    aired_dates = None
    if anime.aired_from or anime.aired_to:
        aired_dates = {
            "aired_from": normalize_to_utc(anime.aired_from),
            "aired_to": normalize_to_utc(anime.aired_to),
        }

    broadcast = None
    if any([anime.broadcast_day, anime.broadcast_time, anime.broadcast_timezone]):
        broadcast = {
            "day": anime.broadcast_day,
            "time": anime.broadcast_time,
            "timezone": anime.broadcast_timezone,
        }

    # Images
    images: dict[str, list[str]] = {}
    if anime.images:
        cover_url = anime.images.get("large_jpg") or anime.images.get("jpg")
        if cover_url:
            images["covers"] = [cover_url]
    if anime.picture_urls:
        images["gallery"] = anime.picture_urls

    # Relations
    related_animes: dict[str, list[dict[str, Any]]] = {}
    related_original_works: dict[str, list[dict[str, Any]]] = {}

    for entry in anime.related_entries:
        rel_raw = entry.relation.lower()
        if entry.is_anime:
            rel_type = ANIME_RELATION.get(rel_raw, "OTHER")
            if rel_type not in related_animes:
                related_animes[rel_type] = []
            related_animes[rel_type].append({
                "title": entry.title,
                "type": ANIME_TYPE.get(entry.entry_type.lower() if entry.entry_type else "", "UNKNOWN"),
                "sources": [entry.source],
            })
        else:
            rel_type = SOURCE_RELATION.get(rel_raw, "OTHER")
            if rel_type not in related_original_works:
                related_original_works[rel_type] = []
            related_original_works[rel_type].append({
                "title": entry.title,
                "type": SOURCE_MATERIAL.get(entry.entry_type.lower() if entry.entry_type else "", "UNKNOWN"),
                "sources": [entry.source],
            })

    # Statistics
    stats: dict[str, Any] = {}
    for field in ("score", "scored_by", "rank", "members", "favorites", "popularity"):
        v = getattr(anime, field)
        if v is not None:
            stats[field] = v

    # Theme Songs
    opening_themes = [
        {
            "title": s.title,
            "artist": s.artist,
            "episodes": [{"start": r.start, "end": r.end} for r in s.episodes],
        }
        for s in anime.opening_themes
    ]
    ending_themes = [
        {
            "title": s.title,
            "artist": s.artist,
            "episodes": [{"start": r.start, "end": r.end} for r in s.episodes],
        }
        for s in anime.ending_themes
    ]

    # Links
    external_links = {link.name: link.source for link in anime.external_links}
    streaming_info = [
        {"platform": link.name, "url": link.source} for link in anime.streaming
    ]

    # Build final canonical object
    result = Anime(
        id=str(anime.mal_id),
        background=background,
        duration=duration,
        episode_count=episode_count,
        rating=rating,
        season=season,
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
        opening_themes=opening_themes,
        producers=producers,
        related_animes=related_animes,
        related_original_works=related_original_works,
        sources=sources,
        streaming_info=streaming_info,
        studios=studios,
        synonyms=synonyms,
        themes=themes,
        trailers=trailers,
        aired_dates=aired_dates,
        broadcast=broadcast,
        external_links=external_links,
        images=images,
        statistics={"mal": stats} if stats else {},
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
    if char.url:
        result["sources"] = [char.url]

    # ── Role normalization (handled by Character.normalize_role validator) ─
    if char.role:
        result["role"] = char.role  # Validator will normalize "Main" → "MAIN"

    # ── Images ────────────────────────────────────────────────────────────
    if char.images:
        result["images"] = char.images

    # ── Voice actors ──────────────────────────────────────────────────────
    if char.voice_actors:
        result["voice_actors"] = [
            {"name": va.name, "language": va.language} for va in char.voice_actors
        ]

    # ── Free-form biographical info ───────────────────────────────────────
    if char.character_info:
        result["character_info"] = char.character_info

    # ── Ography ───────────────────────────────────────────────────────────
    # Validate through OgraphyEntry so role gets normalized to CharacterRole enum.
    if char.animeography:
        result["animeography"] = [
            OgraphyEntry.model_validate(e.model_dump()).model_dump(
                mode="json", exclude_none=True
            )
            for e in char.animeography
        ]
    if char.mangaography:
        result["mangaography"] = [
            OgraphyEntry.model_validate(e.model_dump()).model_dump(
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
        EpisodeCharacterAppearance(
            name=c.name,
            role=c.role,
            sources=[f"https://myanimelist.net/character/{c.mal_id}"],
            voice_actors=[
                SimpleVoiceActor(name=va.name, language=va.language)
                for va in c.voice_actors
            ],
        )
        for c in ep.characters
    ]

    staff = [
        EpisodeStaffCredit(
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
        sources=[ep.url],
    )

    return episode.model_dump(mode="json", exclude_none=True)
