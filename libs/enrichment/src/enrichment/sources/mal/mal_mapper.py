"""MAL → canonical model mapper.

Pure value normalization functions. No I/O, no side effects.

Since MalAnime/Character/Episode field names already match the canonical
models wherever possible, these functions only:
  1. Normalize string values ("Currently Airing" → "ONGOING")
  2. Parse raw strings into typed values ("24 min." → 1440)
  3. Split relations into anime vs source-material buckets
  4. Build nested structures (statistics dict, images dict, etc.)

Field names are NOT renamed here — that's already handled at the model level.
"""

from collections.abc import Sequence
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
from common.utils.datetime_utils import (
    normalize_to_utc,
    parse_broadcast_time,
    parse_iana_timezone,
    parse_iso_weekday,
)
from enrichment.sources.base.companies import companies_from_roles
from enrichment.sources.base.external_links import external_link
from enrichment.sources.base.page_fetcher import (
    extract_canonical_url,
    find_paginated_responses,
    index_responses_by_name,
    parse_html_response,
)
from enrichment.sources.mal.mal_models import (
    MalAnime,
    MalCharacter,
    MalEpisode,
)
from shared_proto.v1.merge_pb2 import PageResponse


def anime_from_mal(anime: MalAnime) -> dict[str, Any]:
    """Normalize a MalAnime into canonical Anime field values.

    Args:
        anime: Scraped MAL anime model.

    Returns:
        Dict of canonical field name → normalized value.
    """
    # ── Scalars ──────────────────────────────────────────────────────────
    background = anime.background

    duration_seconds = anime.duration
    episode_count = anime.episode_count or 0
    rating = AnimeRating(anime.rating) if anime.rating else AnimeRating.UNKNOWN
    season = AnimeSeason(anime.season.upper()) if anime.season else None
    source_material = (
        SourceMaterialType(anime.source_material) if anime.source_material else None
    )
    status = AnimeStatus(anime.status)
    synopsis = anime.synopsis
    title = anime.title
    title_english = anime.title_english
    title_japanese = anime.title_japanese
    anime_type = AnimeType(anime.type)
    year = anime.year

    # ── Arrays ───────────────────────────────────────────────────────────
    demographics = anime.demographics
    genres = anime.genres
    licensors = [
        CompanyEntry(name=lic.name, sources=[lic.source]) for lic in anime.licensors
    ]
    producers = [CompanyEntry(name=p.name, sources=[p.source]) for p in anime.producers]
    sources = [anime.source] if anime.source else []
    studios = [CompanyEntry(name=s.name, sources=[s.source]) for s in anime.studios]
    synonyms = anime.synonyms
    themes = [ThemeEntry(name=t) for t in anime.themes]
    trailers = (
        [
            TrailerEntry(
                source=anime.trailer.source,
                title=anime.trailer.title,
                thumbnail_source=anime.trailer.thumbnail,
            )
        ]
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

    broadcast_day = parse_iso_weekday(anime.broadcast_day)
    broadcast_time = parse_broadcast_time(anime.broadcast_time)
    broadcast_timezone = parse_iana_timezone(anime.broadcast_timezone)
    broadcast = None
    if (
        broadcast_day is not None
        or broadcast_time is not None
        or broadcast_timezone is not None
    ):
        from common.models.anime import Broadcast

        broadcast = Broadcast(
            day=broadcast_day,
            jp_time=broadcast_time,
            timezone=broadcast_timezone,
        )

    images = AnimeImages(covers=anime.picture_urls)

    # Relations
    related_anime: dict[AnimeRelationType, list[RelatedAnime]] = {}
    related_source_material: dict[
        SourceMaterialRelationType, list[RelatedSourceMaterial]
    ] = {}

    for entry in anime.related_entries:
        if entry.is_anime:
            rel_type = AnimeRelationType(entry.relation)
            if rel_type not in related_anime:
                related_anime[rel_type] = []
            related_anime[rel_type].append(
                RelatedAnime(
                    title=entry.title,
                    type=AnimeType(entry.entry_type),
                    sources=[entry.source],
                )
            )
        else:
            rel_type = SourceMaterialRelationType(entry.relation)
            if rel_type not in related_source_material:
                related_source_material[rel_type] = []
            related_source_material[rel_type].append(
                RelatedSourceMaterial(
                    title=entry.title,
                    type=SourceMaterialType(entry.entry_type),
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
            episodes=[
                EpisodeRange(range_start=r.start, range_end=r.end) for r in s.episodes
            ],
        )
        for s in anime.opening_themes
    ]
    ending_themes = [
        ThemeSong(
            title=s.title,
            artist=s.artist,
            episodes=[
                EpisodeRange(range_start=r.start, range_end=r.end) for r in s.episodes
            ],
        )
        for s in anime.ending_themes
    ]

    # Links
    external_sources = [
        entry
        for link in anime.external_sources
        if (entry := external_link(link.source, label=link.name))
    ]
    streaming_sources = [
        StreamingEntry(platform=link.name, source=link.source)
        for link in anime.streaming
    ]

    # Build final canonical object
    result = Anime(
        background=background,
        duration_seconds=duration_seconds,
        episode_count=episode_count,
        month=anime.month,
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
        opening_themes=opening_themes,
        related_anime=related_anime,
        related_source_material=related_source_material,
        sources=sources,
        streaming_sources=streaming_sources,
        companies=companies_from_roles(
            studios=studios,
            producers=producers,
            licensors=licensors,
        ),
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


def character_from_mal(char: MalCharacter) -> dict[str, Any]:
    """Normalize a MalCharacter into canonical Character field values.

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
    ep: MalEpisode,
    *,
    anime_id: str | None = None,
) -> dict[str, Any]:
    """Normalize a MalEpisode into canonical Episode field values.

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
    aired_at = None
    if ep.aired:
        aired_dt = normalize_to_utc(ep.aired)
        if aired_dt:
            aired_at = aired_dt

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
        aired_at=aired_at,
        anime_id=anime_id,
        duration_seconds=ep.duration,
        episode_number=ep.episode_number,
        is_filler=ep.filler,
        is_recap=ep.recap,
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


def map_mal_anime_responses(
    responses: Sequence[PageResponse],
) -> dict[str, Any]:
    """Map stored MAL anime HTML responses into a canonical Anime dict.

    Args:
        responses: Stored PageResponse sequence containing a ``main`` HTML response.

    Returns:
        Dict of canonical Anime field values.

    Raises:
        ValueError: If the ``main`` response is missing or yields no anime data.
    """
    from enrichment.sources.mal.mal_anime_crawler import (
        _build_anime_from_raw,
        _extract_anime_from_html,
        _extract_pics_from_html,
    )

    by_name = index_responses_by_name(responses)
    main_response = by_name.get("main")
    if main_response is None:
        error_message = "MAL anime responses missing required 'main' response"
        raise ValueError(error_message)
    raw = _extract_anime_from_html(parse_html_response(main_response))
    if raw is None:
        error_message = "MAL 'main' HTML yielded no anime data"
        raise ValueError(error_message)

    picture_urls = (
        _extract_pics_from_html(parse_html_response(pics_response))
        if (pics_response := by_name.get("pics")) is not None
        else []
    )
    resolved_url = raw.get("canonical_url") or ""
    return anime_from_mal(_build_anime_from_raw(raw, resolved_url, picture_urls))


def map_mal_character_responses(
    responses: Sequence[PageResponse],
) -> dict[str, Any]:
    """Map stored MAL character HTML responses into a canonical Character dict.

    Args:
        responses: Stored PageResponse sequence containing a ``main`` HTML response.

    Returns:
        Dict of canonical Character field values.

    Raises:
        ValueError: If the ``main`` response is missing or yields no character data.
    """
    from enrichment.sources.mal.mal_character_crawler import (
        _build_character_from_raw,
        _extract_character_from_html,
    )

    by_name = index_responses_by_name(responses)
    main_response = by_name.get("main")
    if main_response is None:
        error_message = "MAL character responses missing required 'main' response"
        raise ValueError(error_message)
    html = parse_html_response(main_response)
    raw = _extract_character_from_html(html)
    if raw is None:
        error_message = "MAL 'main' HTML yielded no character data"
        raise ValueError(error_message)
    resolved_url = extract_canonical_url(html) or ""
    return character_from_mal(_build_character_from_raw(raw, resolved_url))


def map_mal_anime_episode_responses(
    responses: Sequence[PageResponse],
) -> list[dict[str, Any]]:
    """Map a MAL work's stored episode list pages into canonical Episode dicts.

    The list pages are stored as ``episodes_page_<n>`` responses, one per 100
    episodes. A work whose list was not fetched has none, and maps to no episodes.

    Args:
        responses: Stored PageResponse sequence of a MAL anime page.

    Returns:
        One canonical Episode dict per listed episode, in list order.
    """
    from enrichment.sources.mal.mal_episode_crawler import (
        _extract_episode_list_from_html,
    )

    return [
        episode_from_mal(episode)
        for list_page in find_paginated_responses(responses, "episodes_page_")
        for episode in _extract_episode_list_from_html(parse_html_response(list_page))
    ]


def map_mal_episode_responses(
    responses: Sequence[PageResponse],
) -> dict[str, Any]:
    """Map stored MAL episode HTML responses into a canonical Episode dict.

    Args:
        responses: Stored PageResponse sequence containing a ``main`` HTML response.

    Returns:
        Dict of canonical Episode field values.

    Raises:
        ValueError: If the ``main`` response is missing or yields no episode data.
    """
    import re

    from enrichment.sources.mal.mal_episode_crawler import (
        _build_episode_from_raw,
        _extract_episode_from_html,
    )

    by_name = index_responses_by_name(responses)
    main_response = by_name.get("main")
    if main_response is None:
        error_message = "MAL episode responses missing required 'main' response"
        raise ValueError(error_message)
    html = parse_html_response(main_response)
    raw = _extract_episode_from_html(html)
    if raw is None:
        error_message = "MAL 'main' HTML yielded no episode data"
        raise ValueError(error_message)
    resolved_url = extract_canonical_url(html) or ""
    if not re.search(r"/episode/(\d+)", resolved_url):
        og_match = re.search(
            r'<meta[^>]+property=["\']og:url["\'][^>]+content=["\']([^"\']+)["\']',
            html,
            re.IGNORECASE,
        )
        if og_match:
            resolved_url = og_match.group(1).strip()
    ep_match = re.search(r"/episode/(\d+)", resolved_url) or re.search(
        r"<h2[^>]*>.*?#(\d+)\s*-", html, re.DOTALL
    )
    episode_number = int(ep_match.group(1)) if ep_match else 1
    return episode_from_mal(_build_episode_from_raw(raw, episode_number, resolved_url))
