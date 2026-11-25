"""AniList → canonical model mapper.

Pure value normalization functions. No I/O, no side effects.

Maps raw AniList API response (validated as AniListAnime / AniListCharacterEdge)
to canonical Anime / Character field dicts.
"""

import logging
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from common.models.anime import (
    AiredDates,
    Anime,
    AnimeImages,
    AnimeRelationType,
    AnimeSeason,
    AnimeStatus,
    AnimeType,
    Broadcast,
    Character,
    CharacterRole,
    CompanyEntry,
    ContextualRank,
    ExternalLink,
    Ography,
    RelatedAnime,
    RelatedSourceMaterial,
    SourceMaterialRelationType,
    SourceMaterialType,
    Statistics,
    StreamingEntry,
    ThemeEntry,
    TrailerEntry,
    VoiceActor,
)
from common.utils.datetime_utils import normalize_to_utc
from enrichment.sources.anilist.anilist_anime_models import (
    AniListAnime,
    AniListRelationEdge,
)
from enrichment.sources.anilist.anilist_character_models import (
    AniListCharacterEdge,
    AniListFuzzyDate,
)
from enrichment.sources.base.companies import companies_from_roles
from enrichment.sources.base.external_links import external_link
from enrichment.sources.base.page_fetcher import (
    collect_paginated_responses,
    index_responses_by_name,
    parse_json_response,
)
from enrichment.utils.text_utils import normalize_score
from shared_proto.v1.merge_pb2 import PageResponse

logger = logging.getLogger(__name__)

# AniList relation types that represent the anime being the SOURCE of a relation
# (i.e. the related item is what the anime was adapted FROM)
_SOURCE_RELATION_TYPES = {"SOURCE", "ADAPTATION"}


def _best_cover(cover_image) -> str | None:  # type: ignore[no-untyped-def]
    if not cover_image:
        return None
    return cover_image.extra_large or cover_image.large or None


def _fuzzy_date_str(d: AniListFuzzyDate | None) -> str | None:
    """Format a FuzzyDate as YYYY-MM-DD, YYYY-MM, or YYYY depending on available parts.

    >>> from enrichment.sources.anilist.anilist_character_models import (
    ...     AniListFuzzyDate,
    ... )
    >>> _fuzzy_date_str(AniListFuzzyDate(year=1999, month=10, day=20))
    '1999-10-20'
    >>> _fuzzy_date_str(AniListFuzzyDate(year=1999, month=10))
    '1999-10'
    >>> _fuzzy_date_str(AniListFuzzyDate(year=1999))
    '1999'

    A date without its year gives None:

    >>> print(_fuzzy_date_str(AniListFuzzyDate(month=10, day=20)))
    None
    """
    if not d or d.year is None:
        return None
    if d.month and d.day:
        return f"{d.year:04d}-{d.month:02d}-{d.day:02d}"
    if d.month:
        return f"{d.year:04d}-{d.month:02d}"
    return str(d.year)


def _split_relations(
    edges: list[AniListRelationEdge],
) -> tuple[
    dict[AnimeRelationType, list[RelatedAnime]],
    dict[SourceMaterialRelationType, list[RelatedSourceMaterial]],
]:
    """Split relation edges into anime vs source-material buckets using canonical enums."""
    related_anime: dict[AnimeRelationType, list[RelatedAnime]] = defaultdict(list)
    related_source: dict[SourceMaterialRelationType, list[RelatedSourceMaterial]] = (
        defaultdict(list)
    )

    for edge in edges:
        node = edge.node
        node_format = node.format or ""
        rel_type_str = edge.relation_type
        title = (node.title.romaji if node.title else None) or ""

        # An anime format (TV, MOVIE, OVA, ...) maps to an anime type; MANGA,
        # NOVEL and ONE_SHOT do not, and are source material.
        resolved_anime_type = AnimeType.from_provider_value(node_format)
        if resolved_anime_type not in (None, AnimeType.UNKNOWN):
            # It's a media format (TV, MOVIE, OVA, etc.) → related_anime
            anime_rel_type = AnimeRelationType(rel_type_str)
            cover = _best_cover(node.cover_image)
            related_anime[anime_rel_type].append(
                RelatedAnime(
                    title=title,
                    type=resolved_anime_type,
                    sources=[f"https://anilist.co/anime/{node.id}"],
                    status=AnimeStatus(node.status) if node.status else None,
                    year=node.season_year,
                    score=normalize_score(node.average_score)
                    if node.average_score
                    else None,
                    images=[cover] if cover else [],
                    episode_count=node.episodes,
                )
            )
        else:
            # Not a media format → source material (MANGA, NOVEL, ONE_SHOT, etc.)
            src_rel_type = SourceMaterialRelationType(rel_type_str)
            mat_type = SourceMaterialType(node_format)
            cover = _best_cover(node.cover_image)
            related_source[src_rel_type].append(
                RelatedSourceMaterial(
                    title=title,
                    type=mat_type,
                    sources=[f"https://anilist.co/manga/{node.id}"],
                    status=AnimeStatus(node.status) if node.status else None,
                    score=normalize_score(node.average_score)
                    if node.average_score
                    else None,
                    images=[cover] if cover else [],
                    chapters=node.chapters,
                    volumes=node.volumes,
                )
            )

    return dict(related_anime), dict(related_source)


def anime_from_anilist(anime: AniListAnime) -> dict[str, Any]:
    """Normalize an AniListAnime into canonical Anime field values.

    Args:
        anime: Validated AniList anime model.

    Returns:
        Dict of canonical field name → normalized value.
    """
    # ── Scalars ──────────────────────────────────────────────────────────────
    title = (anime.title.romaji if anime.title else None) or ""
    title_english = anime.title.english if anime.title else None
    title_japanese = anime.title.native if anime.title else None
    synopsis = anime.description
    anime_type = AnimeType(anime.format)
    source_material = SourceMaterialType(anime.source) if anime.source else None
    status = AnimeStatus(anime.status)
    episode_count = anime.episodes or 0
    duration_seconds = (
        (anime.duration * 60) if anime.duration else None
    )  # minutes → seconds
    is_nsfw = anime.is_adult
    start, end = anime.start_date, anime.end_date
    year = anime.season_year or (start.year if start else None)
    season = AnimeSeason(anime.season) if anime.season else None
    # normalize_to_utc reads only full dates, so "1977-10" or "2027" give no date;
    # a month without a day is stated as the month instead.
    aired_from = normalize_to_utc(_fuzzy_date_str(start))
    aired_to = normalize_to_utc(_fuzzy_date_str(end))
    aired_dates = (
        AiredDates(aired_from=aired_from, aired_to=aired_to)
        if aired_from or aired_to
        else None
    )
    month = start.month if start and start.month and not start.day else None
    country_of_origin = anime.country_of_origin

    # ── Sources ───────────────────────────────────────────────────────────────
    sources: list[str] = [f"https://anilist.co/anime/{anime.id}"]
    if anime.id_mal:
        sources.append(f"https://myanimelist.net/anime/{anime.id_mal}")

    # ── Genres & tags ─────────────────────────────────────────────────────────
    genres = list(anime.genres)
    synonyms = list(anime.synonyms)
    demographics: list[str] = []
    themes: list[ThemeEntry] = []
    tags: list[str] = []
    content_warnings: list[str] = []

    for tag in anime.tags:
        if tag.is_adult:
            content_warnings.append(tag.name)
        elif tag.category == "Demographic":
            demographics.append(tag.name)
        elif tag.category and tag.category.startswith("Theme-"):
            themes.append(ThemeEntry(name=tag.name, description=tag.description))
        else:
            # Cast-*, Setting*, Technical → flat tags
            tags.append(tag.name)

    # ── Studios & producers ───────────────────────────────────────────────────
    studios: list[CompanyEntry] = []
    producers: list[CompanyEntry] = []
    for edge in anime.studios:
        company = CompanyEntry(
            name=edge.node.name,
            sources=[f"https://anilist.co/studio/{edge.node.id}"],
        )
        if edge.node.is_animation_studio:
            studios.append(company)
        else:
            producers.append(company)

    # ── Streaming & external links ────────────────────────────────────────────
    streaming_sources: list[StreamingEntry] = []
    external_sources: list[ExternalLink] = []
    for link in anime.external_links:
        if not link.url or not link.site:
            continue
        if link.type == "STREAMING":
            streaming_sources.append(
                StreamingEntry(platform=link.site, source=link.url)
            )
        elif link.type in ("INFO", "SOCIAL"):
            entry = external_link(link.url, label=link.site, language=link.language)
            if entry:
                external_sources.append(entry)

    # ── Trailer ───────────────────────────────────────────────────────────────
    trailers: list[TrailerEntry] = []
    if anime.trailer and anime.trailer.site == "youtube" and anime.trailer.id:
        trailers.append(
            TrailerEntry(
                source=f"https://youtu.be/{anime.trailer.id}",
                thumbnail_source=anime.trailer.thumbnail,
            )
        )

    # ── Images ───────────────────────────────────────────────────────────────
    covers: list[str] = []
    banners: list[str] = []
    cover = _best_cover(anime.cover_image)
    if cover:
        covers.append(cover)
    if anime.banner_image:
        banners.append(anime.banner_image)

    # ── Statistics ────────────────────────────────────────────────────────────
    contextual_ranks: list[ContextualRank] = [
        ContextualRank(
            rank=r.rank,
            context=r.context,
            format=r.format,
            year=r.year,
            season=r.season,
            is_all_time=r.all_time,
        )
        for r in anime.rankings
    ]
    anilist_stats = Statistics(
        score=normalize_score(anime.average_score) if anime.average_score else None,
        scored_by=anime.scored_by,
        members=anime.popularity,
        favorites=anime.favourites,
        contextual_ranks=contextual_ranks or None,
    )

    # ── Broadcast (next episode airing) ──────────────────────────────────────
    broadcast: Broadcast | None = None
    if anime.next_airing_episode and anime.next_airing_episode.airing_at:
        broadcast = Broadcast(
            next_episode_at=datetime.fromtimestamp(
                anime.next_airing_episode.airing_at, tz=UTC
            ),
            next_episode_number=anime.next_airing_episode.episode,
        )

    # ── Relations ─────────────────────────────────────────────────────────────
    related_anime, related_source_material = _split_relations(anime.relations)

    result = Anime(
        title=title,
        title_english=title_english,
        title_japanese=title_japanese,
        synopsis=synopsis,
        type=anime_type,
        source_material=source_material,
        status=status,
        episode_count=episode_count,
        duration_seconds=duration_seconds,
        is_nsfw=is_nsfw,
        country_of_origin=country_of_origin,
        year=year,
        month=month,
        season=season,
        aired_dates=aired_dates,
        sources=sources,
        genres=genres,
        synonyms=synonyms,
        demographics=demographics,
        themes=themes,
        tags=tags,
        content_warnings=content_warnings,
        companies=companies_from_roles(
            studios=studios,
            producers=producers,
        ),
        streaming_sources=streaming_sources,
        external_sources=external_sources,
        trailers=trailers,
        images=AnimeImages(covers=covers, banners=banners),
        statistics={"anilist": anilist_stats},
        broadcast=broadcast,
        related_anime=related_anime,
        related_source_material=related_source_material,
    )

    return result.model_dump(mode="json", exclude_none=True)


def character_from_anilist(edge: AniListCharacterEdge) -> dict[str, Any]:
    """Normalize an AniListCharacterEdge into canonical Character field values.

    Args:
        edge: Validated AniList character edge model.

    Returns:
        Dict of canonical field name → normalized value.
    """
    node = edge.node
    name_data = node.name

    name = (name_data.full if name_data else None) or ""
    name_native = name_data.native if name_data else None
    name_variations = list(name_data.alternative if name_data else [])
    nicknames = list(name_data.alternative_spoiler if name_data else [])

    images: list[str] = []
    if node.image and node.image.large:
        images.append(node.image.large)

    description = node.description_prose
    favorites = node.favourites

    # Biographical attributes — explicit AniList fields take precedence over parsed ones
    attributes: dict[str, str] = {**node.description_attributes}
    if node.gender:
        attributes["gender"] = node.gender
    if node.age:
        attributes["age"] = node.age
    if node.blood_type:
        attributes["blood_type"] = node.blood_type
    birth_date = _fuzzy_date_str(node.date_of_birth)
    if birth_date:
        attributes["date_of_birth"] = birth_date

    roles: list[CharacterRole] = [CharacterRole(edge.role)]

    # All voice actors across all languages via voiceActorRoles
    voice_actors: list[VoiceActor] = []
    for var in edge.voice_actor_roles:
        va = var.voice_actor
        if not va:
            continue
        va_name = (va.name.full if va.name else None) or ""
        if not va_name:
            continue
        va_image = va.image.large if va.image else None
        voice_actors.append(
            VoiceActor(
                name=va_name,
                native_name=va.name.native if va.name else None,
                language=va.language_v2,
                image=va_image,
                sources=[va.site_url or f"https://anilist.co/staff/{va.id}"],
            )
        )

    sources = [node.site_url or f"https://anilist.co/character/{node.id}"]

    result = Character(
        name=name,
        name_native=name_native,
        name_variations=name_variations,
        nicknames=nicknames,
        images=images,
        description=description,
        favorites=favorites,
        attributes=attributes,
        spoilers=node.description_spoilers,
        roles=roles,
        voice_actors=voice_actors,
        sources=sources,
    )

    return result.model_dump(mode="json", exclude_none=True)


def map_anilist_anime_responses(
    responses: Sequence[PageResponse],
) -> dict[str, Any]:
    """Map stored AniList anime page responses into a canonical Anime dict.

    Args:
        responses: Stored PageResponse sequence containing a ``main`` response.

    Returns:
        Dict of canonical Anime field values.

    Raises:
        ValueError: If the ``main`` response or its ``Media`` payload is missing.
    """
    by_name = index_responses_by_name(responses)
    main_response = by_name.get("main")
    if main_response is None:
        error_message = "AniList anime responses missing required 'main' response"
        raise ValueError(error_message)
    payload = parse_json_response(main_response)
    media = payload.get("data", {}).get("Media")
    if not isinstance(media, dict):
        error_message = "AniList 'main' response missing 'data.Media' object"
        raise TypeError(error_message)
    return anime_from_anilist(AniListAnime.model_validate(media))


def map_anilist_anime_character_responses(
    responses: Sequence[PageResponse],
) -> list[dict[str, Any]]:
    """Map paginated AniList character connection responses into Character dicts.

    Args:
        responses: Stored PageResponse sequence containing ``characters_page_*`` items.

    Returns:
        List of canonical Character field dicts.
    """
    characters: list[dict[str, Any]] = []
    for page_payload in collect_paginated_responses(responses, "characters_page_"):
        edges = (
            page_payload.get("data", {})
            .get("Media", {})
            .get("characters", {})
            .get("edges", [])
        )
        for edge in edges:
            try:
                characters.append(
                    character_from_anilist(AniListCharacterEdge.model_validate(edge))
                )
            except Exception:
                logger.warning("Skipping invalid AniList character edge", exc_info=True)
    return characters


def _extract_anilist_character_media_ography(
    char_data: dict[str, Any],
    responses: Sequence[PageResponse],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract animeography and mangaography from AniList character media pages.

    Args:
        char_data: Main GraphQL ``Character`` object.
        responses: Stored PageResponse sequence containing optional ``media_page_*``.

    Returns:
        Tuple of ``(animeography, mangaography)`` normalized dicts.
    """
    animeography: list[dict[str, Any]] = []
    mangaography: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = list(
        (char_data.get("media") or {}).get("edges") or []
    )
    for page_payload in collect_paginated_responses(responses, "media_page_"):
        page_edges = (
            ((page_payload.get("data") or {}).get("Character") or {}).get("media") or {}
        ).get("edges") or []
        edges.extend(page_edges)

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
    dob = char_data.get("dateOfBirth") or {}
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
    char_data = (main_payload.get("data") or {}).get("Character")
    if not isinstance(char_data, dict):
        error_message = "AniList character 'main' response missing 'Character' object"
        raise TypeError(error_message)

    name_obj = char_data.get("name") or {}
    animeography, mangaography = _extract_anilist_character_media_ography(
        char_data, responses
    )
    spoilers = {
        f"name_spoiler_{idx}": str(spoiler_name)
        for idx, spoiler_name in enumerate(name_obj.get("alternativeSpoiler") or [])
        if spoiler_name
    }
    image_obj = char_data.get("image") or {}
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
    )
    return char_model.model_dump(mode="json", exclude_none=True)
