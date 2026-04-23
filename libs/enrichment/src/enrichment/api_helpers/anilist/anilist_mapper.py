"""AniList → canonical model mapper.

Pure value normalization functions. No I/O, no side effects.

Maps raw AniList API response (validated as AniListAnime / AniListCharacterEdge)
to canonical Anime / Character field dicts.
"""

from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from common.models.anime import (
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

from enrichment.api_helpers.anilist.anilist_anime_models import (
    AniListAnime,
    AniListRelationEdge,
)
from enrichment.api_helpers.anilist.anilist_character_models import (
    AniListCharacterEdge,
    AniListFuzzyDate,
)
from enrichment.utils.text_utils import normalize_score

# AniList relation types that represent the anime being the SOURCE of a relation
# (i.e. the related item is what the anime was adapted FROM)
_SOURCE_RELATION_TYPES = {"SOURCE", "ADAPTATION"}


def _best_cover(cover_image) -> str | None:  # type: ignore[no-untyped-def]
    if not cover_image:
        return None
    return cover_image.extra_large or cover_image.large or None


def _fuzzy_date_str(d: "AniListFuzzyDate | None") -> str | None:
    """Format a FuzzyDate as YYYY-MM-DD, YYYY-MM, or YYYY depending on available parts."""
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

        # Discriminate by whether the format resolves to a known anime media type
        resolved_anime_type = AnimeType(node_format)
        if resolved_anime_type != AnimeType.UNKNOWN:
            # It's a media format (TV, MOVIE, OVA, etc.) → related_anime
            anime_rel_type = AnimeRelationType(rel_type_str)
            cover = _best_cover(node.cover_image)
            related_anime[anime_rel_type].append(
                RelatedAnime(
                    title=title,
                    type=resolved_anime_type,
                    sources=[f"https://anilist.co/anime/{node.id}"],
                    status=AnimeStatus(node.status or "") if node.status else None,
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
                    status=AnimeStatus(node.status or "") if node.status else None,
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
    title_english = (anime.title.english if anime.title else None) or None
    title_japanese = (anime.title.native if anime.title else None) or None
    synopsis = anime.description
    anime_type = AnimeType(anime.format or "")
    source_material = SourceMaterialType(anime.source or "") if anime.source else None
    status = AnimeStatus(anime.status or "")
    episode_count = anime.episodes or 0
    duration = (anime.duration * 60) if anime.duration else None  # minutes → seconds
    nsfw = anime.is_adult
    year = anime.season_year
    season = AnimeSeason(anime.season) if anime.season else None
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
    external_sources: dict[str, str] = {}
    for link in anime.external_links:
        if not link.url or not link.site:
            continue
        if link.type == "STREAMING":
            streaming_sources.append(
                StreamingEntry(platform=link.site, source=link.url)
            )
        elif link.type in ("INFO", "SOCIAL"):
            external_sources[link.site.lower()] = link.url

    # ── Trailer ───────────────────────────────────────────────────────────────
    trailers: list[TrailerEntry] = []
    if anime.trailer and anime.trailer.site == "youtube" and anime.trailer.id:
        trailers.append(
            TrailerEntry(
                source=f"https://youtu.be/{anime.trailer.id}",
                thumbnail=anime.trailer.thumbnail,
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
            all_time=r.all_time,
        )
        for r in anime.rankings
    ]
    anilist_stats = Statistics(
        score=normalize_score(anime.average_score) if anime.average_score else None,
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
            )
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
        duration=duration,
        nsfw=nsfw,
        country_of_origin=country_of_origin,
        year=year,
        season=season,
        sources=sources,
        genres=genres,
        synonyms=synonyms,
        demographics=demographics,
        themes=themes,
        tags=tags,
        content_warnings=content_warnings,
        studios=studios,
        producers=producers,
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
    name_native = (name_data.native if name_data else None) or None
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

    roles: list[CharacterRole] = [CharacterRole(edge.role or "")]

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
                native_name=(va.name.native if va.name else None) or None,
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
