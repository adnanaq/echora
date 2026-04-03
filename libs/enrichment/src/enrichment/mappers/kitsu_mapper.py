"""Kitsu → canonical model mapper.

Pure value normalization functions. No I/O, no side effects.

Maps validated Kitsu models to canonical Anime / Character / Episode field dicts.
"""

import re
from typing import Any

from common.models.anime import (
    AiredDates,
    Anime,
    AnimeImages,
    AnimeRating,
    AnimeStatus,
    AnimeType,
    Broadcast,
    Character,
    CharacterRole,
    Episode,
    Ography,
    Statistics,
    TrailerEntry,
    VoiceActor,
)
from common.utils.datetime_utils import (
    determine_anime_season,
    determine_anime_year,
    normalize_to_utc,
)
from enrichment.api_helpers.kitsu.kitsu_models import (
    KitsuAnime,
    KitsuEpisode,
    KitsuMediaCharacter,
)

# Kitsu locale codes → canonical language name for VoiceActor.language
_LOCALE_TO_LANGUAGE: dict[str, str] = {
    "ja_jp": "Japanese",
    "en": "English",
    "en_us": "English",
    "pt_br": "Portuguese (Brazil)",
    "pt_pt": "Portuguese",
    "es": "Spanish",
    "es_la": "Spanish (Latin America)",
    "de": "German",
    "fr": "French",
    "it": "Italian",
    "ko": "Korean",
    "zh": "Chinese",
    "zh_cn": "Chinese (Simplified)",
    "zh_tw": "Chinese (Traditional)",
    "ru": "Russian",
    "pl": "Polish",
    "hu": "Hungarian",
    "ar": "Arabic",
}


def _strip_html(text: str | None) -> str | None:
    """Strip HTML tags and decode common HTML entities."""
    if not text:
        return None
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&quot;", '"', text)
    text = re.sub(r"&#39;", "'", text)
    return text.strip() or None


def anime_from_kitsu(anime: KitsuAnime) -> dict[str, Any]:
    """Normalize a KitsuAnime into canonical Anime field values.

    ``anime.genres`` and ``anime.themes`` must be populated by the helper
    before calling this function (fetched from ``/genres`` and ``/categories``).

    Args:
        anime: Validated Kitsu anime model with genres and themes attached.

    Returns:
        Dict of canonical field name → normalized value.
    """
    attrs = anime.attributes
    titles = attrs.titles

    title = attrs.canonicalTitle or titles.en_jp or titles.en or "Unknown"

    season = determine_anime_season(attrs.startDate)
    year = determine_anime_year(attrs.startDate)
    start_dt = normalize_to_utc(attrs.startDate)
    end_dt = normalize_to_utc(attrs.endDate)

    poster = attrs.posterImage
    cover = attrs.coverImage
    images = AnimeImages(
        posters=[poster.original] if poster and poster.original else [],
        covers=[cover.original] if cover and cover.original else [],
    )

    kitsu_score = float(attrs.averageRating) / 10.0 if attrs.averageRating else None
    statistics: dict[str, Statistics] = {
        "kitsu": Statistics(
            score=kitsu_score,
            members=attrs.userCount,
            favorites=attrs.favoritesCount,
            popularity=attrs.popularityRank,
            rank=attrs.ratingRank,
        )
    }

    broadcast: Broadcast | None = None
    if attrs.nextRelease:
        broadcast = Broadcast(next_episode_at=normalize_to_utc(attrs.nextRelease))

    trailers: list[TrailerEntry] = []
    if attrs.youtubeVideoId:
        trailers.append(
            TrailerEntry(
                source=f"https://www.youtube.com/watch?v={attrs.youtubeVideoId}"
            )
        )

    aired_dates: AiredDates | None = None
    if start_dt or end_dt:
        aired_dates = AiredDates(aired_from=start_dt, aired_to=end_dt)

    result = Anime(
        title=title,
        title_english=titles.en,
        title_japanese=titles.ja_jp,
        synopsis=_strip_html(attrs.synopsis),
        type=AnimeType(attrs.subtype or ""),
        status=AnimeStatus(attrs.status or ""),
        episode_count=attrs.episodeCount or 0,
        duration=attrs.episodeLength * 60 if attrs.episodeLength else None,
        year=year,
        season=season,
        nsfw=attrs.nsfw,
        rating=AnimeRating(attrs.ageRating or ""),
        genres=anime.genres,
        themes=anime.themes,
        synonyms=attrs.abbreviatedTitles,
        sources=[f"https://kitsu.io/anime/{attrs.slug}"] if attrs.slug else [],
        images=images,
        statistics=statistics,
        broadcast=broadcast,
        trailers=trailers,
        aired_dates=aired_dates,
    )
    return result.model_dump(mode="json", exclude_none=True)


def character_from_kitsu(char: KitsuMediaCharacter) -> dict[str, Any]:
    """Normalize a KitsuMediaCharacter into canonical Character field values.

    ``char.character``, ``char.voices``, and ``char.animeography``
    must all be populated by the helper before calling this function.

    Args:
        char: Validated media character model with all sub-resources attached.

    Returns:
        Dict of canonical field name → normalized value.

    Raises:
        ValueError: If ``char.character`` is not resolved.
    """
    character = char.character
    if not character:
        raise ValueError(f"char {char.id} has no resolved character")  # noqa: TRY003

    attrs = character.attributes
    names = attrs.names

    sources: list[str] = (
        [f"https://kitsu.io/characters/{attrs.slug}"] if attrs.slug else []
    )
    if attrs.malId:
        sources.append(f"https://myanimelist.net/character/{attrs.malId}")

    voice_actors: list[VoiceActor] = []
    for voice in char.voices:
        person = voice.person
        if not person:
            continue
        p_attrs = person.attributes
        locale = voice.attributes.locale or ""
        voice_actors.append(
            VoiceActor(
                name=p_attrs.name or "Unknown",
                language=_LOCALE_TO_LANGUAGE.get(locale, locale) or None,
                image=p_attrs.image.original if p_attrs.image else None,
                biography=_strip_html(p_attrs.description),
                sources=[f"https://kitsu.app/people/{person.id}"],
            )
        )

    anime_entries: list[Ography] = []
    manga_entries: list[Ography] = []
    for entry in char.animeography:
        media = entry.media
        if not media:
            continue
        m_attrs = media.attributes
        role = CharacterRole(entry.attributes.role or "")
        ography = Ography(
            title=m_attrs.canonicalTitle or "Unknown",
            role=role,
            sources=(
                [f"https://kitsu.io/{entry.media_type}/{m_attrs.slug}"]
                if m_attrs.slug and entry.media_type
                else []
            ),
        )
        if entry.media_type == "manga":
            manga_entries.append(ography)
        else:
            anime_entries.append(ography)

    result = Character(
        name=attrs.canonicalName or attrs.name or "Unknown",
        name_native=names.ja_jp,
        name_variations=attrs.otherNames,
        description=_strip_html(attrs.description),
        images=[attrs.image.original] if attrs.image and attrs.image.original else [],
        roles=[CharacterRole(char.attributes.role or "")],
        sources=sources,
        voice_actors=voice_actors,
        animeography=anime_entries,
        mangaography=manga_entries,
    )
    return result.model_dump(mode="json", exclude_none=True)


def episode_from_kitsu(ep: KitsuEpisode, anime_slug: str | None = None) -> dict[str, Any]:
    """Normalize a KitsuEpisode into canonical Episode field values.

    Args:
        ep: Validated Kitsu episode model.
        anime_slug: Kitsu anime slug (e.g. ``"DAN-DA-DAN"``). When provided, the
            source URL is ``https://kitsu.app/anime/{slug}/episodes/{number}``.
            Falls back to the numeric episode ID URL when omitted.

    Returns:
        Dict of canonical field name → normalized value.
    """
    attrs = ep.attributes
    titles = attrs.titles

    title = (
        attrs.canonicalTitle
        or titles.en_us
        or titles.en_jp
        or f"Episode {attrs.number or '?'}"
    )

    if anime_slug and attrs.number is not None:
        source = f"https://kitsu.app/anime/{anime_slug}/episodes/{attrs.number}"
    else:
        source = None

    result = Episode(
        title=title,
        title_japanese=titles.ja_jp,
        title_romaji=titles.en_jp,
        episode_number=attrs.number or 0,
        season_number=attrs.seasonNumber,
        synopsis=_strip_html(attrs.synopsis or attrs.description),
        aired=normalize_to_utc(attrs.airdate),
        duration=attrs.length * 60 if attrs.length else None,
        images=(
            [attrs.thumbnail.original]
            if attrs.thumbnail and attrs.thumbnail.original
            else []
        ),
        sources=[source] if source else [],
    )
    return result.model_dump(mode="json", exclude_none=True)
