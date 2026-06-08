"""AniDB → canonical model mapper.

Pure value normalization functions. No I/O, no side effects.

AniDBAnime field names already match canonical fields where possible, so these
functions only:
  1. Normalize enum strings ("TV Series" → AnimeType.TV)
  2. Build nested structures (statistics dict, images dict, external_sources, etc.)
  3. Derive computed fields (status, season, year from dates)
  4. Convert units (episode length: minutes → seconds)
"""

from typing import Any

from common.models.anime import (
    AiredDates,
    Anime,
    AnimeImages,
    AnimeRelationType,
    AnimeStatus,
    AnimeType,
    Character,
    CharacterRole,
    Episode,
    Ography,
    RelatedAnime,
    Statistics,
    VoiceActor,
)
from common.utils.datetime_utils import (
    determine_anime_season,
    determine_anime_status,
    determine_anime_year,
    normalize_to_utc,
)
from enrichment.sources.anidb.anidb_models import (
    AniDBAnime,
    AniDBCharacter,
    AniDBCharacterPage,
    AniDBEpisode,
)

_CDN_BASE = "https://cdn-eu.anidb.net/images/main"

# External resource type → (canonical key, url template).
# {} is replaced with the first identifier or url value.
# Types not listed are silently skipped.
_RESOURCE_MAP: dict[str, tuple[str, str]] = {
    "1":  ("anime_news_network", "https://www.animenewsnetwork.com/encyclopedia/anime.php?id={}"),
    "2":  ("myanimelist", "https://myanimelist.net/anime/{}"),
    "4":  ("official_website", "{}"),   # type 4 supplies a full url, not an identifier
    "6":  ("wikipedia_en", "https://en.wikipedia.org/wiki/{}"),
    "7":  ("wikipedia_jp", "https://ja.wikipedia.org/wiki/{}"),
    "8":  ("syoboi", "http://cal.syoboi.jp/tid/{}"),
    "26": ("youtube", "https://www.youtube.com/{}"),
    "32": ("amazon", "https://www.amazon.com/dp/{}"),
    "41": ("netflix", "https://www.netflix.com/title/{}"),
    "43": ("imdb", "https://www.imdb.com/title/{}"),
    "45": ("hulu", "https://www.hulu.com/series/{}"),
    "28": ("crunchyroll", "https://www.crunchyroll.com/series/{}"),
    "38": ("bangumi", "https://bgm.tv/subject/{}"),
    "39": ("douban", "https://movie.douban.com/subject/{}"),
    "47": ("bilibili", "https://www.bilibili.com/{}"),
}


def anime_from_anidb(anime: AniDBAnime, *, anidb_url: str) -> dict[str, Any]:
    """Normalize an AniDBAnime into canonical Anime field values.

    Args:
        anime: Parsed AniDB source model.
        anidb_url: Original AniDB URL from the seed database, used directly
            as the canonical source URL without reconstruction from the ID.

    Returns:
        Dict of canonical field name → normalized value, suitable for merging
        into the pipeline anime record.
    """
    # ── Scalars ──────────────────────────────────────────────────────────────
    episode_count = anime.episode_count
    synopsis = anime.description
    title = anime.title or anime.title_english or ""
    title_english = anime.title_english
    title_japanese = anime.title_japanese
    nsfw = anime.restricted or None

    anime_type = AnimeType(anime.type or "")
    status = determine_anime_status(anime.start_date, anime.end_date)
    season = determine_anime_season(anime.start_date)
    year = determine_anime_year(anime.start_date)

    # ── Arrays ───────────────────────────────────────────────────────────────
    sources = [anidb_url]
    synonyms = anime.synonyms
    tags = list(anime.tags)

    # Append non-hentai category names to tags
    for cat in anime.categories:
        if not cat.hentai and cat.name not in tags:
            tags.append(cat.name)

    # ── Objects / Dicts ──────────────────────────────────────────────────────
    aired_dates = None
    if anime.start_date or anime.end_date:
        aired_dates = AiredDates(
            aired_from=normalize_to_utc(anime.start_date),
            aired_to=normalize_to_utc(anime.end_date),
        )

    images = AnimeImages(
        covers=[f"{_CDN_BASE}/{anime.picture}"] if anime.picture else []
    )

    # Official titles in other languages (BCP 47) → canonical Anime.titles
    titles: dict[str, str] = dict(anime.title_others)

    # External sources from <resources>
    external_sources: dict[str, str] = {}
    if anime.url:
        external_sources["official_website"] = anime.url
    for resource in anime.resources:
        if resource.type == "33":
            # Baidu Baike identifier may have ?fromModule=... query string — strip it
            if resource.identifiers:
                slug = resource.identifiers[0].split("?")[0]
                external_sources["baidu_baike"] = f"https://baike.baidu.com/item/{slug}"
            continue
        if resource.type == "44":
            # TMDB has two identifiers: numeric id + media type ("tv" or "movie")
            if len(resource.identifiers) >= 2:
                tmdb_id, tmdb_type = resource.identifiers[0], resource.identifiers[1]
                external_sources["themoviedb"] = f"https://www.themoviedb.org/{tmdb_type}/{tmdb_id}"
            continue
        mapping = _RESOURCE_MAP.get(resource.type)
        if mapping is None:
            continue
        key, template = mapping
        # type 4 uses urls list; all others use identifiers list
        value = (resource.urls or resource.identifiers or [None])[0]
        if value:
            external_sources[key] = template.format(value)

    # Statistics from <ratings>
    statistics: dict[str, Statistics] = {}
    if anime.ratings and anime.ratings.permanent is not None:
        stats: dict[str, Any] = {"score": anime.ratings.permanent}
        if anime.ratings.permanent_count:
            stats["scored_by"] = anime.ratings.permanent_count
        statistics["anidb"] = Statistics(**stats)

    # Related anime
    related_anime: dict[AnimeRelationType, list[RelatedAnime]] = {}
    for rel in anime.related_anime:
        rel_type = AnimeRelationType(rel.relation_type)
        entry = RelatedAnime(
            title=rel.title or "",
            type=AnimeType.UNKNOWN,
            sources=[f"https://anidb.net/anime/{rel.id}"],
        )
        related_anime.setdefault(rel_type, []).append(entry)

    # ── Build canonical object ────────────────────────────────────────────────
    result = Anime(
        episode_count=episode_count,
        nsfw=nsfw,
        status=status or AnimeStatus.UNKNOWN,
        synopsis=synopsis,
        title=title,
        title_english=title_english,
        title_japanese=title_japanese,
        type=anime_type,
        year=year,
        season=season,
        sources=sources,
        synonyms=synonyms,
        tags=tags,
        titles=titles,
        aired_dates=aired_dates,
        external_sources=external_sources,
        images=images,
        related_anime=related_anime,
        statistics=statistics,
    )

    return result.model_dump(mode="json", exclude_none=True)


def episode_from_anidb(
    ep: AniDBEpisode,
    *,
    anime_id: str | None = None,
) -> dict[str, Any] | None:
    """Normalize an AniDBEpisode into canonical Episode field values.

    Only regular episodes (episode_type == 1) are mapped; all others return None
    so the caller can filter them out.

    Args:
        ep: Parsed AniDB episode model.
        anime_id: Optional UUID of the parent anime to inject as episode.anime_id.

    Returns:
        Dict of canonical Episode field name → value, or None for non-regular episodes.
    """
    if ep.episode_type != 1:
        return None

    if not isinstance(ep.episode_number, int):
        return None

    # Title fallback: english → romaji → empty string
    title = ep.titles.get("en") or ep.titles.get("romaji") or ""
    title_japanese = ep.titles.get("ja") or None
    title_romaji = ep.titles.get("romaji") or None

    # Remaining non-standard lang codes → Episode.titles
    skip = {"en", "ja", "romaji"}
    extra_titles = {k: v for k, v in ep.titles.items() if k not in skip}

    aired = normalize_to_utc(ep.airdate) if ep.airdate else None
    duration = ep.length * 60 if ep.length else None  # minutes → seconds

    sources = [f"https://anidb.net/episode/{ep.id}"] if ep.id else []

    episode = Episode(
        aired=aired,
        anime_id=anime_id,
        duration=duration,
        episode_number=ep.episode_number,
        filler=False,
        recap=False,
        synopsis=ep.summary,
        title=title,
        title_japanese=title_japanese,
        title_romaji=title_romaji,
        titles=extra_titles,
        sources=sources,
        streaming=ep.streaming,
    )

    return episode.model_dump(mode="json", exclude_none=True)


def character_from_anidb(
    char: AniDBCharacter,
    page_data: AniDBCharacterPage | None = None,
) -> dict[str, Any]:
    """Normalize an AniDBCharacter into canonical Character field values.

    Args:
        char: Parsed AniDB character model from XML.
        page_data: Optional enrichment from anidb_character_crawler web page.

    Returns:
        Dict of canonical Character field name → value.
    """
    result: dict[str, Any] = {
        "name": char.name or "",
    }

    if char.id:
        result["sources"] = [f"https://anidb.net/character/{char.id}"]

    if char.description:
        result["description"] = char.description

    if char.picture:
        result["images"] = [f"{_CDN_BASE}/{char.picture}"]

    if char.type:
        role = CharacterRole(char.type)
        result["roles"] = [role.value]

    if char.seiyuu:
        result["voice_actors"] = [
            VoiceActor(
                name=s.name or "",
                image=f"{_CDN_BASE}/{s.picture}" if s.picture else None,
                sources=[f"https://anidb.net/creator/{s.id}"] if s.id else [],
            ).model_dump(mode="json", exclude_none=True)
            for s in char.seiyuu
        ]

    if page_data:
        _apply_page_data(result, page_data)

    character = Character.model_validate(result)
    return character.model_dump(mode="json", exclude_none=True)


def _apply_page_data(result: dict[str, Any], page: AniDBCharacterPage) -> None:
    """Merge AniDBCharacterPage fields into a canonical character result dict.

    All fields here come exclusively from the web page — not from the XML API.
    Called by character_from_anidb.
    """
    if page.name_kanji:
        result["name_native"] = page.name_kanji
    if page.description:
        result.setdefault("description", page.description)
    if page.nicknames:
        result["nicknames"] = page.nicknames
    if page.official_names:
        result["name_variations"] = page.official_names

    traits = (
        page.abilities
        + page.looks
        + page.personality
        + page.role
        + page.supernatural_abilities
    )
    if traits:
        result["traits"] = traits

    if page.animeography:
        ography_entries = [
            Ography(
                title=e["title"],
                role=CharacterRole(e.get("role", "")),
                sources=[e["url"]] if e.get("url") else [],
            )
            for e in page.animeography
            if e.get("title")
        ]
        result["animeography"] = ography_entries

        # Derive unique roles from all anime appearances and merge with any
        # role already set from the XML API (e.g. role for the queried anime).
        existing = list(result.get("roles", []))
        from_ography = [e.role.value for e in ography_entries if e.role != CharacterRole.UNKNOWN]
        merged = list(dict.fromkeys(existing + from_ography))
        if merged:
            result["roles"] = merged

    if page.gender:
        result["attributes"] = {"gender": page.gender}
