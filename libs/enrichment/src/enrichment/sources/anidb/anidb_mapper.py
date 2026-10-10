"""AniDB → canonical model mapper.

Pure value normalization functions. No I/O, no side effects.

AniDBAnime field names already match canonical fields where possible, so these
functions only:
  1. Normalize enum strings ("TV Series" → AnimeType.TV)
  2. Build nested structures (statistics dict, images dict, external_sources, etc.)
  3. Derive computed fields (status, season, year from dates)
  4. Convert units (episode length: minutes → seconds)
"""

import logging
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
    CompanyEntry,
    Episode,
    ExternalLink,
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
    AniDBExternalResource,
)
from enrichment.sources.base.companies import companies_from_roles
from enrichment.sources.base.external_links import (
    OFFICIAL_SITE,
    PageKind,
    external_link,
    page_link,
)

_CDN_BASE = "https://cdn-eu.anidb.net/images/main"
# AniDB sends this as the start date of works whose date it does not know yet.
_UNKNOWN_DATE = "1970-01-01"

logger = logging.getLogger(__name__)

# External resource type → the platform and page kind its identifier names; the
# page address itself comes from `page_link`. Types not listed are skipped; see
# docs/anidb_type_mappings.md for the full list, including the ones left out.
_RESOURCE_PAGES: dict[str, PageKind] = {
    "1": ("anime_news_network", None),
    "2": ("myanimelist", "anime"),
    "6": ("wikipedia", "en"),
    "7": ("wikipedia", "ja"),
    "8": ("syoboi", None),
    "9": ("allcinema", None),
    "10": ("anison", None),
    "11": ("lain", None),
    "16": ("animemorial", None),
    "17": ("tv_animation_museum", None),
    "19": ("wikipedia", "ko"),
    "20": ("wikipedia", "zh"),
    "22": ("facebook", None),
    "23": ("twitter", None),
    "26": ("youtube", None),
    "28": ("crunchyroll", None),
    "32": ("amazon", None),
    "38": ("bangumi", None),
    "39": ("douban", None),
    "41": ("netflix", None),
    "42": ("hidive", None),
    "43": ("imdb", None),
    "45": ("funimation", None),
    "46": ("qq_video", None),
    "47": ("bilibili", None),
    "48": ("prime_video", None),
}

# Types that supply a full url rather than an identifier, with the language of
# the page where the host cannot reveal it. Type 4 is the Japanese official
# site, as the anime-level <url> is, so the same address collapses into one.
_RESOURCE_URL_TYPES: dict[str, str | None] = {
    "4": "Japanese",
    "5": "English",
    "34": None,
    "35": None,
}

# Url types that are the work's own site: both official sites and its blog.
_OFFICIAL_RESOURCE_TYPES = frozenset({"4", "5", "35"})


def _known_date(value: str | None) -> str | None:
    """Return the date, or None for AniDB's unknown-date placeholder."""
    return None if value == _UNKNOWN_DATE else value


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
    start_date = _known_date(anime.start_date)
    end_date = _known_date(anime.end_date)
    status = determine_anime_status(start_date, end_date)
    season = determine_anime_season(start_date)
    year = determine_anime_year(start_date)

    # ── Arrays ───────────────────────────────────────────────────────────────
    sources = [anidb_url]
    synonyms = anime.synonyms
    tags = list(anime.tags)

    # Append non-hentai category names to tags
    for cat in anime.categories:
        if not cat.hentai and cat.name not in tags:
            tags.append(cat.name)

    # ── Objects / Dicts ──────────────────────────────────────────────────────
    aired_from = normalize_to_utc(start_date)
    aired_to = normalize_to_utc(end_date)
    aired_dates = (
        AiredDates(aired_from=aired_from, aired_to=aired_to)
        if aired_from or aired_to
        else None
    )
    images = AnimeImages(
        covers=[f"{_CDN_BASE}/{anime.picture}"] if anime.picture else []
    )

    # Official titles in other languages (BCP 47) → canonical Anime.titles
    titles: dict[str, str] = dict(anime.title_others)

    external_sources = _external_sources(anime.url, anime.resources, anidb_url)

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

    # ── Companies ─────────────────────────────────────────────────────────────
    # <creators> mixes companies and people under one list, told apart only by
    # the type attribute. Measured over 219 cached responses, "Animation Work"
    # is 72 distinct names and all companies, and "Work" is 66 and all but one.
    # Every other type is people, including two that read like company fields:
    # "Animation Production" held only Shinkai Makoto, and "Original Plan" mixes
    # Bandai and Bushiroad with Tezuka Osamu and Jules Verne.
    def _companies(role: str) -> list[CompanyEntry]:
        return [
            CompanyEntry(
                name=creator.name,
                sources=(
                    [f"https://anidb.net/creator/{creator.id}"] if creator.id else []
                ),
            )
            for creator in anime.creators
            if creator.role == role and creator.name
        ]

    studios = _companies("Animation Work")
    producers = _companies("Work")

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
        companies=companies_from_roles(
            studios=studios,
            producers=producers,
        ),
    )

    return result.model_dump(mode="json", exclude_none=True)


def _external_sources(
    official_site: str | None,
    resources: list[AniDBExternalResource],
    anidb_url: str,
) -> list[ExternalLink]:
    """Build links from the anime's <url> and its <resources>.

    Args:
        official_site: The anime's <url>, its Japanese official site.
        resources: The anime's <resources> entries.
        anidb_url: The AniDB page, for the log line about skipped resources.

    Returns:
        One link per usable resource, in AniDB's order.

    Examples:
        >>> links = _external_sources(
        ...     "http://www.toei-anim.co.jp/tv/onep/",
        ...     [
        ...         AniDBExternalResource(type="2", identifiers=["21"]),
        ...         AniDBExternalResource(type="44", identifiers=["37854", "tv"]),
        ...     ],
        ...     "https://anidb.net/anime/69",
        ... )
        >>> [(link.platform, link.source) for link in links]
        [('official_site', 'http://www.toei-anim.co.jp/tv/onep/'), ('myanimelist', 'https://myanimelist.net/anime/21'), ('themoviedb', 'https://www.themoviedb.org/tv/37854')]
    """
    links = [external_link(official_site, language="Japanese", platform=OFFICIAL_SITE)]
    for resource in resources:
        identifiers = resource.identifiers
        match resource.type:
            case resource_type if resource_type in _RESOURCE_URL_TYPES:
                links.append(
                    external_link(
                        resource.urls[0] if resource.urls else None,
                        language=_RESOURCE_URL_TYPES[resource_type],
                        platform=(
                            OFFICIAL_SITE
                            if resource_type in _OFFICIAL_RESOURCE_TYPES
                            else None
                        ),
                    )
                )
            case "14" if len(identifiers) >= 2:
                # VNDB gives the number and the entry letter apart: ["7721", "v"].
                links.append(page_link("vndb", f"{identifiers[1]}{identifiers[0]}"))
            case "44" if len(identifiers) >= 2:
                # TMDB gives the number and the media type: ["37854", "tv"].
                links.append(
                    page_link("themoviedb", identifiers[0], kind=identifiers[1])
                )
            case "33" if identifiers:
                # Baidu Baike ids can carry a "?fromModule=..." tail.
                links.append(page_link("baidu_baike", identifiers[0].split("?")[0]))
            case resource_type if page := _RESOURCE_PAGES.get(resource_type):
                if len(identifiers) == 1:
                    platform, kind = page
                    links.append(page_link(platform, identifiers[0], kind=kind))
                elif identifiers:
                    # Each identifier is a separate entry on that platform, and
                    # nothing marks which one is this work: taking the first
                    # linked One Piece to MAL 62593, a 2025 special. Lowest-id
                    # was right in only 7 of 8 ambiguous cases, so skip.
                    logger.debug(
                        f"Skipping ambiguous {page[0]} resource for AniDB "
                        f"{anidb_url}: {len(identifiers)} candidates"
                    )
    return [link for link in links if link]


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

    if char.gender:
        result["attributes"] = {"gender": char.gender}

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

    Fields here come from the web page. Gender is the exception: the XML also
    carries it, so this merges into any existing attributes rather than
    replacing them. Called by character_from_anidb.
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
        from_ography = [
            e.role.value for e in ography_entries if e.role != CharacterRole.UNKNOWN
        ]
        merged = list(dict.fromkeys(existing + from_ography))
        if merged:
            result["roles"] = merged

    if page.gender:
        result.setdefault("attributes", {})["gender"] = page.gender
