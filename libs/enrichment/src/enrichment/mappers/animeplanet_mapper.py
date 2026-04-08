"""AnimePlanet → canonical model mapper.

Pure value normalization — no I/O, no side effects.

Receives a validated AnimePlanetAnime source model and returns a canonical
Anime dict.  All field renaming, derivation (year, season, status), and
stat normalization happen here — not in the crawler.

Key AP-specific details:
- Ratings: AP uses 0–5 scale → multiply × 2 for canonical 0–10
- Season: derived from /anime/seasons/<season>-<year> href slug
- Type:   JSON-LD @type ("TVSeries" → TV, "Movie" → MOVIE) — coarse but reliable
- Related anime subtype mapping documented in docs/source_api_field_mappings.md
- Manga entries with subtype "Original Manga" → related_source_material (ADAPTATION)
- All other manga entries → related_source_material (OTHER)
"""

import re
from typing import Any

from common.models.anime import (
    AiredDates,
    Anime,
    AnimeImages,
    AnimeRelationType,
    AnimeSeason,
    AnimeType,
    CompanyEntry,
    RelatedAnime,
    RelatedSourceMaterial,
    SourceMaterialRelationType,
    SourceMaterialType,
    Statistics,
)
from common.utils.datetime_utils import (
    determine_anime_season,
    determine_anime_status,
    determine_anime_year,
    normalize_to_utc,
)

from enrichment.crawlers.anime_planet.anime_planet_models import AnimePlanetAnime

_SEASON_SLUG_RE = re.compile(r"/seasons/([^/?#]+)")
_RANK_RE = re.compile(r"#(\d+)")


def _parse_season_from_url(season_url: str | None) -> AnimeSeason | None:
    """Parse AnimeSeason from AP season href like '/anime/seasons/fall-1999'."""
    if not season_url:
        return None
    match = _SEASON_SLUG_RE.search(season_url)
    if not match:
        return None
    season_name = match.group(1).split("-")[0]  # "fall-1999" → "fall"
    return AnimeSeason(season_name)             # _missing_ handles lowercase


def _parse_rank(rank_text: str | None) -> int | None:
    """Parse rank integer from text like 'Rank #157' → 157."""
    if not rank_text:
        return None
    match = _RANK_RE.search(rank_text)
    return int(match.group(1)) if match else None


def _parse_aka(aka: str | None) -> str | None:
    """Strip 'Alt title: ' prefix and return the bare title."""
    if not aka:
        return None
    text = aka.strip()
    lower = text.lower()
    if lower.startswith("alt title:"):
        text = text[len("alt title:"):].strip()
    return text or None


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

    # Season: prefer the dedicated season URL slug; fall back to start_date
    season = _parse_season_from_url(anime.season_url)
    if season is None and anime.start_date:
        season = determine_anime_season(anime.start_date)

    year = determine_anime_year(anime.start_date) if anime.start_date else None
    status = determine_anime_status(anime.start_date, anime.end_date)
    title_japanese = _parse_aka(anime.aka)

    # ── Aired dates ───────────────────────────────────────────────────────
    aired_dates = None
    if anime.start_date or anime.end_date:
        aired_dates = AiredDates(
            aired_from=normalize_to_utc(anime.start_date),
            aired_to=normalize_to_utc(anime.end_date),
        )

    # ── Statistics ────────────────────────────────────────────────────────
    statistics: dict[str, Statistics] = {}
    rank = _parse_rank(anime.rank_text)
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
        title_japanese=title_japanese,
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
