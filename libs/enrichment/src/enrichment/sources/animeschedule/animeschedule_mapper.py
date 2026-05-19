"""AnimSchedule → canonical model mapper.

Pure value normalization. No I/O, no side effects.

AnimSchedule returns clean structured JSON, so mapping is straightforward.
Key quirks handled here:
- Zero-date sentinel "0001-01-01T00:00:00Z" means "not set"
- ``websites`` values are partial URLs (no https:// scheme)
- ``stats.averageScore`` is 0–100 → normalize to 0–10
- ``season.season`` may be "" for year-only entries
- ``relations`` values are slugs; slug is used as fallback title until
  MAL/AniList fill in the real title during consolidation
"""

import re
from typing import Any

from common.models.anime import (
    AiredDates,
    Anime,
    AnimeHiatus,
    AnimeImages,
    AnimeRelationType,
    AnimeSeason,
    AnimeStatus,
    AnimeType,
    Broadcast,
    CompanyEntry,
    RelatedAnime,
    SourceMaterialType,
    Statistics,
    StreamingEntry,
)
from common.utils.datetime_utils import normalize_to_utc

from enrichment.sources.animeschedule.animeschedule_models import AnimScheduleAnime
from enrichment.utils.text_utils import normalize_score

# ── Constants ────────────────────────────────────────────────────────────────

_ZERO_DATE = "0001-01-01"
_IMAGE_CDN = "https://img.animeschedule.net/production/assets/public/img/"
_BASE_URL = "https://animeschedule.net/anime/"

# AnimSchedule website keys that map to cross-source source URLs
_WEBSITE_KEYS = ("mal", "aniList", "kitsu", "animePlanet", "anidb")

_HTML_TAG_RE = re.compile(r"<[^>]+>")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _is_zero_date(s: str | None) -> bool:
    """Return True if the date string is the AnimSchedule zero-date sentinel."""
    return not s or s.startswith(_ZERO_DATE)


def _full_url(partial: str) -> str:
    """Prepend https:// to a partial URL if no scheme is present."""
    return partial if partial.startswith("http") else f"https://{partial}"


def _strip_html(html: str) -> str:
    """Strip HTML tags and normalize whitespace."""
    return _HTML_TAG_RE.sub(" ", html).replace("\r", "").strip()


# ── Public mapper ────────────────────────────────────────────────────────────


def anime_from_animeschedule(anime: AnimScheduleAnime) -> dict[str, Any]:
    """Normalize an AnimScheduleAnime into canonical Anime field values.

    Args:
        anime: Validated AnimSchedule API response model.

    Returns:
        Dict of canonical field name → normalized value (exclude_none applied).
    """
    # ── Scalars ───────────────────────────────────────────────────────────
    title = anime.title
    title_japanese = anime.names.native if anime.names else None
    synonyms = anime.names.synonyms if anime.names else []
    status = AnimeStatus(anime.status or "")
    year = anime.year
    episode_count = anime.episodes or 0
    duration = anime.length_min * 60 if anime.length_min else None

    season_str = anime.season.season if anime.season else None
    season = AnimeSeason(season_str.upper()) if season_str else None

    source_material: SourceMaterialType | None = None
    if anime.sources:
        source_material = SourceMaterialType(anime.sources[0]["name"])

    anime_type = AnimeType(anime.media_types[0]["name"] if anime.media_types else "")

    synopsis: str | None = None
    if anime.description:
        synopsis = _strip_html(anime.description) or None

    # ── Sources list (own URL + cross-source links) ───────────────────────
    sources: list[str] = [f"{_BASE_URL}{anime.route}"]
    for key in _WEBSITE_KEYS:
        raw = anime.websites.get(key)
        if isinstance(raw, str) and raw:
            sources.append(_full_url(raw))

    # ── External sources (official website) ──────────────────────────────
    external_sources: dict[str, str] = {}
    official = anime.websites.get("official")
    if isinstance(official, str) and official:
        external_sources["official"] = _full_url(official)

    # ── Genres ───────────────────────────────────────────────────────────
    genres = [g["name"] for g in anime.genres if g.get("name")]

    # ── Studios ──────────────────────────────────────────────────────────
    studios = [
        CompanyEntry(
            name=s["name"],
            sources=[f"https://animeschedule.net/studios/{s['route']}"],
        )
        for s in anime.studios
        if s.get("name")
    ]

    # ── Images ───────────────────────────────────────────────────────────
    covers: list[str] = []
    if anime.image_version_route:
        covers.append(f"{_IMAGE_CDN}{anime.image_version_route}")
    images = AnimeImages(covers=covers)

    # ── Statistics ───────────────────────────────────────────────────────
    statistics: dict[str, Statistics] = {}
    if anime.stats and anime.stats.rating_count:
        stats_data: dict[str, Any] = {}
        if anime.stats.average_score is not None:
            stats_data["score"] = normalize_score(anime.stats.average_score)
        if anime.stats.rating_count is not None:
            stats_data["scored_by"] = anime.stats.rating_count
        if anime.stats.tracked_count is not None:
            stats_data["members"] = anime.stats.tracked_count
        statistics["animeschedule"] = Statistics(**stats_data)

    # ── Aired dates ───────────────────────────────────────────────────────
    aired_dates: AiredDates | None = None
    if not _is_zero_date(anime.premier):
        aired_dates = AiredDates(aired_from=normalize_to_utc(anime.premier))

    # ── Broadcast ────────────────────────────────────────────────────────
    broadcast_fields: dict[str, Any] = {}
    if not _is_zero_date(anime.jpn_time):
        broadcast_fields["jp_time"] = anime.jpn_time
    if not _is_zero_date(anime.sub_time):
        broadcast_fields["sub_time"] = anime.sub_time
    if not _is_zero_date(anime.dub_time):
        broadcast_fields["dub_time"] = anime.dub_time
    if not _is_zero_date(anime.premier):
        broadcast_fields["premiere_jp"] = normalize_to_utc(anime.premier)
    if not _is_zero_date(anime.sub_premier):
        broadcast_fields["premiere_sub"] = normalize_to_utc(anime.sub_premier)
    if not _is_zero_date(anime.dub_premier):
        broadcast_fields["premiere_dub"] = normalize_to_utc(anime.dub_premier)
    broadcast = Broadcast(**broadcast_fields) if broadcast_fields else None

    # ── Hiatus ───────────────────────────────────────────────────────────
    hiatus: AnimeHiatus | None = None
    if not _is_zero_date(anime.delayed_from):
        hiatus = AnimeHiatus(
            reason=anime.delayd_reason,
            hiatus_from=anime.delayed_from,
            hiatus_until=anime.delayed_until
            if not _is_zero_date(anime.delayed_until)
            else None,
        )

    # ── Streaming ────────────────────────────────────────────────────────
    streaming_sources: list[StreamingEntry] = []
    raw_streams = anime.websites.get("streams")
    if isinstance(raw_streams, list):
        for s in raw_streams:
            if isinstance(s, dict) and s.get("platform") and s.get("url"):
                streaming_sources.append(
                    StreamingEntry(platform=s["platform"], source=_full_url(s["url"]))
                )

    # ── Related anime ─────────────────────────────────────────────────────
    # AnimSchedule provides slugs only; slug is used as fallback title until
    # MAL/AniList consolidation fills in real titles and types.
    related_anime: dict[AnimeRelationType, list[RelatedAnime]] = {}
    for rel_key, slugs in anime.relations.items():
        rel_type = AnimeRelationType(rel_key)
        entries = related_anime.setdefault(rel_type, [])
        for slug in slugs:
            entries.append(
                RelatedAnime(
                    title=slug,
                    type=AnimeType.UNKNOWN,
                    sources=[f"{_BASE_URL}{slug}"],
                )
            )

    # ── Assemble ──────────────────────────────────────────────────────────
    result = Anime(
        title=title,
        title_japanese=title_japanese,
        status=status,
        type=anime_type,
        year=year,
        season=season,
        episode_count=episode_count,
        duration=duration,
        source_material=source_material,
        synopsis=synopsis,
        synonyms=synonyms,
        genres=genres,
        studios=studios,
        sources=sources,
        images=images,
        statistics=statistics,
        aired_dates=aired_dates,
        broadcast=broadcast,
        hiatus=hiatus,
        streaming_sources=streaming_sources,
        related_anime=related_anime,
        external_sources=external_sources,
    )

    return result.model_dump(mode="json", exclude_none=True)
