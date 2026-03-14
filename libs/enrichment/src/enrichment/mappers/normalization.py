
"""Shared normalization lookup tables for all 7 enrichment sources.
All lookups are keyed by lowercase source strings → canonical enum value string.
Used by all source-specific mappers (mal_mapper, anilist_mapper, kitsu_mapper, etc.).
Sources covered: MAL, AniList, Kitsu, AniDB, AniSearch, Anime-Planet, AnimSchedule.
"""
import re
from typing import Any

# =============================================================================
# SOURCE MATERIAL
# =============================================================================

SOURCE_MATERIAL: dict[str, str] = {
    "manga": "MANGA",
    "one_shot": "MANGA",
    "4-koma manga": "MANGA",
    "doujinshi": "MANGA",
    "light novel": "LIGHT NOVEL",
    "light_novel": "LIGHT NOVEL",
    "novel": "NOVEL",
    "visual novel": "VISUAL NOVEL",
    "visual_novel": "VISUAL NOVEL",
    "game": "GAME",
    "video_game": "GAME",
    "card game": "CARD GAME",
    "card_game": "CARD GAME",
    "music": "MUSIC",
    "radio": "RADIO",
    "book": "BOOK",
    "picture book": "PICTURE BOOK",
    "picture_book": "PICTURE BOOK",
    "original": "ORIGINAL",
    "mixed media": "MIXED MEDIA",
    "mixed_media": "MIXED MEDIA",
    "web manga": "WEB MANGA",
    "web_manga": "WEB MANGA",
    "web novel": "WEB NOVEL",
    "web_novel": "WEB NOVEL",
    "other": "OTHER",
    "unknown": "UNKNOWN",
    "": "UNKNOWN",
}


# =============================================================================
# ANIME RELATION TYPE  (anime ↔ anime)
# =============================================================================

ANIME_RELATION: dict[str, str] = {
    # MAL
    "sequel": "SEQUEL",
    "prequel": "PREQUEL",
    "alternative version": "ALTERNATIVE VERSION",
    "alternative setting": "ALTERNATIVE SETTING",
    "side story": "SIDE_STORY",
    "full story": "FULL_STORY",
    "parent story": "PARENT_STORY",
    "spin-off": "SPIN_OFF",
    "spinoff": "SPIN_OFF",
    "summary": "SUMMARY",
    "adaptation": "ADAPTATION",
    "character": "CHARACTER",
    "crossover": "CROSSOVER",
    "other": "OTHER",
    # AniList — maps single ALTERNATIVE to both; use version as default
    "alternative": "ALTERNATIVE VERSION",
    "side_story": "SIDE_STORY",
    "parent": "PARENT_STORY",
    "spin_off": "SPIN_OFF",
}


# =============================================================================
# SOURCE MATERIAL RELATION TYPE  (anime ↔ source)
# =============================================================================

SOURCE_RELATION: dict[str, str] = {
    "adaptation": "ADAPTATION",
    "source": "SOURCE",
    "alternative": "ALTERNATIVE",
    "spin-off": "SPIN_OFF",
    "spinoff": "SPIN_OFF",
    "other": "OTHER",
}


# =============================================================================
# CONTENT WARNING TAGS  (from MAL Genres / AniList tags)
# =============================================================================

CONTENT_WARNING_TAGS: set[str] = {
    "ecchi",
    "hentai",
    "erotica",
    "nudity",
    "sexual content",
    "explicit sexual content",
    "graphic violence",
    "gore",
    "sexual violence",
    "rx - hentai",
}


# =============================================================================
# DURATION PARSING
# =============================================================================

# Matches "24 min." or "1 hr. 30 min." or "00:24:37"
DURATION_RE = re.compile(
    r"(?:(\d+)\s*hr\.?)?\s*(?:(\d+)\s*min\.?)?|(\d{1,2}):(\d{2}):(\d{2})"
)


def parse_duration(raw: str | None) -> int | None:
    """Parse a human-readable duration string to seconds.

    Handles MAL sidebar format ("24 min.", "1 hr. 30 min.") and episode page
    format ("00:24:37").

    Args:
        raw: Raw duration string.

    Returns:
        Duration in seconds, or None if unparseable.
    """
    if not raw:
        return None

    raw = raw.strip()

    # HH:MM:SS (episode pages)
    hms = re.match(r"^(\d{1,2}):(\d{2}):(\d{2})$", raw)
    if hms:
        return int(hms.group(1)) * 3600 + int(hms.group(2)) * 60 + int(hms.group(3))

    hours = re.search(r"(\d+)\s*hr", raw, re.IGNORECASE)
    minutes = re.search(r"(\d+)\s*min", raw, re.IGNORECASE)
    h = int(hours.group(1)) if hours else 0
    m = int(minutes.group(1)) if minutes else 0
    total = h * 3600 + m * 60
    return total if total > 0 else None


# =============================================================================
# THEME SONG PARSING
# =============================================================================

# MAL format: '1: "Title" by Artist (eps 1-26)'
THEME_RE = re.compile(
    r"(?:\d+:\s*)?"  # Optional "1: " prefix
    r'"([^"]+)"'  # Title in double quotes
    r"(?:\s+by\s+([^(]+?))?"  # Optional "by Artist"
    r"(?:\s+\(eps?\s+([^)]+)\))?"  # Optional "(eps 1-26)"
    r"\s*$",
    re.IGNORECASE,
)


def parse_theme_song(raw: str | None) -> dict[str, Any] | None:
    """Parse a raw MAL theme song string into a structured dict.

    Args:
        raw: Raw theme string like '1: "Hands Up!" by V6 (eps 1-26)'.

    Returns:
        Dict with keys title, artist, episodes, or None if unparseable.
    """
    if not raw:
        return None

    match = THEME_RE.search(raw.strip())
    if not match:
        # Fallback: return raw title
        return {"title": raw.strip(), "artist": None, "episodes": None}

    return {
        "title": match.group(1).strip(),
        "artist": match.group(2).strip() if match.group(2) else None,
        "episodes": match.group(3).strip() if match.group(3) else None,
    }
