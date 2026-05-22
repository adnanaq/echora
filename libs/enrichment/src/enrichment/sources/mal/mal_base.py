"""Shared infrastructure for all MAL crawlers.

Provides:
- Sidebar parser utilities (text-anchor extraction for MAL sidebar fields)
- ID extraction helpers (URL → numeric MAL ID)
- Number parsing utilities ("2,644,378" → int, "#17" → int)
- Model-level diffing (field-by-field comparison, no HTML hashing)
- Rate limiters: scraping (2s/25rpm), shared singleton (0.5s/60rpm)

All crawlers import from this module — no duplicated boilerplate.
"""

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from enrichment.sources.base.crawler_config import CrawlerRateLimiter
from enrichment.sources.base.utils import (
    parse_broadcast_string as parse_broadcast_string,
)  # noqa: F401
from enrichment.sources.base.utils import parse_iso_date as parse_iso_date  # noqa: F401
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# MAL base URL
MAL_BASE_URL = "https://myanimelist.net"


# =============================================================================
# RATE LIMITER
# =============================================================================


def get_mal_scraping_limiter() -> CrawlerRateLimiter:
    """Create a MAL scraping rate limiter with conservative timing.

    Uses 2s intervals and 25 requests/minute (vs 0.5s/60rpm for Jikan).
    Scraping is heavier than API calls — be respectful to MAL servers.
    """
    return CrawlerRateLimiter(min_interval_seconds=2.0, max_per_minute=25)


@lru_cache(maxsize=1)
def get_shared_mal_rate_limiter() -> CrawlerRateLimiter:
    """Return a process-wide shared limiter instance for all MAL requests."""
    return CrawlerRateLimiter(min_interval_seconds=0.5, max_per_minute=60)


# =============================================================================
# SIDEBAR PARSING UTILITIES
# =============================================================================


def parse_sidebar_field(sidebar_html: str, label: str) -> str | None:
    """Extract text value following a label in MAL sidebar HTML.

    MAL sidebar format:
        <span class="dark_text">Label:</span> Value text here<br>

    Args:
        sidebar_html: Raw HTML of the sidebar div (div.leftside or similar).
        label: The label text to search for (e.g., "Episodes", "Status").

    Returns:
        Stripped text value, or None if not found.

    Example:
        >>> parse_sidebar_field(html, "Episodes")
        "1122"
        >>> parse_sidebar_field(html, "Status")
        "Currently Airing"
    """
    # Escape label for regex
    escaped = re.escape(label)
    # Find the span with dark_text containing our label, then grab text after it
    # MAL uses: <span class="dark_text">Label:</span>\n value text
    pattern = rf'<span[^>]*class="dark_text"[^>]*>{escaped}:?</span>\s*(.*?)(?:<br|</div|<span)'
    match = re.search(pattern, sidebar_html, re.DOTALL | re.IGNORECASE)
    if not match:
        return None

    raw = match.group(1)
    # Strip HTML tags from the value portion
    text = re.sub(r"<[^>]+>", " ", raw)
    cleaned = " ".join(text.split()).strip(" ,")
    return cleaned if cleaned and cleaned not in ("N/A", "?", "None", "") else None


# =============================================================================
# URL / NUMBER UTILITIES
# =============================================================================


def normalize_mal_anime_url(url: str) -> tuple[str, bool]:
    """Validate and inspect a MAL anime URL.

    Args:
        url: MAL anime URL — slugged or slugless.
            e.g. "https://myanimelist.net/anime/21" or
                 "https://myanimelist.net/anime/21/One_Piece"

    Returns:
        (url, has_slug) — url unchanged, has_slug True when a slug segment
        follows the numeric ID.

    Raises:
        ValueError: If url does not start with MAL_BASE_URL or does not
            match the /anime/{id} pattern.
    """
    if not url.startswith(MAL_BASE_URL):
        raise ValueError(f"URL must start with {MAL_BASE_URL!r}, got: {url!r}")
    if not re.search(r"/anime/\d+", url):
        raise ValueError(f"URL does not match /anime/{{id}} pattern: {url!r}")
    has_slug = bool(re.search(r"/anime/\d+/[^/]", url))
    return url, has_slug


def parse_number(s: str | None) -> int | None:
    """Parse a formatted number string to int.

    Handles:
        "2,644,378" → 2644378
        "#17"       → 17
        "#54"       → 54
        "N/A"       → None
        None        → None

    Args:
        s: String representation of a number (possibly with commas or # prefix).

    Returns:
        Integer value, or None if parsing fails.
    """
    if not s:
        return None
    cleaned = re.sub(r"[,#\s]", "", s.strip())
    try:
        return int(cleaned)
    except ValueError:
        return None


def parse_duration_seconds(raw: str | None) -> int | None:
    """Parse a MAL duration string to seconds.

    Handles:
        "24 min."       → 1440
        "1 hr. 30 min." → 5400
        "00:24:37"      → 1477 (HH:MM:SS format from episode pages)
        "2 min."        → 120

    Args:
        raw: Raw duration string from MAL.

    Returns:
        Duration in seconds, or None if parsing fails.
    """
    if not raw:
        return None

    # HH:MM:SS format (episode pages)
    hms = re.match(r"^(\d+):(\d{2}):(\d{2})$", raw.strip())
    if hms:
        h, m, s = int(hms.group(1)), int(hms.group(2)), int(hms.group(3))
        return h * 3600 + m * 60 + s

    # "X hr. Y min." or "X hr." or "Y min."
    hours = re.search(r"(\d+)\s*hr", raw, re.IGNORECASE)
    minutes = re.search(r"(\d+)\s*min", raw, re.IGNORECASE)
    h = int(hours.group(1)) if hours else 0
    m = int(minutes.group(1)) if minutes else 0
    total = h * 3600 + m * 60
    return total if total > 0 else None


def parse_aired_string(aired_raw: str | None) -> tuple[str | None, str | None]:
    """Parse a MAL aired date range string into (from_date, to_date) ISO strings.

    Handles:
        "Oct 20, 1999 to ?"           → ("1999-10-20", None)
        "Oct 20, 1999 to Nov 5, 2000" → ("1999-10-20", "2000-11-05")
        "Apr 5, 2003"                 → ("2003-04-05", None)  (movie / single date)

    Args:
        aired_raw: Raw aired string from MAL sidebar.

    Returns:
        Tuple of (ISO from_date, ISO to_date), either may be None.
    """
    if not aired_raw:
        return None, None

    if " to " in aired_raw:
        parts = aired_raw.split(" to ", 1)
        from_date = parse_iso_date(parts[0].strip())
        to_date = parse_iso_date(parts[1].strip())
    else:
        from_date = parse_iso_date(aired_raw.strip())
        to_date = None

    return from_date, to_date


def parse_premiered(premiered_raw: str | None) -> tuple[str | None, int | None]:
    """Parse a MAL 'Premiered' value into (season, year).

    Handles:
        "Fall 1999"   → ("fall", 1999)
        "Spring 2024" → ("spring", 2024)
        None          → (None, None)

    Args:
        premiered_raw: Raw premiered string from MAL sidebar link text.

    Returns:
        Tuple of (lowercased season string, year int), either may be None.
    """
    if not premiered_raw:
        return None, None

    match = re.match(
        r"(spring|summer|fall|winter)\s+(\d{4})", premiered_raw.strip(), re.IGNORECASE
    )
    if match:
        return match.group(1).lower(), int(match.group(2))
    return None, None


def parse_episode_ranges(raw: str | None) -> list[tuple[int, int | None]]:
    """Parse MAL episode range strings into a list of (start, end) tuples.

    Handles: "1-30", "492", "1139-", "1-30, 492, 1139-".
    Returns tuples of (start, end) where end is None for open ranges.
    """
    if not raw:
        return []

    # Strip non-range characters (like "eps", "(", ")")
    clean = re.sub(r"[^\d\s\-,]", "", str(raw))
    parts = [p.strip() for p in clean.split(",") if p.strip()]

    ranges = []
    for part in parts:
        if "-" in part:
            subparts = part.split("-")
            try:
                # Handle start
                s_str = subparts[0].strip()
                start = int(s_str) if s_str else None

                # Handle end
                e_str = subparts[1].strip() if len(subparts) > 1 else None
                end = int(e_str) if e_str else None

                if start is not None:
                    ranges.append((start, end))
            except ValueError:
                continue
        else:
            try:
                val = int(part.strip())
                ranges.append((val, val))
            except ValueError:
                continue
    return ranges


# =============================================================================
# MODEL DIFFING
# =============================================================================


@dataclass
class FieldChange:
    """A single changed field between two scraped models."""

    field: str
    old_value: Any
    new_value: Any


@dataclass
class ModelDiff:
    """Field-level diff result for a scraped model."""

    entity_type: str  # "anime", "character", "episode"
    entity_id: str | int  # URL for anime, numeric mal_id for characters/episodes
    changes: list[FieldChange] = field(default_factory=list)
    is_new: bool = False

    @property
    def has_changes(self) -> bool:
        """True if the entity is new or has any changed fields."""
        return self.is_new or len(self.changes) > 0


@dataclass
class ListDiff:
    """Set-level diff for a list of scraped models keyed by url or mal_id."""

    added: list[str | int] = field(default_factory=list)
    removed: list[str | int] = field(default_factory=list)
    updated: list[ModelDiff] = field(default_factory=list)


def _get_entity_id(model: BaseModel) -> str | int:
    """Extract a unique entity key from any scraped model.

    Tries episode_number first (MalEpisode), then falls back
    to url (MalAnime, MalCharacter).

    Returns 0 when none are found.
    """
    for attr in ("episode_number",):
        v = getattr(model, attr, None)
        if v is not None:
            return int(v)
    source = getattr(model, "source", None)
    if source:
        return str(source)
    return 0


def diff_models(old: BaseModel | None, new: BaseModel, entity_type: str) -> ModelDiff:
    """Compare two Pydantic models field-by-field.

    This approach is more precise than HTML hashing — it produces exact field
    names for database updates and ignores cosmetic HTML changes (ads, tokens).

    Args:
        old: Previous scraped model (None if this is a first-time scrape).
        new: Newly scraped model.
        entity_type: String like "anime", "character", or "episode".

    Returns:
        ModelDiff containing all changed fields, or is_new=True if no prior version.
    """
    entity_id: str | int = _get_entity_id(new)
    if old is None:
        return ModelDiff(entity_type=entity_type, entity_id=entity_id, is_new=True)

    old_data = old.model_dump()
    new_data = new.model_dump()
    changes = []
    for key in new_data:
        if old_data.get(key) != new_data[key]:
            changes.append(
                FieldChange(
                    field=key,
                    old_value=old_data.get(key),
                    new_value=new_data[key],
                )
            )
    return ModelDiff(entity_type=entity_type, entity_id=entity_id, changes=changes)


def diff_model_lists(
    old_models: list[BaseModel],
    new_models: list[BaseModel],
    entity_type: str,
) -> ListDiff:
    """Set-level diff for a list of models keyed by mal_id.

    Detects additions, removals, and changed fields across both lists.

    Args:
        old_models: Previously scraped list.
        new_models: Newly scraped list.
        entity_type: String label for the entity type.

    Returns:
        ListDiff with added/removed/updated model IDs and diffs.
    """

    def _get_id(m: BaseModel) -> str | int | None:
        v = _get_entity_id(m)
        return v if v != 0 else None

    old_by_id: dict[str | int, BaseModel] = {
        mid: m for m in old_models if (mid := _get_id(m)) is not None
    }
    new_by_id: dict[str | int, BaseModel] = {
        mid: m for m in new_models if (mid := _get_id(m)) is not None
    }

    added = [mid for mid in new_by_id if mid not in old_by_id]
    removed = [mid for mid in old_by_id if mid not in new_by_id]
    updated = []
    for mid in new_by_id:
        if mid in old_by_id:
            d = diff_models(old_by_id[mid], new_by_id[mid], entity_type)
            if d.has_changes:
                updated.append(d)

    return ListDiff(added=added, removed=removed, updated=updated)
