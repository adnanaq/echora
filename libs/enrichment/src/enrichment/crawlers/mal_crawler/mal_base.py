"""Shared infrastructure for all MAL crawlers.

Provides:
- Anti-detection layers (stealth → curl_cffi → warmup cookie → undetected)
- Sidebar parser utilities (text-anchor extraction for MAL sidebar fields)
- ID extraction helpers (URL → numeric MAL ID)
- Number parsing utilities ("2,644,378" → int, "#17" → int)
- Model-level diffing (field-by-field comparison, no HTML hashing)
- Rate limiter factory (2s intervals for scraping vs 0.5s for Jikan)

All crawlers import from this module — no duplicated boilerplate.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, cast

from crawl4ai import BrowserConfig
from pydantic import BaseModel

from enrichment.api_helpers.mal_rate_limiter import MalRateLimiter

logger = logging.getLogger(__name__)

# MAL base URL
MAL_BASE_URL = "https://myanimelist.net"

# Default browser config for MAL crawling
_MAL_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


# =============================================================================
# ANTI-DETECTION LAYERS
# =============================================================================


class AntiDetectionLayer(str, Enum):
    """Progressive escalation strategy for bypassing MAL bot detection.

    Start with STEALTH (cheapest). On 403/block, escalate to the next layer.
    Most MAL pages work fine with STEALTH alone.
    """

    STEALTH = "stealth"  # Layer 1: Crawl4AI headless + enable_stealth=True
    CURL_CFFI = "curl_cffi"  # Layer 2: TLS impersonation (no browser overhead)
    WARMUP_COOKIE = "warmup"  # Layer 3: Solve challenge once, reuse cf_clearance cookie
    UNDETECTED = "undetected"  # Layer 4: Camoufox / fingerprint spoofing
    RESIDENTIAL_PROXY = "proxy"  # Layer 5: Rotating residential proxies


def get_browser_config(
    layer: AntiDetectionLayer = AntiDetectionLayer.STEALTH,
    cookies: list[dict[str, str]] | None = None,
) -> BrowserConfig:
    """Build a BrowserConfig for the given anti-detection layer.

    Args:
        layer: Which anti-detection layer to configure. Defaults to STEALTH.
        cookies: Optional cookies to inject (used for WARMUP_COOKIE layer).

    Returns:
        BrowserConfig ready for AsyncWebCrawler.
    """
    base_kwargs: dict[str, Any] = {
        "headless": True,
        "verbose": False,
        "headers": _MAL_BROWSER_HEADERS,
        "viewport_width": 1920,
        "viewport_height": 1080,
    }

    if layer == AntiDetectionLayer.STEALTH:
        base_kwargs["enable_stealth"] = True

    elif layer == AntiDetectionLayer.WARMUP_COOKIE:
        base_kwargs["enable_stealth"] = True
        if cookies:
            base_kwargs["cookies"] = cookies

    elif layer == AntiDetectionLayer.UNDETECTED:
        # Camoufox / UndetectedAdapter — full fingerprint spoofing
        # For now falls back to stealth with extra settings
        base_kwargs["enable_stealth"] = True
        base_kwargs["override_navigator"] = True

    return BrowserConfig(**base_kwargs)


# =============================================================================
# RATE LIMITER
# =============================================================================


def get_mal_docker_browser_config() -> dict[str, Any]:
    """Browser config dict for the crawl4ai Docker REST API (MAL stealth layer).

    Mirrors ``get_browser_config(AntiDetectionLayer.STEALTH)`` but serialized as
    the type-params wrapper required by the Docker REST API.
    """
    return {
        "type": "BrowserConfig",
        "params": {
            "headless": True,
            "verbose": False,
            "enable_stealth": True,
            "user_agent_mode": "random",
            "headers": _MAL_BROWSER_HEADERS,
            "viewport_width": 1920,
            "viewport_height": 1080,
        },
    }


def get_mal_docker_crawler_config(
    schema: dict[str, Any],
    *,
    strategy_type: str = "JsonXPathExtractionStrategy",
    wait_until: str = "domcontentloaded",
    delay: float = 1.0,
    magic: bool = False,
) -> dict[str, Any]:
    """Crawler config dict for the crawl4ai Docker REST API with structured extraction.

    Args:
        schema: Extraction schema dict (from the caller's ``_get_*_schema()``).
        strategy_type: Extraction strategy class name — ``"JsonXPathExtractionStrategy"``
            (default) or ``"JsonCssExtractionStrategy"``.
        wait_until: Page load event to wait for before extracting.
        delay: Seconds to wait before returning HTML (allows JS rendering).
        magic: Enable crawl4ai magic mode (extra anti-bot measures).
    """
    params: dict[str, Any] = {
        "extraction_strategy": {
            "type": strategy_type,
            "params": {"schema": schema},
        },
        "delay_before_return_html": delay,
        # "simulate_user": True,  # TEST: disabled to check simulation overhead
        "wait_until": wait_until,
    }
    if magic:
        params["magic"] = True
    return {"type": "CrawlerRunConfig", "params": params}


def get_mal_scraping_limiter() -> MalRateLimiter:
    """Create a MAL scraping rate limiter with conservative timing.

    Uses 2s intervals and 25 requests/minute (vs 0.5s/60rpm for Jikan).
    Scraping is heavier than API calls — be respectful to MAL servers.
    """
    return MalRateLimiter(min_interval_seconds=2.0, max_per_minute=25)


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


def parse_sidebar_links(sidebar_html: str, label: str) -> list[dict[str, Any]]:
    """Extract links with mal_id from a MAL sidebar section.

    Used for producers, studios, genres, themes, demographics — any sidebar
    field that contains multiple anchor tags pointing to MAL entities.

    Args:
        sidebar_html: Raw HTML of the sidebar.
        label: The label text (e.g., "Studios", "Genres").

    Returns:
        List of dicts with keys: name (str), url (str), mal_id (int | None).
    """
    escaped = re.escape(label)
    # Find the block between this label and the next span.dark_text
    block_pattern = (
        rf'<span[^>]*class="dark_text"[^>]*>{escaped}:?</span>(.*?)'
        rf'(?:<span[^>]*class="dark_text"|$)'
    )
    block_match = re.search(block_pattern, sidebar_html, re.DOTALL | re.IGNORECASE)
    if not block_match:
        return []

    block = block_match.group(1)
    results = []
    for link_match in re.finditer(
        r'<a[^>]*href="([^"]*myanimelist[^"]*)"[^>]*>(.*?)</a>', block, re.DOTALL
    ):
        url = link_match.group(1)
        name = re.sub(r"<[^>]+>", "", link_match.group(2)).strip()
        if not name:
            continue
        results.append(
            {
                "name": name,
                "url": url,
                "mal_id": extract_id_from_url(url),
            }
        )
    return results


def parse_sidebar_link_texts(sidebar_html: str, label: str) -> list[str]:
    """Extract just the text values from sidebar links (for genres, themes, etc.)."""
    return [item["name"] for item in parse_sidebar_links(sidebar_html, label)]


# =============================================================================
# URL / NUMBER UTILITIES
# =============================================================================


def extract_id_from_url(url: str) -> int | None:
    """Extract a numeric MAL entity ID from a URL.

    Handles paths like:
        /anime/21/One_Piece     → 21
        /character/40           → 40
        /people/123/Name        → 123
        /manga/456              → 456

    Args:
        url: MAL URL string.

    Returns:
        Integer ID, or None if not found.
    """
    match = re.search(r"/(?:anime|character|people|manga|producer)/(\d+)", url)
    return int(match.group(1)) if match else None


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


def parse_iso_date(raw: str | None) -> str | None:
    """Parse a MAL date string to an ISO 8601 date string (YYYY-MM-DD).

    Handles:
        "Oct 20, 1999"    → "1999-10-20"
        "Oct  20, 1999"   → "1999-10-20" (extra space)
        "?"               → None
        None              → None

    Args:
        raw: Raw date string from MAL.

    Returns:
        ISO date string "YYYY-MM-DD", or None if not parseable.
    """
    if not raw or raw.strip() in ("?", "N/A", ""):
        return None

    months = {
        "jan": "01",
        "feb": "02",
        "mar": "03",
        "apr": "04",
        "may": "05",
        "jun": "06",
        "jul": "07",
        "aug": "08",
        "sep": "09",
        "oct": "10",
        "nov": "11",
        "dec": "12",
    }

    raw = raw.strip()

    # "Oct 20, 1999" or "Oct  20, 1999"
    match = re.match(r"(\w{3})\s+(\d{1,2}),\s*(\d{4})", raw, re.IGNORECASE)
    if match:
        mon_str = match.group(1).lower()
        day = match.group(2).zfill(2)
        year = match.group(3)
        mon = months.get(mon_str)
        if mon:
            return f"{year}-{mon}-{day}"

    # Already ISO-ish "1999-10-20"
    if re.match(r"^\d{4}-\d{2}-\d{2}$", raw):
        return raw

    return None


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


def parse_broadcast_string(
    broadcast_raw: str | None,
) -> tuple[str | None, str | None, str | None]:
    """Parse a MAL broadcast string into (day, time, timezone).

    Handles:
        "Sundays at 23:15 (JST)"    → ("Sundays", "23:15", "JST")
        "Saturdays at 00:00 (JST)"  → ("Saturdays", "00:00", "JST")
        "Unknown"                   → (None, None, None)

    Args:
        broadcast_raw: Raw broadcast string from MAL sidebar.

    Returns:
        Tuple of (day, time, timezone), any may be None.
    """
    if not broadcast_raw or broadcast_raw.strip().lower() in ("unknown", "n/a", ""):
        return None, None, None

    match = re.match(r"(\w+)\s+at\s+(\d{2}:\d{2})\s+\((\w+)\)", broadcast_raw.strip())
    if match:
        return match.group(1), match.group(2), match.group(3)
    return None, None, None


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

    Tries episode_number first (MalScrapedEpisode), then falls back
    to url (MalScrapedAnime, MalScrapedCharacter).

    Returns 0 when none are found.
    """
    for attr in ("episode_number",):
        v = getattr(model, attr, None)
        if v is not None:
            return int(v)
    url = getattr(model, "url", None)
    if url:
        return str(url)
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
