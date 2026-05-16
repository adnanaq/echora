"""
Common utility functions for crawler modules.

Provides shared functionality for path sanitization, validation, and other
common operations used across multiple crawler implementations.
"""

import re
from datetime import datetime
from pathlib import Path

_BROADCAST_WITH_AT_RE = re.compile(r"(\w+)\s+at\s+(\d{1,2}:\d{2})\s+\((\w+)\)")
_BROADCAST_WITHOUT_AT_RE = re.compile(r"(\w+)\s+(\d{1,2}:\d{2})\s*\((\w+)\)")


def sanitize_output_path(output_path: str) -> str:
    """
    Sanitize output path to prevent path traversal attacks.

    Args:
        output_path: User-provided output file path

    Returns:
        Sanitized absolute path

    Raises:
        ValueError: If relative path escapes working directory
    """
    p = Path(output_path)
    abs_path = p.resolve()

    # Absolute paths are allowed as-is
    if p.is_absolute():
        return str(abs_path)

    # Relative paths must remain within CWD after resolution
    try:
        abs_path.relative_to(Path.cwd())
    except ValueError as err:
        raise ValueError(
            f"Output path escapes working directory: {output_path}"
        ) from err

    return str(abs_path)


def parse_iso_date(raw: str | None) -> str | None:
    """Parse a date string to an ISO 8601 date string (YYYY-MM-DD).

    Handles:
        "Oct 20, 1999"  → "1999-10-20"   (MAL month-name format)
        "1999-10-20"    → "1999-10-20"   (already ISO)
        "2026"          → "2026-01-01"   (year-only, upcoming anime)
        "?"             → None
        None            → None

    Args:
        raw: Raw date string from any supported source.

    Returns:
        ISO date string "YYYY-MM-DD", or None if not parseable.
    """
    if not raw or raw.strip() in ("?", "N/A", ""):
        return None

    raw = raw.strip()

    # "Oct 20, 1999" or "Oct  20, 1999"
    try:
        return datetime.strptime(re.sub(r"\s+", " ", raw), "%b %d, %Y").strftime("%Y-%m-%d")
    except ValueError:
        pass

    # "20. Oct 1999" (AniSearch episode format)
    try:
        return datetime.strptime(raw, "%d. %b %Y").strftime("%Y-%m-%d")
    except ValueError:
        pass

    # Already ISO "1999-10-20"
    if re.match(r"^\d{4}-\d{2}-\d{2}$", raw):
        return raw

    # Year-only "2026" — upcoming anime with no specific date yet
    if re.match(r"^\d{4}$", raw):
        return f"{raw}-01-01"

    return None


def parse_broadcast_string(
    broadcast_raw: str | None,
) -> tuple[str | None, str | None, str | None]:
    """Parse a broadcast schedule string into (day, time, timezone).

    Handles:
        "Sundays at 23:15 (JST)"  → ("Sundays", "23:15", "JST")   [MAL]
        "Sunday 23:15 (JST)"      → ("Sunday",  "23:15", "JST")   [AniSearch]
        "Unknown"                 → (None, None, None)

    Args:
        broadcast_raw: Raw broadcast string from any supported source.

    Returns:
        Tuple of (day, time, timezone), any may be None.
    """
    if not broadcast_raw or broadcast_raw.strip().lower() in ("unknown", "n/a", ""):
        return None, None, None
    s = broadcast_raw.strip()
    m = _BROADCAST_WITH_AT_RE.match(s) or _BROADCAST_WITHOUT_AT_RE.match(s)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None, None, None
