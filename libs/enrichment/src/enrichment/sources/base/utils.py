"""
Common utility functions for crawler modules.

Provides shared functionality for path sanitization, validation, and other
common operations used across multiple crawler implementations.
"""

import calendar
import re
from datetime import datetime
from pathlib import Path

_DATE_RANGE_SEPARATOR_RE = re.compile(r"\s+to\s+|\s*[–‑]\s*|\s+-\s+")
_MONTH_NUMBERS = {
    name.lower(): number for number, name in enumerate(calendar.month_abbr) if name
}
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
    """Parse a full date string to an ISO 8601 date string (YYYY-MM-DD).

    Only a date that states its day is returned. A month and year or a year
    alone is not a date: making one up (such as 1 January) would publish a
    premiere that never happened. Use ``parse_partial_date`` for those.

    Handles:
        "Oct 20, 1999"  → "1999-10-20"   (MAL month-name format)
        "1999-10-20"    → "1999-10-20"   (already ISO)
        "20.10.1999"    → "1999-10-20"   (AniSearch anime format)
        "Oct 1999"      → None           (month and year only)
        "2026"          → None           (year only)
        "?"             → None
        None            → None

    Args:
        raw: Raw date string from any supported source.

    Returns:
        ISO date string "YYYY-MM-DD", or None if the string is not a full date.
    """
    if not raw or raw.strip() in ("?", "N/A", ""):
        return None

    raw = raw.strip()

    # "Oct 20, 1999" or "Oct  20, 1999"
    try:
        return datetime.strptime(re.sub(r"\s+", " ", raw), "%b %d, %Y").strftime(
            "%Y-%m-%d"
        )
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

    # "20.10.1999" (AniSearch anime format)
    try:
        return datetime.strptime(raw, "%d.%m.%Y").strftime("%Y-%m-%d")
    except ValueError:
        pass

    return None


def split_date_range(raw: str | None) -> tuple[str, str]:
    """Split a date range into its start and end text.

    Handles each provider's separator:
        "Oct 20, 1999 to Nov 5, 2000"  (MAL)
        "20.10.1999 ‑ ?" / "2027‑2028"  (AniSearch: en dash or non-breaking hyphen)
        "1999 - ?"                      (Anime-Planet: hyphen between spaces)

    A plain hyphen only separates when spaced, so an ISO date is never split.

    Args:
        raw: Date or date range text.

    Returns:
        ``(start, end)``, both stripped; ``end`` is "" when there is no range.
    """
    parts = _DATE_RANGE_SEPARATOR_RE.split((raw or "").strip(), maxsplit=1)
    return parts[0].strip(), parts[1].strip() if len(parts) > 1 else ""


def parse_partial_date(raw: str | None) -> tuple[int | None, int | None]:
    """Read the year and month a date string states, at whatever precision it has.

    Handles:
        "Oct 20, 1999" / "Oct 1999" / "1999"  (MAL)
        "20.10.1999" / "10.1999"              (AniSearch)
        "20. Oct 1999"                        (AniSearch episodes)
        "1999-10-20" / "1999-10"              (ISO)
        "?" / None                            → (None, None)

    Args:
        raw: Raw date string from any supported source.

    Returns:
        ``(year, month)``; ``month`` is ``None`` when only a year is stated, and
        both are ``None`` when the string states neither.
    """
    if not raw:
        return None, None
    text = re.sub(r"\s+", " ", raw.strip())
    named = re.match(
        r"^(?:\d{1,2}\. )?([A-Za-z]{3})[a-z]*\.?(?: \d{1,2},)? (\d{4})$", text
    )
    if named and named.group(1).lower() in _MONTH_NUMBERS:
        return int(named.group(2)), _MONTH_NUMBERS[named.group(1).lower()]
    dotted = re.match(r"^(?:\d{1,2}\.)?(\d{1,2})\.(\d{4})$", text)
    if dotted and 1 <= int(dotted.group(1)) <= 12:
        return int(dotted.group(2)), int(dotted.group(1))
    iso = re.match(r"^(\d{4})(?:-(\d{2}))?(?:-\d{2})?$", text)
    if iso:
        month = int(iso.group(2)) if iso.group(2) else None
        return int(iso.group(1)), month if month and 1 <= month <= 12 else None
    return None, None


def month_name(month: int | None) -> str | None:
    """Return the English month name for a month number, as ``Anime.month`` holds it.

    Args:
        month: Month number from 1 to 12, or ``None``.

    Returns:
        The month name (e.g. "October"), or ``None`` when no month is given.
    """
    return calendar.month_name[month] if month else None


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
