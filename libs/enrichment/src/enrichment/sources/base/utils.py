"""
Common utility functions for crawler modules.

Provides shared functionality for path sanitization, validation, and other
common operations used across multiple crawler implementations.
"""

import re
from datetime import date
from pathlib import Path

_DATE_RANGE_SEPARATOR_RE = re.compile(r"\s+to\s+|\s*[–‑]\s*|\s+-\s+")
# English month names, fixed rather than read from the process locale: provider
# pages are in English whatever locale the service runs in.
_MONTH_NAMES = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
_MONTH_NUMBERS = {
    name[:3].lower(): number for number, name in enumerate(_MONTH_NAMES, start=1)
}
# "Oct 20, 1999", "Sept 5, 2020", "Sep. 5, 2020", "September 5, 2020"
_MONTH_DAY_YEAR = re.compile(r"^([A-Za-z]{3})[a-z]*\.? (\d{1,2}), (\d{4})$")
# "20. Oct 1999", "5. Sept 2020"
_DAY_MONTH_YEAR = re.compile(r"^(\d{1,2})\. ([A-Za-z]{3})[a-z]*\.? (\d{4})$")
_ISO_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
# "20.10.1999"
_DOTTED_DATE = re.compile(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})$")
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

    Month names are read in English whatever the process locale.

    Args:
        raw: Raw date string from any supported source.

    Returns:
        ISO date string "YYYY-MM-DD", or None if the string is not a full date.

    Examples:
        MAL writes the month first, short or long:

        >>> parse_iso_date("Oct 20, 1999")
        '1999-10-20'
        >>> parse_iso_date("Sept 5, 2020")
        '2020-09-05'

        AniSearch episodes write the day first; AniSearch anime use dots:

        >>> parse_iso_date("20. Oct 1999")
        '1999-10-20'
        >>> parse_iso_date("20.10.1999")
        '1999-10-20'

        An ISO date comes back unchanged:

        >>> parse_iso_date("1999-10-20")
        '1999-10-20'

        A day that does not exist, a date without its day, and placeholders
        give None:

        >>> print(parse_iso_date("2026-02-31"))
        None
        >>> print(parse_iso_date("Oct 1999"))
        None
        >>> print(parse_iso_date("2026"))
        None
        >>> print(parse_iso_date("?"))
        None
    """
    if not raw:
        return None
    text = re.sub(r"\s+", " ", raw.strip())
    if match := _MONTH_DAY_YEAR.match(text):
        month_text, day, year = match.groups()
        return _iso_date(int(year), _MONTH_NUMBERS.get(month_text.lower()), int(day))
    if match := _DAY_MONTH_YEAR.match(text):
        day, month_text, year = match.groups()
        return _iso_date(int(year), _MONTH_NUMBERS.get(month_text.lower()), int(day))
    if match := _ISO_DATE.match(text):
        year, month, day = match.groups()
        return _iso_date(int(year), int(month), int(day))
    if match := _DOTTED_DATE.match(text):
        day, month, year = match.groups()
        return _iso_date(int(year), int(month), int(day))
    return None


def _iso_date(year: int, month: int | None, day: int) -> str | None:
    """Return the ISO date, or None when the month or day does not exist.

    >>> _iso_date(1999, 10, 20)
    '1999-10-20'
    >>> print(_iso_date(2026, 2, 31))
    None
    >>> print(_iso_date(1999, None, 20))
    None
    """
    if month is None:
        return None
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def split_date_range(raw: str | None) -> tuple[str, str]:
    """Split a date range into its start and end text.

    Each provider uses its own separator: MAL "to", AniSearch an en dash or
    non-breaking hyphen, Anime-Planet a hyphen between spaces. A plain hyphen
    only separates when spaced, so an ISO date is never split.

    Args:
        raw: Date or date range text.

    Returns:
        ``(start, end)``, both stripped; ``end`` is "" when there is no range.

    Examples:
        >>> split_date_range("Oct 20, 1999 to Nov 5, 2000")
        ('Oct 20, 1999', 'Nov 5, 2000')
        >>> split_date_range("20.10.1999 \u2011 ?")
        ('20.10.1999', '?')
        >>> split_date_range("1999 - ?")
        ('1999', '?')

        A single date, including an ISO date, has no end:

        >>> split_date_range("1999-10-20")
        ('1999-10-20', '')
        >>> split_date_range(None)
        ('', '')
    """
    parts = _DATE_RANGE_SEPARATOR_RE.split((raw or "").strip(), maxsplit=1)
    return parts[0].strip(), parts[1].strip() if len(parts) > 1 else ""


def parse_partial_date(raw: str | None) -> tuple[int | None, int | None]:
    """Read the year and month a date string states, at whatever precision it has.

    Takes the same formats as ``parse_iso_date``, and also a month and year
    or a year alone. The day is never returned.

    Args:
        raw: Raw date string from any supported source.

    Returns:
        ``(year, month)``; ``month`` is ``None`` when only a year is stated, and
        both are ``None`` when the string states neither.

    Examples:
        MAL, at each precision it gives:

        >>> parse_partial_date("Oct 20, 1999")
        (1999, 10)
        >>> parse_partial_date("Oct 1999")
        (1999, 10)
        >>> parse_partial_date("1999")
        (1999, None)

        AniSearch and ISO:

        >>> parse_partial_date("20.10.1999")
        (1999, 10)
        >>> parse_partial_date("10.1999")
        (1999, 10)
        >>> parse_partial_date("20. Oct 1999")
        (1999, 10)
        >>> parse_partial_date("1999-10")
        (1999, 10)

        A month that does not exist, or no date at all:

        >>> parse_partial_date("1999-13")
        (None, None)
        >>> parse_partial_date("?")
        (None, None)
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
        if month is not None and not 1 <= month <= 12:
            return None, None
        return int(iso.group(1)), month
    return None, None


def month_name(month: int | None) -> str | None:
    """Return the English month name for a month number, as ``Anime.month`` holds it.

    Args:
        month: Month number from 1 to 12, or ``None``.

    Returns:
        The month name, or ``None`` when no month, or a number outside 1 to 12,
        is given.

    Examples:
        >>> month_name(10)
        'October'
        >>> print(month_name(13))
        None
        >>> print(month_name(None))
        None
    """
    return _MONTH_NAMES[month - 1] if month and 1 <= month <= 12 else None


def parse_broadcast_string(
    broadcast_raw: str | None,
) -> tuple[str | None, str | None, str | None]:
    """Parse a broadcast schedule string into (day, time, timezone).

    Args:
        broadcast_raw: Raw broadcast string from any supported source.

    Returns:
        Tuple of (day, time, timezone); all three are None unless the day, the
        time and the timezone are all present.

    Examples:
        MAL writes "at"; AniSearch leaves it out. The day is kept as written:

        >>> parse_broadcast_string("Sundays at 23:15 (JST)")
        ('Sundays', '23:15', 'JST')
        >>> parse_broadcast_string("Sunday 23:15 (JST)")
        ('Sunday', '23:15', 'JST')

        A placeholder or a schedule without a time gives nothing:

        >>> parse_broadcast_string("Unknown")
        (None, None, None)
        >>> parse_broadcast_string("Sundays (JST)")
        (None, None, None)
    """
    if not broadcast_raw or broadcast_raw.strip().lower() in ("unknown", "n/a", ""):
        return None, None, None
    s = broadcast_raw.strip()
    m = _BROADCAST_WITH_AT_RE.match(s) or _BROADCAST_WITHOUT_AT_RE.match(s)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None, None, None
