"""Datetime utility functions for anime data processing."""

from datetime import UTC, datetime

from common.models.anime import AnimeSeason, AnimeStatus

# Module-level constant for season mapping (winter, spring, summer, fall)
_SEASONS = (
    AnimeSeason.WINTER,
    AnimeSeason.SPRING,
    AnimeSeason.SUMMER,
    AnimeSeason.FALL,
)


def determine_anime_status(
    start_date: str | None,
    end_date: str | None,
    current_date: datetime | None = None,
) -> AnimeStatus:
    """Determine anime airing status based on start and end dates.

    Analyzes the start and end dates to classify anime into one of four
    statuses: UNKNOWN (no dates), UPCOMING (not yet aired), ONGOING
    (currently airing), or FINISHED (completed airing).

    Args:
        start_date: Anime start date in ISO format (e.g., "2024-01-15" or
            "2024-01-15T00:00:00Z"). Pass None or empty string for unknown
            start dates.
        end_date: Anime end date in ISO format. Pass None or empty string
            for ongoing anime or unknown end dates.
        current_date: Reference datetime for comparison. If not provided,
            uses current UTC time. Should be timezone-aware.

    Returns:
        The determined status as an enum value. Returns UNKNOWN if start_date
            is missing, UPCOMING if start_date is in the future, FINISHED if
            both dates exist and end_date has passed, or ONGOING if anime has
            started but not finished.

    Examples:
        >>> from datetime import datetime, timezone
        >>> current = datetime(2025, 1, 1, tzinfo=timezone.utc)
        >>> # Finished anime
        >>> determine_anime_status("2024-10-04", "2024-12-20", current)
        <AnimeStatus.FINISHED: 'FINISHED'>
        >>> # Ongoing anime (no end date)
        >>> determine_anime_status("1999-10-20", None, current)
        <AnimeStatus.ONGOING: 'ONGOING'>
        >>> # Upcoming anime
        >>> determine_anime_status("2025-04-01", None, current)
        <AnimeStatus.UPCOMING: 'UPCOMING'>
        >>> # Unknown (no dates)
        >>> determine_anime_status(None, None, current)
        <AnimeStatus.UNKNOWN: 'UNKNOWN'>
    """
    # Use current UTC time if not provided
    if current_date is None:
        current_date = datetime.now(UTC)

    # Ensure current_date is timezone-aware
    if current_date.tzinfo is None:
        current_date = current_date.replace(tzinfo=UTC)

    # Handle None or empty string dates
    if not start_date:
        return AnimeStatus.UNKNOWN

    # Parse start_date
    try:
        start_dt = _parse_date(start_date)
    except (ValueError, TypeError):
        return AnimeStatus.UNKNOWN

    # Check if upcoming (start date in future)
    if start_dt > current_date:
        return AnimeStatus.UPCOMING

    # At this point, anime has started (start_date <= current_date)
    # Check end_date to determine if FINISHED or ONGOING
    if not end_date:
        return AnimeStatus.ONGOING

    # Parse end_date
    try:
        end_dt = _parse_date(end_date)
    except (ValueError, TypeError):
        # If end_date exists but can't be parsed, treat as ongoing
        return AnimeStatus.ONGOING

    # Compare end_date with current_date
    # If end_date is current or past, it's finished
    if end_dt <= current_date:
        return AnimeStatus.FINISHED

    # end_date is in the future, still ongoing
    return AnimeStatus.ONGOING


def _parse_date(date_str: str) -> datetime:
    """Parse ISO format date string to timezone-aware datetime.

    Handles both date-only (YYYY-MM-DD) and full datetime formats with or
    without timezone information. Automatically converts 'Z' timezone
    indicator to '+00:00'. Naive datetimes (without timezone) are assumed
    to be in UTC.

    Supported formats include:
        - Date only: "2024-04-20"
        - Datetime with Z: "2024-04-20T00:00:00Z"
        - Datetime with offset: "2024-04-20T00:00:00+09:00"
        - Naive datetime: "2024-04-20T00:00:00" (treated as UTC)

    Args:
        date_str: ISO 8601 format date or datetime string. Must be parseable
            by datetime.fromisoformat() after Z normalization.

    Returns:
        Timezone-aware datetime object in UTC or specified timezone. Naive
            input datetimes are converted to UTC timezone.

    Raises:
        ValueError: If the date string format is invalid or cannot be parsed
            by datetime.fromisoformat().
        TypeError: If date_str is not a string.
    """
    # Handle Z timezone indicator
    normalized = date_str.replace("Z", "+00:00")

    dt = datetime.fromisoformat(normalized)

    # Ensure timezone-aware (assume UTC for naive)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)

    return dt


def determine_anime_season(date_str: str | None) -> AnimeSeason | None:
    """Determine anime season from a date string.

    Maps the month from a date string to the corresponding anime broadcast
    season using the standard seasonal classification. Season is determined
    solely by the month, regardless of day, time, or timezone.

    Seasonal mappings:
        - December, January, February → WINTER
        - March, April, May → SPRING
        - June, July, August → SUMMER
        - September, October, November → FALL

    Args:
        date_str: Date string in ISO format (e.g., "2024-04-20" or
            "2024-04-20T00:00:00Z"). Pass None for no date.

    Returns:
        The anime season enum value (WINTER, SPRING, SUMMER, or FALL) based
            on the date's month, or None if the date cannot be parsed or is
            invalid.

    Examples:
        >>> determine_anime_season("2024-04-20")
        <AnimeSeason.SPRING: 'SPRING'>
        >>> determine_anime_season("2024-10-04")
        <AnimeSeason.FALL: 'FALL'>
        >>> determine_anime_season("2024-12-25")
        <AnimeSeason.WINTER: 'WINTER'>
        >>> determine_anime_season(None)
        None
        >>> determine_anime_season("invalid")
        None
    """
    if not date_str:
        return None

    try:
        dt = _parse_date(date_str)
    except (ValueError, TypeError):
        return None

    # Use mathematical mapping: (month % 12) // 3
    # Winter (12,1,2): (12%12)//3=0, (1%12)//3=0, (2%12)//3=0
    # Spring (3,4,5): (3%12)//3=1, (4%12)//3=1, (5%12)//3=1
    # Summer (6,7,8): (6%12)//3=2, (7%12)//3=2, (8%12)//3=2
    # Fall (9,10,11): (9%12)//3=3, (10%12)//3=3, (11%12)//3=3
    return _SEASONS[(dt.month % 12) // 3]


def determine_anime_year(date_str: str | None) -> int | None:
    """Determine the year from a date string.

    Extracts the year component from a date string using proper ISO 8601
    parsing. This is more robust than regex extraction as it handles all
    valid date formats and validates the date structure.

    Args:
        date_str: Date string in ISO format (e.g., "2024-04-20" or
            "2024-04-20T00:00:00Z"). Pass None for no date.

    Returns:
        The year as an integer (e.g., 2024), or None if the date cannot
            be parsed or is invalid.

    Examples:
        >>> determine_anime_year("2024-04-20")
        2024
        >>> determine_anime_year("1999-10-20T00:00:00Z")
        1999
        >>> determine_anime_year(None)
        None
    """
    if not date_str:
        return None

    try:
        dt = _parse_date(date_str)
    except (ValueError, TypeError):
        return None

    return dt.year
