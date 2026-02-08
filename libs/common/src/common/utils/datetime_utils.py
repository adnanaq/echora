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
        The determined status as an enum value.

    Examples:
        >>> from datetime import UTC, datetime
        >>> current = datetime(2025, 1, 1, tzinfo=UTC)
        >>> determine_anime_status("2024-10-04", "2024-12-20", current)
        <AnimeStatus.FINISHED: 'FINISHED'>
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


def normalize_to_utc(date_input: str | datetime | int | float | None) -> datetime | None:
    """Standardize any date input to a UTC-aware datetime object.

    Handles strings in multiple formats (ISO, AniSearch), Unix timestamps (int/float),
    and converts them to UTC. If the input is a date-only string or a midnight 
    timestamp (Pseudo-UTC), it applies the 'Midnight JST' rule.

    Args:
        date_input: Date as ISO 8601 string, AniSearch string, unix timestamp, or datetime object.

    Returns:
        A UTC-aware datetime object, or None if input is invalid.
    """
    if date_input is None:
        return None

    try:
        if isinstance(date_input, str):
            if not date_input:
                return None
            dt = _parse_date(date_input)
        elif isinstance(date_input, int | float):
            dt = datetime.fromtimestamp(date_input, tz=UTC)
        else:
            dt = date_input

        # Ensure timezone-aware (assume UTC if naive)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)

        # Convert to UTC
        return dt.astimezone(UTC)
    except (ValueError, TypeError, OSError):
        return None


def _parse_date(date_str: str) -> datetime:
    """Parse various date formats to a timezone-aware UTC datetime.

    Internal helper used by normalization and classification functions.

    Handles:
        - ISO 8601: "2024-04-20", "2024-04-20T00:00:00Z", "2024-04-20T00:00:00+09:00"
        - AniSearch format: "20.04.2024" (DD.MM.YYYY)
        - Date-only or Midnight strings: Treated as Midnight JST (+09:00) per project standard.

    Args:
        date_str: Date string to parse.

    Returns:
        Timezone-aware datetime object in UTC.

    Raises:
        ValueError: If the date format is unsupported or invalid.
    """
    # 1. Normalize AniSearch format: DD.MM.YYYY -> YYYY-MM-DD
    if "." in date_str and len(date_str) == 10:
        parts = date_str.split(".")
        if len(parts) == 3:
            date_str = f"{parts[2]}-{parts[1]}-{parts[0]}"

    # 2. Handle Z timezone indicator
    normalized = date_str.replace("Z", "+00:00")

    # 3. Detect Date-Only or Pseudo-UTC Midnight
    # If it's date-only (length 10 like YYYY-MM-DD) OR exactly Midnight (T00:00:00),
    # we treat it as a Japanese Air Date (JST) regardless of the label.
    if len(normalized) == 10 or "T00:00:00" in normalized:
        # Standardize to JST Midnight before UTC shift
        normalized = normalized[:10] + "T00:00:00+09:00"

    dt = datetime.fromisoformat(normalized)

    # 4. Final normalization to UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)

    return dt.astimezone(UTC)


def determine_anime_season(date_str: str | None) -> AnimeSeason | None:
    """Determine anime season from a date string.

    Maps the month from a date string to the corresponding anime broadcast
    season. Uses the month from the raw input string to ensure correct
    classification regardless of subsequent UTC shifts (e.g., Dec 1 JST -> Nov 30 UTC).

    Seasonal mappings:
        - December, January, February → WINTER
        - March, April, May → SPRING
        - June, July, August → SUMMER
        - September, October, November → FALL

    Args:
        date_str: Date string in various formats (ISO, AniSearch).

    Returns:
        The determined anime season enum or None.
    """
    if not date_str:
        return None

    try:
        # Extract month from string formats before UTC normalization
        if "-" in date_str:
            # YYYY-MM-DD...
            month = int(date_str.split("-")[1])
        elif "." in date_str:
            # DD.MM.YYYY...
            month = int(date_str.split(".")[1])
        else:
            # Fallback for full ISO strings with timezone
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            month = dt.month

        # Validation: Ensure month is valid (1-12)
        if not (1 <= month <= 12):
            return None
    except (ValueError, TypeError, IndexError):
        return None

    return _SEASONS[(month % 12) // 3]


def determine_anime_year(date_str: str | None) -> int | None:
    """Determine the release year from a date string.

    Extracts the year component, ensuring it reflects the territory-local year
    to maintain seasonal consistency (important for shows airing on Jan 1 JST).

    Args:
        date_str: Date string in various formats (ISO, AniSearch).

    Returns:
        The release year as an integer or None.
    """
    if not date_str:
        return None

    try:
        if "-" in date_str:
            year = int(date_str.split("-")[0])
        elif "." in date_str:
            year = int(date_str.split(".")[2][:4])
        else:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            year = dt.year

        # Basic year validation
        if year <= 0:
            return None
        return year
    except (ValueError, TypeError, IndexError):
        return None