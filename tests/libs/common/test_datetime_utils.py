"""Tests for datetime utility functions."""

from datetime import UTC, datetime

import pytest
from common.models.anime import AnimeSeason, AnimeStatus
from common.utils.datetime_utils import (
    determine_anime_season,
    determine_anime_status,
    determine_anime_year,
)


class TestDetermineAnimeStatus:
    """Test suite for determine_anime_status function."""

    def test_returns_unknown_when_no_dates(self):
        """Returns UNKNOWN when both start_date and end_date are None."""
        status = determine_anime_status(start_date=None, end_date=None)
        assert status == AnimeStatus.UNKNOWN

    def test_returns_unknown_when_empty_strings(self):
        """Returns UNKNOWN when both dates are empty strings."""
        status = determine_anime_status(start_date="", end_date="")
        assert status == AnimeStatus.UNKNOWN

    def test_returns_upcoming_when_start_date_in_future(self):
        """Returns UPCOMING when start_date is in the future."""
        current = datetime(2024, 1, 1, tzinfo=UTC)
        future_start = "2024-06-01"

        status = determine_anime_status(
            start_date=future_start, end_date=None, current_date=current
        )
        assert status == AnimeStatus.UPCOMING

    def test_returns_upcoming_even_with_future_end_date(self):
        """Returns UPCOMING when start_date is future, regardless of end_date."""
        current = datetime(2024, 1, 1, tzinfo=UTC)
        future_start = "2024-06-01"
        future_end = "2024-09-01"

        status = determine_anime_status(
            start_date=future_start, end_date=future_end, current_date=current
        )
        assert status == AnimeStatus.UPCOMING

    def test_returns_finished_when_both_dates_in_past(self):
        """Returns FINISHED when both start and end dates are in the past."""
        current = datetime(2025, 1, 1, tzinfo=UTC)
        past_start = "2024-10-04"
        past_end = "2024-12-20"

        status = determine_anime_status(
            start_date=past_start, end_date=past_end, current_date=current
        )
        assert status == AnimeStatus.FINISHED

    def test_returns_finished_when_end_date_is_today(self):
        """Returns FINISHED when end_date is same as current date."""
        current = datetime(2024, 12, 20, 12, 0, 0, tzinfo=UTC)
        start = "2024-10-04"
        end = "2024-12-20"

        status = determine_anime_status(
            start_date=start, end_date=end, current_date=current
        )
        assert status == AnimeStatus.FINISHED

    def test_returns_ongoing_when_started_but_no_end_date(self):
        """Returns ONGOING when start_date is past but no end_date."""
        current = datetime(2024, 12, 1, tzinfo=UTC)
        past_start = "1999-10-20"  # One Piece example

        status = determine_anime_status(
            start_date=past_start, end_date=None, current_date=current
        )
        assert status == AnimeStatus.ONGOING

    def test_returns_ongoing_when_end_date_in_future(self):
        """Returns ONGOING when started but end_date is in future."""
        current = datetime(2024, 11, 1, tzinfo=UTC)
        past_start = "2024-10-04"
        future_end = "2024-12-20"

        status = determine_anime_status(
            start_date=past_start, end_date=future_end, current_date=current
        )
        assert status == AnimeStatus.ONGOING

    def test_handles_iso_datetime_format(self):
        """Handles ISO datetime format with time and timezone."""
        current = datetime(2024, 12, 25, tzinfo=UTC)
        start = "2024-10-04T00:00:00Z"
        end = "2024-12-20T23:59:59Z"

        status = determine_anime_status(
            start_date=start, end_date=end, current_date=current
        )
        assert status == AnimeStatus.FINISHED

    def test_handles_naive_datetime_assumes_utc(self):
        """Handles naive datetime strings by assuming UTC."""
        current = datetime(2024, 12, 25, tzinfo=UTC)
        start = "2024-10-04T00:00:00"
        end = "2024-12-20T23:59:59"

        status = determine_anime_status(
            start_date=start, end_date=end, current_date=current
        )
        assert status == AnimeStatus.FINISHED

    def test_returns_unknown_on_invalid_date_format(self):
        """Returns UNKNOWN when date format cannot be parsed."""
        status = determine_anime_status(
            start_date="invalid-date",
            end_date="also-invalid",
            current_date=datetime(2024, 1, 1, tzinfo=UTC),
        )
        assert status == AnimeStatus.UNKNOWN

    def test_uses_current_time_when_no_current_date_provided(self):
        """Uses current time when current_date is not provided."""
        # This test ensures the function works without explicit current_date
        very_old_start = "1999-01-01"
        status = determine_anime_status(start_date=very_old_start, end_date=None)
        # Should be ONGOING since it started long ago and has no end
        assert status == AnimeStatus.ONGOING

    def test_returns_ongoing_when_start_today_no_end(self):
        """Returns ONGOING when anime starts today with no end date."""
        current = datetime(2024, 10, 4, 12, 0, 0, tzinfo=UTC)
        start = "2024-10-04"

        status = determine_anime_status(
            start_date=start, end_date=None, current_date=current
        )
        assert status == AnimeStatus.ONGOING


class TestDetermineAnimeSeason:
    """Test suite for determine_anime_season function."""

    @pytest.mark.parametrize(
        "date_str, expected_season",
        [
            # Winter: Dec, Jan, Feb
            ("2024-12-01", AnimeSeason.WINTER),
            ("2024-12-31", AnimeSeason.WINTER),
            ("2024-01-01", AnimeSeason.WINTER),
            ("2024-01-15", AnimeSeason.WINTER),
            ("2024-02-14", AnimeSeason.WINTER),
            # Spring: Mar, Apr, May
            ("2024-03-01", AnimeSeason.SPRING),
            ("2024-04-20", AnimeSeason.SPRING),
            ("2024-05-31", AnimeSeason.SPRING),
            # Summer: Jun, Jul, Aug
            ("2024-06-01", AnimeSeason.SUMMER),
            ("2024-07-15", AnimeSeason.SUMMER),
            ("2024-08-01", AnimeSeason.SUMMER),
            # Fall: Sep, Oct, Nov
            ("2024-09-01", AnimeSeason.FALL),
            ("2024-10-15", AnimeSeason.FALL),
            ("2024-11-30", AnimeSeason.FALL),
            # ISO and timezone formats
            ("2024-04-20T00:00:00Z", AnimeSeason.SPRING),
            ("2024-10-04T12:30:00+00:00", AnimeSeason.FALL),
            ("2024-01-15T00:00:00+09:00", AnimeSeason.WINTER),
        ],
    )
    def test_determine_anime_season_mapping(self, date_str, expected_season):
        """Returns correct season for various date formats and months."""
        assert determine_anime_season(date_str) == expected_season

    @pytest.mark.parametrize(
        "invalid_input",
        [
            "",
            None,
            "invalid-date",
            "2024/12/01",
            "2024-00-15",
            "2024-13-01",
        ],
    )
    def test_returns_none_for_invalid_inputs(self, invalid_input):
        """Returns None for invalid, empty, or None inputs."""
        assert determine_anime_season(invalid_input) is None


class TestDetermineAnimeYear:
    """Test suite for determine_anime_year function."""

    def test_extracts_year_from_basic_date(self):
        """Extracts year from basic ISO date format."""
        assert determine_anime_year("2024-04-20") == 2024

    def test_returns_none_for_none_input(self):
        """Returns None when input is None."""
        assert determine_anime_year(None) is None

    def test_returns_none_for_empty_string(self):
        """Returns None when input is empty string."""
        assert determine_anime_year("") is None

    def test_returns_none_for_invalid_format(self):
        """Returns None when date format is invalid."""
        assert determine_anime_year("invalid-date") is None

    def test_extracts_year_from_datetime_with_timezone(self):
        """Extracts year from full ISO datetime format with timezone."""
        assert determine_anime_year("1999-10-20T00:00:00Z") == 1999

    def test_extracts_year_from_different_years(self):
        """Extracts year correctly from various years."""
        assert determine_anime_year("2025-01-15") == 2025
        assert determine_anime_year("1998-06-01") == 1998


class TestNormalizeToUtc:
    """Test suite for normalize_to_utc function."""

    def test_normalizes_utc_string(self):
        """Normalizes a UTC string with Z suffix."""
        from common.utils.datetime_utils import normalize_to_utc
        date_str = "2024-04-20T12:00:00Z"
        result = normalize_to_utc(date_str)
        assert result.tzinfo == UTC
        assert result.hour == 12
        assert result.day == 20

    def test_normalizes_jst_string_to_utc(self):
        """Normalizes a JST (+09:00) string to UTC."""
        from common.utils.datetime_utils import normalize_to_utc
        # 2024-04-20 09:00 JST should be 2024-04-20 00:00 UTC
        date_str = "2024-04-20T09:00:00+09:00"
        result = normalize_to_utc(date_str)
        assert result.tzinfo == UTC
        assert result.hour == 0
        assert result.day == 20

    def test_normalizes_jst_midnight_to_previous_day_utc(self):
        """Normalizes JST midnight to the previous day in UTC."""
        from common.utils.datetime_utils import normalize_to_utc
        # 2024-04-20 00:00 JST should be 2024-04-19 15:00 UTC
        date_str = "2024-04-20T00:00:00+09:00"
        result = normalize_to_utc(date_str)
        assert result.tzinfo == UTC
        assert result.hour == 15
        assert result.day == 19
        assert result.month == 4

    def test_normalizes_anisearch_format_to_utc(self):
        """Normalizes AniSearch format (DD.MM.YYYY) using the Midnight JST rule."""
        from common.utils.datetime_utils import normalize_to_utc
        # 04.10.2024 (AniSearch) -> Midnight JST -> 15:00 UTC previous day
        date_str = "04.10.2024"
        result = normalize_to_utc(date_str)
        assert result.tzinfo == UTC
        assert result.year == 2024
        assert result.month == 10
        assert result.day == 3
        assert result.hour == 15

    def test_normalizes_iso_date_only_using_midnight_jst_rule(self):
        """Normalizes ISO date-only (YYYY-MM-DD) using the Midnight JST rule."""
        from common.utils.datetime_utils import normalize_to_utc
        # 2024-10-04 (Standard) -> Midnight JST -> 15:00 UTC previous day
        date_str = "2024-10-04"
        result = normalize_to_utc(date_str)
        assert result.tzinfo == UTC
        assert result.day == 3
        assert result.hour == 15

    def test_handles_naive_datetime_string(self):
        """Handles naive datetime strings by assuming UTC and making them aware."""
        from common.utils.datetime_utils import normalize_to_utc
        date_str = "2024-04-20T12:00:00"
        result = normalize_to_utc(date_str)
        assert result.tzinfo == UTC
        assert result.hour == 12

    def test_handles_datetime_object(self):
        """Handles already aware datetime objects by converting to UTC."""
        from datetime import timedelta, timezone

        from common.utils.datetime_utils import normalize_to_utc
        jst = timezone(timedelta(hours=9))
        dt = datetime(2024, 4, 20, 9, 0, tzinfo=jst)
        result = normalize_to_utc(dt)
        assert result.tzinfo == UTC
        assert result.hour == 0

    def test_returns_none_for_none_input(self):
        """Returns None when input is None."""
        from common.utils.datetime_utils import normalize_to_utc
        assert normalize_to_utc(None) is None

    def test_returns_none_for_empty_string(self):
        """Returns None when input is empty string."""
        from common.utils.datetime_utils import normalize_to_utc
        assert normalize_to_utc("") is None

    def test_returns_none_for_invalid_format(self):
        """Returns None when date format is invalid."""
        from common.utils.datetime_utils import normalize_to_utc
        assert normalize_to_utc("not-a-date") is None

    def test_handles_unix_timestamp_int(self):
        """Handles integer Unix timestamps from AniList."""
        from common.utils.datetime_utils import normalize_to_utc
        # 1727969160 is 2024-10-03 15:26:00 UTC
        ts = 1727969160
        result = normalize_to_utc(ts)
        assert result.tzinfo == UTC
        assert result.year == 2024
        assert result.month == 10
        assert result.day == 3
        assert result.hour == 15
        assert result.minute == 26

    def test_normalizes_pseudo_utc_midnight_to_jst_shift(self):
        """Treats a UTC-labeled midnight string as a JST air date and shifts it."""
        from common.utils.datetime_utils import normalize_to_utc
        # MAL/AnimSchedule often return this for air dates
        date_str = "2024-10-04T00:00:00+00:00"
        result = normalize_to_utc(date_str)
        # Should be treated as 2024-10-04 00:00 JST -> 2024-10-03 15:00 UTC
        assert result.tzinfo == UTC
        assert result.day == 3
        assert result.hour == 15

    def test_normalizes_z_suffix_midnight_to_jst_shift(self):
        """Treats a Z-suffixed midnight string as a JST air date."""
        from common.utils.datetime_utils import normalize_to_utc
        date_str = "2024-10-04T00:00:00Z"
        result = normalize_to_utc(date_str)
        assert result.tzinfo == UTC
        assert result.day == 3
        assert result.hour == 15
