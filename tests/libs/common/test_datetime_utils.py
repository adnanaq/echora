"""Tests for datetime utility functions."""

from datetime import datetime, timezone

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
        current = datetime(2024, 1, 1, tzinfo=timezone.utc)
        future_start = "2024-06-01"

        status = determine_anime_status(
            start_date=future_start,
            end_date=None,
            current_date=current
        )
        assert status == AnimeStatus.UPCOMING

    def test_returns_upcoming_even_with_future_end_date(self):
        """Returns UPCOMING when start_date is future, regardless of end_date."""
        current = datetime(2024, 1, 1, tzinfo=timezone.utc)
        future_start = "2024-06-01"
        future_end = "2024-09-01"

        status = determine_anime_status(
            start_date=future_start,
            end_date=future_end,
            current_date=current
        )
        assert status == AnimeStatus.UPCOMING

    def test_returns_finished_when_both_dates_in_past(self):
        """Returns FINISHED when both start and end dates are in the past."""
        current = datetime(2025, 1, 1, tzinfo=timezone.utc)
        past_start = "2024-10-04"
        past_end = "2024-12-20"

        status = determine_anime_status(
            start_date=past_start,
            end_date=past_end,
            current_date=current
        )
        assert status == AnimeStatus.FINISHED

    def test_returns_finished_when_end_date_is_today(self):
        """Returns FINISHED when end_date is same as current date."""
        current = datetime(2024, 12, 20, 12, 0, 0, tzinfo=timezone.utc)
        start = "2024-10-04"
        end = "2024-12-20"

        status = determine_anime_status(
            start_date=start,
            end_date=end,
            current_date=current
        )
        assert status == AnimeStatus.FINISHED

    def test_returns_ongoing_when_started_but_no_end_date(self):
        """Returns ONGOING when start_date is past but no end_date."""
        current = datetime(2024, 12, 1, tzinfo=timezone.utc)
        past_start = "1999-10-20"  # One Piece example

        status = determine_anime_status(
            start_date=past_start,
            end_date=None,
            current_date=current
        )
        assert status == AnimeStatus.ONGOING

    def test_returns_ongoing_when_end_date_in_future(self):
        """Returns ONGOING when started but end_date is in future."""
        current = datetime(2024, 11, 1, tzinfo=timezone.utc)
        past_start = "2024-10-04"
        future_end = "2024-12-20"

        status = determine_anime_status(
            start_date=past_start,
            end_date=future_end,
            current_date=current
        )
        assert status == AnimeStatus.ONGOING

    def test_handles_iso_datetime_format(self):
        """Handles ISO datetime format with time and timezone."""
        current = datetime(2024, 12, 25, tzinfo=timezone.utc)
        start = "2024-10-04T00:00:00Z"
        end = "2024-12-20T23:59:59Z"

        status = determine_anime_status(
            start_date=start,
            end_date=end,
            current_date=current
        )
        assert status == AnimeStatus.FINISHED

    def test_handles_naive_datetime_assumes_utc(self):
        """Handles naive datetime strings by assuming UTC."""
        current = datetime(2024, 12, 25, tzinfo=timezone.utc)
        start = "2024-10-04T00:00:00"
        end = "2024-12-20T23:59:59"

        status = determine_anime_status(
            start_date=start,
            end_date=end,
            current_date=current
        )
        assert status == AnimeStatus.FINISHED

    def test_returns_unknown_on_invalid_date_format(self):
        """Returns UNKNOWN when date format cannot be parsed."""
        status = determine_anime_status(
            start_date="invalid-date",
            end_date="also-invalid",
            current_date=datetime(2024, 1, 1, tzinfo=timezone.utc)
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
        current = datetime(2024, 10, 4, 12, 0, 0, tzinfo=timezone.utc)
        start = "2024-10-04"

        status = determine_anime_status(
            start_date=start,
            end_date=None,
            current_date=current
        )
        assert status == AnimeStatus.ONGOING


class TestDetermineAnimeSeason:
    """Test suite for determine_anime_season function."""

    def test_returns_winter_for_december(self):
        """Returns WINTER for December dates."""
        assert determine_anime_season("2024-12-01") == AnimeSeason.WINTER
        assert determine_anime_season("2024-12-31") == AnimeSeason.WINTER

    def test_returns_winter_for_january(self):
        """Returns WINTER for January dates."""
        assert determine_anime_season("2024-01-01") == AnimeSeason.WINTER
        assert determine_anime_season("2024-01-15") == AnimeSeason.WINTER

    def test_returns_winter_for_february(self):
        """Returns WINTER for February dates."""
        assert determine_anime_season("2024-02-14") == AnimeSeason.WINTER

    def test_returns_spring_for_march_to_may(self):
        """Returns SPRING for March, April, May dates."""
        assert determine_anime_season("2024-03-01") == AnimeSeason.SPRING
        assert determine_anime_season("2024-04-20") == AnimeSeason.SPRING
        assert determine_anime_season("2024-05-31") == AnimeSeason.SPRING

    def test_returns_summer_for_june_to_august(self):
        """Returns SUMMER for June, July, August dates."""
        assert determine_anime_season("2024-06-01") == AnimeSeason.SUMMER
        assert determine_anime_season("2024-07-15") == AnimeSeason.SUMMER
        assert determine_anime_season("2024-08-01") == AnimeSeason.SUMMER

    def test_returns_fall_for_september_to_november(self):
        """Returns FALL for September, October, November dates."""
        assert determine_anime_season("2024-09-01") == AnimeSeason.FALL
        assert determine_anime_season("2024-10-15") == AnimeSeason.FALL
        assert determine_anime_season("2024-11-30") == AnimeSeason.FALL

    def test_returns_none_for_empty_string(self):
        """Returns None for empty string."""
        assert determine_anime_season("") is None

    def test_returns_none_for_none(self):
        """Returns None for None input."""
        assert determine_anime_season(None) is None

    def test_returns_none_for_invalid_date_format(self):
        """Returns None for invalid date format."""
        assert determine_anime_season("invalid-date") is None
        assert determine_anime_season("2024/12/01") is None  # Wrong separator
        # Note: "20241201" is actually valid for fromisoformat() and returns WINTER

    def test_returns_none_for_invalid_month_zero(self):
        """Returns None for month 00."""
        assert determine_anime_season("2024-00-15") is None

    def test_returns_none_for_invalid_month_thirteen(self):
        """Returns None for month 13."""
        assert determine_anime_season("2024-13-01") is None

    def test_handles_iso_datetime_format(self):
        """Handles full ISO datetime format."""
        assert determine_anime_season("2024-04-20T00:00:00Z") == AnimeSeason.SPRING
        assert determine_anime_season("2024-10-04T12:30:00+00:00") == AnimeSeason.FALL

    def test_handles_datetime_with_timezone(self):
        """Handles datetime strings with timezone info."""
        assert determine_anime_season("2024-01-15T00:00:00+09:00") == AnimeSeason.WINTER


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
