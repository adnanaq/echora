"""Unit tests for enrichment.sources.base.utils — shared crawler utilities."""

import pytest

from enrichment.sources.base.utils import parse_broadcast_string, parse_iso_date


# =============================================================================
# parse_broadcast_string
# =============================================================================


def test_parse_broadcast_string_mal_format() -> None:
    day, time, tz = parse_broadcast_string("Sundays at 23:15 (JST)")
    assert day == "Sundays"
    assert time == "23:15"
    assert tz == "JST"


def test_parse_broadcast_string_anisearch_format() -> None:
    day, time, tz = parse_broadcast_string("Sunday 23:15 (JST)")
    assert day == "Sunday"
    assert time == "23:15"
    assert tz == "JST"


def test_parse_broadcast_string_unknown() -> None:
    day, time, tz = parse_broadcast_string("Unknown")
    assert day is None
    assert time is None
    assert tz is None


def test_parse_broadcast_string_none() -> None:
    day, time, tz = parse_broadcast_string(None)
    assert day is None
    assert time is None
    assert tz is None


def test_parse_broadcast_string_no_match() -> None:
    day, time, tz = parse_broadcast_string("Irregular schedule")
    assert day is None
    assert time is None
    assert tz is None


# =============================================================================
# parse_iso_date
# =============================================================================


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Oct 20, 1999", "1999-10-20"),
        ("Apr 5, 2003", "2003-04-05"),
        ("Jan 1, 2000", "2000-01-01"),
        ("?", None),
        ("N/A", None),
        (None, None),
        ("", None),
        ("1999-10-20", "1999-10-20"),  # Already ISO
        ("2026", "2026-01-01"),        # Year-only (upcoming anime)
        ("20. Oct 1999", "1999-10-20"),  # AniSearch episode format
        ("5. Apr 2003", "2003-04-05"),
        ("1. Jan 2000", "2000-01-01"),
    ],
)
def test_parse_iso_date(raw: str | None, expected: str | None) -> None:
    assert parse_iso_date(raw) == expected


def test_parse_iso_date_unrecognized_returns_none() -> None:
    assert parse_iso_date("Some Random String") is None
