"""Unit tests for enrichment.sources.base.utils — shared crawler utilities."""

import pytest
from enrichment.sources.base.utils import (
    month_name,
    parse_broadcast_string,
    parse_iso_date,
    parse_partial_date,
    split_date_range,
)

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
        ("20.10.1999", "1999-10-20"),
        ("20. Oct 1999", "1999-10-20"),  # AniSearch episode format
        ("5. Apr 2003", "2003-04-05"),
        ("1. Jan 2000", "2000-01-01"),
    ],
)
def test_parse_iso_date(raw: str | None, expected: str | None) -> None:
    assert parse_iso_date(raw) == expected


def test_parse_iso_date_unrecognized_returns_none() -> None:
    assert parse_iso_date("Some Random String") is None


@pytest.mark.parametrize("raw", ["2026", "2026 to ?"])
def test_parse_iso_date_year_only_returns_none(raw: str) -> None:
    assert parse_iso_date(raw) is None


@pytest.mark.parametrize("raw", ["Oct 1977", "1977-10", "10.1977"])
def test_parse_iso_date_month_and_year_returns_none(raw: str) -> None:
    assert parse_iso_date(raw) is None


@pytest.mark.parametrize(
    "raw", ["Oct 20, 1999", "Oct  20, 1999", "20.10.1999", "20. Oct 1999", "1999-10-20"]
)
def test_parse_partial_date_full_date_returns_year_and_month(raw: str) -> None:
    assert parse_partial_date(raw) == (1999, 10)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Oct 1977", (1977, 10)),
        ("11.2008", (2008, 11)),
        ("01.2027", (2027, 1)),
        ("1977-10", (1977, 10)),
    ],
)
def test_parse_partial_date_month_and_year_returns_year_and_month(
    raw: str, expected: tuple[int, int]
) -> None:
    assert parse_partial_date(raw) == expected


@pytest.mark.parametrize("raw", ["1988", " 1988 "])
def test_parse_partial_date_year_only_returns_year_without_month(raw: str) -> None:
    assert parse_partial_date(raw) == (1988, None)


@pytest.mark.parametrize("raw", ["?", "Not available", "", None, "13.2008", "Foo 2008"])
def test_parse_partial_date_without_year_or_valid_month_returns_nothing(
    raw: str | None,
) -> None:
    assert parse_partial_date(raw) == (None, None)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Oct 20, 1999 to Nov 5, 2000", ("Oct 20, 1999", "Nov 5, 2000")),
        ("2026 to ?", ("2026", "?")),
        ("20.10.1999 ‑ ?", ("20.10.1999", "?")),
        ("20.10.1999‑31.03.2002", ("20.10.1999", "31.03.2002")),
        ("2019 – 2021", ("2019", "2021")),
        (" 1999 - ? ", ("1999", "?")),
    ],
)
def test_split_date_range_range_returns_start_and_end(
    raw: str, expected: tuple[str, str]
) -> None:
    assert split_date_range(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1999-10-20", "1999-10-20"), ("Apr 5, 2003", "Apr 5, 2003"), ("?", "?")],
)
def test_split_date_range_single_date_returns_empty_end(
    raw: str, expected: str
) -> None:
    assert split_date_range(raw) == (expected, "")


def test_split_date_range_none_returns_empty_parts() -> None:
    assert split_date_range(None) == ("", "")


def test_month_name_number_returns_english_name() -> None:
    assert month_name(10) == "October"


def test_month_name_none_returns_none() -> None:
    assert month_name(None) is None
