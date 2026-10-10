from pathlib import Path

import pytest
from enrichment.sources.base.utils import (
    month_name,
    parse_broadcast_string,
    parse_iso_date,
    parse_partial_date,
    sanitize_output_path,
    split_date_range,
)


def test_sanitize_output_path_absolute_path_returns_resolved_path(
    tmp_path: Path,
) -> None:
    target = tmp_path / "nested" / ".." / "out.json"
    assert sanitize_output_path(str(target)) == str(tmp_path / "out.json")


def test_sanitize_output_path_relative_path_inside_working_directory_returns_absolute_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert sanitize_output_path("data/out.json") == str(
        tmp_path.resolve() / "data" / "out.json"
    )


def test_sanitize_output_path_relative_path_outside_working_directory_raises_value_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="escapes working directory"):
        sanitize_output_path("../out.json")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Sundays at 23:15 (JST)", ("Sundays", "23:15", "JST")),
        ("Sunday 23:15 (JST)", ("Sunday", "23:15", "JST")),
    ],
)
def test_parse_broadcast_string_day_time_and_zone_returns_all_three(
    raw: str, expected: tuple[str, str, str]
) -> None:
    assert parse_broadcast_string(raw) == expected


@pytest.mark.parametrize("raw", ["Unknown", None, "Irregular schedule"])
def test_parse_broadcast_string_without_schedule_returns_nothing(
    raw: str | None,
) -> None:
    assert parse_broadcast_string(raw) == (None, None, None)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Oct 20, 1999", "1999-10-20"),
        ("Apr 5, 2003", "2003-04-05"),
        ("Jan 1, 2000", "2000-01-01"),
        ("1999-10-20", "1999-10-20"),
        ("20.10.1999", "1999-10-20"),
        ("20. Oct 1999", "1999-10-20"),
        ("5. Apr 2003", "2003-04-05"),
        ("1. Jan 2000", "2000-01-01"),
    ],
)
def test_parse_iso_date_full_date_returns_iso_date(raw: str, expected: str) -> None:
    assert parse_iso_date(raw) == expected


@pytest.mark.parametrize("raw", ["?", "N/A", None, "", "Some Random String"])
def test_parse_iso_date_without_date_returns_none(raw: str | None) -> None:
    assert parse_iso_date(raw) is None


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


@pytest.mark.parametrize(
    "raw", ["?", "Not available", "", None, "13.2008", "2026-13", "Foo 2008"]
)
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
        ("2027‑2028", ("2027", "2028")),
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
