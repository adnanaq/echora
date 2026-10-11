import pytest
from enrichment.sources.base.field_types import (
    NumberFromText,
    PlainText,
    StrippedText,
    html_to_plain_text,
    number_from_display_text,
    strip_text_or_none,
)
from pydantic import BaseModel, ValidationError


class _ProviderFields(BaseModel):
    title: StrippedText = None
    synopsis: PlainText = None
    rank: NumberFromText = None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("  One Piece ", "One Piece"), ("", None), ("   ", None), (None, None)],
)
def test_stripped_text_strips_and_empties_to_none(
    raw: str | None, expected: str | None
) -> None:
    assert _ProviderFields(title=raw).title == expected


def test_strip_text_or_none_non_string_returns_unchanged() -> None:
    assert strip_text_or_none(12) == 12


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("<p>Luffy &amp; Zoro</p>", "Luffy & Zoro"),
        ("worse...<br><br>(Source: sazuma.com)", "worse... (Source: sazuma.com)"),
        ("see &#34;black entities&#34; that", 'see "black entities" that'),
        ("season of <i>Log Horizon</i>.", "season of Log Horizon."),
        ("Twitter\n<a href='x'>@</a>GabeKunda", "Twitter\n@GabeKunda"),
        ("Birthplace\n<dl><dt>Turin</dt></dl>", "Birthplace\nTurin"),
        ("Age: 18  Luffy is a pirate", "Age: 18  Luffy is a pirate"),
        ("a < b and c > d", "a < b and c > d"),
        ("Cast in:\r\n(French version)", "Cast in:\n(French version)"),
        ("x<p>next</p>", "x next"),
        ("<br>", None),
        ("", None),
        (None, None),
    ],
)
def test_plain_text_removes_tags_and_decodes_entities(
    raw: str | None, expected: str | None
) -> None:
    assert _ProviderFields(synopsis=raw).synopsis == expected


def test_html_to_plain_text_cp1252_mojibake_is_repaired() -> None:
    assert html_to_plain_text("PokÃ©mon") == "Pokémon"


def test_html_to_plain_text_real_japanese_is_kept() -> None:
    assert html_to_plain_text("ワンピース") == "ワンピース"


def test_html_to_plain_text_non_string_returns_unchanged() -> None:
    assert html_to_plain_text(3) == 3


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Rank #157", 157),
        ("Rank #1,005", 1005),
        ("#10,563", 10563),
        ("#1.087", 1087),
        ("36,485 users", 36485),
        ("1,234,567 users", 1234567),
        ("#2.345.678", 2345678),
        ("Rank #12,345,678", 12345678),
        ("Rank", None),
        ("", None),
        (None, None),
        (42, 42),
    ],
)
def test_number_from_display_text_reads_number(
    raw: str | int | None, expected: int | None
) -> None:
    assert _ProviderFields(rank=raw).rank == expected


def test_number_from_display_text_non_string_returns_unchanged() -> None:
    assert number_from_display_text(7) == 7


def test_number_from_display_text_field_fractional_number_fails() -> None:
    with pytest.raises(ValidationError):
        _ProviderFields(rank=1.5)
