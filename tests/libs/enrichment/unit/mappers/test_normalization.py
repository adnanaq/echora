"""Unit tests for normalization.py — shared lookup tables and parsing helpers."""

import pytest

from enrichment.mappers.normalization import SOURCE_MATERIAL, parse_duration, parse_theme_song


# =============================================================================
# SOURCE_MATERIAL lookup table
# =============================================================================


def test_source_material_manga() -> None:
    assert SOURCE_MATERIAL["manga"] == "MANGA"


def test_source_material_light_novel() -> None:
    assert SOURCE_MATERIAL["light novel"] == "LIGHT NOVEL"


def test_source_material_visual_novel() -> None:
    assert SOURCE_MATERIAL["visual novel"] == "VISUAL NOVEL"


def test_source_material_original() -> None:
    assert SOURCE_MATERIAL["original"] == "ORIGINAL"


def test_source_material_game() -> None:
    assert SOURCE_MATERIAL["game"] == "GAME"


# =============================================================================
# parse_duration
# =============================================================================


def test_parse_duration_minutes() -> None:
    assert parse_duration("24 min.") == 1440


def test_parse_duration_hours_and_minutes() -> None:
    assert parse_duration("1 hr. 30 min.") == 5400


def test_parse_duration_hms_format() -> None:
    assert parse_duration("00:24:37") == 1477


def test_parse_duration_none() -> None:
    assert parse_duration(None) is None


def test_parse_duration_empty_string() -> None:
    assert parse_duration("") is None


# =============================================================================
# parse_theme_song
# =============================================================================


def test_parse_theme_song_full() -> None:
    result = parse_theme_song('1: "Hands Up!" by V6 (eps 1-26)')
    assert result is not None
    assert result["title"] == "Hands Up!"
    assert result["artist"] == "V6"
    assert result["episodes"] == "1-26"


def test_parse_theme_song_no_artist() -> None:
    result = parse_theme_song('"We Are!"')
    assert result is not None
    assert result["title"] == "We Are!"
    assert result["artist"] is None


def test_parse_theme_song_no_episodes() -> None:
    result = parse_theme_song('"We Are!" by Kitadani Hiroshi')
    assert result is not None
    assert result["title"] == "We Are!"
    assert result["artist"] == "Kitadani Hiroshi"
    assert result["episodes"] is None


def test_parse_theme_song_none() -> None:
    assert parse_theme_song(None) is None
