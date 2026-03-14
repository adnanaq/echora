"""Unit tests for mal_episode_crawler.py — title parsing and filler/recap detection."""

import pytest

from enrichment.crawlers.mal_crawler.mal_episode_crawler import _parse_title_info


# =============================================================================
# _parse_title_info — filler / recap suffix stripping
# =============================================================================


def test_parse_title_normal() -> None:
    """Regular episode — no flags set, title unchanged."""
    title, jp, romaji, filler, recap = _parse_title_info(
        "#1 - I'm Luffy! The Man Who Will Become the Pirate King!",
        None,
        1,
    )
    assert title == "I'm Luffy! The Man Who Will Become the Pirate King!"
    assert filler is False
    assert recap is False


def test_parse_title_filler_stripped() -> None:
    """'Filler' suffix is stripped from title and flag is set (ep 50)."""
    title, jp, romaji, filler, recap = _parse_title_info(
        "#50 - Usopp vs. Daddy the Parent! Showdown at High! Filler",
        None,
        50,
    )
    assert title == "Usopp vs. Daddy the Parent! Showdown at High!"
    assert filler is True
    assert recap is False


def test_parse_title_recap_stripped() -> None:
    """'Recap' suffix is stripped from title and flag is set (ep 279)."""
    title, jp, romaji, filler, recap = _parse_title_info(
        "#279 - Jump Towards the Falls! Luffy's Feelings! Recap",
        None,
        279,
    )
    assert title == "Jump Towards the Falls! Luffy's Feelings!"
    assert recap is True
    assert filler is False


def test_parse_title_filler_case_insensitive() -> None:
    """Suffix detection is case-insensitive."""
    title, *_, filler, recap = _parse_title_info("#1 - Title FILLER", None, 1)
    assert filler is True
    assert "FILLER" not in title
    assert "filler" not in title.lower()


def test_parse_title_subtitle_parsing() -> None:
    """Romaji and kanji are split from p.fn-grey2 text correctly."""
    title, jp, romaji, filler, recap = _parse_title_info(
        "#1 - I'm Luffy!",
        "Ore wa Luffy! Kaizoku Ou ni Naru Otoko Da! (俺はルフィ！海賊王になる男だ！)",
        1,
    )
    assert jp == "俺はルフィ！海賊王になる男だ！"
    assert romaji == "Ore wa Luffy! Kaizoku Ou ni Naru Otoko Da!"
    assert filler is False


def test_parse_title_fallback_when_no_header() -> None:
    """Falls back to 'Episode N' when no header is provided."""
    title, *_, filler, recap = _parse_title_info(None, None, 42)
    assert title == "Episode 42"
    assert filler is False
    assert recap is False
