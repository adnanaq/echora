"""Japanese text normalization utilities.

This module provides utilities for normalizing Japanese text (Hiragana, Katakana, Kanji)
to Romaji for consistent text processing and comparison.
"""

import logging
from functools import cache

import jaconv
import pykakasi

logger = logging.getLogger(__name__)

__all__ = ["normalize_japanese_text"]


@cache
def _get_kakasi() -> pykakasi.kakasi:
    """Get cached pykakasi instance (lazy singleton).

    Thread-safe, lazy initialization - instance created on first call.
    """
    return pykakasi.kakasi()


def normalize_japanese_text(text: str) -> str:
    """Convert Japanese text (Hiragana, Katakana, Kanji) to Romaji.

    Uses pykakasi and jaconv to convert Japanese characters to romanized ASCII.
    If the text does not contain Japanese characters, it returns the lowercase
    stripped version of the original text.

    Args:
        text: Input text that may contain Japanese characters

    Returns:
        Romanized lowercase text with stripped whitespace

    Example:
        >>> normalize_japanese_text("ワンピース")
        'wanpi-su'
        >>> normalize_japanese_text("ONE PIECE")
        'one piece'
    """
    if not text:
        return ""

    # Check if text contains Japanese characters (Hiragana, Katakana, Kanji)
    has_japanese = any(
        "\u3040" <= char <= "\u30ff" or "\u4e00" <= char <= "\u9faf" for char in text
    )

    if has_japanese:
        try:
            # Convert Katakana to Hiragana first for consistent romaji conversion
            hiragana = jaconv.kata2hira(text)
            # Use modern pykakasi API (convert returns list of dicts with 'hepburn' key)
            # Use cached instance for performance (dictionary loads once)
            kks = _get_kakasi()
            result = kks.convert(hiragana)
            romaji = "".join(item["hepburn"] for item in result)
            # Replace Japanese punctuation with ASCII equivalents
            romaji = romaji.replace("・", " ")  # Middle dot to space
            romaji = romaji.replace("　", " ")  # Full-width space to regular space
            return romaji.lower().strip()
        except Exception:  # normalization must be best-effort, never fail caller
            logger.exception("Japanese normalization failed for %r", text[:50])
            return text.lower().strip()

    return text.lower().strip()
