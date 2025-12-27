"""Japanese text normalization utilities.

This module provides utilities for normalizing Japanese text (Hiragana, Katakana, Kanji)
to Romaji for consistent text processing and comparison.
"""

import logging

import jaconv
import pykakasi

logger = logging.getLogger(__name__)

# Setup pykakasi for Japanese text normalization
_kakasi = pykakasi.kakasi()
_kakasi.setMode("H", "a")  # Hiragana to ascii
_kakasi.setMode("K", "a")  # Katakana to ascii
_kakasi.setMode("J", "a")  # Kanji to ascii
_kakasi.setMode("r", "Hepburn")  # Romanization system

_kakasi_converter = _kakasi.getConverter()

__all__ = ["normalize_japanese_text"]


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
            romaji = _kakasi_converter.do(hiragana)
            return romaji.lower().strip()
        except Exception as e:
            logger.warning(f"Japanese normalization failed for '{text[:50]}': {e}")
            return text.lower().strip()

    return text.lower().strip()
