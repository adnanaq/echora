"""Text and data normalization utilities for enrichment."""

import logging
import re
import unicodedata
from functools import cache

import jaconv
import pykakasi

logger = logging.getLogger(__name__)

__all__ = [
    "fold_for_comparison",
    "has_cjk",
    "is_shouting",
    "normalize_japanese_text",
    "normalize_score",
    "strip_synopsis_markup",
]

# Apostrophe variants that providers use interchangeably in the same title.
_APOSTROPHES = str.maketrans({"`": "'", "‘": "'", "’": "'", "ʼ": "'"})

_NON_ALNUM = re.compile(r"[^\w\s']", re.UNICODE)

# Kana and CJK ideographs. Enough to tell a Japanese title from its romaji,
# which is all the caller needs - not a general script classifier.
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿ｦ-ﾟ]")

_HTML_TAG = re.compile(r"<[^>]+>")

# Providers sign their synopses: "[Written by MAL Rewrite]", "(Source: ...)",
# "Source: www.anisearch.com/...". The words are required, so a synopsis simply
# ending in a parenthetical keeps it.
_ATTRIBUTION = re.compile(
    r"\s*(?:\[\s*(?:written by|source)\b[^\]]*\]"
    r"|\(\s*(?:written by|source)\b[^)]*\)"
    r"|(?:written by|source)\s*:[^\n]*)\s*$",
    re.IGNORECASE,
)

_BLANK_RUN = re.compile(r"\n{3,}")


def fold_for_comparison(text: str) -> str:
    """Reduce a title to a key that ignores cosmetic difference.

    Applies NFKC so full-width and half-width forms agree, folds the apostrophe
    variants providers mix within one title, drops remaining punctuation and
    collapses whitespace. Deliberately deterministic: measured against the real
    synonym pool this removes every duplicate an embedding model found that was
    genuinely a duplicate, and none of the ones it got wrong.

    Args:
        text: A title or synonym.

    Returns:
        A comparison key, empty when the text carries no comparable characters.
    """
    folded = unicodedata.normalize("NFKC", text).translate(_APOSTROPHES).casefold()
    return " ".join(_NON_ALNUM.sub(" ", folded).split())


def has_cjk(text: str) -> bool:
    """Report whether text contains kana or CJK ideographs.

    Args:
        text: Text to inspect.

    Returns:
        ``True`` when at least one character is kana or a CJK ideograph.
    """
    return bool(_CJK.search(text))


def is_shouting(text: str) -> bool:
    """Report whether a title is written in all capitals.

    Args:
        text: Title to inspect.

    Returns:
        ``True`` when the text has cased letters and none of them are lowercase.
    """
    return any(char.isalpha() for char in text) and text == text.upper()


def strip_synopsis_markup(text: str) -> str:
    """Remove provider markup and attribution from a synopsis.

    Providers deliver the same prose differently - AniList in HTML, MAL and
    Kitsu with a ``[Written by ...]`` footer, AniSearch and AnimeSchedule with a
    ``Source:`` line. Comparing raw lengths therefore measures markup as much as
    content, and the markup should not reach an embedding either way.

    Args:
        text: Synopsis as the provider supplied it.

    Returns:
        The prose alone, with blank-line runs collapsed.
    """
    cleaned = _ATTRIBUTION.sub("", _HTML_TAG.sub(" ", text))
    lines = [" ".join(line.split()) for line in cleaned.splitlines()]
    return _BLANK_RUN.sub("\n\n", "\n".join(lines)).strip()


def normalize_score(raw: float | None, *, source_max: float = 100.0) -> float | None:
    """Normalize a score onto the canonical 0–10 scale, to 2 decimal places.

    Providers publish on different scales: AniList and Kitsu on 0–100,
    AniSearch on 0–5 stars, the rest already on 0–10.

    Args:
        raw: The provider's score.
        source_max: Top of the provider's own scale.

    Returns:
        The score on 0–10, or None when there is nothing to convert.
    """
    if raw is None:
        return None
    return min(10.0, max(0.0, round(raw * 10 / source_max, 2)))


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
            logger.exception(f"Japanese normalization failed for {text[:50]!r}")
            return text.lower().strip()

    return text.lower().strip()
