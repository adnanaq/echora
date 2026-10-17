"""Reduce a title to the form two titles are compared in.

Titles of one work differ in ways that carry no meaning: case ("CUE!" and
"Cue!"), punctuation and spacing, accents ("Pokémon"), "½" against "1/2",
full-width letters, and season wording ("2nd Season" against "Season 2").
Letters of every script are kept, because dropping non-Latin letters would make
every Chinese and Japanese title compare equal.
"""

import re
import unicodedata
from functools import lru_cache

_NOT_LETTER_OR_DIGIT = re.compile(r"[\W_]+")
_ORDINAL_SEASON = re.compile(r"\b(\d+)(?:st|nd|rd|th) season\b")


@lru_cache(maxsize=65536)
def comparable_title(title: str) -> str:
    """Return a title in the form titles are compared in.

    Args:
        title: A title as a provider or the seed writes it.

    Returns:
        The title in lower case, without accents on Latin letters, with
        punctuation turned into single spaces and "Nth Season" written as
        "season N". Empty when the title has no letters or digits.

    Examples:
        >>> comparable_title("Ranma ½")
        'ranma 1 2'
        >>> comparable_title("Shingeki no Kyojin 2nd Season")
        'shingeki no kyojin season 2'
    """
    folded = unicodedata.normalize("NFKC", title).casefold()
    without_accents = "".join(map(_without_accent, folded))
    spaced = " ".join(_NOT_LETTER_OR_DIGIT.sub(" ", without_accents).split())
    return _ORDINAL_SEASON.sub(r"season \1", spaced)


@lru_cache(maxsize=4096)
def _without_accent(character: str) -> str:
    """Return a Latin letter without its accents; any other character unchanged.

    Only Latin letters lose their marks: the voiced mark of a Japanese kana
    decomposes the same way, and dropping it would turn "ガ" into "カ".

    Examples:
        >>> _without_accent("é")
        'e'
        >>> _without_accent("ガ")
        'ガ'
    """
    if character.isascii():
        return character
    base, *marks = unicodedata.normalize("NFD", character)
    if base.isascii() and base.isalpha() and all(map(unicodedata.combining, marks)):
        return base
    return character
