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
def fold_title_for_comparison(title: str) -> str:
    """Return a title in the form titles are compared in.

    Args:
        title: A title as a provider or the seed writes it.

    Returns:
        The title in lower case, without accents on Latin letters, with
        punctuation turned into single spaces and "Nth Season" written as
        "season N". Empty when the title has no letters or digits.

    Examples:
        >>> fold_title_for_comparison("Ranma ½")
        'ranma 1 2'
        >>> fold_title_for_comparison("Shingeki no Kyojin 2nd Season")
        'shingeki no kyojin season 2'
    """
    casefolded_title = unicodedata.normalize("NFKC", title).casefold()
    title_without_accents = "".join(map(_remove_accent, casefolded_title))
    title_words = " ".join(_NOT_LETTER_OR_DIGIT.sub(" ", title_without_accents).split())
    return _ORDINAL_SEASON.sub(r"season \1", title_words)


@lru_cache(maxsize=4096)
def _remove_accent(character: str) -> str:
    """Return a Latin letter without its accents; any other character unchanged.

    Only Latin letters lose their marks: the voiced mark of a Japanese kana
    decomposes the same way, and dropping it would turn "ガ" into "カ".

    Examples:
        >>> _remove_accent("é")
        'e'
        >>> _remove_accent("ガ")
        'ガ'
    """
    if character.isascii():
        return character
    base_letter, *accent_marks = unicodedata.normalize("NFD", character)
    if (
        base_letter.isascii()
        and base_letter.isalpha()
        and all(map(unicodedata.combining, accent_marks))
    ):
        return base_letter
    return character
