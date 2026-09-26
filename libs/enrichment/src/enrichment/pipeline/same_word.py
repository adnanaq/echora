"""Decide when two differently-written words are the same word.

Providers write the same thing several ways. Folding case and punctuation
catches most of it — ``action`` and ``Action``, ``cross-dressing`` and
``Cross Dressing`` — but not the ones where the text genuinely differs:
``science fiction`` against ``Sci-Fi``, ``Superpowers`` against
``Super Power``.

Those are listed below. Every one was found by comparing all 3,124 words the
seven providers publish — MAL, AniList, AniDB, Kitsu, Anime-Planet, AniSearch
and AnimeSchedule — and each was checked by hand. That check is the whole
point: close-looking words are usually not the same word. ``France`` scores
77% against ``Romance``, ``sentai`` 83% against ``Hentai``, and ``Shounen Ai``
82% against ``Shounen`` while meaning something else entirely. Of 26 candidates
a similarity threshold proposed, 24 were wrong.

There is deliberately no list of genres or themes here. Which field a word
belongs in is decided by what the providers called it, not by a vocabulary we
maintain — see ``merge_categories``.
"""

from __future__ import annotations

from enrichment.utils.text_utils import fold_for_comparison

# Written differently, meaning the same. Left side and right side both fold to
# the same key, so either spelling can arrive from any provider.
_SAME: dict[str, str] = {
    # abbreviation and expansion
    "science fiction": "sci fi",
    # singular against plural
    "delinquent": "delinquents",
    "detectives": "detective",
    "high stakes games": "high stakes game",
    "pet": "pets",
    "strategy games": "strategy game",
    "superpowers": "super power",
    "vampires": "vampire",
    "video games": "video game",
    # one word against two
    "cross dressing": "crossdressing",
    "iyashi kei": "iyashikei",
    # different ending, same idea
    "anthropomorphism": "anthropomorphic",
}


def word_key(value: str) -> str:
    """Return the key every spelling of this word shares.

    Args:
        value: A genre, theme, demographic or tag as a provider wrote it.

    Returns:
        A key for grouping. Two values that mean the same word share it.
    """
    folded = fold_for_comparison(value)
    return _SAME.get(folded, folded)
