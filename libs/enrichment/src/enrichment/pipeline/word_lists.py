"""The word lists that decide whether a value is a genre, theme or demographic.

Providers disagree about which field a word belongs in. AnimeSchedule files
`Shounen` under genres, Kitsu files `Super Power` there, and AniList calls
`School` a setting where MAL calls it a theme. Where a word arrived is a habit
of the provider, not a fact about the word, so the merge decides for itself by
looking the word up here.

These lists come from MyAnimeList, read from
``https://api.jikan.moe/v4/genres/anime`` on 2026-09-21. That one address
returns all 78 words; split the way MAL splits them they are 5 demographics,
21 genres and 52 themes, matching what ``docs/stage1_metadata_merge.md``
counted separately.

AniList's `Theme-*` words are not listed here, and do not need to be. AniList's
own mapper already files them under ``themes`` (``anilist_mapper.py:165-174``),
so they arrive sorted and stay themes unless a list above claims them.
"""

from __future__ import annotations

# Smallest and most specific list, so it is consulted first: a demographic must
# never be left to rest as a genre.
DEMOGRAPHICS: frozenset[str] = frozenset(
    {"Josei", "Kids", "Seinen", "Shoujo", "Shounen"}
)

# MAL's genres plus its three explicit ones. MAL and AniList agree on every
# genre in the measured sample, which is why this list outranks themes.
GENRES: frozenset[str] = frozenset(
    {
        "Action",
        "Adventure",
        "Avant Garde",
        "Award Winning",
        "Boys Love",
        "Comedy",
        "Drama",
        "Ecchi",
        "Erotica",
        "Fantasy",
        "Girls Love",
        "Gourmet",
        "Hentai",
        "Horror",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Slice of Life",
        "Sports",
        "Supernatural",
        "Suspense",
    }
)

THEMES: frozenset[str] = frozenset(
    {
        "Adult Cast",
        "Anthropomorphic",
        "CGDCT",
        "Childcare",
        "Combat Sports",
        "Crossdressing",
        "Delinquents",
        "Detective",
        "Educational",
        "Gag Humor",
        "Gore",
        "Harem",
        "High Stakes Game",
        "Historical",
        "Idols (Female)",
        "Idols (Male)",
        "Isekai",
        "Iyashikei",
        "Love Polygon",
        "Love Status Quo",
        "Magical Sex Shift",
        "Mahou Shoujo",
        "Martial Arts",
        "Mecha",
        "Medical",
        "Military",
        "Music",
        "Mythology",
        "Organized Crime",
        "Otaku Culture",
        "Parody",
        "Performing Arts",
        "Pets",
        "Psychological",
        "Racing",
        "Reincarnation",
        "Reverse Harem",
        "Samurai",
        "School",
        "Showbiz",
        "Space",
        "Strategy Game",
        "Super Power",
        "Survival",
        "Team Sports",
        "Time Travel",
        "Urban Fantasy",
        "Vampire",
        "Video Game",
        "Villainess",
        "Visual Arts",
        "Workplace",
    }
)

# Checked in this order, smallest list first. A word goes to the first list
# that holds it, no matter which field the provider put it in.
_SEARCH_ORDER: tuple[tuple[str, frozenset[str]], ...] = (
    ("demographics", DEMOGRAPHICS),
    ("genres", GENRES),
    ("themes", THEMES),
)

# Maps a loosened spelling back to the list's own, so the merge stores
# "Shounen" rather than whichever of "Shounen"/"shounen" happened to arrive.
_LOOKUP: dict[str, tuple[str, str]] = {
    word.casefold().replace("-", " ").replace("_", " "): (field, word)
    for field, words in _SEARCH_ORDER
    for word in words
}


def which_field(value: str, *, default: str = "tags") -> tuple[str, str]:
    """Say which field a word belongs in, and how it should be spelled.

    Case and separators are loosened before the lookup, so ``shounen``,
    ``Shounen`` and ``SHOUNEN`` count as one word rather than three.

    Args:
        value: A genre, theme, demographic or tag as a provider sent it.
        default: Where to put a word that is on no list. ``tags`` is the
            catch-all; pass ``themes`` for a word that arrived as a theme, so
            AniList's own sorting is not thrown away.

    Returns:
        ``(field, spelling)``. The provider's spelling is kept only when the
        word is on no list.
    """
    key = value.strip().casefold().replace("-", " ").replace("_", " ")
    return _LOOKUP.get(key) or (default, value.strip())
