"""Reduce a provider page address to a key naming the page, whatever its spelling.

Providers decorate the same page differently: MAL appends a title slug,
AniSearch appends ``,slug``, MAL still links AniDB's old ``animedb.pl``
address, and ANN keeps the id in the query. Two spellings of one page must give
one key, so pages are compared by key, never by address.

The patterns are a copy of those in
``enrichment.pipeline.same_work.canonical_url_key``. They are copied rather than
shared so this library imports nothing from enrichment, and the two must give
the same key for every address.
"""

import re
from functools import lru_cache

# Every pattern is anchored and captures the id explicitly. Prefix comparison is
# not safe: "/anime/466" is a prefix of "/anime/46633", two unrelated works.
_PAGE_PATTERNS: dict[str, tuple[tuple[re.Pattern[str], str], ...]] = {
    "myanimelist.net": (
        (
            re.compile(r"^https?://(?:www\.)?myanimelist\.net/(anime|manga)/(\d+)"),
            "mal",
        ),
    ),
    "anisearch.com": (
        (
            re.compile(r"^https?://(?:www\.)?anisearch\.com/(anime|manga)/(\d+)"),
            "anisearch",
        ),
    ),
    "anilist.co": (
        (re.compile(r"^https?://(?:www\.)?anilist\.co/(anime|manga)/(\d+)"), "anilist"),
    ),
    "anidb.net": (
        (re.compile(r"^https?://(?:www\.)?anidb\.net/(anime)/(\d+)"), "anidb"),
        # AniDB's old address for the same page, which MAL still links.
        (
            re.compile(
                r"^https?://(?:www\.)?anidb\.net/perl-bin/animedb\.pl\?"
                r"(?=[^#]*\bshow=(anime)\b)[^#]*\baid=(\d+)"
            ),
            "anidb",
        ),
    ),
    "kitsu.app": (
        (
            re.compile(
                r"^https?://(?:www\.)?kitsu\.(?:app|io)/(anime|manga)/([^/?#]+)"
            ),
            "kitsu",
        ),
    ),
    "anime-planet.com": (
        (
            re.compile(
                r"^https?://(?:www\.)?anime-planet\.com/(anime|manga)/([^/?#]+)"
            ),
            "animeplanet",
        ),
    ),
    # ANN puts the id in the query, where its position and the scheme vary.
    "animenewsnetwork.com": (
        (
            re.compile(
                r"^https?://(?:www\.)?animenewsnetwork\.com/encyclopedia/(anime|manga)\.php\?"
                r"(?:[^#]*&)?id=(\d+)"
            ),
            "ann",
        ),
    ),
    "livechart.me": (
        (re.compile(r"^https?://(?:www\.)?livechart\.me/(anime)/(\d+)"), "livechart"),
    ),
    "simkl.com": (
        (re.compile(r"^https?://(?:www\.)?simkl\.com/(anime)/(\d+)"), "simkl"),
    ),
    "animecountdown.com": (
        (
            re.compile(r"^https?://(?:www\.)?animecountdown\.com/(\d+)()"),
            "animecountdown",
        ),
    ),
}
_PAGE_PATTERNS["kitsu.io"] = _PAGE_PATTERNS["kitsu.app"]

PROVIDERS: frozenset[str] = frozenset(
    provider for patterns in _PAGE_PATTERNS.values() for _, provider in patterns
)

# Which page names an entry: the providers that seed merged anime ids come
# first, in the same order as ``link_rules._ID_SEED_ORDER``.
REFERENCE_ORDER: tuple[str, ...] = (
    "mal",
    "anilist",
    "anidb",
    "kitsu",
    "animeplanet",
    "anisearch",
    "ann",
    "livechart",
    "simkl",
    "animecountdown",
)


@lru_cache(maxsize=65536)
def provider_page_key(address: str) -> str:
    """Return the key naming the page an address points to.

    Args:
        address: A provider page address.

    Returns:
        A ``provider:kind:id`` key when the address is recognised, otherwise the
        lowercased address with ``www.`` and any trailing slash removed.

    Examples:
        >>> provider_page_key("https://myanimelist.net/anime/21/One_Piece")
        'mal:anime:21'
        >>> provider_page_key("https://anidb.net/perl-bin/animedb.pl?show=anime&aid=69")
        'anidb:anime:69'
    """
    host = address.partition("://")[2].partition("/")[0].lower().removeprefix("www.")
    for pattern, provider in _PAGE_PATTERNS.get(host, ()):
        if match := pattern.match(address):
            kind, identifier = match.group(1), match.group(2)
            return f"{provider}:{kind}:{identifier.lower()}"
    return address.lower().replace("://www.", "://").rstrip("/")


def provider_of_key(key: str) -> str | None:
    """Return the provider a page key belongs to, or None for an unrecognised address.

    Examples:
        >>> provider_of_key("kitsu:anime:12")
        'kitsu'
        >>> provider_of_key("https://example.com/show") is None
        True
    """
    provider = key.partition(":")[0]
    return provider if provider in PROVIDERS else None
