"""Reduce a provider page address to a key naming the page, whatever its spelling.

Providers decorate the same page differently: MAL appends a title slug,
AniSearch appends ``,slug``, MAL still links AniDB's old ``animedb.pl``
address, and ANN keeps the id in the query. Two spellings of one page must give
one key, so pages are compared by key, never by address. A page that has both
a number and a slug, as on Kitsu, gets one key for each: telling them apart
needs the provider, so the seed's number form is the one to record.

Enrichment uses the same keys to merge provider records and to derive anime ids,
so a change to a key changes those too.
"""

import re
from collections import Counter
from collections.abc import Iterable
from functools import lru_cache

# A numeric id ends where the address does or at a character that cannot be
# part of it, so "/anime/21extra" is not page 21.
_NUMERIC_ID = r"(?P<id>\d+)(?!\w)"
# A slug id runs to the next path, query or fragment separator.
_SLUG_ID = r"(?P<id>[^/?#]+)"
_NUMBERED_PAGE = re.compile(rf"/(?P<kind>anime|manga)/{_NUMERIC_ID}")
_SLUG_PAGE = re.compile(rf"/(?P<kind>anime|manga)/{_SLUG_ID}")
_NUMBERED_ANIME_PAGE = re.compile(rf"/(?P<kind>anime)/{_NUMERIC_ID}")

# The pages each host serves, matched against the address after the host. Every
# pattern names the id explicitly; prefix comparison is not safe, because
# "/anime/466" is a prefix of "/anime/46633", two unrelated works. A pattern
# without a kind names an anime page.
_PAGE_PATTERNS: dict[str, tuple[tuple[str, re.Pattern[str]], ...]] = {
    "myanimelist.net": (("mal", _NUMBERED_PAGE),),
    "anisearch.com": (("anisearch", _NUMBERED_PAGE),),
    "anilist.co": (("anilist", _NUMBERED_PAGE),),
    "anidb.net": (
        ("anidb", _NUMBERED_ANIME_PAGE),
        # AniDB's old address for the same page, which MAL still links.
        (
            "anidb",
            re.compile(
                r"/perl-bin/animedb\.pl\?(?=[^#]*\bshow=(?P<kind>anime)\b)"
                rf"[^#]*\baid={_NUMERIC_ID}"
            ),
        ),
    ),
    "kitsu.app": (("kitsu", _SLUG_PAGE),),
    "kitsu.io": (("kitsu", _SLUG_PAGE),),
    "anime-planet.com": (("animeplanet", _SLUG_PAGE),),
    # ANN puts the id in the query, where its position varies.
    "animenewsnetwork.com": (
        (
            "ann",
            re.compile(
                rf"/encyclopedia/(?P<kind>anime|manga)\.php\?(?:[^#]*&)?id={_NUMERIC_ID}"
            ),
        ),
    ),
    "livechart.me": (("livechart", _NUMBERED_ANIME_PAGE),),
    "simkl.com": (("simkl", _NUMBERED_ANIME_PAGE),),
    "animecountdown.com": (("animecountdown", re.compile(rf"/{_NUMERIC_ID}")),),
}

PROVIDERS: frozenset[str] = frozenset(
    provider for patterns in _PAGE_PATTERNS.values() for provider, _ in patterns
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


def rank_by_provider_trust(key: str) -> int:
    """Rank a page by how far its provider is trusted to name a work, MAL first.

    Pages of providers outside ``REFERENCE_ORDER`` rank after all of them.

    Examples:
        >>> rank_by_provider_trust("mal:anime:1") < rank_by_provider_trust("ann:anime:1")
        True
    """
    provider = get_page_provider(key)
    return (
        REFERENCE_ORDER.index(provider)
        if provider in REFERENCE_ORDER
        else len(REFERENCE_ORDER)
    )


@lru_cache(maxsize=65536)
def get_page_key(address: str) -> str:
    """Return the key naming the page an address points to.

    Args:
        address: A provider page address.

    Returns:
        A ``provider:kind:id`` key when the address is recognised, otherwise the
        address with its scheme and host in lower case, ``www.`` and any
        trailing slash removed. The path keeps its case, since another site can
        serve different pages at ``/Show`` and ``/show``.

    Examples:
        >>> get_page_key("https://myanimelist.net/anime/21/One_Piece")
        'mal:anime:21'
        >>> get_page_key("https://anidb.net/perl-bin/animedb.pl?show=anime&aid=69")
        'anidb:anime:69'
    """
    scheme, scheme_separator, address_after_scheme = address.partition("://")
    host, _, path = address_after_scheme.partition("/")
    host = host.lower().removeprefix("www.")
    if scheme.lower() in ("http", "https"):
        for provider, pattern in _PAGE_PATTERNS.get(host, ()):
            if page_match := pattern.match(f"/{path}"):
                page_kind = page_match.groupdict().get("kind") or "anime"
                return f"{provider}:{page_kind}:{page_match['id'].lower()}"
    if not scheme_separator:
        return address.rstrip("/")
    return f"{scheme.lower()}://{host}/{path}".rstrip("/")


def get_page_provider(key: str) -> str | None:
    """Return the provider a page key belongs to, or None for an unrecognised address.

    Examples:
        >>> get_page_provider("kitsu:anime:12")
        'kitsu'
        >>> get_page_provider("https://example.com/show") is None
        True
    """
    provider = key.partition(":")[0]
    return provider if provider in PROVIDERS else None


def get_page_id(key: str) -> str:
    """Return the id part of a page key, or the key itself when it is unrecognised.

    A key is ``provider:kind:id``, and everything after the kind is the id,
    even a slug holding a colon.

    Examples:
        >>> get_page_id("mal:anime:21")
        '21'
    """
    return key.split(":", 2)[2] if get_page_provider(key) else key


def has_repeated_provider(keys: Iterable[str]) -> bool:
    """Report whether two or more of the page keys belong to one provider.

    Unrecognised keys belong to no provider, so they never count.

    Examples:
        >>> has_repeated_provider(["mal:anime:1", "kitsu:anime:1", "mal:anime:2"])
        True
        >>> has_repeated_provider(["mal:anime:1", "kitsu:anime:1"])
        False
    """
    providers = Counter(filter(None, map(get_page_provider, keys)))
    return any(page_count > 1 for page_count in providers.values())
