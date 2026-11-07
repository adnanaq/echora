"""Reduce a provider page address to a canonical key regardless of URL spelling.

Providers format the same page address in multiple ways across external links.
MyAnimeList appends a title slug, AniSearch appends a comma-separated slug, and
Anime News Network places the identifier in the query string. MyAnimeList also
links to AniDB using legacy ``animedb.pl`` URLs. Comparing pages reliably
requires reducing every spelling of a page to a single canonical key. When a
provider such as Kitsu supports both numeric and slug identifiers for the same
page, the seed's numeric identifier is recorded as the canonical form.

The enrichment pipeline uses these same keys to merge provider records and
derive anime identifiers.
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

# Each provider's page address as the seed spells it, filled with a key's kind
# and id. Links are kept as page keys, so fetching a linked page needs its
# address back; the seed's spelling keeps the cache shared with seed pages.
_SEED_ADDRESS_BY_PROVIDER = {
    "mal": "https://myanimelist.net/{kind}/{id}",
    "anisearch": "https://anisearch.com/{kind}/{id}",
    "anilist": "https://anilist.co/{kind}/{id}",
    "anidb": "https://anidb.net/{kind}/{id}",
    "kitsu": "https://kitsu.app/{kind}/{id}",
    "animeplanet": "https://anime-planet.com/{kind}/{id}",
    "ann": "https://animenewsnetwork.com/encyclopedia/{kind}.php?id={id}",
    "livechart": "https://livechart.me/{kind}/{id}",
    "simkl": "https://simkl.com/{kind}/{id}",
    "animecountdown": "https://animecountdown.com/{id}",
}

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


def rank_by_provider_trust(page_key: str) -> int:
    """Rank a page by how far its provider is trusted to name a work, MAL first.

    Pages of providers outside ``REFERENCE_ORDER`` rank after all of them.

    Args:
        page_key: A page key, as ``resolve_page_key`` gives it.

    Returns:
        The zero-based position of the page's provider in ``REFERENCE_ORDER``,
        or ``len(REFERENCE_ORDER)`` for unrecognised providers.

    Examples:
        >>> rank_by_provider_trust("mal:anime:1") < rank_by_provider_trust("ann:anime:1")
        True
    """
    provider = read_page_provider(page_key)
    return (
        REFERENCE_ORDER.index(provider)
        if provider in REFERENCE_ORDER
        else len(REFERENCE_ORDER)
    )


@lru_cache(maxsize=65536)
def resolve_page_key(address: str) -> str:
    """Return the canonical key naming the page an address points to.

    Unrecognised URLs keep their path case because external hosts can serve
    distinct pages at ``/Show`` and ``/show``.

    Args:
        address: A provider page address.

    Returns:
        A ``provider:kind:id`` key for recognised provider addresses, or the
        normalised URL with a lowercase scheme and host, ``www.`` removed, and
        any trailing slash stripped.

    Examples:
        >>> resolve_page_key("https://myanimelist.net/anime/21/One_Piece")
        'mal:anime:21'
        >>> resolve_page_key("https://anidb.net/perl-bin/animedb.pl?show=anime&aid=69")
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


def build_page_address(page_key: str) -> str | None:
    """Build the canonical seed URL for a recognised provider page key.

    This function reverses ``resolve_page_key`` so a page key extracted from
    cross-provider links can be fetched over HTTP.

    Args:
        page_key: A page key, as ``resolve_page_key`` gives it.

    Returns:
        The page address, or None when the key is not a known provider's page.

    Examples:
        >>> build_page_address("mal:anime:21")
        'https://myanimelist.net/anime/21'
        >>> build_page_address("https://example.com/show") is None
        True
    """
    provider = read_page_provider(page_key)
    if provider is None:
        return None
    _, page_kind, page_id = page_key.split(":", 2)
    return _SEED_ADDRESS_BY_PROVIDER[provider].format(kind=page_kind, id=page_id)


def read_page_provider(page_key: str) -> str | None:
    """Return the provider a page key belongs to, or None for an unrecognised address.

    Args:
        page_key: A page key, as ``resolve_page_key`` gives it.

    Returns:
        The provider name, or None when the key does not belong to a known
        provider.

    Examples:
        >>> read_page_provider("kitsu:anime:12")
        'kitsu'
        >>> read_page_provider("https://example.com/show") is None
        True
    """
    provider = page_key.partition(":")[0]
    return provider if provider in PROVIDERS else None


def is_mal_page(page_key: str) -> bool:
    """Check whether a page key is a MyAnimeList page.

    A work is what MAL lists, so several rules treat MAL pages apart.

    Args:
        page_key: A page key, as ``resolve_page_key`` gives it.

    Returns:
        True when the key belongs to MAL.
    """
    return read_page_provider(page_key) == "mal"


def read_page_kind(page_key: str) -> str | None:
    """Return the media kind (``anime`` or ``manga``) from a recognised page key.

    Args:
        page_key: A page key, as ``resolve_page_key`` gives it.

    Returns:
        ``anime`` or ``manga``, or None when the key is not a known provider's page.

    Examples:
        >>> read_page_kind("anilist:manga:119534")
        'manga'
        >>> read_page_kind("https://example.com/show") is None
        True
    """
    return page_key.split(":", 2)[1] if read_page_provider(page_key) else None


def read_page_id(page_key: str) -> str:
    """Return the identifier segment of a page key, or the key itself if unrecognised.

    A recognised key follows the ``provider:kind:id`` format where everything
    after the second colon belongs to the identifier, including colons in slugs.

    Args:
        page_key: A page key, as ``resolve_page_key`` gives it.

    Returns:
        The identifier segment of a recognised page key, or ``page_key`` unchanged.

    Examples:
        >>> read_page_id("mal:anime:21")
        '21'
    """
    return page_key.split(":", 2)[2] if read_page_provider(page_key) else page_key


def has_repeated_provider(page_keys: Iterable[str]) -> bool:
    """Report whether two or more page keys belong to the same provider.

    Unrecognised keys do not belong to any provider and are ignored.

    Args:
        page_keys: Page keys to inspect.

    Returns:
        True when at least two recognised keys share the same provider.

    Examples:
        >>> has_repeated_provider(["mal:anime:1", "kitsu:anime:1", "mal:anime:2"])
        True
        >>> has_repeated_provider(["mal:anime:1", "kitsu:anime:1"])
        False
    """
    page_count_by_provider = Counter(filter(None, map(read_page_provider, page_keys)))
    return any(page_count > 1 for page_count in page_count_by_provider.values())
