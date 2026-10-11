"""Build canonical ExternalLink entries from provider links.

Providers name the same site differently — AniList sends a category
("Official Site"), MAL the account handle ("@heroaca_anime"), AniSearch the
page's title ("Toei Animation"). The platform is therefore derived from the
url's domain, and the provider's own name is kept as the label.

The domain gives the name for almost every site: ``www.imdb.com`` is ``imdb``,
``cal.syoboi.jp`` is ``syoboi``, ``toei-anim.co.jp`` is ``toei_anim``. Only the
few sites whose domain is not the name used for them are listed here.
"""

from urllib.parse import unquote, urlsplit

import tldextract
from common.models.anime import ExternalLink

# The suffix list bundled with tldextract, so naming never goes online and
# gives the same answer on every run.
_DOMAIN_PARTS = tldextract.TLDExtract(suffix_list_urls=())

OFFICIAL_SITE = "official_site"

# Sites whose domain is not the name used for them, by host or by domain.
_PLATFORM_NAMES: dict[str, str] = {
    "animenewsnetwork.com": "anime_news_network",
    "baike.baidu.com": "baidu_baike",
    "bgm.tv": "bangumi",
    "disneyplus.com": "disney_plus",
    "primevideo.com": "prime_video",
    "mediaarts-db.artmuseums.go.jp": "media_arts_database",
    "home-aki.la.coocan.jp": "tv_animation_museum",
    "ani.gamer.com.tw": "bahamut",
    "iq.com": "iqiyi",
    "v.qq.com": "qq_video",
    "nicovideo.jp": "niconico",
    "tubitv.com": "tubi",
    "x.com": "twitter",
    "youtu.be": "youtube",
}

# Link shorteners: the domain names no service, so a caller that knows one
# from elsewhere, such as the provider's label, has to supply it.
_SHORTENERS = frozenset({"amzn.to", "apple.co"})

# Provider labels that mark a link as the work's own site.
_OFFICIAL_LABELS = frozenset({"official site", "official website", "official"})

# A site and, for sites with several, the kind of page: ("anilist", "manga").
type PageKind = tuple[str, str | None]

# Address of a work's page on a site, for providers that give only an
# identifier: AniDB's typed resources and Kitsu's mappings. Looked up through
# `_page_key`, so "animenewsnetwork" and "anime_news_network" are one site.
_PAGE_ADDRESSES: dict[PageKind, str] = {
    ("myanimelist", "anime"): "https://myanimelist.net/anime/{}",
    ("anilist", "anime"): "https://anilist.co/anime/{}",
    ("anilist", "manga"): "https://anilist.co/manga/{}",
    ("anidb", None): "https://anidb.net/anime/{}",
    (
        "anime_news_network",
        None,
    ): "https://www.animenewsnetwork.com/encyclopedia/anime.php?id={}",
    ("thetvdb", "series"): "https://thetvdb.com/dereferrer/series/{}",
    ("thetvdb", "season"): "https://thetvdb.com/dereferrer/season/{}",
    # Trakt ids from Kitsu are show ids, on movies too, where they name the
    # film's series; Trakt's movie ids are a separate list.
    ("trakt", None): "https://trakt.tv/shows/{}",
    ("imdb", None): "https://www.imdb.com/title/{}",
    ("themoviedb", "tv"): "https://www.themoviedb.org/tv/{}",
    ("themoviedb", "movie"): "https://www.themoviedb.org/movie/{}",
    ("vndb", None): "https://vndb.org/{}",
    ("wikipedia", "en"): "https://en.wikipedia.org/wiki/{}",
    ("wikipedia", "ja"): "https://ja.wikipedia.org/wiki/{}",
    ("wikipedia", "ko"): "https://ko.wikipedia.org/wiki/{}",
    ("wikipedia", "zh"): "https://zh.wikipedia.org/wiki/{}",
    ("syoboi", None): "https://cal.syoboi.jp/tid/{}/time",
    ("allcinema", None): "https://www.allcinema.net/cinema/{}",
    ("anison", None): "http://anison.info/data/program/{}.html",
    ("lain", None): "http://lain.gr.jp/{}",
    ("animemorial", None): "http://www.animemorial.net/ja/{}-a",
    ("tv_animation_museum", None): "http://home-aki.la.coocan.jp/anime-list/{}.htm",
    ("baidu_baike", None): "https://baike.baidu.com/item/{}",
    ("bangumi", None): "https://bgm.tv/subject/{}",
    ("douban", None): "https://movie.douban.com/subject/{}",
    ("facebook", None): "https://www.facebook.com/{}",
    ("twitter", None): "https://twitter.com/{}",
    ("youtube", None): "https://www.youtube.com/{}",
    ("crunchyroll", None): "https://www.crunchyroll.com/series/{}",
    ("amazon", None): "https://www.amazon.com/dp/{}",
    ("netflix", None): "https://www.netflix.com/title/{}",
    ("hidive", None): "https://www.hidive.com/{}",
    ("funimation", None): "https://www.funimation.com/shows/{}",
    ("qq_video", None): "https://v.qq.com/detail/{}",
    ("bilibili", None): "https://www.bilibili.com/{}",
    ("prime_video", None): "https://www.primevideo.com/detail/{}",
}

# Share and analytics parameters identify the referrer, not the resource.
_TRACKING = frozenset(
    {
        "t",
        "s",
        "ref",
        "ref_src",
        "igshid",
        "fbclid",
        "gclid",
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
    }
)


def host_of(url: str) -> str:
    """Return the url's host without a leading ``www.``.

    Accepts an address with or without its scheme.

    Args:
        url: The address, such as ``"https://www.abc.com/page"`` or ``"abc.com"``.

    Returns:
        The lowercase host without ``www.``, or an empty string for anything
        that is not a web address on a public domain, such as an IP address or
        ``localhost``.

    Examples:
        >>> host_of("https://www.Abc.com:8080/page")
        'abc.com'
        >>> host_of("abc.com")
        'abc.com'
        >>> host_of("https://192.168.0.1/")
        ''
    """
    return _DOMAIN_PARTS(url.strip()).fqdn.lower().removeprefix("www.")


def canonical_platform(url: str) -> str:
    """Return the platform name for a url, derived from its domain.

    A name listed for the host or one of its parent domains wins; otherwise the
    domain gives the name, with a hyphen turned into an underscore. Wikipedia
    keeps its language.

    Args:
        url: The link, with or without its scheme.

    Returns:
        The platform name, or ``"unknown"`` for a url on no public domain or
        on a link shortener.

    Examples:
        >>> canonical_platform("https://www.imdb.com/title/tt0388629")
        'imdb'
        >>> canonical_platform("https://www.toei-anim.co.jp/tv/onep/")
        'toei_anim'
        >>> canonical_platform("https://baike.baidu.com/item/x")
        'baidu_baike'
        >>> canonical_platform("https://zhidao.baidu.com/question/1")
        'baidu'
        >>> canonical_platform("https://ja.wikipedia.org/wiki/x")
        'wikipedia_ja'
        >>> canonical_platform("https://amzn.to/3xyz")
        'unknown'
    """
    parts = _DOMAIN_PARTS(url.strip())
    domain = parts.top_domain_under_public_suffix.lower()
    if not domain or domain in _SHORTENERS:
        return "unknown"
    if domain == "wikipedia.org":
        return f"wikipedia_{parts.subdomain.lower().partition('.')[0]}"
    candidate = parts.fqdn.lower().removeprefix("www.")
    while candidate != domain:
        if name := _PLATFORM_NAMES.get(candidate):
            return name
        candidate = candidate.partition(".")[2]
    return _PLATFORM_NAMES.get(domain) or parts.domain.lower().replace("-", "_")


def normalize_link_url(url: str) -> str:
    """Collapse urls that differ only cosmetically, for comparison.

    The scheme is dropped rather than kept: providers publish the same page as
    both ``http://`` and ``https://`` — MAL and AniDB each link the Toei site
    one way and AniSearch the other — and keeping it filed the same page twice.
    The result is a comparison key, not a url to visit.

    Args:
        url: The link to fold.

    Returns:
        The host without ``www.`` (``x.com`` as ``twitter.com``), the path
        without its trailing slash, and the query without tracking parameters.

    Examples:
        >>> normalize_link_url("http://www.toei-anim.co.jp/tv/onep/")
        'toei-anim.co.jp/tv/onep'
        >>> normalize_link_url("https://x.com/OnePieceAnime?ref=share")
        'twitter.com/OnePieceAnime'
    """
    parts = urlsplit(unquote(url.strip()))
    host = parts.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if host == "x.com":
        host = "twitter.com"
    query = "&".join(
        kv
        for kv in parts.query.split("&")
        if kv and kv.split("=")[0].lower() not in _TRACKING
    )
    path = parts.path.rstrip("/")
    return f"{host}{path}" + (f"?{query}" if query else "")


def external_link(
    url: str | None,
    *,
    label: str | None = None,
    language: str | None = None,
    platform: str | None = None,
) -> ExternalLink | None:
    """Build an ExternalLink, or None when the url is unusable.

    The url is stored as the provider gave it. Normalizing here would mean
    recording an address the provider never published, and stripping ``www.``
    breaks hosts that require it; ``normalize_link_url`` is for the merge, where
    two links are being compared rather than recorded.

    A link is the work's own site only when the provider says so, by labelling
    it "Official Site" or by passing ``platform=OFFICIAL_SITE``; any other link
    is named after its domain.

    Args:
        url: The link itself.
        label: The provider's own name for the link.
        language: Language of the linked page, when the provider states it.
        platform: Overrides the domain-derived platform. Only for providers
            that already know the service, such as AniDB's typed resources.

    Returns:
        An ExternalLink carrying the provider url, or None.

    Examples:
        >>> external_link("https://www.imdb.com/title/tt0388629").platform
        'imdb'
        >>> external_link("https://one-piece.com/", label="Official Site").platform
        'official_site'
        >>> external_link("not a link") is None
        True
    """
    if not url or not url.strip():
        return None
    source = url.strip()
    if not host_of(source):
        return None
    label = label.strip() if label and label.strip() else None
    if platform is None and label and label.lower() in _OFFICIAL_LABELS:
        platform = OFFICIAL_SITE
    return ExternalLink(
        platform=platform or canonical_platform(source),
        source=source,
        label=label,
        language=language,
    )


def page_link(
    platform: str,
    identifier: str,
    *,
    kind: str | None = None,
    language: str | None = None,
) -> ExternalLink | None:
    """Build the link to a work's page on a site from the site's identifier.

    Args:
        platform: The site, such as ``"myanimelist"`` or ``"thetvdb"``; spacing
            is ignored, so ``"animenewsnetwork"`` matches ``"anime_news_network"``.
        identifier: The work's identifier on that site.
        kind: Which kind of page, for sites with more than one, such as
            ``"anime"`` or ``"manga"`` on AniList.
        language: Language of the linked page, when known.

    Returns:
        An ExternalLink to the page, or None when the site and kind have no
        known address or the identifier is empty.

    Examples:
        >>> page_link("anilist", "30013", kind="manga").source
        'https://anilist.co/manga/30013'
        >>> page_link("animenewsnetwork", "836").platform
        'anime_news_network'
        >>> page_link("aozora", "a70pbqlcBK") is None
        True
    """
    template = _PAGE_TEMPLATES.get(_page_key(platform, kind))
    identifier = identifier.strip()
    if not template or not identifier:
        return None
    return external_link(template.format(identifier), language=language)


def _page_key(platform: str, kind: str | None) -> PageKind:
    """Fold a site name so providers' spellings of it match one entry.

    Args:
        platform: The site name as a provider or the address list spells it.
        kind: The page kind, where the site has more than one.

    Returns:
        The site name in lowercase without underscores or hyphens, and the kind
        with an empty string taken as no kind.

    Examples:
        >>> _page_key("anime_news_network", None) == _page_key("animenewsnetwork", "")
        True
        >>> _page_key("Anime-Planet", "anime")
        ('animeplanet', 'anime')
    """
    return platform.lower().replace("_", "").replace("-", ""), kind or None


_PAGE_TEMPLATES: dict[PageKind, str] = {
    _page_key(platform, kind): template
    for (platform, kind), template in _PAGE_ADDRESSES.items()
}
