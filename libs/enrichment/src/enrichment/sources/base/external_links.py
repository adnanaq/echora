"""Build canonical ExternalLink entries from provider links.

Providers name the same site differently — AniList sends a category
("Official Site"), MAL the account handle ("@heroaca_anime"), AniSearch the
page's title ("Toei Animation"). The platform is therefore derived from the
url's host, and the provider's own name is kept as the label.
"""

from urllib.parse import unquote, urlsplit

from common.models.anime import ExternalLink

# Hosts shared by many anime. Anything else is the work's own site and folds
# into `official_site`; keying those by host would invent a platform per anime.
_SERVICES: dict[str, str] = {
    "twitter.com": "twitter",
    "x.com": "twitter",
    "youtube.com": "youtube",
    "youtu.be": "youtube",
    "instagram.com": "instagram",
    "tiktok.com": "tiktok",
    "facebook.com": "facebook",
    "weibo.com": "weibo",
    "bgm.tv": "bangumi",
    "bangumi.tv": "bangumi",
    "douban.com": "douban",
    "baike.baidu.com": "baidu_baike",
    "imdb.com": "imdb",
    "themoviedb.org": "themoviedb",
    "allcinema.net": "allcinema",
    "anison.info": "anison",
    "cal.syoboi.jp": "syoboi",
    "lain.gr.jp": "lain",
    "vndb.org": "vndb",
    "animemorial.net": "animemorial",
    "mediaarts-db.artmuseums.go.jp": "media_arts_database",
    "tumblr.com": "tumblr",
    "note.com": "note",
    "ameblo.jp": "ameblo",
    "line.me": "line",
    "discord.gg": "discord",
    "vimeo.com": "vimeo",
    "nicovideo.jp": "niconico",
    "pixiv.net": "pixiv",
    "crunchyroll.com": "crunchyroll",
    "funimation.com": "funimation",
    "netflix.com": "netflix",
    "hidive.com": "hidive",
    "amazon.com": "amazon",
    "primevideo.com": "prime_video",
    "hulu.com": "hulu",
    "disneyplus.com": "disney_plus",
    "max.com": "max",
    "v.qq.com": "qq_video",
    "bilibili.com": "bilibili",
    "iq.com": "iqiyi",
    "iqiyi.com": "iqiyi",
    "wetv.vip": "wetv",
    "youku.com": "youku",
    "abema.tv": "abema",
    "u-next.jp": "u_next",
    "ani.gamer.com.tw": "bahamut",
    "tubitv.com": "tubi",
    "viki.com": "viki",
    "vrv.co": "vrv",
    "animenewsnetwork.com": "anime_news_network",
    "myanimelist.net": "myanimelist",
    "anidb.net": "anidb",
    "anilist.co": "anilist",
    "kitsu.app": "kitsu",
    "kitsu.io": "kitsu",
    "anime-planet.com": "anime_planet",
    "anisearch.com": "anisearch",
    "animeschedule.net": "animeschedule",
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
    """Return the url's host without a leading ``www.``."""
    host = urlsplit(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def canonical_platform(url: str) -> str:
    """Return the platform name for a url, derived from its host."""
    host = host_of(url)
    if not host:
        return "unknown"
    if host.endswith("wikipedia.org"):
        return f"wikipedia_{host.split('.')[0]}"
    for suffix, name in _SERVICES.items():
        if host == suffix or host.endswith("." + suffix):
            return name
    return "official_site"


def normalize_link_url(url: str) -> str:
    """Collapse urls that differ only cosmetically."""
    parts = urlsplit(unquote(url.strip()))
    host = parts.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if host == "x.com":
        host = "twitter.com"
    scheme = parts.scheme or "https"
    query = "&".join(
        kv
        for kv in parts.query.split("&")
        if kv and kv.split("=")[0].lower() not in _TRACKING
    )
    path = parts.path.rstrip("/")
    return f"{scheme}://{host}{path}" + (f"?{query}" if query else "")


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

    Args:
        url: The link itself.
        label: The provider's own name for the link.
        language: Language of the linked page, when the provider states it.
        platform: Overrides the host-derived platform. Only for providers that
            already know the service, such as AniDB's typed resources.

    Returns:
        An ExternalLink carrying the provider url, or None.
    """
    if not url or not url.strip():
        return None
    source = url.strip()
    if not host_of(source):
        return None
    return ExternalLink(
        platform=platform or canonical_platform(source),
        source=source,
        label=label.strip() if label and label.strip() else None,
        language=language,
    )
