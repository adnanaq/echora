import pytest
from enrichment.sources.base.external_links import (
    _PAGE_ADDRESSES,
    OFFICIAL_SITE,
    canonical_platform,
    external_link,
    host_of,
    normalize_link_url,
    page_link,
)


@pytest.mark.parametrize(
    "url",
    [
        "abc.com",
        "www.abc.com",
        "http://abc.com",
        "https://www.abc.com/path?q=1",
        "HTTPS://WWW.ABC.COM:8080/x",
        "  https://abc.com/  ",
    ],
)
def test_host_of_any_address_form_returns_host_without_www(url: str) -> None:
    assert host_of(url) == "abc.com"


@pytest.mark.parametrize(
    "url",
    ["https://192.168.0.1/", "https://localhost/x", "not a link", ""],
)
def test_host_of_address_on_no_public_domain_returns_empty_string(url: str) -> None:
    assert host_of(url) == ""


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.imdb.com/title/tt0388629", "imdb"),
        ("https://cal.syoboi.jp/tid/350/time", "syoboi"),
        ("https://thetvdb.com/dereferrer/series/81797", "thetvdb"),
        ("https://trakt.tv/shows/37696", "trakt"),
        ("https://kitsu.app/anime/12", "kitsu"),
        ("https://kitsu.io/anime/one-piece", "kitsu"),
        ("https://myanimelist.net/anime/21", "myanimelist"),
        ("https://www.toei-anim.co.jp/tv/onep/", "toei_anim"),
        ("https://www.anime-planet.com/anime/one-piece", "anime_planet"),
        ("https://www.u-next.jp/title/1", "u_next"),
        ("abc.com", "abc"),
        ("https://m.youtube.com/watch?v=1", "youtube"),
    ],
)
def test_canonical_platform_names_site_after_its_domain(
    url: str, expected: str
) -> None:
    assert canonical_platform(url) == expected


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "https://www.animenewsnetwork.com/encyclopedia/anime.php?id=836",
            "anime_news_network",
        ),
        ("https://baike.baidu.com/item/x", "baidu_baike"),
        ("https://bgm.tv/subject/975", "bangumi"),
        ("https://www.disneyplus.com/series/x", "disney_plus"),
        ("https://www.primevideo.com/detail/x", "prime_video"),
        ("https://mediaarts-db.artmuseums.go.jp/id/1", "media_arts_database"),
        ("http://home-aki.la.coocan.jp/anime-list/1.htm", "tv_animation_museum"),
        ("https://ani.gamer.com.tw/animeRef.php?sn=1", "bahamut"),
        ("https://www.iq.com/album/x", "iqiyi"),
        ("https://v.qq.com/detail/x", "qq_video"),
        ("https://www.nicovideo.jp/watch/x", "niconico"),
        ("https://tubitv.com/series/x", "tubi"),
        ("https://x.com/OnePieceAnime", "twitter"),
        ("https://youtu.be/abc123", "youtube"),
    ],
)
def test_canonical_platform_site_with_different_name_uses_listed_name(
    url: str, expected: str
) -> None:
    assert canonical_platform(url) == expected


def test_canonical_platform_other_host_on_listed_domain_uses_domain_name() -> None:
    assert canonical_platform("https://zhidao.baidu.com/question/1") == "baidu"


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://en.wikipedia.org/wiki/One_Piece", "wikipedia_en"),
        ("https://ja.wikipedia.org/wiki/x", "wikipedia_ja"),
        ("https://ja.m.wikipedia.org/wiki/x", "wikipedia_ja"),
    ],
)
def test_canonical_platform_wikipedia_keeps_language(url: str, expected: str) -> None:
    assert canonical_platform(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "https://amzn.to/3xyz",
        "https://apple.co/3abc",
        "https://192.168.0.1/",
        "not a link",
    ],
)
def test_canonical_platform_shortener_or_address_on_no_domain_returns_unknown(
    url: str,
) -> None:
    assert canonical_platform(url) == "unknown"


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("http://www.toei-anim.co.jp/tv/onep/", "https://toei-anim.co.jp/tv/onep"),
        ("https://x.com/OnePieceAnime", "https://twitter.com/OnePieceAnime"),
        (
            "https://example.com/page?utm_source=x&id=7&ref=y",
            "https://example.com/page?id=7",
        ),
        ("https://example.com/%7Euser", "https://example.com/~user"),
    ],
)
def test_normalize_link_url_cosmetic_differences_give_same_key(
    first: str, second: str
) -> None:
    assert normalize_link_url(first) == normalize_link_url(second)


def test_normalize_link_url_different_paths_give_different_keys() -> None:
    assert normalize_link_url("https://shingeki.tv") != normalize_link_url(
        "https://shingeki.tv/season1"
    )


def test_external_link_keeps_address_as_given_with_domain_platform() -> None:
    link = external_link("  https://www.imdb.com/title/tt0388629  ", language="English")
    assert (link.platform, link.source, link.label, link.language) == (
        "imdb",
        "https://www.imdb.com/title/tt0388629",
        None,
        "English",
    )


@pytest.mark.parametrize("url", [None, "", "   ", "not a link", "https://localhost/x"])
def test_external_link_unusable_address_returns_none(url: str | None) -> None:
    assert external_link(url) is None


@pytest.mark.parametrize("label", ["Official Site", "official website", " official "])
def test_external_link_official_label_gives_official_site(label: str) -> None:
    link = external_link("https://one-piece.com/", label=label)
    assert (link.platform, link.label) == (OFFICIAL_SITE, label.strip())


def test_external_link_other_label_keeps_domain_platform() -> None:
    link = external_link("https://one-piece.com/", label="One Piece")
    assert (link.platform, link.label) == ("one_piece", "One Piece")


def test_external_link_blank_label_gives_no_label() -> None:
    assert external_link("https://one-piece.com/", label="  ").label is None


def test_external_link_stated_platform_wins_over_label_and_domain() -> None:
    link = external_link(
        "https://gkids.com/", label="Official Site", platform="distributor"
    )
    assert link.platform == "distributor"


@pytest.mark.parametrize(
    ("platform", "identifier", "kind", "expected"),
    [
        ("myanimelist", "21", "anime", "https://myanimelist.net/anime/21"),
        ("anilist", "30013", "manga", "https://anilist.co/manga/30013"),
        ("anidb", "69", None, "https://anidb.net/anime/69"),
        ("thetvdb", "482226", "season", "https://thetvdb.com/dereferrer/season/482226"),
        ("trakt", "37696", None, "https://trakt.tv/shows/37696"),
        ("themoviedb", "37854", "tv", "https://www.themoviedb.org/tv/37854"),
        ("wikipedia", "One_Piece", "ja", "https://ja.wikipedia.org/wiki/One_Piece"),
    ],
)
def test_page_link_site_and_identifier_give_page_address(
    platform: str, identifier: str, kind: str | None, expected: str
) -> None:
    assert page_link(platform, identifier, kind=kind).source == expected


@pytest.mark.parametrize(
    "platform", ["animenewsnetwork", "anime_news_network", "Anime-News-Network"]
)
def test_page_link_site_name_spelled_any_way_finds_same_page(platform: str) -> None:
    link = page_link(platform, "836")
    assert (link.platform, link.source) == (
        "anime_news_network",
        "https://www.animenewsnetwork.com/encyclopedia/anime.php?id=836",
    )


def test_page_link_empty_kind_counts_as_no_kind() -> None:
    assert page_link("anidb", "69", kind="").source == "https://anidb.net/anime/69"


def test_page_link_keeps_language_and_trims_identifier() -> None:
    link = page_link("myanimelist", " 21 ", kind="anime", language="English")
    assert (link.source, link.language) == (
        "https://myanimelist.net/anime/21",
        "English",
    )


@pytest.mark.parametrize(
    ("platform", "identifier", "kind"),
    [
        ("aozora", "a70pbqlcBK", None),
        ("hulu", "50024059", None),
        ("anilist", "21", "character"),
        ("myanimelist", "  ", "anime"),
    ],
)
def test_page_link_unknown_page_or_empty_identifier_returns_none(
    platform: str, identifier: str, kind: str | None
) -> None:
    assert page_link(platform, identifier, kind=kind) is None


@pytest.mark.parametrize(("platform", "kind"), list(_PAGE_ADDRESSES))
def test_page_link_every_listed_page_is_named_after_its_own_site(
    platform: str, kind: str | None
) -> None:
    named = page_link(platform, "1", kind=kind).platform.replace("_", "")
    assert named in {platform.replace("_", ""), f"{platform}{kind}".replace("_", "")}
