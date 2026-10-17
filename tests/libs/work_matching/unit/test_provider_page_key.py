import pytest
from work_matching.provider_page_key import (
    REFERENCE_ORDER,
    has_repeated_provider,
    page_id_of_key,
    provider_of_key,
    provider_page_key,
)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (
            "https://myanimelist.net/anime/21/One_Piece",
            "https://myanimelist.net/anime/21",
        ),
        (
            "https://www.anisearch.com/anime/2227,one-piece",
            "https://anisearch.com/anime/2227",
        ),
        (
            "https://anidb.net/perl-bin/animedb.pl?show=anime&aid=69",
            "https://anidb.net/anime/69",
        ),
        (
            "http://anidb.net/perl-bin/animedb.pl?aid=69&show=anime",
            "https://anidb.net/anime/69",
        ),
        ("https://WWW.MyAnimeList.net/anime/21", "https://myanimelist.net/anime/21"),
        ("https://kitsu.io/anime/12", "https://kitsu.app/anime/12"),
        (
            "https://www.anime-planet.com/anime/one-piece",
            "https://anime-planet.com/anime/one-piece",
        ),
        (
            "http://www.animenewsnetwork.com/encyclopedia/anime.php?page=2&id=836",
            "https://animenewsnetwork.com/encyclopedia/anime.php?id=836",
        ),
        ("https://www.animecountdown.com/38636", "https://animecountdown.com/38636"),
    ],
)
def test_provider_page_key_two_spellings_give_one_key(first: str, second: str) -> None:
    assert provider_page_key(first) == provider_page_key(second)


def test_provider_page_key_recognised_page_gives_provider_kind_and_id() -> None:
    assert (
        provider_page_key("https://myanimelist.net/anime/21/One_Piece")
        == "mal:anime:21"
    )


def test_provider_page_key_kitsu_slug_and_number_stay_apart() -> None:
    assert provider_page_key("https://kitsu.io/anime/one-piece") != provider_page_key(
        "https://kitsu.app/anime/12"
    )


def test_provider_page_key_id_prefix_of_another_id_stays_apart() -> None:
    assert provider_page_key("https://myanimelist.net/anime/466") != provider_page_key(
        "https://myanimelist.net/anime/46633"
    )


def test_provider_page_key_animecountdown_page_gives_anime_kind() -> None:
    assert (
        provider_page_key("https://animecountdown.com/38636")
        == "animecountdown:anime:38636"
    )


def test_provider_page_key_id_followed_by_letters_keeps_address() -> None:
    address = "https://myanimelist.net/anime/21extra"
    assert provider_page_key(address) == address


def test_provider_page_key_unrecognised_address_drops_www_host_case_and_trailing_slash() -> (
    None
):
    assert (
        provider_page_key("HTTPS://WWW.Example.com/Show/") == "https://example.com/Show"
    )


def test_provider_page_key_address_without_scheme_keeps_address() -> None:
    assert provider_page_key("myanimelist.net/anime/21/") == "myanimelist.net/anime/21"


def test_provider_page_key_ann_manga_page_gives_manga_key() -> None:
    assert (
        provider_page_key(
            "https://www.animenewsnetwork.com/encyclopedia/manga.php?id=1"
        )
        == "ann:manga:1"
    )


def test_provider_page_key_ann_page_without_id_keeps_address() -> None:
    assert (
        provider_page_key(
            "https://www.animenewsnetwork.com/encyclopedia/anime.php?list=A"
        )
        == "https://animenewsnetwork.com/encyclopedia/anime.php?list=A"
    )


def test_provider_page_key_anidb_legacy_non_anime_page_keeps_address() -> None:
    address = "https://anidb.net/perl-bin/animedb.pl?show=character&aid=69"
    assert provider_page_key(address) == address


def test_provider_of_key_recognised_key_gives_provider() -> None:
    assert provider_of_key(provider_page_key("https://anidb.net/anime/69")) == "anidb"


def test_provider_of_key_unrecognised_key_gives_none() -> None:
    assert provider_of_key(provider_page_key("https://example.com/show")) is None


def test_reference_order_starts_with_providers_that_name_merged_ids() -> None:
    assert REFERENCE_ORDER[:6] == (
        "mal",
        "anilist",
        "anidb",
        "kitsu",
        "animeplanet",
        "anisearch",
    )


@pytest.mark.parametrize(
    ("keys", "expected"),
    [
        (["mal:anime:1", "kitsu:anime:1", "mal:anime:2"], True),
        (["mal:anime:1", "kitsu:anime:1"], False),
        (["https://example.com/first", "https://example.com/second"], False),
    ],
)
def test_has_repeated_provider_counts_pages_of_known_providers(
    keys: list[str], expected: bool
) -> None:
    assert has_repeated_provider(keys) is expected


def test_page_id_of_key_recognised_key_gives_id() -> None:
    assert page_id_of_key("animecountdown:anime:38636") == "38636"


def test_page_id_of_key_unrecognised_key_gives_key() -> None:
    assert page_id_of_key("https://example.com/show") == "https://example.com/show"
