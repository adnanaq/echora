"""Tests for deciding when two provider URLs denote the same work.

Identity is what keeps a remake apart from its original and fuses a romaji
title with its English one, so a wrong answer here is invisible in the output:
two works merge into one, or one work ships twice.
"""

import json

from enrichment.pipeline.same_work import OfflineDatabaseResolver, canonical_url_key


def _url(anime_id: int) -> str:
    return f"https://myanimelist.net/anime/{anime_id}"


def test_anidb_legacy_url_resolves_to_the_same_work() -> None:
    # MAL still links AniDB works through the old perl-bin address. Left
    # unrecognised it resolves to nothing and reads as a separate work.
    modern = canonical_url_key("https://anidb.net/anime/69")
    assert (
        canonical_url_key("https://anidb.net/perl-bin/animedb.pl?show=anime&aid=69")
        == modern
    )
    assert (
        canonical_url_key("http://anidb.net/perl-bin/animedb.pl?aid=69&show=anime")
        == modern
    )


def test_anidb_legacy_non_anime_page_is_not_a_work() -> None:
    url = "https://anidb.net/perl-bin/animedb.pl?show=character&charid=474"
    assert canonical_url_key(url) == url


def test_the_offline_database_places_every_url_of_one_work_together() -> None:
    resolver = OfflineDatabaseResolver(
        [{"sources": [_url(21), "https://anidb.net/anime/69"]}]
    )
    assert resolver.resolve(_url(21)) == resolver.resolve("https://anidb.net/anime/69")


def test_the_offline_database_indexes_one_key_per_url() -> None:
    resolver = OfflineDatabaseResolver(
        [
            {"sources": [_url(21), "https://anidb.net/anime/69"]},
            {"sources": [_url(1)]},
            {"sources": []},
            {},
        ]
    )
    assert len(resolver) == 3


def test_an_unlisted_url_resolves_to_nothing() -> None:
    resolver = OfflineDatabaseResolver([{"sources": [_url(21)]}])
    assert resolver.resolve(_url(9999)) is None


def test_two_works_never_share_a_work_id() -> None:
    resolver = OfflineDatabaseResolver(
        [{"sources": [_url(21)]}, {"sources": [_url(1)]}]
    )
    assert resolver.resolve(_url(21)) != resolver.resolve(_url(1))


def test_the_work_id_does_not_depend_on_the_order_sources_are_listed_in() -> None:
    forward = OfflineDatabaseResolver(
        [{"sources": [_url(21), "https://anidb.net/anime/69"]}]
    )
    reversed_order = OfflineDatabaseResolver(
        [{"sources": ["https://anidb.net/anime/69", _url(21)]}]
    )
    assert forward.resolve(_url(21)) == reversed_order.resolve(_url(21))


def test_a_decorated_url_resolves_to_the_same_work_as_the_bare_one() -> None:
    resolver = OfflineDatabaseResolver([{"sources": [_url(21)]}])
    assert resolver.resolve("https://myanimelist.net/anime/21/One_Piece") is not None


def test_the_offline_database_loads_from_a_file(tmp_path) -> None:
    path = tmp_path / "anime-offline-database.json"
    path.write_text(json.dumps({"data": [{"sources": [_url(21)]}]}))
    resolver = OfflineDatabaseResolver.from_file(path)
    assert len(resolver) == 1
    assert resolver.resolve(_url(21)) is not None
