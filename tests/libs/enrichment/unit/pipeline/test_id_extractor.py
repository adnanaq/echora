"""Unit tests for PlatformIDExtractor."""

import pytest

from enrichment.pipeline.id_extractor import PlatformIDExtractor  # direct import avoids deep pipeline chain


@pytest.fixture
def extractor() -> PlatformIDExtractor:
    return PlatformIDExtractor()


# ---------------------------------------------------------------------------
# extract_all_ids — URL-based keys
# ---------------------------------------------------------------------------


def test_mal_url_extracted(extractor: PlatformIDExtractor) -> None:
    data = {"sources": ["https://myanimelist.net/anime/21"]}
    assert extractor.extract_all_ids(data)["mal_url"] == "https://myanimelist.net/anime/21"


def test_anilist_url_extracted(extractor: PlatformIDExtractor) -> None:
    data = {"sources": ["https://anilist.co/anime/21"]}
    assert extractor.extract_all_ids(data)["anilist_url"] == "https://anilist.co/anime/21"


def test_anime_planet_url_extracted(extractor: PlatformIDExtractor) -> None:
    data = {"sources": ["https://www.anime-planet.com/anime/one-piece"]}
    assert extractor.extract_all_ids(data)["anime_planet_url"] == "https://www.anime-planet.com/anime/one-piece"


# ---------------------------------------------------------------------------
# anisearch_url — full URL, not a numeric ID
# ---------------------------------------------------------------------------


def test_anisearch_url_not_set_from_anilist(extractor: PlatformIDExtractor) -> None:
    data = {"sources": ["https://anilist.co/anime/21"]}
    result = extractor.extract_all_ids(data)
    assert result["anisearch_url"] is None


def test_anisearch_url_extracted(extractor: PlatformIDExtractor) -> None:
    data = {"sources": ["https://www.anisearch.com/anime/458,one-piece"]}
    assert extractor.extract_all_ids(data)["anisearch_url"] == "https://www.anisearch.com/anime/458,one-piece"


def test_anisearch_url_and_anilist_url_both_present(extractor: PlatformIDExtractor) -> None:
    data = {
        "sources": [
            "https://anilist.co/anime/21",
            "https://www.anisearch.com/anime/458,one-piece",
        ]
    }
    result = extractor.extract_all_ids(data)
    assert result["anilist_url"] == "https://anilist.co/anime/21"
    assert result["anisearch_url"] == "https://www.anisearch.com/anime/458,one-piece"


# ---------------------------------------------------------------------------
# extract_all_ids — regex-based IDs
# ---------------------------------------------------------------------------


def test_kitsu_id_extracted(extractor: PlatformIDExtractor) -> None:
    data = {"sources": ["https://kitsu.app/anime/one-piece"]}
    assert extractor.extract_all_ids(data)["kitsu_id"] == "one-piece"


def test_anidb_id_extracted(extractor: PlatformIDExtractor) -> None:
    data = {"sources": ["https://anidb.net/anime/69"]}
    assert extractor.extract_all_ids(data)["anidb_id"] == "69"


def test_livechart_id_extracted(extractor: PlatformIDExtractor) -> None:
    data = {"sources": ["https://www.livechart.me/anime/10959"]}
    assert extractor.extract_all_ids(data)["livechart_id"] == "10959"


def test_notify_id_extracted(extractor: PlatformIDExtractor) -> None:
    data = {"sources": ["https://notify.moe/anime/0-A-5Fimg"]}
    assert extractor.extract_all_ids(data)["notify_id"] == "0-A-5Fimg"


def test_empty_sources_returns_all_none(extractor: PlatformIDExtractor) -> None:
    result = extractor.extract_all_ids({"sources": []})
    assert all(v is None for v in result.values())


def test_missing_sources_key_returns_all_none(extractor: PlatformIDExtractor) -> None:
    result = extractor.extract_all_ids({})
    assert all(v is None for v in result.values())


def test_first_match_wins_for_url_keys(extractor: PlatformIDExtractor) -> None:
    """When multiple MAL URLs appear only the first is kept."""
    data = {
        "sources": [
            "https://myanimelist.net/anime/21",
            "https://myanimelist.net/anime/999",
        ]
    }
    assert extractor.extract_all_ids(data)["mal_url"] == "https://myanimelist.net/anime/21"


def test_full_mixed_sources(extractor: PlatformIDExtractor) -> None:
    data = {
        "sources": [
            "https://myanimelist.net/anime/21",
            "https://anilist.co/anime/21",
            "https://www.anime-planet.com/anime/one-piece",
            "https://www.anisearch.com/anime/458,one-piece",
            "https://anidb.net/anime/69",
            "https://kitsu.app/anime/one-piece",
            "https://notify.moe/anime/0-A-5Fimg",
            "https://www.livechart.me/anime/10959",
        ]
    }
    result = extractor.extract_all_ids(data)
    assert result["mal_url"] == "https://myanimelist.net/anime/21"
    assert result["anilist_url"] == "https://anilist.co/anime/21"
    assert result["anime_planet_url"] == "https://www.anime-planet.com/anime/one-piece"
    assert result["anisearch_url"] == "https://www.anisearch.com/anime/458,one-piece"
    assert result["anidb_id"] == "69"
    assert result["kitsu_id"] == "one-piece"
    assert result["notify_id"] == "0-A-5Fimg"
    assert result["livechart_id"] == "10959"


# ---------------------------------------------------------------------------
# validate_ids
# ---------------------------------------------------------------------------


def test_validate_ids_keeps_numeric_anidb(extractor: PlatformIDExtractor) -> None:
    assert extractor.validate_ids({"anidb_id": "69"}) == {"anidb_id": "69"}


def test_validate_ids_rejects_non_numeric_anidb(extractor: PlatformIDExtractor) -> None:
    assert extractor.validate_ids({"anidb_id": "not-a-number"}) == {}


def test_validate_ids_drops_none_values(extractor: PlatformIDExtractor) -> None:
    assert extractor.validate_ids({"mal_url": None, "anilist_url": "https://anilist.co/anime/21"}) == {
        "anilist_url": "https://anilist.co/anime/21"
    }


def test_validate_ids_drops_empty_string(extractor: PlatformIDExtractor) -> None:
    assert extractor.validate_ids({"kitsu_id": "   "}) == {}
