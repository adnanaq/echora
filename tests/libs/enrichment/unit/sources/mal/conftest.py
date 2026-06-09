"""Fixtures for MAL crawler unit tests.

Fixtures are real captured HTML from live MAL pages:
- mal_anime_html:               https://myanimelist.net/anime/21 (2026-06-09)
- mal_anime_pics_html:          https://myanimelist.net/anime/21/One_Piece/pics (2026-06-09)
- mal_anime_extracted:          XPath-extracted dict from mal_anime_html (with _url/_picture_urls)
- mal_character_html:           https://myanimelist.net/character/40 (2026-06-09, with scroll)
- mal_character_extracted:      XPath-extracted dict from mal_character_html
- mal_episode_raw:              https://myanimelist.net/anime/21/One_Piece/episode/1 (2026-04-17)
- mal_episode_filler_raw:       https://myanimelist.net/anime/21/One_Piece/episode/50 (2026-04-17)
- mal_episode_recap_raw:        https://myanimelist.net/anime/21/One_Piece/episode/279 (2026-04-17)
- mal_episode_no_synopsis_raw:  https://myanimelist.net/anime/21/One_Piece/episode/1152 (2026-04-17)
"""

import json
from pathlib import Path

import pytest

_FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def mal_anime_html() -> str:
    return (_FIXTURES / "mal_anime_21.html").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def mal_anime_pics_html() -> str:
    return (_FIXTURES / "mal_anime_21_pics.html").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def mal_anime_extracted(mal_anime_html, mal_anime_pics_html) -> dict:
    from enrichment.sources.mal.mal_anime_crawler import (
        _extract_anime_from_html,
        _extract_pics_from_html,
    )

    raw = _extract_anime_from_html(mal_anime_html)
    assert raw is not None, "HTML fixture produced no extraction — fixture may be stale"
    raw["_url"] = "https://myanimelist.net/anime/21/One_Piece"
    raw["_picture_urls"] = _extract_pics_from_html(mal_anime_pics_html)
    return raw


@pytest.fixture(scope="session")
def mal_character_html() -> str:
    return (_FIXTURES / "mal_character_40.html").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def mal_character_extracted(mal_character_html) -> dict:
    from enrichment.sources.mal.mal_character_crawler import _extract_character_from_html

    raw = _extract_character_from_html(mal_character_html)
    assert raw is not None, "HTML fixture produced no extraction — fixture may be stale"
    return raw


@pytest.fixture(scope="session")
def mal_episode_raw() -> dict:
    return json.loads((_FIXTURES / "mal_episode_raw.json").read_text())


@pytest.fixture(scope="session")
def mal_episode_filler_raw() -> dict:
    return json.loads((_FIXTURES / "mal_episode_filler_raw.json").read_text())


@pytest.fixture(scope="session")
def mal_episode_recap_raw() -> dict:
    return json.loads((_FIXTURES / "mal_episode_recap_raw.json").read_text())


@pytest.fixture(scope="session")
def mal_episode_no_synopsis_raw() -> dict:
    return json.loads((_FIXTURES / "mal_episode_no_synopsis_raw.json").read_text())
