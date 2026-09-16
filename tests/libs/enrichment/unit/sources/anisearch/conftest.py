"""Fixtures for AniSearch crawler unit tests.

Fixtures are real XPath extraction output captured from:
- https://www.anisearch.com/anime/2227,one-piece (2026-04-17, HTML 2026-06-09)
- https://www.anisearch.com/anime/2227,one-piece/relations?show=overall (2026-06-09)
- https://www.anisearch.com/character/4852,monkey-d-luffy (2026-04-23, HTML 2026-06-09)
- https://www.anisearch.com/character/4852,monkey-d-luffy/anime (2026-06-09)
- https://www.anisearch.com/character/4852,monkey-d-luffy/manga (2026-06-09)
- https://www.anisearch.com/anime/2227,one-piece/characters (2026-04-23)
- https://www.anisearch.com/anime/2227,one-piece/episodes (2026-04-28)
"""

import json
from pathlib import Path

import pytest

_FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def one_piece_main_raw() -> dict:
    return json.loads((_FIXTURES / "one_piece_main_raw.json").read_text())


@pytest.fixture(scope="session")
def one_piece_relations_raw() -> dict:
    return json.loads((_FIXTURES / "one_piece_relations_raw.json").read_text())


@pytest.fixture(scope="session")
def luffy_char_raw() -> dict:
    """Raw XPath extraction output for Monkey D. Luffy character page.

    favorites is a string ("678") and anime_roles URLs are relative — exactly
    as they appear before _post_process_character runs.
    """
    return json.loads((_FIXTURES / "luffy_char_raw.json").read_text())


@pytest.fixture(scope="session")
def luffy_char_html() -> str:
    """Full page HTML for https://www.anisearch.com/character/4852,monkey-d-luffy."""
    return (_FIXTURES / "luffy_char.html").read_text()


@pytest.fixture(scope="session")
def luffy_anime_ography_html() -> str:
    """Full page HTML for https://www.anisearch.com/character/4852,monkey-d-luffy/anime.

    Contains 49 ography entries inside a ul.covers list.
    """
    return (_FIXTURES / "luffy_anime_ography.html").read_text()


@pytest.fixture(scope="session")
def luffy_manga_ography_html() -> str:
    """Full page HTML for https://www.anisearch.com/character/4852,monkey-d-luffy/manga.

    Contains 10 ography entries inside a ul.covers list.
    """
    return (_FIXTURES / "luffy_manga_ography.html").read_text()


@pytest.fixture(scope="session")
def one_piece_main_html() -> str:
    """Full page HTML for https://www.anisearch.com/anime/2227,one-piece."""
    return (_FIXTURES / "one_piece_main.html").read_text()


@pytest.fixture(scope="session")
def one_piece_relations_html() -> str:
    """Full page HTML for https://www.anisearch.com/anime/2227,one-piece/relations?show=overall."""
    return (_FIXTURES / "one_piece_relations.html").read_text()


@pytest.fixture(scope="session")
def one_piece_refs_raw() -> dict:
    """Raw XPath extraction output for One Piece characters page.

    Keys are section IDs (chara1…chara50); values are lists of {url} dicts
    with relative hrefs — exactly as they appear before processing.
    """
    return json.loads((_FIXTURES / "one_piece_refs_raw.json").read_text())


@pytest.fixture(scope="session")
def one_piece_episodes_raw() -> dict:
    """Raw extraction output for One Piece /episodes page.

    7 representative rows: ep 1 (4Kids dub prefix), ep 2, ep 279 (filler+recap),
    ep 457 (recap only), ep 50 (filler only), ep 1144 (partial), ep 1200 (future).
    Format: {"episodes": [...rows...]} — matches _extract_episodes_from_html output.
    """
    return json.loads((_FIXTURES / "one_piece_episodes_raw.json").read_text())


@pytest.fixture(scope="session")
def one_piece_episodes_html() -> str:
    """Full page HTML for https://www.anisearch.com/anime/2227,one-piece/episodes."""
    return (_FIXTURES / "one_piece_episodes.html").read_text()
