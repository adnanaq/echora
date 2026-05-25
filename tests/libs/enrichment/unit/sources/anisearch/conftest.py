"""Fixtures for AniSearch crawler unit tests.

Fixtures are real XPath extraction output captured from:
- https://www.anisearch.com/anime/2227,one-piece (2026-04-17)
- https://www.anisearch.com/character/4852,monkey-d-luffy (2026-04-23)
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
    """Raw crawl4ai extracted_content for Monkey D. Luffy character page.

    favorites is a string ("678") and anime_roles URLs are relative — exactly
    as crawl4ai returns them before _post_process_character runs.
    """
    return json.loads((_FIXTURES / "luffy_char_raw.json").read_text())


@pytest.fixture(scope="session")
def one_piece_refs_raw() -> dict:
    """Raw crawl4ai extracted_content for One Piece characters page.

    Keys are section IDs (chara1…chara50); values are lists of {url} dicts
    with relative hrefs — exactly as crawl4ai returns them.
    """
    return json.loads((_FIXTURES / "one_piece_refs_raw.json").read_text())


@pytest.fixture(scope="session")
def one_piece_episodes_raw() -> list:
    """Raw crawl4ai extracted_content for One Piece /episodes page.

    5 representative rows: ep 1 (4Kids dub prefix), ep 2, ep 50 (filler),
    ep 1144 (partial — no runtime/date/title_ja), ep 1200 (future — number only).
    Exactly as crawl4ai returns them before _parse_episode_row runs.
    """
    return json.loads((_FIXTURES / "one_piece_episodes_raw.json").read_text())
