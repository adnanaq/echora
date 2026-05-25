"""Fixtures for MAL crawler unit tests.

Fixtures are real XPath extraction output captured from live MAL pages:
- mal_anime_raw:                https://myanimelist.net/anime/21/One_Piece (2026-04-17)
- mal_character_raw:            https://myanimelist.net/character/40/Luffy_Monkey_D (2026-04-17)
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
def mal_anime_raw() -> dict:
    return json.loads((_FIXTURES / "mal_anime_raw.json").read_text())


@pytest.fixture(scope="session")
def mal_character_raw() -> dict:
    return json.loads((_FIXTURES / "mal_character_raw.json").read_text())


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
