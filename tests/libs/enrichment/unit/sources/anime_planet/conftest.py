"""Fixtures for Anime-Planet crawler unit tests.

Fixtures are real output captured from live AP pages:
- ap_anime_raw:      https://www.anime-planet.com/anime/one-piece (2026-04-17)
  (output of _fetch_animeplanet_anime_data — merged JSON-LD + XPath primitives)
- ap_character_raw:  https://www.anime-planet.com/characters/monkey-d-luffy (2026-04-17)
  (output of _fetch_character_data — XPath fields + _html)
"""

import json
from pathlib import Path

import pytest

_FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def ap_anime_raw() -> dict:
    return json.loads((_FIXTURES / "ap_anime_raw.json").read_text())


@pytest.fixture(scope="session")
def ap_character_raw() -> dict:
    return json.loads((_FIXTURES / "ap_character_raw.json").read_text())
