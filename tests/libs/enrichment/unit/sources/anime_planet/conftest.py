"""Fixtures for Anime-Planet crawler unit tests.

HTML fixtures are real pages captured from live AP pages:
- ap_anime_html:          https://www.anime-planet.com/anime/one-piece (2026-06-10)
- ap_anime_extracted:     result of _extract_anime_from_html(ap_anime_html) + slug injected
- ap_character_html:      https://www.anime-planet.com/characters/monkey-d-luffy (2026-06-10)
- ap_character_extracted: result of _extract_character_from_html(ap_character_html)
- ap_character_raw:       https://www.anime-planet.com/characters/monkey-d-luffy (2026-04-17)
- ap_char_refs_html:      https://www.anime-planet.com/anime/dandadan/characters (2026-06-10)
"""

from pathlib import Path

import pytest

_FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def ap_anime_html() -> str:
    return (_FIXTURES / "ap_anime_one-piece.html").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def ap_anime_extracted(ap_anime_html: str) -> dict:
    from enrichment.sources.anime_planet.anime_planet_anime_crawler import (
        _extract_anime_from_html,
    )

    raw = _extract_anime_from_html(ap_anime_html)
    assert raw is not None, "HTML fixture failed to extract — fixture may be stale"
    raw["slug"] = "one-piece"
    return raw


@pytest.fixture(scope="session")
def ap_character_html() -> str:
    return (_FIXTURES / "ap_char_monkey-d-luffy.html").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def ap_character_extracted(ap_character_html: str) -> dict:
    from enrichment.sources.anime_planet.anime_planet_character_crawler import (
        _extract_character_from_html,
    )

    raw = _extract_character_from_html(ap_character_html)
    assert raw is not None, "character HTML fixture failed to extract — fixture may be stale"
    return raw


@pytest.fixture(scope="session")
def ap_char_refs_html() -> str:
    return (_FIXTURES / "ap_char_refs_dandadan.html").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def ap_character_raw() -> dict:
    import json

    return json.loads((_FIXTURES / "ap_character_raw.json").read_text())
