"""Fixtures for AniDB source unit tests.

Fixtures are real captured data from AniDB:
- onepiece_anidb_raw.xml: https://anidb.net/anime/69 (One Piece, aid=69, captured 2026-06-05)
                          Note: <enddate>2030-06-01</enddate> injected — One Piece is ongoing.
"""

from pathlib import Path

import pytest
from enrichment.sources.anidb.anidb_models import AniDBAnime
from enrichment.sources.anidb.anidb_xml_parser import parse_anime_xml

_FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def onepiece_xml() -> str:
    """Raw XML string from the AniDB HTTP API for One Piece (aid=69)."""
    return (_FIXTURES / "onepiece_anidb_raw.xml").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def onepiece_anime(onepiece_xml: str) -> AniDBAnime:
    """Parsed AniDBAnime model for One Piece."""
    return parse_anime_xml(onepiece_xml)
