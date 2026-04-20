"""Fixtures for AniSearch crawler unit tests.

Fixtures are real XPath extraction output captured from
https://www.anisearch.com/anime/2227,one-piece (2026-04-17).
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
