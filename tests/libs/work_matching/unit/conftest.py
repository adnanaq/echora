import json
from pathlib import Path

import pytest

_FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def seed_entries() -> list[dict]:
    return json.loads((_FIXTURES / "seed_entries.json").read_text(encoding="utf-8"))[
        "data"
    ]


@pytest.fixture(scope="session")
def seed_file() -> Path:
    return _FIXTURES / "seed_entries.json"


@pytest.fixture(scope="session")
def canonical_records() -> dict[str, dict]:
    return json.loads(
        (_FIXTURES / "canonical_records.json").read_text(encoding="utf-8")
    )
