from pathlib import Path

import pytest
from work_matching import KnownWorkEntriesFromSeed

_FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def seed_works() -> KnownWorkEntriesFromSeed:
    return KnownWorkEntriesFromSeed.from_file(_FIXTURES / "seed_entries.json")
