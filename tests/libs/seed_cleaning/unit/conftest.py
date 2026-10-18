from pathlib import Path

import pytest
from work_matching import KnownWorksFromSeed

_FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def seed_works() -> KnownWorksFromSeed:
    return KnownWorksFromSeed.from_file(_FIXTURES / "seed_entries.json")
