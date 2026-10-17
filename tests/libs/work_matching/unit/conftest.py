import json
from collections.abc import Callable
from pathlib import Path

import pytest
from work_matching.provider_page import ProviderPage
from work_matching.provider_page_key import provider_page_key
from work_matching.work_entry import WorkEntry, WorkType

_FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def linked_page() -> Callable[..., ProviderPage]:
    def build_linked_page(
        address: str, *links: str, title: str = "Test"
    ) -> ProviderPage:
        return ProviderPage.from_canonical_record(
            address,
            {"title": title, "external_sources": [{"source": link} for link in links]},
        )

    return build_linked_page


@pytest.fixture(scope="session")
def work_entry() -> Callable[..., WorkEntry]:
    def build_work_entry(
        *addresses: str, work_type: WorkType = "TV", year: int | None = 2020
    ) -> WorkEntry:
        return WorkEntry(
            title="Test",
            type=work_type,
            year=year,
            pages={provider_page_key(address): address for address in addresses},
        )

    return build_work_entry


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


@pytest.fixture(scope="session")
def matching_cases() -> dict[str, tuple[list[WorkEntry], list[ProviderPage]]]:
    cases = json.loads((_FIXTURES / "matching_cases.json").read_text(encoding="utf-8"))
    return {
        name: (
            [WorkEntry.from_seed_entry(entry) for entry in case["entries"]],
            [ProviderPage.model_validate(page) for page in case["pages"]],
        )
        for name, case in cases.items()
    }
