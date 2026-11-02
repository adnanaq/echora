import json
from collections.abc import Callable
from pathlib import Path

import pytest
from work_matching.doubled_provider_check import decide_doubled_providers
from work_matching.entry_group import combine_group_entries
from work_matching.known_work_entries_from_seed import KnownWorkEntriesFromSeed
from work_matching.linked_page_check import LinkingEntryDetails, decide_linked_page
from work_matching.provider_page import ProviderPage
from work_matching.provider_page_key import get_page_key
from work_matching.review_cases import ReviewCase
from work_matching.split_work_check import list_split_work_review_reasons
from work_matching.work_entry import WorkEntry, WorkType

_FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def linked_page() -> Callable[..., ProviderPage]:
    def build_linked_page(
        address: str, *linked_addresses: str, title: str = "Test"
    ) -> ProviderPage:
        return ProviderPage.from_canonical_record(
            address,
            {
                "title": title,
                "external_sources": [
                    {"source": linked_address} for linked_address in linked_addresses
                ],
            },
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
            addresses_by_page_key={
                get_page_key(address): address for address in addresses
            },
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


@pytest.fixture(scope="session")
def review_case_lines() -> list[str]:
    return (
        (_FIXTURES / "review_cases.jsonl")
        .read_text(encoding="utf-8")
        .splitlines(keepends=True)
    )


@pytest.fixture(scope="session")
def review_cases(review_case_lines) -> dict[str, ReviewCase]:
    cases = [ReviewCase.model_validate_json(line) for line in review_case_lines]
    page_key_by_name = {
        "atashinchi_next": "mal:anime:63013",
        "gochuumon_movie": "mal:anime:61115",
        "nami": "kitsu:anime:10561",
        "aware_meisaku_kun": "mal:anime:32956",
    }
    return {
        name: next(case for case in cases if page_key in case.case_name)
        for name, page_key in page_key_by_name.items()
    }


@pytest.fixture(scope="session")
def build_group_case() -> Callable[[list[WorkEntry], list[ProviderPage]], ReviewCase]:
    def build(entries: list[WorkEntry], pages: list[ProviderPage]) -> ReviewCase:
        decisions = decide_doubled_providers(
            combine_group_entries(entries), pages, KnownWorkEntriesFromSeed([])
        )
        return ReviewCase.from_entry_group(
            entries, decisions, list_split_work_review_reasons(entries, pages), pages
        )

    return build


@pytest.fixture
def doubled_case(matching_cases) -> ReviewCase:
    entries, pages = matching_cases["genshin_sigewinne"]
    decisions = decide_doubled_providers(
        entries[0], pages, KnownWorkEntriesFromSeed([])
    )
    return ReviewCase.from_entry_group([entries[0]], decisions, (), pages)


@pytest.fixture
def split_case(linked_page, work_entry, build_group_case) -> ReviewCase:
    entries = [
        work_entry("https://myanimelist.net/anime/1"),
        work_entry("https://anilist.co/anime/1"),
    ]
    return build_group_case(entries, [linked_page("https://myanimelist.net/anime/1")])


@pytest.fixture
def linked_page_case() -> ReviewCase:
    entry = WorkEntry.from_seed_entry(
        {
            "title": "The Boxer",
            "type": "TV",
            "sources": ["https://anilist.co/anime/163794"],
        }
    )
    entry_page = ProviderPage(
        key="anilist:anime:163794",
        address="https://anilist.co/anime/163794",
        titles=("The Boxer",),
        type="TV",
        episode_count=12,
        linked_page_keys=("mal:anime:65144",),
    )
    linked_page = ProviderPage(
        key="mal:anime:65144",
        address="https://myanimelist.net/anime/65144",
        titles=("The Boxer",),
        type="TV",
        episode_count=1,
    )
    decision = decide_linked_page(
        linked_page,
        [
            LinkingEntryDetails(
                entry, entry_page.episode_count, frozenset({linked_page.key})
            )
        ],
        KnownWorkEntriesFromSeed([]),
    )
    return ReviewCase.from_linked_page(decision, [linked_page, entry_page])
