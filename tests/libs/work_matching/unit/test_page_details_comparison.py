import pytest
from work_matching.page_details_comparison import (
    get_seed_type,
    is_page_details_matching,
)
from work_matching.provider_page import ProviderPage


@pytest.mark.parametrize(
    ("page_type", "page_year", "expected"),
    [
        ("TV SHORT", 2020, True),
        ("TV", None, True),
        ("MOVIE", 2020, False),
        ("TV", 2019, False),
    ],
)
def test_is_page_details_matching_compares_year_and_seed_type(
    work_entry, page_type: str, page_year: int | None, expected: bool
) -> None:
    page = ProviderPage(
        key="anidb:anime:1",
        address="https://anidb.net/anime/1",
        type=page_type,
        year=page_year,
    )
    entry = work_entry("https://myanimelist.net/anime/1", work_type="TV", year=2020)
    assert is_page_details_matching(page, entry) is expected


@pytest.mark.parametrize(
    ("page_type", "seed_type"),
    [("TV SHORT", "TV"), ("MUSIC", "SPECIAL"), ("UNKNOWN", None), (None, None)],
)
def test_get_seed_type_provider_type_falls_under_seed_type(
    page_type: str | None, seed_type: str | None
) -> None:
    page = ProviderPage(key="mal:anime:1", address="https://x", type=page_type)
    assert get_seed_type(page) == seed_type
