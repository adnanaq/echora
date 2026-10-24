import pytest
from work_matching.doubled_provider_pages import (
    find_doubled_provider_page_keys,
    get_decided_provider,
    has_several_page_ids,
)


@pytest.mark.parametrize(
    ("key", "provider"),
    [
        ("animecountdown:anime:5", "simkl"),
        ("simkl:anime:5", "simkl"),
        ("mal:anime:1", "mal"),
        ("https://example.com/anime/1", None),
    ],
)
def test_get_decided_provider_animecountdown_counts_as_simkl(
    key: str, provider: str | None
) -> None:
    assert get_decided_provider(key) == provider


def test_find_doubled_provider_page_keys_provider_with_two_ids_listed() -> None:
    assert find_doubled_provider_page_keys(
        ["mal:anime:1", "kitsu:anime:1", "mal:anime:2"]
    ) == {"mal": ("mal:anime:1", "mal:anime:2")}


def test_find_doubled_provider_page_keys_simkl_and_animecountdown_same_number_not_doubled() -> (
    None
):
    assert (
        find_doubled_provider_page_keys(["simkl:anime:5", "animecountdown:anime:5"])
        == {}
    )


def test_find_doubled_provider_page_keys_simkl_and_animecountdown_other_numbers_doubled() -> (
    None
):
    assert find_doubled_provider_page_keys(
        ["simkl:anime:5", "animecountdown:anime:6"]
    ) == {"simkl": ("simkl:anime:5", "animecountdown:anime:6")}


@pytest.mark.parametrize(
    ("page_keys", "expected"),
    [
        (["mal:anime:1", "mal:anime:2"], True),
        (["simkl:anime:5", "animecountdown:anime:5"], False),
        (["mal:anime:1"], False),
    ],
)
def test_has_several_page_ids_counts_distinct_ids(
    page_keys: list[str], expected: bool
) -> None:
    assert has_several_page_ids(page_keys) is expected
