from seed_cleaning.doubled_provider_entries import find_doubled_provider_entries


def test_find_doubled_provider_entries_lists_entry_with_two_pages_of_one_provider(
    seed_works,
) -> None:
    assert [entry.title for entry in find_doubled_provider_entries(seed_works)] == [
        "Boku Dake no Rock Star"
    ]


def test_find_doubled_provider_entries_simkl_and_animecountdown_same_number_not_listed(
    seed_works,
) -> None:
    cue = next(work for work in seed_works if work.title == "Cue!")
    assert {"simkl:anime:1467793", "animecountdown:anime:1467793"} <= cue.pages.keys()
    assert find_doubled_provider_entries([cue]) == []
