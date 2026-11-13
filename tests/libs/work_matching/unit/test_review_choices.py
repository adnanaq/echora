import dataclasses

import pytest
from work_matching.doubled_provider_check import decide_doubled_providers
from work_matching.entry_group import combine_group_entries
from work_matching.known_work_entries_from_seed import KnownWorkEntriesFromSeed
from work_matching.models import LinkedPageAnswerRule, PlacementRule, WorkPartialSources
from work_matching.review_cases import ReviewCase, TargetEntrySuggestion
from work_matching.review_choices import (
    choose_group_page_place,
    choose_partial_sources,
    find_linked_page_conflicts,
    list_group_page_choices,
    list_linked_page_options,
    list_required_page_keys,
    list_work_partial_sources,
)

MAL = "https://myanimelist.net/anime/1"
OTHER_MAL = "https://myanimelist.net/anime/2"
GOCHUUMON_DOUBLED_PAGES = ("anidb:anime:19188", "anidb:anime:19792")


def _index_choices_by_page_key(case, answer):
    return {
        page_choices.page_key: page_choices
        for page_choices in list_group_page_choices(case, answer)
    }


def _read_disabling_rules(options):
    return {option.value: option.disabling_rule for option in options}


@pytest.fixture
def two_mal_group(work_entry) -> ReviewCase:
    entries = [work_entry(MAL), work_entry(OTHER_MAL)]
    decisions = decide_doubled_providers(
        combine_group_entries(entries), [], KnownWorkEntriesFromSeed([])
    )
    return ReviewCase.build_entry_group_case(entries, decisions, [], [])


def test_list_group_page_choices_no_place_yet_offers_first_work_and_other_work(
    review_cases,
) -> None:
    choices = _index_choices_by_page_key(review_cases["atashinchi_next"], {})
    assert [option.value for option in choices["mal:anime:63013"].work_options] == [
        "work 1",
        "other work",
        "dropped",
    ]


def test_list_group_page_choices_offers_one_work_past_highest_in_use(
    review_cases,
) -> None:
    choices = _index_choices_by_page_key(
        review_cases["atashinchi_next"], {"anidb:anime:19895": "work 2"}
    )
    assert [option.value for option in choices["mal:anime:63013"].work_options] == [
        "work 1",
        "work 2",
        "work 3",
        "other work",
        "dropped",
    ]


def test_list_group_page_choices_never_offers_more_works_than_pages(
    review_cases,
) -> None:
    choices = _index_choices_by_page_key(
        review_cases["atashinchi_next"],
        {"anidb:anime:19895": "work 3", "mal:anime:63013": "work 2"},
    )
    assert [option.value for option in choices["mal:anime:63013"].work_options] == [
        "work 1",
        "work 2",
        "work 3",
        "other work",
        "dropped",
    ]


def test_list_group_page_choices_split_work_requires_every_page(review_cases) -> None:
    choices = list_group_page_choices(review_cases["atashinchi_next"], {})
    assert all(page_choices.is_required for page_choices in choices)


def test_list_group_page_choices_starts_on_given_answer(review_cases) -> None:
    choices = _index_choices_by_page_key(
        review_cases["gochuumon_movie"],
        {
            "anidb:anime:19188": "work 2",
            "anidb:anime:19792": "work 1, partial in work 2",
        },
    )
    assert (
        choices["anidb:anime:19188"].work,
        choices["anidb:anime:19792"].work,
        choices["anidb:anime:19792"].partial_works,
    ) == ("work 2", "work 1", ("work 2",))


def test_list_group_page_choices_other_work_named_entry_keeps_named_key(
    review_cases,
) -> None:
    choices = _index_choices_by_page_key(
        review_cases["gochuumon_movie"],
        {"anidb:anime:19188": "other work mal:anime:21273"},
    )
    assert (
        choices["anidb:anime:19188"].work,
        choices["anidb:anime:19188"].named_entry_page_key,
    ) == ("other work", "mal:anime:21273")


def test_list_group_page_choices_second_page_of_provider_in_work_disabled(
    two_mal_group,
) -> None:
    choices = _index_choices_by_page_key(two_mal_group, {"mal:anime:1": "work 1"})
    assert (
        _read_disabling_rules(choices["mal:anime:2"].work_options)["work 1"]
        is PlacementRule.WORK_WITH_TWO_PAGES_OF_ONE_PROVIDER
    )


def test_list_group_page_choices_page_keeps_its_own_work_enabled(
    two_mal_group,
) -> None:
    choices = _index_choices_by_page_key(two_mal_group, {"mal:anime:1": "work 1"})
    assert _read_disabling_rules(choices["mal:anime:1"].work_options)["work 1"] is None


@pytest.mark.parametrize(
    ("answer", "disabling_rule"),
    [
        ({}, PlacementRule.DROPPED_PAGE_WITHOUT_KEPT_PAGE),
        (
            {"anidb:anime:19792": "other work"},
            PlacementRule.DROPPED_PAGE_WITHOUT_KEPT_PAGE,
        ),
        ({"anidb:anime:19792": "work 1"}, None),
    ],
)
def test_list_group_page_choices_dropped_needs_kept_page_of_provider(
    review_cases, answer, disabling_rule
) -> None:
    choices = _index_choices_by_page_key(review_cases["gochuumon_movie"], answer)
    assert (
        _read_disabling_rules(choices["anidb:anime:19188"].work_options)["dropped"]
        is disabling_rule
    )


def test_list_required_page_keys_doubled_provider_first_work_needs_doubled_pages(
    review_cases,
) -> None:
    assert list_required_page_keys(
        review_cases["gochuumon_movie"], {"anidb:anime:19188": "work 1"}
    ) == frozenset(GOCHUUMON_DOUBLED_PAGES)


def test_list_required_page_keys_doubled_provider_later_work_needs_every_page(
    review_cases,
) -> None:
    case = review_cases["gochuumon_movie"]
    required_page_keys = list_required_page_keys(case, {"anidb:anime:19188": "work 2"})
    assert len(required_page_keys) == len(list_group_page_choices(case, {})) > 2


@pytest.mark.parametrize(
    ("work", "placement_text"),
    [
        ("work 3", "work 3, partial in work 2"),
        ("work 2", "work 2"),
        ("other work", "other work"),
        ("dropped", "dropped"),
    ],
)
def test_choose_group_page_place_keeps_partial_works_except_own_work(
    work, placement_text
) -> None:
    answer = choose_group_page_place(
        {"anidb:anime:19188": "work 1, partial in work 2", "mal:anime:1": "work 2"},
        "anidb:anime:19188",
        work,
    )
    assert answer["anidb:anime:19188"] == placement_text


def test_choose_group_page_place_emptied_work_leaves_partial_works() -> None:
    answer = choose_group_page_place(
        {
            "anidb:anime:19188": "work 1, partial in work 2, other work mal:anime:9",
            "mal:anime:1": "work 2",
        },
        "mal:anime:1",
        "other work",
    )
    assert answer == {
        "anidb:anime:19188": "work 1, partial in other work mal:anime:9",
        "mal:anime:1": "other work",
    }


def test_choose_group_page_place_no_work_removes_page(review_cases) -> None:
    answer = choose_group_page_place(
        {"anidb:anime:19188": "work 1"},
        "anidb:anime:19188",
        None,
    )
    assert answer == {}


def test_list_linked_page_options_one_linking_entry_offers_add_without_entry(
    review_cases,
) -> None:
    options = list_linked_page_options(review_cases["nami"])
    assert _read_disabling_rules(options) == {
        "add": None,
        "add as partial": None,
        "skip": None,
    }


def test_list_linked_page_options_entry_holding_provider_disables_whole_add(
    review_cases, work_entry
) -> None:
    case = review_cases["nami"]
    [decision] = case.decisions
    entry_with_anilist = decision.linking_entries[0].model_copy(
        update={
            "addresses_by_page_key": {
                **decision.linking_entries[0].addresses_by_page_key,
                "anilist:anime:1": "https://anilist.co/anime/1",
            }
        }
    )
    case = case.model_copy(
        update={
            "decisions": (
                dataclasses.replace(decision, linking_entries=(entry_with_anilist,)),
            )
        }
    )
    assert (
        _read_disabling_rules(list_linked_page_options(case))["add"]
        is LinkedPageAnswerRule.PROVIDER_ALREADY_IN_ENTRY
    )


def test_list_linked_page_options_several_entries_name_each_entry(
    review_cases, work_entry
) -> None:
    case = review_cases["nami"]
    [decision] = case.decisions
    other_entry = work_entry("https://kitsu.app/anime/99")
    case = case.model_copy(
        update={
            "decisions": (
                dataclasses.replace(
                    decision,
                    linking_entries=(*decision.linking_entries, other_entry),
                ),
            )
        }
    )
    assert [option.value for option in list_linked_page_options(case)] == [
        "add to kitsu:anime:10561",
        "add as partial to kitsu:anime:10561",
        "add to kitsu:anime:99",
        "add as partial to kitsu:anime:99",
        "skip",
    ]


def test_find_linked_page_conflicts_two_whole_pages_of_provider_to_one_entry(
    review_cases,
) -> None:
    case = review_cases["nami"]
    [decision] = case.decisions
    second_case = case.model_copy(
        update={
            "case_name": "linked page: anilist:anime:5 from kitsu:anime:10561",
            "decisions": (dataclasses.replace(decision, page_key="anilist:anime:5"),),
            "answer": {"anilist:anime:5": "add"},
        }
    )
    first_case = case.model_copy(update={"answer": {"anilist:anime:102289": "add"}})
    [conflict] = find_linked_page_conflicts([first_case, second_case])
    assert (conflict.entry_reference_page, conflict.provider, conflict.case_names) == (
        "kitsu:anime:10561",
        "anilist",
        (first_case.case_name, second_case.case_name),
    )


def test_find_linked_page_conflicts_partial_sources_never_conflict(
    review_cases,
) -> None:
    case = review_cases["nami"]
    [decision] = case.decisions
    second_case = case.model_copy(
        update={
            "case_name": "linked page: anilist:anime:5 from kitsu:anime:10561",
            "decisions": (dataclasses.replace(decision, page_key="anilist:anime:5"),),
            "answer": {"anilist:anime:5": "add as partial"},
        }
    )
    first_case = case.model_copy(
        update={"answer": {"anilist:anime:102289": "add as partial"}}
    )
    assert find_linked_page_conflicts([first_case, second_case]) == ()


def test_list_work_partial_sources_lists_partial_sources_and_options(
    review_cases,
) -> None:
    case = review_cases["gochuumon_movie"].model_copy(
        update={"target_entry_suggestions": ()}
    )
    answer = {
        "anidb:anime:19188": "work 1, partial in work 2",
        "anidb:anime:19792": "work 2",
        "mal:anime:61115": "work 1",
    }
    assert list_work_partial_sources(case, answer) == (
        WorkPartialSources("work 1", (), ("anidb:anime:19792",)),
        WorkPartialSources("work 2", ("anidb:anime:19188",), ("mal:anime:61115",)),
    )


@pytest.mark.parametrize(
    ("page_key", "work", "is_partial_source", "placement_text"),
    [
        ("mal:anime:1", "work 2", True, "work 1, partial in work 3, work 2"),
        ("mal:anime:1", "work 3", False, "work 1"),
        (
            "mal:anime:1",
            "other work mal:anime:9",
            True,
            "work 1, partial in work 3, other work mal:anime:9",
        ),
        ("mal:anime:1", "work 1", True, "work 1, partial in work 3"),
        ("kitsu:anime:1", "work 2", True, "other work"),
    ],
)
def test_choose_partial_sources_adds_or_removes_partial_work(
    page_key, work, is_partial_source, placement_text
) -> None:
    answer = choose_partial_sources(
        {
            "mal:anime:1": "work 1, partial in work 3",
            "kitsu:anime:1": "other work",
            "anilist:anime:1": "work 2",
            "anidb:anime:1": "work 3",
        },
        [page_key],
        work,
        is_partial_source=is_partial_source,
    )
    assert answer[page_key] == placement_text


def test_list_work_partial_sources_lone_work_hidden_and_covered_entry_listed(
    review_cases,
) -> None:
    covered_entry = TargetEntrySuggestion(
        entry_reference_page="mal:anime:21273",
        title="Gochuumon wa Usagi desu ka?",
        providers=("mal",),
    )
    case = review_cases["gochuumon_movie"].model_copy(
        update={"covered_outside_entries": (covered_entry,)}
    )
    answer = {"anidb:anime:19188": "work 1", "anidb:anime:19792": "other work"}
    assert list_work_partial_sources(case, answer) == (
        WorkPartialSources("other work mal:anime:21273", (), ("anidb:anime:19188",)),
    )
