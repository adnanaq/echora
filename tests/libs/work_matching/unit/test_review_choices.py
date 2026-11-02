import dataclasses

import pytest
from work_matching.answer_rules import LinkedPageAnswerRule, PlacementRule
from work_matching.doubled_provider_check import decide_doubled_providers
from work_matching.entry_group import combine_group_entries
from work_matching.known_work_entries_from_seed import KnownWorkEntriesFromSeed
from work_matching.review_cases import ReviewCase
from work_matching.review_choices import (
    MAIN_MARK,
    NO_MARK,
    PART_MARK,
    ChoiceRule,
    choose_group_page_place,
    find_linked_page_conflicts,
    list_group_page_choices,
    list_linked_page_options,
    list_required_page_keys,
)

MAL = "https://myanimelist.net/anime/1"
OTHER_MAL = "https://myanimelist.net/anime/2"
GOCHUUMON_DOUBLED_PAGES = ("anidb:anime:19188", "anidb:anime:19792")


def _choices_by_page_key(case, answer):
    return {
        page_choices.page_key: page_choices
        for page_choices in list_group_page_choices(case, answer)
    }


def _disabling_rules(options):
    return {option.value: option.disabling_rule for option in options}


@pytest.fixture
def two_mal_group(work_entry) -> ReviewCase:
    entries = [work_entry(MAL), work_entry(OTHER_MAL)]
    decisions = decide_doubled_providers(
        combine_group_entries(entries), [], KnownWorkEntriesFromSeed([])
    )
    return ReviewCase.from_entry_group(entries, decisions, [], [])


def test_list_group_page_choices_no_place_yet_offers_first_work_and_other_work(
    review_cases,
) -> None:
    choices = _choices_by_page_key(review_cases["atashinchi_next"], {})
    assert [option.value for option in choices["mal:anime:63013"].work_options] == [
        "work 1",
        "other work",
    ]


def test_list_group_page_choices_offers_one_work_past_highest_in_use(
    review_cases,
) -> None:
    choices = _choices_by_page_key(
        review_cases["atashinchi_next"], {"anidb:anime:19895": "work 2"}
    )
    assert [option.value for option in choices["mal:anime:63013"].work_options] == [
        "work 1",
        "work 2",
        "work 3",
        "other work",
    ]


def test_list_group_page_choices_never_offers_more_works_than_pages(
    review_cases,
) -> None:
    choices = _choices_by_page_key(
        review_cases["atashinchi_next"],
        {"anidb:anime:19895": "work 3", "mal:anime:63013": "work 2"},
    )
    assert [option.value for option in choices["mal:anime:63013"].work_options] == [
        "work 1",
        "work 2",
        "work 3",
        "other work",
    ]


def test_list_group_page_choices_split_work_requires_every_page(review_cases) -> None:
    choices = list_group_page_choices(review_cases["atashinchi_next"], {})
    assert all(page_choices.is_required for page_choices in choices)


def test_list_group_page_choices_starts_on_given_answer(review_cases) -> None:
    choices = _choices_by_page_key(
        review_cases["gochuumon_movie"],
        {"anidb:anime:19188": "other work", "anidb:anime:19792": "work 1, part"},
    )
    assert (
        choices["anidb:anime:19188"].work,
        choices["anidb:anime:19792"].work,
        choices["anidb:anime:19792"].mark,
    ) == ("other work", "work 1", PART_MARK)


def test_list_group_page_choices_second_mal_page_in_work_disabled(
    two_mal_group,
) -> None:
    choices = _choices_by_page_key(two_mal_group, {"mal:anime:1": "work 1"})
    assert (
        _disabling_rules(choices["mal:anime:2"].work_options)["work 1"]
        is PlacementRule.WORK_WITH_TWO_MAL_PAGES
    )


def test_list_group_page_choices_mal_page_keeps_its_own_work_enabled(
    two_mal_group,
) -> None:
    choices = _choices_by_page_key(two_mal_group, {"mal:anime:1": "work 1"})
    assert _disabling_rules(choices["mal:anime:1"].work_options)["work 1"] is None


def test_list_group_page_choices_only_page_of_provider_cannot_be_main(
    review_cases,
) -> None:
    choices = _choices_by_page_key(
        review_cases["gochuumon_movie"],
        {"anidb:anime:19188": "other work", "anidb:anime:19792": "work 1"},
    )
    assert _disabling_rules(choices["anidb:anime:19792"].mark_options) == {
        NO_MARK: None,
        MAIN_MARK: PlacementRule.SINGLE_PAGE_MARKED_MAIN,
        PART_MARK: None,
    }


def test_list_group_page_choices_second_page_of_provider_offers_main(
    review_cases,
) -> None:
    choices = _choices_by_page_key(
        review_cases["gochuumon_movie"],
        dict.fromkeys(GOCHUUMON_DOUBLED_PAGES, "work 1"),
    )
    assert all(
        _disabling_rules(choices[page_key].mark_options)[MAIN_MARK] is None
        for page_key in GOCHUUMON_DOUBLED_PAGES
    )


def test_list_group_page_choices_part_pages_offer_main(review_cases) -> None:
    choices = _choices_by_page_key(
        review_cases["gochuumon_movie"],
        dict.fromkeys(GOCHUUMON_DOUBLED_PAGES, "work 1, part"),
    )
    assert (
        _disabling_rules(choices["anidb:anime:19188"].mark_options)[MAIN_MARK] is None
    )


def test_list_group_page_choices_main_page_offers_part_to_others(
    review_cases,
) -> None:
    choices = _choices_by_page_key(
        review_cases["gochuumon_movie"],
        {"anidb:anime:19188": "work 1, main", "anidb:anime:19792": "work 1"},
    )
    assert (
        _disabling_rules(choices["anidb:anime:19792"].mark_options)[PART_MARK] is None
    )


def test_choose_group_page_place_part_after_main_marks_all_part(review_cases) -> None:
    answer = choose_group_page_place(
        {"anidb:anime:19188": "work 1, main", "anidb:anime:19792": "work 1"},
        "anidb:anime:19792",
        "work 1",
        PART_MARK,
    )
    assert answer == dict.fromkeys(GOCHUUMON_DOUBLED_PAGES, "work 1, part")


def test_list_group_page_choices_other_work_offers_no_mark(review_cases) -> None:
    choices = _choices_by_page_key(
        review_cases["gochuumon_movie"], {"anidb:anime:19188": "other work"}
    )
    assert _disabling_rules(choices["anidb:anime:19188"].mark_options) == {
        NO_MARK: None,
        MAIN_MARK: ChoiceRule.MARK_IN_OTHER_WORK,
        PART_MARK: ChoiceRule.MARK_IN_OTHER_WORK,
    }


def test_list_group_page_choices_page_without_work_offers_no_mark(
    review_cases,
) -> None:
    choices = _choices_by_page_key(review_cases["gochuumon_movie"], {})
    assert _disabling_rules(choices["anidb:anime:19188"].mark_options) == {
        NO_MARK: None,
        MAIN_MARK: ChoiceRule.MARK_WITHOUT_WORK,
        PART_MARK: ChoiceRule.MARK_WITHOUT_WORK,
    }


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


def test_choose_group_page_place_part_spreads_to_provider_pages_in_work(
    review_cases,
) -> None:
    answer = choose_group_page_place(
        dict.fromkeys(GOCHUUMON_DOUBLED_PAGES, "work 1"),
        "anidb:anime:19792",
        "work 1",
        PART_MARK,
    )
    assert [answer[page_key] for page_key in GOCHUUMON_DOUBLED_PAGES] == [
        "work 1, part",
        "work 1, part",
    ]


def test_choose_group_page_place_main_clears_other_provider_marks(
    review_cases,
) -> None:
    answer = choose_group_page_place(
        dict.fromkeys(GOCHUUMON_DOUBLED_PAGES, "work 1, part"),
        "anidb:anime:19792",
        "work 1",
        NO_MARK,
    )
    answer = choose_group_page_place(answer, "anidb:anime:19188", "work 1", MAIN_MARK)
    assert [answer[page_key] for page_key in GOCHUUMON_DOUBLED_PAGES] == [
        "work 1, main",
        "work 1",
    ]


def test_choose_group_page_place_removing_part_clears_provider_parts(
    review_cases,
) -> None:
    answer = choose_group_page_place(
        dict.fromkeys(GOCHUUMON_DOUBLED_PAGES, "work 1, part"),
        "anidb:anime:19188",
        "work 1",
        NO_MARK,
    )
    assert [answer[page_key] for page_key in GOCHUUMON_DOUBLED_PAGES] == [
        "work 1",
        "work 1",
    ]


def test_choose_group_page_place_lone_main_page_loses_mark(review_cases) -> None:
    answer = choose_group_page_place(
        {"anidb:anime:19188": "work 1, main", "anidb:anime:19792": "work 1"},
        "anidb:anime:19792",
        "other work",
        NO_MARK,
    )
    assert answer == {"anidb:anime:19188": "work 1", "anidb:anime:19792": "other work"}


def test_choose_group_page_place_no_work_removes_page(review_cases) -> None:
    answer = choose_group_page_place(
        {"anidb:anime:19188": "work 1"},
        "anidb:anime:19188",
        None,
        NO_MARK,
    )
    assert answer == {}


def test_list_linked_page_options_one_linking_entry_offers_add_without_entry(
    review_cases,
) -> None:
    options = list_linked_page_options(review_cases["nami"])
    assert _disabling_rules(options) == {
        "add": None,
        "add as part": None,
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
        _disabling_rules(list_linked_page_options(case))["add"]
        is LinkedPageAnswerRule.PROVIDER_IN_ENTRY
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
        "add as part to kitsu:anime:10561",
        "add to kitsu:anime:99",
        "add as part to kitsu:anime:99",
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


def test_find_linked_page_conflicts_part_pages_never_conflict(review_cases) -> None:
    case = review_cases["nami"]
    [decision] = case.decisions
    second_case = case.model_copy(
        update={
            "case_name": "linked page: anilist:anime:5 from kitsu:anime:10561",
            "decisions": (dataclasses.replace(decision, page_key="anilist:anime:5"),),
            "answer": {"anilist:anime:5": "add as part"},
        }
    )
    first_case = case.model_copy(
        update={"answer": {"anilist:anime:102289": "add as part"}}
    )
    assert find_linked_page_conflicts([first_case, second_case]) == ()
