import pytest
from pydantic import JsonValue
from work_matching.answer_placement_check import (
    check_answer_placements,
    read_answer_placements,
)
from work_matching.models import PlacementRule
from work_matching.page_placement import PagePlacement, list_placeable_page_keys

ENTRY = PagePlacement("work 1")
OTHER_WORK = PagePlacement("other work")


@pytest.mark.parametrize(
    ("answer", "placements"),
    [
        (
            {"kitsu:anime:1": "work 1", "kitsu:anime:2": "other work"},
            {"kitsu:anime:1": ENTRY, "kitsu:anime:2": OTHER_WORK},
        ),
        (
            {"kitsu:anime:1": "work 1", "kitsu:anime:2": "dropped"},
            {"kitsu:anime:1": ENTRY, "kitsu:anime:2": PagePlacement("dropped")},
        ),
        (
            {
                "kitsu:anime:1": "work 1",
                "kitsu:anime:2": "work 2",
                "mal:anime:1": "work 1, partial in work 2",
            },
            {
                "kitsu:anime:1": ENTRY,
                "kitsu:anime:2": PagePlacement("work 2"),
                "mal:anime:1": PagePlacement("work 1", partial_works=("work 2",)),
            },
        ),
        (
            {
                "kitsu:anime:1": "work 1",
                "kitsu:anime:2": "other work",
                "mal:anime:1": "other work",
            },
            {
                "kitsu:anime:1": ENTRY,
                "kitsu:anime:2": OTHER_WORK,
                "mal:anime:1": OTHER_WORK,
            },
        ),
        (
            {
                "kitsu:anime:1": "work 1",
                "kitsu:anime:2": "work 2",
                "mal:anime:1": "work 1",
            },
            {
                "kitsu:anime:1": ENTRY,
                "kitsu:anime:2": PagePlacement("work 2"),
                "mal:anime:1": ENTRY,
            },
        ),
    ],
)
def test_read_answer_placements_valid_answer_gives_placements(
    shared_link, answer: JsonValue, placements: dict[str, PagePlacement]
) -> None:
    assert read_answer_placements(answer, *shared_link[:2]) == placements


@pytest.mark.parametrize(
    "answer",
    [
        None,
        "one work",
        {"kitsu:anime:1": "work 1"},
        {"kitsu:anime:1": "work 1", "kitsu:anime:2": "work 1"},
        {"kitsu:anime:1": "work 1, part", "kitsu:anime:2": "other work"},
        {"kitsu:anime:1": "work 1", "kitsu:anime:2": "other work, partial in work 1"},
        {"kitsu:anime:1": "dropped", "kitsu:anime:2": "other work"},
        {"kitsu:anime:1": "dropped", "kitsu:anime:2": "dropped"},
        {"kitsu:anime:1": "work 1", "kitsu:anime:2": "dropped, partial in work 1"},
        {"kitsu:anime:1": "entry", "kitsu:anime:2": "other work"},
        {"kitsu:anime:1": "work 0", "kitsu:anime:2": "other work"},
        {"kitsu:anime:1": "work 1", "kitsu:anime:2": "work 2"},
        {"kitsu:anime:1": "work 1", "kitsu:anime:2": "work 1", "mal:anime:1": "work 1"},
        {
            "kitsu:anime:1": "work 1",
            "kitsu:anime:2": "work 01",
            "mal:anime:1": "work 1",
        },
        {"kitsu:anime:1": "work 1", "kitsu:anime:2": "maybe"},
        {
            "kitsu:anime:1": "work 1",
            "kitsu:anime:2": "other work",
            "mal:anime:5": "work 1",
        },
        {"kitsu:anime:1": 1, "kitsu:anime:2": ["work 1"]},
    ],
)
def test_read_answer_placements_invalid_answer_gives_none(
    shared_link, answer: JsonValue
) -> None:
    assert read_answer_placements(answer, *shared_link[:2]) is None


@pytest.mark.parametrize(
    ("answer", "broken_rules"),
    [
        (None, (PlacementRule.INVALID_PLACEMENT_MAP,)),
        ("one work", (PlacementRule.INVALID_PLACEMENT_MAP,)),
        ({"kitsu:anime:1": "work 1"}, (PlacementRule.MISSING_REQUIRED_PLACEMENT,)),
        (
            {"kitsu:anime:1": "work 1", "kitsu:anime:2": "work 1"},
            (PlacementRule.WORK_WITH_TWO_PAGES_OF_ONE_PROVIDER,),
        ),
        (
            {"kitsu:anime:1": "dropped", "kitsu:anime:2": "other work"},
            (PlacementRule.DROPPED_PAGE_WITHOUT_KEPT_PAGE,),
        ),
        (
            {"kitsu:anime:1": "work 1, partial in work 1", "kitsu:anime:2": "dropped"},
            (PlacementRule.PARTIAL_SOURCE_OF_OWN_WORK,),
        ),
        (
            {"kitsu:anime:1": "work 1, partial in work 3", "kitsu:anime:2": "dropped"},
            (PlacementRule.PARTIAL_SOURCE_OF_EMPTY_WORK,),
        ),
        (
            {
                "kitsu:anime:1": "work 1",
                "kitsu:anime:2": "other work, partial in work 1",
            },
            (
                PlacementRule.INVALID_PAGE_PLACEMENT,
                PlacementRule.MISSING_REQUIRED_PLACEMENT,
            ),
        ),
        (
            {"kitsu:anime:1": "work 1", "kitsu:anime:2": "work 2"},
            (PlacementRule.MISSING_REQUIRED_PLACEMENT,),
        ),
        (
            {
                "kitsu:anime:1": "work 1",
                "kitsu:anime:2": "other work",
                "mal:anime:5": "work 1",
            },
            (PlacementRule.PAGE_NOT_IN_CASE,),
        ),
        (
            {"kitsu:anime:1": 1, "kitsu:anime:2": ["work 1"]},
            (
                PlacementRule.INVALID_PAGE_PLACEMENT,
                PlacementRule.MISSING_REQUIRED_PLACEMENT,
            ),
        ),
        ({"kitsu:anime:1": "work 1", "kitsu:anime:2": "other work"}, ()),
    ],
)
def test_check_answer_placements_reports_broken_rules(
    shared_link, answer: JsonValue, broken_rules: tuple[PlacementRule, ...]
) -> None:
    assert (
        check_answer_placements(answer, *shared_link[:2]).broken_rules == broken_rules
    )


def test_check_answer_placements_two_pages_of_providers_in_one_work_reports_rule_once(
    linked_apart,
) -> None:
    entry, decisions, _ = linked_apart
    answer: dict[str, JsonValue] = dict.fromkeys(
        list_placeable_page_keys(entry, decisions), "work 1"
    )
    assert check_answer_placements(answer, entry, decisions).broken_rules == (
        PlacementRule.WORK_WITH_TWO_PAGES_OF_ONE_PROVIDER,
    )


def test_check_answer_placements_valid_answer_keeps_placements(shared_link) -> None:
    answer: dict[str, JsonValue] = {
        "kitsu:anime:1": "dropped",
        "kitsu:anime:2": "work 1",
    }
    assert check_answer_placements(answer, *shared_link[:2]).placements == {
        "kitsu:anime:1": PagePlacement("dropped"),
        "kitsu:anime:2": ENTRY,
    }


def test_check_answer_placements_named_entry_in_seed_keeps_placement(
    shared_link,
) -> None:
    answer: JsonValue = {
        "kitsu:anime:1": "work 1",
        "kitsu:anime:2": "other work mal:anime:9",
    }
    placement_check = check_answer_placements(
        answer,
        *shared_link[:2],
        named_entry_reference_pages={"mal:anime:9": "mal:anime:9"},
    )
    assert placement_check.broken_rules == ()
    assert placement_check.placements["kitsu:anime:2"] == PagePlacement(
        "other work", named_entry_page_key="mal:anime:9"
    )


@pytest.mark.parametrize(
    "answer",
    [
        {"kitsu:anime:1": "work 1", "kitsu:anime:2": "other work mal:anime:9"},
        {
            "kitsu:anime:1": "work 1, partial in other work mal:anime:9",
            "kitsu:anime:2": "dropped",
        },
    ],
)
def test_check_answer_placements_named_entry_not_in_seed_reports_rule(
    shared_link, answer: JsonValue
) -> None:
    assert check_answer_placements(answer, *shared_link[:2]).broken_rules == (
        PlacementRule.UNKNOWN_NAMED_ENTRY,
    )


@pytest.mark.parametrize(
    "answer",
    [
        {"kitsu:anime:1": "work 1", "kitsu:anime:2": "other work mal:anime:1"},
        {
            "kitsu:anime:1": "work 1, partial in other work mal:anime:1",
            "kitsu:anime:2": "dropped",
        },
    ],
)
def test_check_answer_placements_named_entry_in_case_reports_rule(
    shared_link, answer: JsonValue
) -> None:
    assert check_answer_placements(
        answer,
        *shared_link[:2],
        named_entry_reference_pages={"mal:anime:1": "mal:anime:1"},
    ).broken_rules == (PlacementRule.NAMED_ENTRY_IN_CASE,)
