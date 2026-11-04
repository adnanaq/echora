import pytest
from pydantic import JsonValue
from work_matching.answer_rules import PlacementRule
from work_matching.doubled_provider_check import decide_doubled_providers
from work_matching.known_work_entries_from_seed import KnownWorkEntriesFromSeed
from work_matching.page_placement import (
    PagePlacement,
    check_answer_placements,
    list_placeable_page_keys,
    place_pages_without_review,
    read_answer_placements,
    read_placement,
    write_placement,
)

ENTRY = PagePlacement("work 1")
OTHER_WORK = PagePlacement("other work")
MAL = "https://myanimelist.net/anime/1"
KITSU_ONE = "https://kitsu.app/anime/1"
KITSU_TWO = "https://kitsu.app/anime/2"
OTHER_MAL = "https://myanimelist.net/anime/9"
OTHER_ANILIST = "https://anilist.co/anime/9"


@pytest.fixture
def known_work_entries() -> KnownWorkEntriesFromSeed:
    return KnownWorkEntriesFromSeed(
        [
            {"title": "Other", "type": "TV", "sources": [OTHER_MAL]},
            {"title": "Third", "type": "TV", "sources": [OTHER_ANILIST]},
        ]
    )


@pytest.fixture
def shared_link(linked_page, work_entry):
    entry = work_entry(MAL, KITSU_ONE, KITSU_TWO)
    pages = [linked_page(KITSU_ONE, MAL), linked_page(KITSU_TWO, MAL)]
    return (
        entry,
        decide_doubled_providers(entry, pages, KnownWorkEntriesFromSeed([])),
        pages,
    )


def test_place_pages_without_review_renamed_page_keeps_current_page(
    matching_case_decisions,
) -> None:
    entry, decisions, pages = matching_case_decisions("cocoon")
    assert place_pages_without_review(decisions) == {
        "animeplanet:anime:cocoon-one-summer-of-girlhood": ENTRY
    }


def test_place_pages_without_review_one_linked_page_places_other_page_elsewhere(
    one_linked,
) -> None:
    entry, decisions, pages = one_linked
    assert place_pages_without_review(decisions) == {
        "kitsu:anime:1": ENTRY,
        "kitsu:anime:2": OTHER_WORK,
    }


def test_place_pages_without_review_pages_sharing_link_gives_none(shared_link) -> None:
    assert place_pages_without_review(shared_link[1]) is None


@pytest.mark.parametrize("case_name", ["puu_neko", "genshin_sigewinne"])
def test_place_pages_without_review_separate_works_or_undecided_gives_none(
    matching_case_decisions, case_name: str
) -> None:
    assert place_pages_without_review(matching_case_decisions(case_name)[1]) is None


def test_place_pages_without_review_one_provider_needing_review_places_nothing(
    linked_page, work_entry, known_work_entries
) -> None:
    planet_one = "https://www.anime-planet.com/anime/one"
    planet_two = "https://www.anime-planet.com/anime/two"
    entry = work_entry(MAL, KITSU_ONE, KITSU_TWO, planet_one, planet_two)
    pages = [linked_page(KITSU_ONE, MAL), linked_page(KITSU_TWO, OTHER_MAL)]
    decisions = decide_doubled_providers(entry, pages, known_work_entries)
    assert [decision.needs_review for decision in decisions] == [False, True]
    assert place_pages_without_review(decisions) is None


def test_list_placeable_page_keys_doubled_pages_first_then_other_pages(
    matching_case_decisions,
) -> None:
    entry, decisions, _ = matching_case_decisions("cocoon")
    assert list_placeable_page_keys(entry, decisions) == (
        "animeplanet:anime:cocoon-one-summer-of-girlhood",
        "anidb:anime:18084",
        "anilist:anime:166527",
        "animecountdown:anime:2196692",
        "ann:anime:28792",
        "anisearch:anime:18492",
        "kitsu:anime:47658",
        "livechart:anime:12223",
        "mal:anime:55779",
        "simkl:anime:2196692",
    )


@pytest.mark.parametrize(
    ("answer", "placements"),
    [
        (
            {"kitsu:anime:1": "work 1", "kitsu:anime:2": "other work"},
            {"kitsu:anime:1": ENTRY, "kitsu:anime:2": OTHER_WORK},
        ),
        (
            {"kitsu:anime:1": "work 1, part", "kitsu:anime:2": "work 1, part"},
            {
                "kitsu:anime:1": PagePlacement("work 1", is_part_page=True),
                "kitsu:anime:2": PagePlacement("work 1", is_part_page=True),
            },
        ),
        (
            {"kitsu:anime:1": "work 1, part", "kitsu:anime:2": "other work"},
            {
                "kitsu:anime:1": PagePlacement("work 1", is_part_page=True),
                "kitsu:anime:2": OTHER_WORK,
            },
        ),
        (
            {"kitsu:anime:1": "work 1, main", "kitsu:anime:2": "work 1"},
            {
                "kitsu:anime:1": PagePlacement("work 1", is_main_page=True),
                "kitsu:anime:2": ENTRY,
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
        {"kitsu:anime:1": "work 1, main", "kitsu:anime:2": "work 1, main"},
        {"kitsu:anime:1": "work 1, main", "kitsu:anime:2": "other work"},
        {"kitsu:anime:1": "work 1", "kitsu:anime:2": "other work, main"},
        {"kitsu:anime:1": "work 1", "kitsu:anime:2": "other work, part"},
        {"kitsu:anime:1": "work 1, part", "kitsu:anime:2": "work 1"},
        {"kitsu:anime:1": "work 1, part", "kitsu:anime:2": "work 1, main"},
        {"kitsu:anime:1": "work 1, main, part", "kitsu:anime:2": "work 1"},
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
        (None, (PlacementRule.ANSWER_NOT_PAGE_PLACES,)),
        ("one work", (PlacementRule.ANSWER_NOT_PAGE_PLACES,)),
        ({"kitsu:anime:1": "work 1"}, (PlacementRule.PAGE_WITHOUT_PLACE,)),
        (
            {"kitsu:anime:1": "work 1", "kitsu:anime:2": "work 1"},
            (PlacementRule.PROVIDER_PAGES_NOT_MARKED_ONCE,),
        ),
        (
            {"kitsu:anime:1": "work 1, part", "kitsu:anime:2": "work 1, main"},
            (PlacementRule.PROVIDER_PAGES_NOT_MARKED_ONCE,),
        ),
        (
            {"kitsu:anime:1": "work 1, main", "kitsu:anime:2": "other work"},
            (PlacementRule.SINGLE_PAGE_MARKED_MAIN,),
        ),
        (
            {"kitsu:anime:1": "work 1", "kitsu:anime:2": "other work, main"},
            (PlacementRule.PLACE_NOT_READABLE, PlacementRule.PAGE_WITHOUT_PLACE),
        ),
        (
            {"kitsu:anime:1": "work 1", "kitsu:anime:2": "work 2"},
            (PlacementRule.PAGE_WITHOUT_PLACE,),
        ),
        (
            {
                "kitsu:anime:1": "work 1",
                "kitsu:anime:2": "other work",
                "mal:anime:5": "work 1",
            },
            (PlacementRule.PAGE_OUTSIDE_CASE,),
        ),
        (
            {"kitsu:anime:1": 1, "kitsu:anime:2": ["work 1"]},
            (PlacementRule.PLACE_NOT_READABLE, PlacementRule.PAGE_WITHOUT_PLACE),
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


def test_check_answer_placements_two_mal_pages_in_one_work_reports_rule(
    linked_apart,
) -> None:
    entry, decisions, _ = linked_apart
    answer: dict[str, JsonValue] = dict.fromkeys(
        list_placeable_page_keys(entry, decisions), "work 1, part"
    )
    assert check_answer_placements(answer, entry, decisions).broken_rules == (
        PlacementRule.WORK_WITH_TWO_MAL_PAGES,
    )


def test_check_answer_placements_valid_answer_keeps_placements(shared_link) -> None:
    answer: dict[str, JsonValue] = {
        "kitsu:anime:1": "work 1, part",
        "kitsu:anime:2": "work 1, part",
    }
    assert check_answer_placements(answer, *shared_link[:2]).placements == {
        "kitsu:anime:1": PagePlacement("work 1", is_part_page=True),
        "kitsu:anime:2": PagePlacement("work 1", is_part_page=True),
    }


def test_read_placement_other_work_naming_page_key_keeps_key() -> None:
    assert read_placement("other work mal:anime:62841") == PagePlacement(
        "other work", named_entry_page_key="mal:anime:62841"
    )


@pytest.mark.parametrize(
    "placement_text", ["other work Kusuriya", "other work mal:anime:", "other work "]
)
def test_read_placement_other_work_naming_no_page_key_gives_none(
    placement_text: str,
) -> None:
    assert read_placement(placement_text) is None


@pytest.mark.parametrize(
    "placement_text",
    ["work 1", "work 2, main", "work 3, part", "other work", "other work mal:anime:9"],
)
def test_write_placement_gives_text_read(placement_text: str) -> None:
    placement = read_placement(placement_text)
    assert placement is not None
    assert write_placement(placement) == placement_text


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


def test_check_answer_placements_named_entry_not_in_seed_reports_rule(
    shared_link,
) -> None:
    answer: JsonValue = {
        "kitsu:anime:1": "work 1",
        "kitsu:anime:2": "other work mal:anime:9",
    }
    assert check_answer_placements(answer, *shared_link[:2]).broken_rules == (
        PlacementRule.NAMED_ENTRY_UNKNOWN,
    )


def test_check_answer_placements_named_entry_in_case_reports_rule(
    shared_link,
) -> None:
    answer: JsonValue = {
        "kitsu:anime:1": "work 1",
        "kitsu:anime:2": "other work mal:anime:1",
    }
    assert check_answer_placements(
        answer,
        *shared_link[:2],
        named_entry_reference_pages={"mal:anime:1": "mal:anime:1"},
    ).broken_rules == (PlacementRule.NAMED_ENTRY_IN_CASE,)
