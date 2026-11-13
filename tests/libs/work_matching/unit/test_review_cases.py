import json

import pytest
from work_matching.doubled_provider_check import decide_doubled_providers
from work_matching.known_work_entries_from_seed import KnownWorkEntriesFromSeed
from work_matching.models import (
    DoubledProviderOutcome,
    DoubledProviderReviewReason,
    LinkedPageAddition,
    LinkedPageAnswerRule,
    LinkedPageReason,
    PlacementRule,
)
from work_matching.page_placement import PagePlacement
from work_matching.provider_page import ProviderPage
from work_matching.provider_page_key import resolve_page_key
from work_matching.review_cases import (
    ReviewCase,
    ReviewCaseKind,
)
from work_matching.split_work_check import list_split_work_review_reasons

SIGEWINNE = "animeplanet:anime:genshin-impact-character-teaser-sigewinne-everyday-correspondence"
VARESA = (
    "animeplanet:anime:genshin-impact-character-teaser-varesa-the-legend-of-a-warrior"
)


def _build_season_page(address: str, *linked_page_keys: str) -> ProviderPage:
    return ProviderPage(
        key=resolve_page_key(address),
        address=address,
        type="TV",
        episode_count=12,
        aired_from="2020-01-05",
        aired_to="2020-03-29",
        linked_page_keys=linked_page_keys,
    )


def test_review_case_build_entry_group_case_single_entry_names_case_by_pages_and_entry(
    doubled_case,
) -> None:
    assert (doubled_case.case_name, doubled_case.reasons) == (
        f"doubled provider: {SIGEWINNE} {VARESA} in {SIGEWINNE}",
        ("UNDECIDED", "UNLINKED_PAGES"),
    )
    assert doubled_case.entries[SIGEWINNE].title == (
        "Genshin Impact: Character Teaser - Sigewinne: Everyday Correspondence"
    )
    assert [page.key for page in doubled_case.pages] == [SIGEWINNE, VARESA]


def test_review_case_build_entry_group_case_single_entry_keeps_entry_and_searched_pages(
    linked_page, work_entry
) -> None:
    entry = work_entry(
        "https://myanimelist.net/anime/1",
        "https://kitsu.app/anime/1",
        "https://kitsu.app/anime/2",
    )
    found_page = ProviderPage.read_canonical_record(
        "https://animeschedule.net/anime/show",
        {"title": "Show"},
        searched_by_address="https://myanimelist.net/anime/1",
    )
    pages = [
        linked_page("https://kitsu.app/anime/1"),
        linked_page("https://kitsu.app/anime/2"),
        linked_page("https://myanimelist.net/anime/1"),
        linked_page("https://anilist.co/anime/9"),
        found_page,
    ]
    decisions = decide_doubled_providers(entry, pages, KnownWorkEntriesFromSeed([]))
    case = ReviewCase.build_entry_group_case([entry], decisions, (), pages)
    assert [page.key for page in case.pages] == [
        "https://animeschedule.net/anime/show",
        "kitsu:anime:1",
        "kitsu:anime:2",
        "mal:anime:1",
    ]


def test_review_case_build_entry_group_case_single_entry_one_case_for_every_provider(
    matching_cases,
) -> None:
    entries, pages = matching_cases["ao_haru_ride_ova"]
    decisions = decide_doubled_providers(
        entries[0], pages, KnownWorkEntriesFromSeed([])
    )
    case = ReviewCase.build_entry_group_case([entries[0]], decisions, (), pages)
    assert case.case_name == (
        "doubled provider: anilist:anime:20837 anilist:anime:20900"
        " animeplanet:anime:blue-spring-ride-page-13"
        " animeplanet:anime:blue-spring-ride-unwritten in mal:anime:24151"
    )
    assert case.reasons == ("PART_PAGES", "UNDECIDED", "UNLINKED_PAGES")


def test_review_case_build_entry_group_case_part_pages_suggest_one_work_per_part(
    matching_cases,
) -> None:
    entries, pages = matching_cases["ao_haru_ride_ova"]
    decisions = decide_doubled_providers(
        entries[0], pages, KnownWorkEntriesFromSeed([])
    )
    case = ReviewCase.build_entry_group_case([entries[0]], decisions, (), pages)
    assert case.suggested_answer == {
        "anilist:anime:20837": "work 1",
        "anilist:anime:20900": "work 2",
    }


def test_review_case_build_entry_group_case_unlinked_entries_suggest_nothing(
    split_case,
) -> None:
    assert split_case.suggested_answer == {}


def test_review_case_build_entry_group_case_linked_entries_suggest_one_work(
    build_group_case, linked_page, work_entry
) -> None:
    entries = [
        work_entry("https://myanimelist.net/anime/1"),
        work_entry("https://anilist.co/anime/1"),
    ]
    pages = [
        _build_season_page("https://myanimelist.net/anime/1"),
        _build_season_page("https://anilist.co/anime/1", "mal:anime:1"),
    ]
    assert build_group_case(entries, pages).suggested_answer == {
        "mal:anime:1": "work 1",
        "anilist:anime:1": "work 1",
    }


def test_review_case_build_entry_group_case_lists_why_suggestions_left_out(
    build_group_case,
    work_entry,
) -> None:
    entries = [
        work_entry("https://myanimelist.net/anime/1"),
        work_entry("https://anilist.co/anime/1"),
    ]
    pages = [
        _build_season_page("https://myanimelist.net/anime/1"),
        _build_season_page("https://anilist.co/anime/1", "mal:anime:1").model_copy(
            update={"type": "MOVIE"}
        ),
    ]
    case = build_group_case(entries, pages)
    assert (case.suggested_answer, case.reasons[-1:]) == (
        {"mal:anime:1": "work 1"},
        ("DETAILS_DIFFER",),
    )


def test_review_case_build_entry_group_case_single_entry_writes_linked_pages_sorted(
    matching_cases,
) -> None:
    entries, pages = matching_cases["puu_neko"]
    decisions = decide_doubled_providers(
        entries[0], pages, KnownWorkEntriesFromSeed([])
    )
    case = ReviewCase.build_entry_group_case([entries[0]], decisions, (), pages)
    written_decisions = json.loads(case.model_dump_json())["decisions"]
    assert all(
        linked_pages == sorted(linked_pages)
        for decision in written_decisions
        for linked_pages in decision["belonging_page_keys"].values()
    )


def test_review_case_build_entry_group_case_single_entry_writes_unrelated_linked_page_keys_sorted(
    linked_page, work_entry
) -> None:
    entry = work_entry(
        "https://myanimelist.net/anime/1",
        "https://kitsu.app/anime/1",
        "https://kitsu.app/anime/2",
    )
    linked_addresses = [
        f"https://myanimelist.net/anime/{number}" for number in range(10, 30)
    ]
    known_work_entries = KnownWorkEntriesFromSeed(
        [{"title": "Other", "type": "TV", "sources": linked_addresses}]
    )
    pages = [
        linked_page("https://kitsu.app/anime/1", "https://myanimelist.net/anime/1"),
        linked_page("https://kitsu.app/anime/2", *linked_addresses),
    ]
    decisions = decide_doubled_providers(entry, pages, known_work_entries)
    case = ReviewCase.build_entry_group_case([entry], decisions, (), pages)
    [decision] = json.loads(case.model_dump_json())["decisions"]
    written_links = decision["unrelated_pages"]["kitsu:anime:2"]["linked_page_keys"]
    assert written_links == sorted(written_links)


def test_review_case_build_entry_group_case_keeps_searched_pages(
    build_group_case, linked_page, work_entry
) -> None:
    entries = [
        work_entry("https://myanimelist.net/anime/1"),
        work_entry("https://anilist.co/anime/1"),
    ]
    found_page = ProviderPage.read_canonical_record(
        "https://animeschedule.net/anime/show",
        {"title": "Show"},
        searched_by_address="https://myanimelist.net/anime/1",
    )
    pages = [linked_page("https://myanimelist.net/anime/1"), found_page]
    case = build_group_case(entries, pages)
    assert [page.key for page in case.pages] == [
        "https://animeschedule.net/anime/show",
        "mal:anime:1",
    ]


def test_review_case_build_entry_group_case_keeps_renamed_page(
    build_group_case, work_entry
) -> None:
    old_address = "https://www.anime-planet.com/anime/old"
    entries = [work_entry(old_address), work_entry("https://myanimelist.net/anime/1")]
    renamed_page = ProviderPage.read_canonical_record(
        old_address,
        {
            "title": "Show",
            "sources": [old_address, "https://www.anime-planet.com/anime/new"],
        },
    )
    case = build_group_case(entries, [renamed_page])
    assert [page.key for page in case.pages] == ["animeplanet:anime:old"]


def test_review_case_build_entry_group_case_names_group_case_by_every_page(
    split_case,
) -> None:
    assert (split_case.case_name, split_case.reasons) == (
        "split work: anilist:anime:1 mal:anime:1",
        ("NOT_TIED_BY_LINKS",),
    )
    assert [page.key for page in split_case.pages] == ["mal:anime:1"]


def test_review_case_build_entry_group_case_keeps_candidate_evidence(
    linked_page, work_entry
) -> None:
    entries = [
        work_entry("https://myanimelist.net/anime/1"),
        work_entry("https://anilist.co/anime/1"),
    ]
    pages = [
        linked_page("https://anilist.co/anime/1", "https://myanimelist.net/anime/1")
    ]
    evidence = ("anilist:anime:1 links mal:anime:1",)
    case = ReviewCase.build_entry_group_case(
        entries,
        [],
        list_split_work_review_reasons(entries, pages),
        pages,
        candidate_evidence=evidence,
    )
    assert case.candidate_evidence == evidence
    assert list(json.loads(case.model_dump_json()))[:5] == [
        "case_name",
        "answer",
        "suggested_answer",
        "reasons",
        "candidate_evidence",
    ]


@pytest.mark.parametrize(
    ("answer", "placements"),
    [
        (
            {SIGEWINNE: "work 1", VARESA: "other work"},
            {SIGEWINNE: PagePlacement("work 1"), VARESA: PagePlacement("other work")},
        ),
        (
            {SIGEWINNE: "work 1", VARESA: "work 10"},
            {SIGEWINNE: PagePlacement("work 1"), VARESA: PagePlacement("work 10")},
        ),
        (
            {SIGEWINNE: "work 1, partial in work 2", VARESA: "work 2"},
            {
                SIGEWINNE: PagePlacement("work 1", partial_works=("work 2",)),
                VARESA: PagePlacement("work 2"),
            },
        ),
    ],
)
def test_review_case_accepted_answer_placement_for_every_page_gives_placements(
    doubled_case, answer: dict[str, str], placements: dict[str, PagePlacement]
) -> None:
    accepted_placements = doubled_case.model_copy(
        update={"answer": answer}
    ).accepted_answer
    assert accepted_placements == placements


@pytest.mark.parametrize(
    "answer",
    [
        None,
        {SIGEWINNE: "work 1"},
        {SIGEWINNE: "work 1", VARESA: "work 1"},
        {SIGEWINNE: "work 1", VARESA: "other work", "mal:anime:1": "work 1"},
        {SIGEWINNE: "work 1", VARESA: "work 1"},
        {SIGEWINNE: "work 1", VARESA: "maybe"},
        {SIGEWINNE: "work 1", VARESA: "work 01"},
        {SIGEWINNE: "work 1", VARESA: "work 0"},
        {SIGEWINNE: "work 1", VARESA: "work 1٢"},
        {SIGEWINNE: 1, VARESA: ["work 1"]},
        "one work",
    ],
)
def test_review_case_accepted_answer_invalid_doubled_provider_answer_gives_none(
    doubled_case, answer: object
) -> None:
    assert doubled_case.model_copy(update={"answer": answer}).accepted_answer is None


@pytest.mark.parametrize(
    ("answer", "accepted_placements"),
    [
        (
            {"mal:anime:1": "work 1", "anilist:anime:1": "work 1"},
            {
                "mal:anime:1": PagePlacement("work 1"),
                "anilist:anime:1": PagePlacement("work 1"),
            },
        ),
        (
            {"mal:anime:1": "work 1", "anilist:anime:1": "work 2"},
            {
                "mal:anime:1": PagePlacement("work 1"),
                "anilist:anime:1": PagePlacement("work 2"),
            },
        ),
        ("one work", None),
        ({"mal:anime:1": "work 1"}, None),
    ],
)
def test_review_case_accepted_answer_entry_group_places_every_page(
    split_case, answer: object, accepted_placements: dict[str, PagePlacement] | None
) -> None:
    assert (
        split_case.model_copy(update={"answer": answer}).accepted_answer
        == accepted_placements
    )


def test_review_case_build_linked_page_case_names_case_by_page_and_linking_entries(
    linked_page_case,
) -> None:
    assert (linked_page_case.case_name, linked_page_case.reasons) == (
        "linked page: mal:anime:65144 from anilist:anime:163794",
        ("EPISODE_COUNTS_DIFFER",),
    )
    assert list(linked_page_case.entries) == ["anilist:anime:163794"]
    assert [page.key for page in linked_page_case.pages] == [
        "anilist:anime:163794",
        "mal:anime:65144",
    ]


def test_review_case_build_linked_page_case_fewer_episodes_suggests_nothing(
    linked_page_case,
) -> None:
    assert linked_page_case.suggested_answer == {}


@pytest.mark.parametrize(
    ("answer", "accepted_answer"),
    [
        (
            {"mal:anime:65144": "add as partial"},
            LinkedPageAddition("anilist:anime:163794", is_partial_source=True),
        ),
        ({"mal:anime:65144": "skip"}, "skip"),
        ({"mal:anime:65144": "work 1"}, None),
        ("one work", None),
    ],
)
def test_review_case_accepted_answer_linked_page_reads_addition(
    linked_page_case, answer: object, accepted_answer: object
) -> None:
    assert (
        linked_page_case.model_copy(update={"answer": answer}).accepted_answer
        == accepted_answer
    )


@pytest.mark.parametrize(
    ("case_name", "kind"),
    [
        ("atashinchi_next", ReviewCaseKind.SPLIT_WORK),
        ("gochuumon_movie", ReviewCaseKind.DOUBLED_PROVIDER),
        ("nami", ReviewCaseKind.LINKED_PAGE),
    ],
)
def test_review_case_kind_and_decisions_read_from_case(
    review_cases, case_name: str, kind: ReviewCaseKind
) -> None:
    case = review_cases[case_name]
    is_linked_page = kind is ReviewCaseKind.LINKED_PAGE
    assert case.kind is kind
    assert case.linked_page_decision is (case.decisions[0] if is_linked_page else None)
    assert case.doubled_provider_decisions == (() if is_linked_page else case.decisions)


def test_review_case_list_broken_answer_rules_group_answer_missing_page(
    review_cases,
) -> None:
    assert review_cases["atashinchi_next"].list_broken_answer_rules(
        {"mal:anime:63013": "work 1"}
    ) == (PlacementRule.MISSING_REQUIRED_PLACEMENT,)


def test_review_case_list_broken_answer_rules_valid_group_answer_breaks_none(
    review_cases,
) -> None:
    assert (
        review_cases["atashinchi_next"].list_broken_answer_rules(
            {
                "mal:anime:63013": "work 1",
                "anilist:anime:203146": "work 1",
                "anidb:anime:19895": "work 1",
            }
        )
        == ()
    )


def test_review_case_list_broken_answer_rules_linked_page_unknown_word(
    review_cases,
) -> None:
    assert review_cases["nami"].list_broken_answer_rules(
        {"anilist:anime:102289": "move"}
    ) == (LinkedPageAnswerRule.INVALID_LINKED_PAGE_ANSWER,)


def test_review_case_json_stores_decision_enums_by_name(review_cases) -> None:
    doubled_decision = json.loads(review_cases["gochuumon_movie"].model_dump_json())[
        "decisions"
    ][0]
    linked_decision = json.loads(review_cases["nami"].model_dump_json())["decisions"][0]
    assert (
        doubled_decision["outcome"] in DoubledProviderOutcome.__members__,
        all(
            reason in DoubledProviderReviewReason.__members__
            for reason in doubled_decision["review_reasons"]
        ),
        linked_decision["reason"] in LinkedPageReason.__members__,
    ) == (True, True, True)
