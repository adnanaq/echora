import json

import pytest
from work_matching.answer_rules import LinkedPageAnswerRule, PlacementRule
from work_matching.doubled_provider_check import decide_doubled_providers
from work_matching.known_work_entries_from_seed import KnownWorkEntriesFromSeed
from work_matching.linked_page_check import (
    LinkedPageAddition,
)
from work_matching.page_placement import PagePlacement
from work_matching.provider_page import ProviderPage
from work_matching.provider_page_key import get_page_key
from work_matching.review_cases import (
    ReviewCase,
    ReviewCaseKind,
)
from work_matching.split_work_check import list_split_work_review_reasons

SIGEWINNE = "animeplanet:anime:genshin-impact-character-teaser-sigewinne-everyday-correspondence"
VARESA = (
    "animeplanet:anime:genshin-impact-character-teaser-varesa-the-legend-of-a-warrior"
)


def _season_page(address: str, *linked_page_keys: str) -> ProviderPage:
    return ProviderPage(
        key=get_page_key(address),
        address=address,
        type="TV",
        episode_count=12,
        aired_from="2020-01-05",
        aired_to="2020-03-29",
        linked_page_keys=linked_page_keys,
    )


def test_review_case_from_entry_group_single_entry_names_case_by_pages_and_entry(
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


def test_review_case_from_entry_group_single_entry_keeps_entry_and_searched_pages(
    linked_page, work_entry
) -> None:
    entry = work_entry(
        "https://myanimelist.net/anime/1",
        "https://kitsu.app/anime/1",
        "https://kitsu.app/anime/2",
    )
    found = ProviderPage.from_canonical_record(
        "https://animeschedule.net/anime/show",
        {"title": "Show"},
        searched_by_address="https://myanimelist.net/anime/1",
    )
    pages = [
        linked_page("https://kitsu.app/anime/1"),
        linked_page("https://kitsu.app/anime/2"),
        linked_page("https://myanimelist.net/anime/1"),
        linked_page("https://anilist.co/anime/9"),
        found,
    ]
    decisions = decide_doubled_providers(entry, pages, KnownWorkEntriesFromSeed([]))
    case = ReviewCase.from_entry_group([entry], decisions, (), pages)
    assert [page.key for page in case.pages] == [
        "https://animeschedule.net/anime/show",
        "kitsu:anime:1",
        "kitsu:anime:2",
        "mal:anime:1",
    ]


def test_review_case_from_entry_group_single_entry_one_case_for_every_provider(
    matching_cases,
) -> None:
    entries, pages = matching_cases["ao_haru_ride_ova"]
    decisions = decide_doubled_providers(
        entries[0], pages, KnownWorkEntriesFromSeed([])
    )
    case = ReviewCase.from_entry_group([entries[0]], decisions, (), pages)
    assert case.case_name == (
        "doubled provider: anilist:anime:20837 anilist:anime:20900"
        " animeplanet:anime:blue-spring-ride-page-13"
        " animeplanet:anime:blue-spring-ride-unwritten in mal:anime:24151"
    )
    assert case.reasons == ("PART_PAGES", "UNDECIDED", "UNLINKED_PAGES")


def test_review_case_from_entry_group_single_entry_suggests_linked_places(
    matching_cases,
) -> None:
    entries, pages = matching_cases["ao_haru_ride_ova"]
    decisions = decide_doubled_providers(
        entries[0], pages, KnownWorkEntriesFromSeed([])
    )
    case = ReviewCase.from_entry_group([entries[0]], decisions, (), pages)
    assert case.suggested_answer["anilist:anime:20900"] == "work 1, part"
    assert "animeplanet:anime:blue-spring-ride-unwritten" not in case.suggested_answer


def test_review_case_from_entry_group_unlinked_entries_suggest_nothing(
    split_case,
) -> None:
    assert split_case.suggested_answer == {}


def test_review_case_from_entry_group_linked_entries_suggest_one_work(
    build_group_case, linked_page, work_entry
) -> None:
    entries = [
        work_entry("https://myanimelist.net/anime/1"),
        work_entry("https://anilist.co/anime/1"),
    ]
    pages = [
        _season_page("https://myanimelist.net/anime/1"),
        _season_page("https://anilist.co/anime/1", "mal:anime:1"),
    ]
    assert build_group_case(entries, pages).suggested_answer == {
        "mal:anime:1": "work 1",
        "anilist:anime:1": "work 1",
    }


def test_review_case_from_entry_group_lists_why_suggestions_left_out(
    build_group_case,
    work_entry,
) -> None:
    entries = [
        work_entry("https://myanimelist.net/anime/1"),
        work_entry("https://anilist.co/anime/1"),
    ]
    pages = [
        _season_page("https://myanimelist.net/anime/1"),
        _season_page("https://anilist.co/anime/1", "mal:anime:1").model_copy(
            update={"type": "MOVIE"}
        ),
    ]
    case = build_group_case(entries, pages)
    assert (case.suggested_answer, case.reasons[-1:]) == (
        {"mal:anime:1": "work 1"},
        ("DETAILS_DIFFER",),
    )


def test_review_case_from_entry_group_single_entry_writes_linked_pages_sorted(
    matching_cases,
) -> None:
    entries, pages = matching_cases["puu_neko"]
    decisions = decide_doubled_providers(
        entries[0], pages, KnownWorkEntriesFromSeed([])
    )
    case = ReviewCase.from_entry_group([entries[0]], decisions, (), pages)
    written_decisions = json.loads(case.model_dump_json())["decisions"]
    assert all(
        linked_pages == sorted(linked_pages)
        for decision in written_decisions
        for linked_pages in decision["belonging_page_keys"].values()
    )


def test_review_case_from_entry_group_single_entry_writes_unrelated_linked_page_keys_sorted(
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
    case = ReviewCase.from_entry_group([entry], decisions, (), pages)
    [decision] = json.loads(case.model_dump_json())["decisions"]
    written_links = decision["unrelated_pages"]["kitsu:anime:2"]["linked_page_keys"]
    assert written_links == sorted(written_links)


def test_review_case_from_entry_group_keeps_searched_pages(
    build_group_case, linked_page, work_entry
) -> None:
    entries = [
        work_entry("https://myanimelist.net/anime/1"),
        work_entry("https://anilist.co/anime/1"),
    ]
    found = ProviderPage.from_canonical_record(
        "https://animeschedule.net/anime/show",
        {"title": "Show"},
        searched_by_address="https://myanimelist.net/anime/1",
    )
    pages = [linked_page("https://myanimelist.net/anime/1"), found]
    case = build_group_case(entries, pages)
    assert [page.key for page in case.pages] == [
        "https://animeschedule.net/anime/show",
        "mal:anime:1",
    ]


def test_review_case_from_entry_group_keeps_renamed_page(
    build_group_case, work_entry
) -> None:
    old_address = "https://www.anime-planet.com/anime/old"
    entries = [work_entry(old_address), work_entry("https://myanimelist.net/anime/1")]
    renamed = ProviderPage.from_canonical_record(
        old_address,
        {
            "title": "Show",
            "sources": [old_address, "https://www.anime-planet.com/anime/new"],
        },
    )
    case = build_group_case(entries, [renamed])
    assert [page.key for page in case.pages] == ["animeplanet:anime:old"]


def test_review_case_from_entry_group_names_group_case_by_every_page(
    split_case,
) -> None:
    assert (split_case.case_name, split_case.reasons) == (
        "split work: anilist:anime:1 mal:anime:1",
        ("NOT_TIED_BY_LINKS",),
    )
    assert [page.key for page in split_case.pages] == ["mal:anime:1"]


def test_review_case_from_entry_group_keeps_candidate_evidence(
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
    case = ReviewCase.from_entry_group(
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
            {SIGEWINNE: "work 1, main", VARESA: "work 1"},
            {
                SIGEWINNE: PagePlacement("work 1", is_main_page=True),
                VARESA: PagePlacement("work 1"),
            },
        ),
    ],
)
def test_review_case_accepted_answer_placement_for_every_page_gives_placements(
    doubled_case, answer: dict[str, str], placements: dict[str, PagePlacement]
) -> None:
    accepted = doubled_case.model_copy(update={"answer": answer}).accepted_answer
    assert accepted == placements


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
    ("answer", "accepted"),
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
    split_case, answer: object, accepted: dict[str, PagePlacement] | None
) -> None:
    assert split_case.model_copy(update={"answer": answer}).accepted_answer == accepted


def test_review_case_from_linked_page_names_case_by_page_and_linking_entries(
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


def test_review_case_from_linked_page_suggests_part_for_fewer_episodes(
    linked_page_case,
) -> None:
    assert linked_page_case.suggested_answer == {"mal:anime:65144": "add as part"}


@pytest.mark.parametrize(
    ("answer", "accepted"),
    [
        (
            {"mal:anime:65144": "add as part"},
            LinkedPageAddition("anilist:anime:163794", is_part_page=True),
        ),
        ({"mal:anime:65144": "skip"}, "skip"),
        ({"mal:anime:65144": "work 1"}, None),
        ("one work", None),
    ],
)
def test_review_case_accepted_answer_linked_page_reads_addition(
    linked_page_case, answer: object, accepted: object
) -> None:
    assert (
        linked_page_case.model_copy(update={"answer": answer}).accepted_answer
        == accepted
    )


@pytest.mark.parametrize(
    ("case_name", "kind"),
    [
        ("atashinchi_next", ReviewCaseKind.SPLIT_WORK),
        ("gochuumon_movie", ReviewCaseKind.DOUBLED_PROVIDER),
        ("nami", ReviewCaseKind.LINKED_PAGE),
    ],
)
def test_review_case_kind_read_from_case(
    review_cases, case_name: str, kind: ReviewCaseKind
) -> None:
    assert review_cases[case_name].kind is kind


def test_review_case_list_broken_answer_rules_group_answer_missing_page(
    review_cases,
) -> None:
    assert review_cases["atashinchi_next"].list_broken_answer_rules(
        {"mal:anime:63013": "work 1"}
    ) == (PlacementRule.PAGE_WITHOUT_PLACE,)


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
    ) == (LinkedPageAnswerRule.ANSWER_NOT_READABLE,)
