import pytest
from work_matching.models import PreviousAnswer
from work_matching.previous_answers import attach_previous_answers

MAL_PAGE = "https://myanimelist.net/anime/1"
ANILIST_PAGE = "https://anilist.co/anime/2"
ANISEARCH_PAGE = "https://www.anisearch.com/anime/5"
KITSU_PAGE = "https://kitsu.app/anime/3"
SEPARATE_WORKS = {"mal:anime:1": "work 1", "anilist:anime:2": "work 2"}


@pytest.fixture
def build_split_case(linked_page, work_entry, build_group_case):
    def _build_split_case(*entry_addresses: tuple[str, ...], answer=None):
        entries = [work_entry(*addresses) for addresses in entry_addresses]
        pages = [
            linked_page(address)
            for addresses in entry_addresses
            for address in addresses
        ]
        return build_group_case(entries, pages).model_copy(update={"answer": answer})

    return _build_split_case


@pytest.fixture
def previous_case(build_split_case):
    return build_split_case((MAL_PAGE,), (ANILIST_PAGE,), answer=SEPARATE_WORKS)


def test_attach_previous_answers_entry_gaining_page_carries_applied_answer(
    build_split_case, previous_case
) -> None:
    case = build_split_case((MAL_PAGE, ANISEARCH_PAGE), (ANILIST_PAGE,))
    [carried_case] = attach_previous_answers([case], [previous_case])
    assert carried_case.previous_answer == PreviousAnswer(
        case_name=previous_case.case_name,
        answer=SEPARATE_WORKS,
        added_page_keys=("anisearch:anime:5",),
        is_applied=True,
    )
    assert carried_case.answer is None
    assert carried_case.suggested_answer.items() >= SEPARATE_WORKS.items()


def test_attach_previous_answers_merged_entries_carries_answer_not_applied(
    build_split_case, previous_case
) -> None:
    case = build_split_case((MAL_PAGE, ANILIST_PAGE), (KITSU_PAGE,))
    [carried_case] = attach_previous_answers([case], [previous_case])
    assert carried_case.previous_answer is not None
    assert not carried_case.previous_answer.is_applied
    assert carried_case.suggested_answer == case.suggested_answer


def test_attach_previous_answers_invalid_previous_answer_carries_answer_not_applied(
    build_split_case,
) -> None:
    previous_case = build_split_case(
        (MAL_PAGE,), (ANILIST_PAGE,), answer={"mal:anime:1": "maybe"}
    )
    case = build_split_case((MAL_PAGE, ANISEARCH_PAGE), (ANILIST_PAGE,))
    [carried_case] = attach_previous_answers([case], [previous_case])
    assert carried_case.previous_answer is not None
    assert not carried_case.previous_answer.is_applied


def test_attach_previous_answers_previous_case_matching_two_cases_carries_nothing(
    build_split_case, previous_case
) -> None:
    cases = [
        build_split_case((MAL_PAGE,), (KITSU_PAGE,)),
        build_split_case((ANILIST_PAGE,), (ANISEARCH_PAGE,)),
    ]
    assert attach_previous_answers(cases, [previous_case]) == tuple(cases)


def test_attach_previous_answers_case_matching_two_previous_cases_carries_nothing(
    build_split_case, previous_case
) -> None:
    other_previous_case = build_split_case(
        (KITSU_PAGE,), (ANISEARCH_PAGE,), answer={"kitsu:anime:3": "work 1"}
    )
    case = build_split_case((MAL_PAGE, KITSU_PAGE), (ANILIST_PAGE, ANISEARCH_PAGE))
    assert attach_previous_answers([case], [previous_case, other_previous_case]) == (
        case,
    )


def test_attach_previous_answers_case_of_other_kind_carries_answer_not_applied(
    linked_page, work_entry, build_group_case, previous_case
) -> None:
    second_mal_page = "https://myanimelist.net/anime/9"
    doubled_provider_case = build_group_case(
        [work_entry(MAL_PAGE, second_mal_page, ANILIST_PAGE)],
        [
            linked_page(MAL_PAGE),
            linked_page(second_mal_page),
            linked_page(ANILIST_PAGE),
        ],
    )
    [carried_case] = attach_previous_answers([doubled_provider_case], [previous_case])
    assert carried_case.previous_answer is not None
    assert not carried_case.previous_answer.is_applied


def test_attach_previous_answers_added_entry_carries_applied_answer(
    build_split_case, previous_case
) -> None:
    case = build_split_case((MAL_PAGE,), (ANILIST_PAGE,), (ANISEARCH_PAGE,))
    [carried_case] = attach_previous_answers([case], [previous_case])
    assert carried_case.previous_answer == PreviousAnswer(
        case_name=previous_case.case_name,
        answer=SEPARATE_WORKS,
        added_page_keys=("anisearch:anime:5",),
        is_applied=True,
    )


def test_attach_previous_answers_unanswered_previous_case_carries_nothing(
    build_split_case,
) -> None:
    unanswered_case = build_split_case((MAL_PAGE,), (ANILIST_PAGE,))
    case = build_split_case((MAL_PAGE, ANISEARCH_PAGE), (ANILIST_PAGE,))
    assert attach_previous_answers([case], [unanswered_case]) == (case,)
