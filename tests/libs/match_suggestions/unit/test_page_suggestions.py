from datetime import UTC, datetime
from pathlib import Path

from match_suggestions.model_pair_judgment import PairJudgment, build_page_key_pair
from match_suggestions.page_suggestions import (
    PageSuggestion,
    list_open_page_keys,
    suggest_page_answers,
    suggest_stored_page_answers,
)
from match_suggestions.pair_judgment_store import JsonlPairJudgmentStore
from match_suggestions.pair_questions import QUESTION_VERSION
from pydantic import JsonValue
from work_matching import ProviderPage, ReviewCase, WorkEntry

MAL = "mal:anime:1"
ANILIST = "anilist:anime:2"
KITSU = "kitsu:anime:3"
OTHER_KITSU = "kitsu:anime:4"
SCHEDULE = "animeschedule:anime:show"

ADDRESSES = {
    MAL: "https://myanimelist.net/anime/1",
    ANILIST: "https://anilist.co/anime/2",
    KITSU: "https://kitsu.app/anime/3",
    OTHER_KITSU: "https://kitsu.app/anime/4",
    SCHEDULE: "https://animeschedule.net/anime/show",
}


def _page(
    page_key: str, day: int = 1, searched_by: tuple[str, ...] = ()
) -> ProviderPage:
    return ProviderPage(
        key=page_key,
        address=ADDRESSES[page_key],
        year=2020,
        aired_from=datetime(2020, 1, day, tzinfo=UTC),
        searched_by=searched_by,
    )


def _case(
    page_keys: list[str],
    suggested_answer: dict[str, str] | None = None,
    answer: JsonValue = None,
    searched_page_keys: tuple[str, ...] = (),
) -> ReviewCase:
    return ReviewCase(
        case_name=f"split work: {' '.join(page_keys)}",
        answer=answer,
        suggested_answer=suggested_answer or {},
        reasons=(),
        entries={
            page_key: WorkEntry(
                title="Show",
                type="TV",
                addresses_by_page_key={page_key: ADDRESSES[page_key]},
            )
            for page_key in page_keys
        },
        decisions=(),
        pages=(
            *(_page(page_key, day) for day, page_key in enumerate(page_keys, start=1)),
            *(_page(page_key, searched_by=(MAL,)) for page_key in searched_page_keys),
        ),
    )


def _judgment(
    page_key: str,
    other_page_key: str,
    is_same_work: float,
    first_part: float = 0.1,
    second_part: float = 0.1,
    question_version: str = QUESTION_VERSION,
) -> tuple[tuple[str, str], PairJudgment]:
    first_page_key, second_page_key = build_page_key_pair(page_key, other_page_key)
    if (first_page_key, second_page_key) != (page_key, other_page_key):
        first_part, second_part = second_part, first_part
    return (first_page_key, second_page_key), PairJudgment(
        first_page_key=first_page_key,
        second_page_key=second_page_key,
        question_version=question_version,
        request_fingerprint="fingerprint",
        is_same_franchise=0.9,
        is_same_work=is_same_work,
        is_first_work_part_of_second_work=first_part,
        is_second_work_part_of_first_work=second_part,
        judged_at=datetime(2026, 11, 23, tzinfo=UTC),
    )


def test_suggest_page_answers_same_work_page_joins_suggested_work() -> None:
    review_case = _case([MAL, ANILIST], {MAL: "work 1"})

    assert suggest_page_answers(review_case, dict([_judgment(ANILIST, MAL, 0.92)])) == (
        PageSuggestion(ANILIST, "work 1", 0.92),
    )


def test_suggest_page_answers_part_page_written_partial_both_ways() -> None:
    review_case = _case([MAL, KITSU], {MAL: "work 1"})

    page_suggestions = suggest_page_answers(
        review_case, dict([_judgment(KITSU, MAL, 0.2, first_part=0.85)])
    )

    assert page_suggestions == (
        PageSuggestion(KITSU, "work 2, partial in work 1", 0.85),
        PageSuggestion(MAL, "work 1, partial in work 2", 0.85),
    )


def test_suggest_page_answers_different_work_page_gets_new_work() -> None:
    review_case = _case([MAL, ANILIST], {MAL: "work 1"})

    assert suggest_page_answers(review_case, dict([_judgment(ANILIST, MAL, 0.1)])) == (
        PageSuggestion(ANILIST, "work 2", 0.9),
    )


def test_suggest_page_answers_open_pages_of_one_new_work_share_work() -> None:
    review_case = _case([MAL, ANILIST, KITSU], {MAL: "work 1"})

    page_suggestions = suggest_page_answers(
        review_case,
        dict(
            [
                _judgment(ANILIST, MAL, 0.1),
                _judgment(KITSU, MAL, 0.2),
                _judgment(ANILIST, KITSU, 0.9),
            ]
        ),
    )

    assert page_suggestions == (
        PageSuggestion(ANILIST, "work 2", 0.9),
        PageSuggestion(KITSU, "work 2", 0.8),
    )


def test_suggest_page_answers_unsettled_page_left_open() -> None:
    review_case = _case([MAL, ANILIST], {MAL: "work 1"})

    assert suggest_page_answers(review_case, dict([_judgment(ANILIST, MAL, 0.5)])) == ()


def test_suggest_page_answers_answered_case_gives_nothing() -> None:
    review_case = _case(
        [MAL, ANILIST], {MAL: "work 1"}, answer={MAL: "work 1", ANILIST: "work 2"}
    )

    assert (
        suggest_page_answers(review_case, dict([_judgment(ANILIST, MAL, 0.95)])) == ()
    )


def test_suggest_page_answers_other_question_version_ignored() -> None:
    review_case = _case([MAL, ANILIST], {MAL: "work 1"})

    assert (
        suggest_page_answers(
            review_case, dict([_judgment(ANILIST, MAL, 0.95, question_version="old")])
        )
        == ()
    )


def test_suggest_page_answers_part_both_ways_left_open() -> None:
    review_case = _case([MAL, KITSU], {MAL: "work 1"})

    assert (
        suggest_page_answers(
            review_case,
            dict([_judgment(KITSU, MAL, 0.2, first_part=0.8, second_part=0.6)]),
        )
        == ()
    )


def test_suggest_page_answers_two_pages_of_one_provider_in_one_work_left_open() -> None:
    review_case = _case([MAL, KITSU, OTHER_KITSU], {MAL: "work 1"})

    page_suggestions = suggest_page_answers(
        review_case,
        dict(
            [
                _judgment(KITSU, MAL, 0.9),
                _judgment(OTHER_KITSU, MAL, 0.9),
                _judgment(KITSU, OTHER_KITSU, 0.5),
            ]
        ),
    )

    assert page_suggestions == ()


def test_suggest_page_answers_without_suggested_work_anchors_most_trusted_page() -> (
    None
):
    review_case = _case([ANILIST, MAL])

    assert suggest_page_answers(review_case, dict([_judgment(ANILIST, MAL, 0.9)])) == (
        PageSuggestion(ANILIST, "work 1", 0.9),
        PageSuggestion(MAL, "work 1", 0.9),
    )


def test_suggest_page_answers_case_without_page_details_gives_nothing() -> None:
    review_case = _case([MAL, ANILIST]).model_copy(update={"pages": ()})

    assert suggest_page_answers(review_case, {}) == ()


def test_list_open_page_keys_leaves_out_suggested_and_searched_pages() -> None:
    review_case = _case(
        [MAL, ANILIST, KITSU], {MAL: "work 1"}, searched_page_keys=(SCHEDULE,)
    )

    assert list_open_page_keys(review_case) == (ANILIST, KITSU)


def test_suggest_stored_page_answers_reads_case_pairs_from_store(
    tmp_path: Path,
) -> None:
    store = JsonlPairJudgmentStore(tmp_path / "page_pair_judgments.jsonl")
    store.add_judgments([_judgment(ANILIST, MAL, 0.92)[1]])

    assert suggest_stored_page_answers(
        _case([MAL, ANILIST], {MAL: "work 1"}), store
    ) == (PageSuggestion(ANILIST, "work 1", 0.92),)


def test_suggest_page_answers_skips_missing_pair_judgments_and_same_provider_different_works() -> (
    None
):
    # Work 1 holds MAL and ANILIST, while KITSU has a judgment only against MAL.
    partial_case = _case([MAL, ANILIST, KITSU], {MAL: "work 1", ANILIST: "work 1"})
    assert suggest_page_answers(partial_case, dict([_judgment(KITSU, MAL, 0.92)])) == (
        PageSuggestion(KITSU, "work 1", 0.92),
    )

    # When KITSU is PARTIAL_IN MAL and work 1 also holds ANILIST with no
    # judgment against KITSU, _has_contradiction_with_work skips the missing pair.
    partial_missing_case = _case(
        [MAL, ANILIST, KITSU],
        {MAL: "work 1", ANILIST: "work 1"},
    )
    assert suggest_page_answers(
        partial_missing_case, dict([_judgment(KITSU, MAL, 0.2, first_part=0.9)])
    ) == (
        PageSuggestion(ANILIST, "work 1, partial in work 2", 0.9),
        PageSuggestion(KITSU, "work 2, partial in work 1", 0.9),
        PageSuggestion(MAL, "work 1, partial in work 2", 0.9),
    )

    # OTHER_KITSU is already in work 1 and Jev says KITSU is DIFFERENT_WORKS
    # even though KITSU looks PARTIAL_IN MAL, and ANILIST has no judgment.
    blocked_case = _case(
        [MAL, ANILIST, KITSU, OTHER_KITSU],
        {MAL: "work 1", ANILIST: "work 1", OTHER_KITSU: "work 1"},
    )
    assert (
        suggest_page_answers(
            blocked_case,
            dict(
                [
                    _judgment(KITSU, MAL, 0.2, first_part=0.9),
                    _judgment(KITSU, OTHER_KITSU, 0.08),
                ]
            ),
        )
        == ()
    )


def test_suggest_page_answers_without_suggested_work_and_no_judgments_returns_empty() -> (
    None
):
    assert suggest_page_answers(_case([ANILIST, MAL]), {}) == ()
