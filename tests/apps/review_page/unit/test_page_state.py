import pytest
from review_page.page_state import (
    CaseEditState,
    CaseFilter,
    CaseRow,
    CaseState,
    SavedChange,
    count_case_rows,
    filter_case_rows,
    find_case_buttons,
    find_case_edit_state,
    find_next_case_name,
    find_page_buttons,
    find_saved_change,
    is_suggestion_complete,
    page_list_rows,
)
from work_matching import ReviewCaseKind

ATASHINCHI_ANSWER = {
    "mal:anime:63013": "work 1",
    "anilist:anime:203146": "work 1",
    "anidb:anime:19895": "work 1",
}


def _row(case_name: str, **fields) -> CaseRow:
    return CaseRow(
        case_name=case_name,
        kind=fields.get("kind", ReviewCaseKind.SPLIT_WORK),
        title=fields.get("title", case_name),
        saved_change=fields.get("saved_change", SavedChange.UNCHANGED),
        is_open=fields.get("is_open", True),
        is_new=fields.get("is_new", False),
        has_suggestion=fields.get("has_suggestion", False),
    )


@pytest.mark.parametrize(
    ("original", "saved", "saved_change"),
    [
        (None, None, SavedChange.UNCHANGED),
        (None, {}, SavedChange.UNCHANGED),
        (None, {"a": "work 1"}, SavedChange.ANSWERED),
        ({"a": "work 1"}, {"a": "work 2"}, SavedChange.CHANGED),
        ({"a": "work 1"}, None, SavedChange.CLEARED),
        ({"a": "work 1"}, {"a": "work 1"}, SavedChange.UNCHANGED),
    ],
)
def test_find_saved_change_compares_saved_answer_with_original(
    original, saved, saved_change: SavedChange
) -> None:
    assert find_saved_change(original, saved) is saved_change


@pytest.mark.parametrize(
    ("unsaved", "is_rejected", "edit_state"),
    [
        ({"a": "work 2"}, False, CaseEditState.EDITED_NOT_SAVED),
        ({"a": "work 1"}, False, CaseEditState.SAVED_APPLIES_ON_RUN),
        (None, False, CaseEditState.SAVED_APPLIES_ON_RUN),
        ({"a": "work 2"}, True, CaseEditState.REJECTED),
    ],
)
def test_find_case_edit_state_saved_change_and_unsaved_edit(
    unsaved, is_rejected: bool, edit_state: CaseEditState
) -> None:
    assert (
        find_case_edit_state(
            original=None,
            saved={"a": "work 1"},
            unsaved=unsaved,
            has_unsaved_edit=unsaved is not None,
            is_rejected=is_rejected,
        )
        is edit_state
    )


def test_find_case_edit_state_saved_back_to_original_unchanged() -> None:
    assert (
        find_case_edit_state(
            original={"a": "work 1"},
            saved={"a": "work 1"},
            unsaved=None,
            has_unsaved_edit=False,
            is_rejected=False,
        )
        is CaseEditState.UNCHANGED
    )


def test_find_case_buttons_save_enabled_after_change(review_cases) -> None:
    buttons = find_case_buttons(
        review_cases["atashinchi_next"],
        original=None,
        saved=None,
        chosen={**ATASHINCHI_ANSWER, "anidb:anime:19895": "work 2"},
        is_run_going=False,
    )
    assert (buttons.is_save_enabled, buttons.is_save_and_next_enabled) == (True, True)


def test_find_case_buttons_untouched_suggestion_disables_save_and_reset(
    review_cases,
) -> None:
    case = review_cases["atashinchi_next"]
    buttons = find_case_buttons(
        case,
        original=None,
        saved=None,
        chosen=dict(case.suggested_answer),
        is_run_going=False,
    )
    assert (
        buttons.is_save_enabled,
        buttons.is_reset_enabled,
        buttons.is_accept_enabled,
    ) == (False, False, True)


def test_find_case_buttons_edited_suggestion_enables_reset(review_cases) -> None:
    buttons = find_case_buttons(
        review_cases["atashinchi_next"],
        original=None,
        saved=None,
        chosen={**ATASHINCHI_ANSWER, "anidb:anime:19895": "work 2"},
        is_run_going=False,
    )
    assert buttons.is_reset_enabled


def test_find_case_buttons_save_disabled_after_changing_back(review_cases) -> None:
    buttons = find_case_buttons(
        review_cases["atashinchi_next"],
        original=None,
        saved=ATASHINCHI_ANSWER,
        chosen=dict(ATASHINCHI_ANSWER),
        is_run_going=False,
    )
    assert (buttons.is_save_enabled, buttons.is_save_and_next_enabled) == (False, False)


def test_find_case_buttons_accept_enabled_for_complete_suggestion(review_cases) -> None:
    case = review_cases["atashinchi_next"]
    buttons = find_case_buttons(
        case, original=None, saved=None, chosen=None, is_run_going=False
    )
    assert is_suggestion_complete(case)
    assert buttons.is_accept_enabled


def test_find_case_buttons_accept_disabled_for_missing_suggestion(review_cases) -> None:
    case = review_cases["aware_meisaku_kun"]
    buttons = find_case_buttons(
        case, original=None, saved=None, chosen=None, is_run_going=False
    )
    assert not is_suggestion_complete(case)
    assert not buttons.is_accept_enabled


def test_find_case_buttons_accept_disabled_for_incomplete_suggestion(
    review_cases,
) -> None:
    case = review_cases["atashinchi_next"].model_copy(
        update={"suggested_answer": {"mal:anime:63013": "work 1"}}
    )
    assert not is_suggestion_complete(case)


def test_find_case_buttons_reset_enabled_for_saved_case(review_cases) -> None:
    buttons = find_case_buttons(
        review_cases["atashinchi_next"],
        original=None,
        saved=ATASHINCHI_ANSWER,
        chosen=ATASHINCHI_ANSWER,
        is_run_going=False,
    )
    assert buttons.is_reset_enabled


def test_find_case_buttons_reset_disabled_at_original(review_cases) -> None:
    buttons = find_case_buttons(
        review_cases["atashinchi_next"],
        original=ATASHINCHI_ANSWER,
        saved=ATASHINCHI_ANSWER,
        chosen=ATASHINCHI_ANSWER,
        is_run_going=False,
    )
    assert not buttons.is_reset_enabled


def test_find_case_buttons_every_button_disabled_while_run_going(review_cases) -> None:
    buttons = find_case_buttons(
        review_cases["atashinchi_next"],
        original=None,
        saved=None,
        chosen=ATASHINCHI_ANSWER,
        is_run_going=True,
    )
    assert not any(
        (
            buttons.is_save_enabled,
            buttons.is_accept_enabled,
            buttons.is_reset_enabled,
            buttons.is_save_and_next_enabled,
        )
    )


def test_find_page_buttons_counts_unsaved_and_saved_changes() -> None:
    buttons = find_page_buttons(
        unsaved_case_names={"a", "b"},
        original_answers={"c": {"x": "work 1"}},
        saved_answers={"c": {"x": "work 2"}, "d": {"x": "work 1"}},
        is_run_going=False,
    )
    assert (
        buttons.unsaved_count,
        buttons.run_count,
        buttons.is_save_all_enabled,
        buttons.is_run_enabled,
    ) == (2, 2, True, False)


def test_find_page_buttons_run_enabled_when_all_saved() -> None:
    buttons = find_page_buttons(
        unsaved_case_names=set(),
        original_answers={},
        saved_answers={"d": {"x": "work 1"}},
        is_run_going=False,
    )
    assert (buttons.is_save_all_enabled, buttons.is_run_enabled) == (False, True)


def test_find_page_buttons_case_saved_back_to_original_not_counted() -> None:
    buttons = find_page_buttons(
        unsaved_case_names=set(),
        original_answers={"c": {"x": "work 1"}},
        saved_answers={"c": {"x": "work 1"}},
        is_run_going=False,
    )
    assert (buttons.run_count, buttons.is_run_enabled) == (0, False)


def test_find_page_buttons_cleared_answer_counted() -> None:
    buttons = find_page_buttons(
        unsaved_case_names=set(),
        original_answers={"c": {"x": "work 1"}},
        saved_answers={},
        is_run_going=False,
    )
    assert buttons.run_count == 1


def test_find_page_buttons_run_going_disables_both() -> None:
    buttons = find_page_buttons(
        unsaved_case_names={"a"},
        original_answers={},
        saved_answers={"d": {"x": "work 1"}},
        is_run_going=True,
    )
    assert (buttons.is_save_all_enabled, buttons.is_run_enabled) == (False, False)


def test_filter_case_rows_kind_and_state_both_apply() -> None:
    rows = [
        _row("a", kind=ReviewCaseKind.SPLIT_WORK, is_new=True),
        _row("b", kind=ReviewCaseKind.LINKED_PAGE, is_new=True),
        _row("c", kind=ReviewCaseKind.SPLIT_WORK),
    ]
    filtered = filter_case_rows(
        rows,
        CaseFilter(
            kinds=frozenset({ReviewCaseKind.SPLIT_WORK}), case_states=frozenset({"new"})
        ),
    )
    assert [row.case_name for row in filtered] == ["a"]


def test_filter_case_rows_edit_state_reads_browser_edits() -> None:
    rows = [_row("a"), _row("b"), _row("c")]
    filtered = filter_case_rows(
        rows,
        CaseFilter(
            edit_states=frozenset({CaseEditState.EDITED_NOT_SAVED}),
            unsaved_case_names=frozenset({"b"}),
        ),
    )
    assert [row.case_name for row in filtered] == ["b"]


def test_filter_case_rows_search_matches_title_and_case_name() -> None:
    rows = [_row("split work: mal:anime:1", title="Nami"), _row("b", title="Other")]
    assert [
        row.case_name for row in filter_case_rows(rows, CaseFilter(search_text="nam"))
    ] == ["split work: mal:anime:1"]
    assert [
        row.title
        for row in filter_case_rows(rows, CaseFilter(search_text="mal:anime:1"))
    ] == ["Nami"]


def test_page_list_rows_gives_page_of_25_and_page_count() -> None:
    rows = [_row(str(number)) for number in range(60)]
    case_page = page_list_rows(rows, page_number=3)
    assert (
        [row.case_name for row in case_page.rows],
        case_page.page_count,
        case_page.first_row_number,
    ) == ([str(number) for number in range(50, 60)], 3, 51)


def test_page_list_rows_page_past_end_gives_last_page() -> None:
    rows = [_row(str(number)) for number in range(30)]
    assert page_list_rows(rows, page_number=9).page_number == 2


def test_find_next_case_name_gives_following_case_in_filtered_list() -> None:
    rows = [_row("a"), _row("b"), _row("c")]
    assert find_next_case_name(rows, "b") == "c"


def test_find_next_case_name_last_case_gives_none() -> None:
    assert find_next_case_name([_row("a"), _row("b")], "b") is None


def test_find_next_case_name_case_left_filter_gives_first_case() -> None:
    assert find_next_case_name([_row("a"), _row("b")], "z") == "a"


def test_count_case_rows_counts_each_filter_value_over_all_rows() -> None:
    rows = [
        _row("a", kind=ReviewCaseKind.SPLIT_WORK, is_new=True),
        _row("b", kind=ReviewCaseKind.LINKED_PAGE, is_open=False),
        _row("c", kind=ReviewCaseKind.SPLIT_WORK, saved_change=SavedChange.ANSWERED),
    ]
    counts = count_case_rows(rows, CaseFilter(unsaved_case_names=frozenset({"a"})))
    assert (
        counts[ReviewCaseKind.SPLIT_WORK.value],
        counts[ReviewCaseKind.LINKED_PAGE.value],
        counts["open"],
        counts["answered"],
        counts["changed"],
        counts["new"],
        counts[CaseEditState.EDITED_NOT_SAVED.value],
        counts[CaseEditState.SAVED_APPLIES_ON_RUN.value],
    ) == (2, 1, 2, 1, 1, 1, 1, 1)


def test_case_state_description_every_state_has_meaning() -> None:
    assert all(state.description for state in CaseState)


def test_case_edit_state_description_every_state_has_meaning() -> None:
    assert all(state.description for state in CaseEditState)
