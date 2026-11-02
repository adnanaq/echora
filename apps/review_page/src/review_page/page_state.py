"""Decide each case's state and which buttons are enabled, from answers alone.

These rules decide what the review page shows and allows. They are plain
functions of the answers, so they are tested without a browser. Each item
gives the rule, then the reason for it.

- An answer with no pages counts as no answer, so a case whose choices are
  all cleared is open again.
- A case's original value is its answer as the last run wrote it. A saved
  answer is answered, changed or cleared against it, and only such cases
  count for Run: a case saved back to its original changes nothing.
- A case's edit state reads in this order: rejected (its last save broke a
  rule), edited and not saved, saved and applying on Run, unchanged.
- An unanswered case starts on its suggestion, so the suggestion stands in
  for a missing saved or original answer: an untouched suggestion is not an
  edit, and Accept is the button that saves it.
- Save and Save & next are enabled while the chosen answer differs from the
  saved one; Accept while a complete suggestion differs from the saved one;
  Reset while the chosen or saved answer differs from the original.
- Save all is enabled while any case has unsaved edits; Run while saved
  answers differ from their originals and nothing is left unsaved, so a Run
  never misses an edit.
- While a run is going every button is disabled; the run rewrites the cases.
- Filters within one group (kinds, case states, edit states) widen the list;
  filters in different groups narrow it.
"""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import JsonValue
from work_matching import ReviewCase, ReviewCaseKind, list_required_page_keys

ROWS_PER_PAGE = 25


class SavedChange(StrEnum):
    """How a case's saved answer differs from its original value."""

    UNCHANGED = "unchanged"
    ANSWERED = "answered"
    CHANGED = "changed"
    CLEARED = "cleared"


class CaseEditState(StrEnum):
    """Where a case's answer stands between the reviewer's edits and the next Run."""

    REJECTED = "rejected"
    EDITED_NOT_SAVED = "edited, not saved"
    SAVED_APPLIES_ON_RUN = "saved, applies on Run"
    UNCHANGED = "unchanged"

    @property
    def description(self) -> str:
        """Return what the state means, as the page's tooltips show it."""
        return _DESCRIPTION_BY_EDIT_STATE[self]


class CaseState(StrEnum):
    """A case's answer as the file holds it, as the state filter names it."""

    OPEN = "open"
    ANSWERED = "answered"
    CHANGED = "changed"
    NEW = "new"

    @property
    def description(self) -> str:
        """Return what the state means, as the page's tooltips show it."""
        return _DESCRIPTION_BY_CASE_STATE[self]


_DESCRIPTION_BY_EDIT_STATE: dict[CaseEditState, str] = {
    CaseEditState.REJECTED: (
        "The last save broke a rule. The edit is kept in this browser, not saved."
    ),
    CaseEditState.EDITED_NOT_SAVED: (
        "Changed in this browser and not saved yet. Save, or Save all, writes it."
    ),
    CaseEditState.SAVED_APPLIES_ON_RUN: (
        "Saved to the review file. The next Run applies it to the seed."
    ),
    CaseEditState.UNCHANGED: "Nothing edited or saved since the last run.",
}

_DESCRIPTION_BY_CASE_STATE: dict[CaseState, str] = {
    CaseState.OPEN: "The review file holds no answer for the case yet.",
    CaseState.ANSWERED: "The review file holds an answer for the case.",
    CaseState.CHANGED: (
        "The saved answer differs from the answer the last run used, so the "
        "next Run changes the seed."
    ),
    CaseState.NEW: "The last run brought the case up for the first time.",
}


@dataclass(frozen=True, slots=True)
class CaseRow:
    """One case as the case list shows it."""

    case_name: str
    kind: ReviewCaseKind
    title: str
    saved_change: SavedChange
    # Whether the file holds no answer for the case.
    is_open: bool
    # Whether the last run brought the case up for the first time.
    is_new: bool
    # Whether the case carries a suggested answer.
    has_suggestion: bool
    # Whether the file keeps the case's answer although the last run no longer
    # brought the case up, as when another answer changed its pages.
    is_answer_unused: bool = False


@dataclass(frozen=True, slots=True)
class CaseFilter:
    """The filters chosen on the page; an empty group lets every case through."""

    kinds: frozenset[ReviewCaseKind] = frozenset()
    case_states: frozenset[str] = frozenset()
    edit_states: frozenset[CaseEditState] = frozenset()
    search_text: str = ""
    # The browser's cases with unsaved edits, and with rejected saves.
    unsaved_case_names: frozenset[str] = frozenset()
    rejected_case_names: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class ListPage[Row]:
    """One page of a filtered list: cases, or corrected entries."""

    rows: list[Row]
    page_number: int
    page_count: int
    # The position of the page's first case in the filtered list, from 1.
    first_row_number: int
    filtered_count: int


@dataclass(frozen=True, slots=True)
class CaseButtons:
    """Which of one case's buttons are enabled."""

    is_save_enabled: bool
    is_accept_enabled: bool
    is_reset_enabled: bool
    is_save_and_next_enabled: bool


@dataclass(frozen=True, slots=True)
class PageButtons:
    """The page's Save all and Run buttons, with the counts they show."""

    unsaved_count: int
    run_count: int
    is_save_all_enabled: bool
    is_run_enabled: bool
    run_changes: dict[SavedChange, int] = field(default_factory=dict)


def find_saved_change(original: JsonValue, saved: JsonValue) -> SavedChange:
    """Return how a saved answer differs from the case's original value."""
    original, saved = _normalize_answer(original), _normalize_answer(saved)
    if saved == original:
        return SavedChange.UNCHANGED
    if original is None:
        return SavedChange.ANSWERED
    return SavedChange.CLEARED if saved is None else SavedChange.CHANGED


def find_case_edit_state(
    *,
    original: JsonValue,
    saved: JsonValue,
    unsaved: JsonValue,
    has_unsaved_edit: bool,
    is_rejected: bool,
) -> CaseEditState:
    """Return a case's edit state, in the order the module docstring gives.

    Args:
        original: The answer as the last run wrote it.
        saved: The answer in the file now.
        unsaved: The browser's unsaved answer, when it has one.
        has_unsaved_edit: Whether the browser holds an unsaved answer.
        is_rejected: Whether the case's last save broke a rule.
    """
    if is_rejected:
        return CaseEditState.REJECTED
    if has_unsaved_edit and _normalize_answer(unsaved) != _normalize_answer(saved):
        return CaseEditState.EDITED_NOT_SAVED
    if find_saved_change(original, saved) is not SavedChange.UNCHANGED:
        return CaseEditState.SAVED_APPLIES_ON_RUN
    return CaseEditState.UNCHANGED


def is_suggestion_complete(case: ReviewCase) -> bool:
    """Report whether a case's suggested answer can be saved as it is.

    It must place every page the answer must place and follow every rule, so
    Accept never saves an answer the run refuses.
    """
    suggested_answer = case.suggested_answer
    if not suggested_answer:
        return False
    if (
        case.kind is not ReviewCaseKind.LINKED_PAGE
        and not list_required_page_keys(case, suggested_answer)
        <= suggested_answer.keys()
    ):
        return False
    suggested_answer_value: dict[str, JsonValue] = {
        page_key: place for page_key, place in suggested_answer.items()
    }
    return not case.list_broken_answer_rules(suggested_answer_value)


def find_case_buttons(
    case: ReviewCase,
    *,
    original: JsonValue,
    saved: JsonValue,
    chosen: JsonValue,
    is_run_going: bool,
) -> CaseButtons:
    """Return which of a case's buttons are enabled, as the module docstring gives.

    Args:
        case: The review case.
        original: The answer as the last run wrote it.
        saved: The answer in the file now.
        chosen: The answer the case's choices show now.
        is_run_going: Whether a run is rewriting the cases.
    """
    if is_run_going:
        return CaseButtons(False, False, False, False)
    original, saved, chosen = map(_normalize_answer, (original, saved, chosen))
    suggestion: JsonValue = _normalize_answer(
        {page_key: place for page_key, place in case.suggested_answer.items()}
    )
    is_save_enabled = chosen != (saved if saved is not None else suggestion)
    return CaseButtons(
        is_save_enabled=is_save_enabled,
        is_accept_enabled=is_suggestion_complete(case) and suggestion != saved,
        is_reset_enabled=saved != original
        or chosen != (original if original is not None else suggestion),
        is_save_and_next_enabled=is_save_enabled,
    )


def find_page_buttons(
    *,
    unsaved_case_names: Collection[str],
    original_answers: Mapping[str, JsonValue],
    saved_answers: Mapping[str, JsonValue],
    is_run_going: bool,
) -> PageButtons:
    """Return the page's Save all and Run states, with their counts.

    Args:
        unsaved_case_names: The cases with unsaved edits in the browser.
        original_answers: Each answered case's answer as the last run wrote it.
        saved_answers: Each answered case's answer in the file now.
        is_run_going: Whether a run is rewriting the cases.
    """
    run_changes: dict[SavedChange, int] = {}
    for case_name in original_answers.keys() | saved_answers.keys():
        saved_change = find_saved_change(
            original_answers.get(case_name), saved_answers.get(case_name)
        )
        if saved_change is not SavedChange.UNCHANGED:
            run_changes[saved_change] = run_changes.get(saved_change, 0) + 1
    unsaved_count = len(unsaved_case_names)
    run_count = sum(run_changes.values())
    return PageButtons(
        unsaved_count=unsaved_count,
        run_count=run_count,
        is_save_all_enabled=unsaved_count > 0 and not is_run_going,
        is_run_enabled=run_count > 0 and unsaved_count == 0 and not is_run_going,
        run_changes=run_changes,
    )


def filter_case_rows(rows: Sequence[CaseRow], case_filter: CaseFilter) -> list[CaseRow]:
    """Return the rows every chosen filter group lets through, in their order."""
    search_text = case_filter.search_text.strip().casefold()
    return [
        row
        for row in rows
        if (not case_filter.kinds or row.kind in case_filter.kinds)
        and (
            not case_filter.case_states
            or _find_case_states(row) & case_filter.case_states
        )
        and (
            not case_filter.edit_states
            or find_row_edit_state(row, case_filter) in case_filter.edit_states
        )
        and (
            not search_text
            or search_text in row.title.casefold()
            or search_text in row.case_name.casefold()
        )
    ]


def count_case_rows(rows: Sequence[CaseRow], case_filter: CaseFilter) -> dict[str, int]:
    """Count the rows each filter chip would let through, over every row.

    Args:
        rows: Every case's row, before filtering.
        case_filter: The browser's unsaved and rejected cases, for edit states.

    Returns:
        Each kind, case state and edit state value mapped to its row count.
    """
    counts = dict.fromkeys(
        [
            *(kind.value for kind in ReviewCaseKind),
            *(state.value for state in CaseState),
            *(state.value for state in CaseEditState),
        ],
        0,
    )
    for row in rows:
        counts[row.kind.value] += 1
        for case_state in _find_case_states(row):
            counts[case_state] += 1
        counts[find_row_edit_state(row, case_filter).value] += 1
    return counts


def page_list_rows[Row](rows: Sequence[Row], page_number: int) -> ListPage[Row]:
    """Return one page of rows; a page past the end gives the last page."""
    page_count = max(1, -(-len(rows) // ROWS_PER_PAGE))
    page_number = min(max(page_number, 1), page_count)
    first_index = (page_number - 1) * ROWS_PER_PAGE
    return ListPage(
        rows=list(rows[first_index : first_index + ROWS_PER_PAGE]),
        page_number=page_number,
        page_count=page_count,
        first_row_number=first_index + 1,
        filtered_count=len(rows),
    )


def find_next_case_name(rows: Sequence[CaseRow], case_name: str) -> str | None:
    """Return the case after this one in the filtered list.

    Returns:
        The following case; the first case when this one is not in the list;
        None after the last case or for an empty list.
    """
    case_names = [row.case_name for row in rows]
    if case_name not in case_names:
        return case_names[0] if case_names else None
    position = case_names.index(case_name)
    return case_names[position + 1] if position + 1 < len(case_names) else None


def _find_case_states(row: CaseRow) -> set[str]:
    """Return every state filter value a row matches."""
    states = {CaseState.OPEN.value if row.is_open else CaseState.ANSWERED.value}
    if row.saved_change is not SavedChange.UNCHANGED:
        states.add(CaseState.CHANGED.value)
    if row.is_new:
        states.add(CaseState.NEW.value)
    return states


def find_row_edit_state(row: CaseRow, case_filter: CaseFilter) -> CaseEditState:
    """Return a row's edit state from the browser's unsaved and rejected cases."""
    if row.case_name in case_filter.rejected_case_names:
        return CaseEditState.REJECTED
    if row.case_name in case_filter.unsaved_case_names:
        return CaseEditState.EDITED_NOT_SAVED
    if row.saved_change is not SavedChange.UNCHANGED:
        return CaseEditState.SAVED_APPLIES_ON_RUN
    return CaseEditState.UNCHANGED


def _normalize_answer(answer: JsonValue) -> JsonValue:
    """Return None for an answer with no pages, so it counts as no answer."""
    return None if answer in (None, {}) else answer
