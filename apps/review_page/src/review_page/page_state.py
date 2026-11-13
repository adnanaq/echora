"""Evaluate review case lifecycle states, filter matches, and button permissions.

State resolution and button enablement rules evaluate exclusively from review
answer payloads and case definitions, ensuring business rules can be verified
independently of the web browser.

Key domain rules and behaviors include:
- Answer normalization: An answer with no placed pages is treated as None,
  reopening any case whose placements have all been removed.
- Baseline comparison: The baseline answer corresponds to the decision recorded
  by the previous clean-seed run. A saved answer is classified as answered,
  modified, or cleared relative to that baseline. Only cases differing from
  their baseline are staged for pipeline execution.
- Edit state precedence: A case's edit state resolves in strict priority order:
  1. Rejected: The most recent save attempt failed rule validation.
  2. Edited not saved: The reviewer modified placements in the browser without saving.
  3. Saved applies on run: The answer differs from baseline and is saved to disk.
  4. Unchanged: The answer matches the baseline or has no edits.
- Suggestions: An unanswered case displays suggested placements as its starting
  point. An untouched suggestion does not count as an edit, and the Accept
  button explicitly saves it.
- Button enablement conditions:
  - Save / Save & Next: Enabled when current panel choices differ from the saved answer.
  - Accept: Enabled when a complete, valid suggestion exists and differs from the saved answer.
  - Reset: Enabled when chosen or saved answers differ from the baseline run answer.
  - Save All: Enabled on the toolbar whenever any cases contain unsaved browser edits.
  - Run: Enabled only when saved answers differ from baseline and zero unsaved edits remain.
  - In-flight execution: All buttons are disabled while a background pipeline run is active.
- Filtering logic: Selecting multiple chips within one category (such as kinds)
  widens the match with an OR condition, whereas selecting across different
  categories narrows the match with an AND condition.
"""

from collections.abc import Collection, Mapping, Sequence

from pydantic import JsonValue
from review_page.models import (
    CaseButtons,
    CaseEditState,
    CaseFilter,
    CaseRow,
    CaseState,
    ListPage,
    PageButtons,
    SavedChange,
)
from review_page.review_choices import list_required_page_keys
from work_matching import ReviewCase, ReviewCaseKind

ROWS_PER_PAGE = 25


def resolve_saved_change(
    original_answer: JsonValue, saved_answer: JsonValue
) -> SavedChange:
    """Resolve the difference between a case's saved answer and its baseline answer.

    Args:
        original_answer: The baseline answer recorded by the last pipeline run.
        saved_answer: The answer currently persisted in the local review file.

    Returns:
        The comparative change status: UNCHANGED, ANSWERED, CHANGED, or CLEARED.
    """
    normalized_original_answer = _normalize_answer(original_answer)
    normalized_saved_answer = _normalize_answer(saved_answer)
    if normalized_saved_answer == normalized_original_answer:
        return SavedChange.UNCHANGED
    if normalized_original_answer is None:
        return SavedChange.ANSWERED
    return (
        SavedChange.CLEARED if normalized_saved_answer is None else SavedChange.CHANGED
    )


def resolve_case_edit_state(
    *,
    original_answer: JsonValue,
    saved_answer: JsonValue,
    unsaved_answer: JsonValue,
    has_unsaved_edit: bool,
    is_rejected: bool,
) -> CaseEditState:
    """Resolve a case's active edit state according to priority precedence rules.

    Evaluates whether a case failed validation, holds in-memory browser edits,
    carries saved changes awaiting a pipeline run, or matches baseline.

    Args:
        original_answer: The baseline answer from the last pipeline run.
        saved_answer: The answer currently persisted in storage.
        unsaved_answer: The in-memory staged answer in the browser, if any.
        has_unsaved_edit: Whether the browser holds an unpersisted answer edit.
        is_rejected: Whether the last save attempt for this case failed validation.

    Returns:
        The resolved CaseEditState enum member.
    """
    return _resolve_edit_state(
        is_rejected=is_rejected,
        has_unsaved_change=has_unsaved_edit
        and _normalize_answer(unsaved_answer) != _normalize_answer(saved_answer),
        saved_change=resolve_saved_change(original_answer, saved_answer),
    )


def is_suggestion_complete(case: ReviewCase) -> bool:
    """Report whether a case's suggested answer can be saved as it is.

    Verifies that the suggestion places every required provider page and satisfies
    all domain rules so that accepting it will not trigger a rule rejection.

    Args:
        case: The review case carrying the suggested answer to evaluate.

    Returns:
        True when the suggestion contains a complete, valid placement.
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
    return not case.list_broken_answer_rules(build_suggested_answer(case))


def is_answer_empty(answer: JsonValue) -> bool:
    """Check whether an answer places no page.

    Args:
        answer: The answer payload, or None.

    Returns:
        True when the answer is None or an empty mapping.
    """
    return answer in (None, {})


def build_suggested_answer(case: ReviewCase) -> dict[str, JsonValue]:
    """Build an editable answer from a case's suggested placements.

    Args:
        case: The review case carrying the suggested answer.

    Returns:
        A new mapping of page keys to suggested placement words.
    """
    return {page_key: place for page_key, place in case.suggested_answer.items()}


def build_case_buttons(
    case: ReviewCase,
    *,
    original_answer: JsonValue,
    saved_answer: JsonValue,
    chosen_answer: JsonValue,
    is_run_going: bool,
) -> CaseButtons:
    """Build the action button enablement model for the active case panel.

    Args:
        case: The review case under inspection.
        original_answer: The baseline answer from the last pipeline run.
        saved_answer: The answer currently persisted in storage.
        chosen_answer: The placements currently selected in the browser panel.
        is_run_going: Whether a pipeline run is currently in progress.

    Returns:
        A CaseButtons instance with booleans for Save, Save & Next, Accept, and Reset.
    """
    if is_run_going:
        return CaseButtons(False, False, False, False)
    (
        normalized_original_answer,
        normalized_saved_answer,
        normalized_chosen_answer,
    ) = map(_normalize_answer, (original_answer, saved_answer, chosen_answer))
    normalized_suggestion = _normalize_answer(build_suggested_answer(case))
    is_save_enabled = normalized_chosen_answer != (
        normalized_saved_answer
        if normalized_saved_answer is not None
        else normalized_suggestion
    )
    return CaseButtons(
        is_save_enabled=is_save_enabled,
        is_accept_enabled=is_suggestion_complete(case)
        and normalized_suggestion != normalized_saved_answer,
        is_reset_enabled=normalized_saved_answer != normalized_original_answer
        or normalized_chosen_answer
        != (
            normalized_original_answer
            if normalized_original_answer is not None
            else normalized_suggestion
        ),
        is_save_and_next_enabled=is_save_enabled,
    )


def build_page_buttons(
    *,
    unsaved_case_names: Collection[str],
    original_answers: Mapping[str, JsonValue],
    saved_answers: Mapping[str, JsonValue],
    is_run_going: bool,
) -> PageButtons:
    """Build the toolbar button state model and counts for staged and saved changes.

    Args:
        unsaved_case_names: Identifiers of cases with unpersisted browser edits.
        original_answers: Baseline answers recorded by the last pipeline run.
        saved_answers: Answers currently persisted on disk.
        is_run_going: Whether a pipeline run is currently in progress.

    Returns:
        A PageButtons instance with execution counts and button enabled flags.
    """
    run_changes: dict[SavedChange, int] = {}
    for case_name in original_answers.keys() | saved_answers.keys():
        saved_change = resolve_saved_change(
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
    """Filter review case rows against selected chips and search text.

    Args:
        rows: The complete list of case rows to filter.
        case_filter: The active filter criteria including kinds, states, and search.

    Returns:
        The filtered subset of case rows matching all active criteria.
    """
    search_text = case_filter.search_text.strip().casefold()
    return [
        row
        for row in rows
        if (not case_filter.kinds or row.kind in case_filter.kinds)
        and (
            not case_filter.case_states
            or _resolve_case_states(row) & case_filter.case_states
        )
        and (
            not case_filter.edit_states
            or resolve_row_edit_state(row, case_filter) in case_filter.edit_states
        )
        and (
            not search_text
            or search_text in row.title.casefold()
            or search_text in row.case_name.casefold()
        )
    ]


def count_case_rows(rows: Sequence[CaseRow], case_filter: CaseFilter) -> dict[str, int]:
    """Aggregate matching row counts for every filter chip across all case rows.

    Args:
        rows: The full list of case rows before filtering.
        case_filter: The active filter providing unsaved and rejected case names.

    Returns:
        A dictionary mapping each kind, case state, and edit state value to its count.
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
        for case_state in _resolve_case_states(row):
            counts[case_state] += 1
        counts[resolve_row_edit_state(row, case_filter).value] += 1
    return counts


def count_resolved_rows_by_kind(rows: Sequence[CaseRow]) -> dict[str, int]:
    """Count the resolved case rows of each kind.

    A case is resolved when the last run applied its saved answer, or when its
    answer is unused because a seed release or another answer already settled
    its pages. Neither leaves anything to decide.

    Args:
        rows: The full list of case rows before filtering.

    Returns:
        Each kind value mapped to its number of resolved rows; kinds without
        any are mapped to 0.
    """
    resolved_case_counts = dict.fromkeys((kind.value for kind in ReviewCaseKind), 0)
    for row in rows:
        if row.is_answer_applied or row.is_answer_unused:
            resolved_case_counts[row.kind.value] += 1
    return resolved_case_counts


def paginate_rows[Row](rows: Sequence[Row], page_number: int) -> ListPage[Row]:
    """Slice a sequence of rows into a single paginated page with navigation metadata.

    Clamps requested page indices to valid boundaries (between 1 and total pages).

    Args:
        rows: The full sequence of filtered rows to paginate.
        page_number: The 1-based page index requested by the user.

    Returns:
        A ListPage model containing the sliced rows, active page index,
        total page count, starting row offset, and total matching count.
    """
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
    """Locate the identifier of the case following the given case in the filtered list.

    Args:
        rows: The sequence of visible case rows in display order.
        case_name: The identifier of the case just reviewed or saved.

    Returns:
        The identifier of the next case, the first case if the given one is
        no longer visible, or None if at the end of the list or if the list is empty.
    """
    case_names = [row.case_name for row in rows]
    if case_name not in case_names:
        return case_names[0] if case_names else None
    position = case_names.index(case_name)
    return case_names[position + 1] if position + 1 < len(case_names) else None


def resolve_row_edit_state(row: CaseRow, case_filter: CaseFilter) -> CaseEditState:
    """Resolve the edit state for a row using active browser state collections.

    Args:
        row: The case row being classified.
        case_filter: Filter criteria holding browser unsaved and rejected case names.

    Returns:
        The resolved CaseEditState enum member for the row.
    """
    return _resolve_edit_state(
        is_rejected=row.case_name in case_filter.rejected_case_names,
        has_unsaved_change=row.case_name in case_filter.unsaved_case_names,
        saved_change=row.saved_change,
    )


def _resolve_case_states(row: CaseRow) -> set[str]:
    """Resolve all case state filter values that apply to a given case row.

    Args:
        row: The case row being classified.

    Returns:
        A set of state strings matching the row (e.g. 'open', 'answered',
        'changed', 'new', 'previous answer').
    """
    states = {CaseState.OPEN.value if row.is_open else CaseState.ANSWERED.value}
    if row.saved_change is not SavedChange.UNCHANGED:
        states.add(CaseState.CHANGED.value)
    if row.is_new:
        states.add(CaseState.NEW.value)
    if row.has_previous_answer:
        states.add(CaseState.PREVIOUS_ANSWER.value)
    return states


def _resolve_edit_state(
    *, is_rejected: bool, has_unsaved_change: bool, saved_change: SavedChange
) -> CaseEditState:
    """Resolve an edit state from its three conditions, in the order the module lists.

    Args:
        is_rejected: Whether the last save of the case failed validation.
        has_unsaved_change: Whether the browser holds an answer that differs
            from the saved answer.
        saved_change: How the saved answer differs from the original answer.

    Returns:
        The first edit state whose condition holds, or UNCHANGED.
    """
    if is_rejected:
        return CaseEditState.REJECTED
    if has_unsaved_change:
        return CaseEditState.EDITED_NOT_SAVED
    if saved_change is not SavedChange.UNCHANGED:
        return CaseEditState.SAVED_APPLIES_ON_RUN
    return CaseEditState.UNCHANGED


def _normalize_answer(answer: JsonValue) -> JsonValue:
    """Normalize empty answer mappings or None to None to indicate no decision.

    Args:
        answer: The candidate answer payload, or None.

    Returns:
        None if the answer is None or an empty dictionary; otherwise the original payload.
    """
    return None if is_answer_empty(answer) else answer
