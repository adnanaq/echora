"""Assemble template display models for review case navigation and inspection.

Transform domain review cases, baseline answers, and browser edit states into
views consumed by the user interface templates. Every presentation rule is
evaluated in this module and helper libraries so that behavior can be validated
without running a browser engine.

Each rule reads `<rule>: <reason>`:
- Panel assembly: Component tables (entries, page rows, comparisons, evidence)
  are constructed by ``case_panel_tables`` and aggregated into a CaseView here.
- Answer resolution: The case panel prioritizes unpersisted browser edits,
  falling back to persisted storage answers, and finally suggested placements.
- Change highlighting: Candidate pages whose assigned placements differ from
  the baseline answer are flagged as changed to show what saving will alter.
- Metadata filtering: Difference-only mode preserves all rows and choices,
  hiding only metadata columns where all candidates share identical values.
- Title derivation: Group cases use the combined title from the highest priority
  provider, while linked page cases display the linking seed entry title.
"""

from collections.abc import Collection, Mapping, Sequence
from typing import Any

from pydantic import JsonValue
from review_page.case_panel_tables import (
    build_comparison_rows,
    build_entry_views,
    build_page_rows,
    find_linked_page_address,
    list_evidence_pages,
    list_shown_columns,
)
from review_page.models import (
    CaseRow,
    CaseView,
    PageRowView,
    RunSummary,
)
from review_page.page_state import (
    build_case_buttons,
    resolve_case_edit_state,
    resolve_saved_change,
)
from work_matching import (
    LinkedPageDecision,
    ReviewCase,
    ReviewCaseKind,
    combine_group_entries,
    describe_case_reasons,
    list_linked_page_options,
)
from work_matching.answer_words import PLACEMENT_WORD_MEANINGS


def find_original_answer(
    case_name: str,
    saved_answer: JsonValue,
    original_answers: Mapping[str, JsonValue] | None,
) -> JsonValue:
    """Retrieve the baseline answer for a case from previous pipeline runs.

    Falls back to the current saved answer when no prior run summary exists,
    treating the persisted answer as the baseline.

    Args:
        case_name: The unique case identifier.
        saved_answer: The answer currently persisted in review storage.
        original_answers: The mapping of case identifiers to baseline answers
            recorded by the last pipeline execution, or None if no run has occurred.

    Returns:
        The baseline answer value for the case, or None if no answer existed.
    """
    if original_answers is None:
        return saved_answer
    return original_answers.get(case_name)


def build_case_rows(
    cases: Sequence[ReviewCase],
    *,
    original_answers: Mapping[str, JsonValue] | None,
    new_case_names: Collection[str],
    unused_answer_case_names: Collection[str] = (),
) -> list[CaseRow]:
    """Construct summary row display models for a sequence of review cases.

    Determines change status, open/closed state, and novelty tags for each
    case relative to recorded pipeline baseline answers.

    Args:
        cases: The review cases to display in the list.
        original_answers: Mapping of case identifiers to baseline answers from
            the previous pipeline run, or None if no prior run exists.
        new_case_names: Identifiers of review cases introduced in the most recent run.
        unused_answer_case_names: Identifiers of previously saved answers that
            no longer correspond to any active case.

    Returns:
        A list of CaseRow models populated with display metadata and change states.
    """
    return [
        CaseRow(
            case_name=case.case_name,
            kind=case.kind,
            title=find_case_title(case),
            saved_change=resolve_saved_change(
                find_original_answer(case.case_name, case.answer, original_answers),
                case.answer,
            ),
            is_open=case.answer in (None, {}),
            is_new=case.case_name in new_case_names,
            has_suggestion=bool(case.suggested_answer),
            is_answer_unused=case.case_name in unused_answer_case_names,
        )
        for case in cases
    ]


def build_case_view(
    case: ReviewCase,
    *,
    original_answer: JsonValue,
    unsaved_answer: JsonValue,
    is_rejected: bool,
    is_run_going: bool,
    show_only_differences: bool,
) -> CaseView:
    """Assemble the detailed inspection panel view for a selected review case.

    Selects the active answer (staged in-browser edit, persisted answer, or suggestion),
    builds candidate page rows and comparison tables, and computes toolbar button states.

    Args:
        case: The review case to present in the inspection panel.
        original_answer: The baseline answer recorded by the previous pipeline run.
        unsaved_answer: The staged browser edit for this case, if any.
        is_rejected: Whether the last save attempt failed validation rules.
        is_run_going: Whether a pipeline cleaning run is currently in progress.
        show_only_differences: Whether to hide metadata attributes identical across entries.

    Returns:
        A CaseView model containing assembled tables, candidate rows, and action button states.
    """
    is_suggestion_shown = unsaved_answer is None and case.answer in (None, {})
    chosen_answer = _choose_shown_answer(case, unsaved_answer, is_suggestion_shown)
    is_linked_page = case.kind is ReviewCaseKind.LINKED_PAGE
    page_rows = (
        ()
        if is_linked_page
        else _build_marked_page_rows(case, chosen_answer, original_answer, is_rejected)
    )
    comparison_rows = (
        build_comparison_rows(case, show_only_differences) if is_linked_page else ()
    )
    return CaseView(
        case=case,
        title=find_case_title(case),
        edit_state=resolve_case_edit_state(
            original_answer=original_answer,
            saved_answer=case.answer,
            unsaved_answer=unsaved_answer,
            has_unsaved_edit=unsaved_answer is not None,
            is_rejected=is_rejected,
        ),
        chosen_answer=chosen_answer,
        is_suggestion_shown=is_suggestion_shown and bool(case.suggested_answer),
        reason_sentences=describe_case_reasons(case),
        entries=build_entry_views(case),
        page_rows=page_rows,
        shown_columns=list_shown_columns(page_rows, show_only_differences),
        evidence_pages=list_evidence_pages(case, page_rows),
        linked_page_options=list_linked_page_options(case) if is_linked_page else (),
        linked_page_address=find_linked_page_address(case) if is_linked_page else "",
        choice_meanings=() if is_linked_page else PLACEMENT_WORD_MEANINGS,
        comparison_rows=comparison_rows,
        buttons=build_case_buttons(
            case,
            original_answer=original_answer,
            saved_answer=case.answer,
            chosen_answer=chosen_answer,
            is_run_going=is_run_going,
        ),
        show_only_differences=show_only_differences,
    )


def _choose_shown_answer(
    case: ReviewCase, unsaved_answer: JsonValue, is_suggestion_shown: bool
) -> JsonValue:
    """Determine the active answer displayed initially in the case panel.

    Prefers unpersisted browser edits, then a copy of the suggested placements
    when no answer has been saved yet, and otherwise the persisted answer.

    Args:
        case: The review case with its persisted answer and suggestion.
        unsaved_answer: The staged browser edit, or None if unedited.
        is_suggestion_shown: Whether the panel should display suggested placements.

    Returns:
        The staged answer, a mutable copy of suggested placements, or the saved answer.
    """
    if unsaved_answer is not None:
        return unsaved_answer
    return _copy_suggested_answer(case) if is_suggestion_shown else case.answer


def _build_marked_page_rows(
    case: ReviewCase,
    chosen_answer: JsonValue,
    original_answer: JsonValue,
    is_rejected: bool,
) -> tuple[PageRowView, ...]:
    """Build candidate page rows for an entry group case, marking altered placements.

    Unanswered cases display suggested placements by default, comparing choices
    against the suggestion rather than an empty baseline.

    Args:
        case: The entry group review case containing candidates to display.
        chosen_answer: The active placement selections shown in the panel.
        original_answer: The baseline answer recorded by the previous pipeline run.
        is_rejected: Whether the last save attempt broke validation rules.

    Returns:
        A tuple of PageRowView models for placeable candidate pages.
    """
    starting_answer = (
        original_answer
        if original_answer not in (None, {}) or not case.suggested_answer
        else _copy_suggested_answer(case)
    )
    return build_page_rows(
        case, chosen_answer, starting_answer, is_rejected=is_rejected
    )


def find_case_title(case: ReviewCase) -> str:
    """Derive the primary display title for a review case.

    Selects the linking entry title for linked page cases and the combined
    canonical title across member entries for split work or entry group cases.

    Args:
        case: The review case whose display title is needed.

    Returns:
        The resolved title string for header and navigation display.
    """
    if case.kind is ReviewCaseKind.LINKED_PAGE:
        [decision] = case.decisions
        if isinstance(decision, LinkedPageDecision):
            return decision.linking_entries[0].title
    return combine_group_entries(list(case.entries.values())).title


def _copy_suggested_answer(case: ReviewCase) -> dict[str, JsonValue]:
    """Create a mutable copy of a case's suggested placements.

    Args:
        case: The review case providing the suggested answer.

    Returns:
        A dictionary mapping page keys to suggested placement values.
    """
    return {page_key: place for page_key, place in case.suggested_answer.items()}


def list_original_answers(
    cases: list[ReviewCase], last_run_summary: RunSummary | None
) -> dict[str, Any]:
    """Extract baseline answers for all cases that have recorded historical values.

    Args:
        cases: The review cases to check for baseline answers.
        last_run_summary: Summary of the previous pipeline run, or None if no run exists.

    Returns:
        A dictionary mapping case identifiers to their baseline answer values.
    """
    original_answers = last_run_summary.original_answers if last_run_summary else None
    return {
        case.case_name: original_answer
        for case in cases
        if (
            original_answer := find_original_answer(
                case.case_name, case.answer, original_answers
            )
        )
    }


def build_all_case_rows(
    cases: Sequence[ReviewCase],
    last_run_summary: RunSummary | None,
) -> list[CaseRow]:
    """Construct summary rows for all review cases using the latest pipeline run state.

    Args:
        cases: The complete list of review cases.
        last_run_summary: Summary of the previous pipeline run, or None if no run exists.

    Returns:
        A list of CaseRow models with novelty and unused-answer tags applied.
    """
    if last_run_summary is None:
        return build_case_rows(cases, original_answers=None, new_case_names=())
    return build_case_rows(
        cases,
        original_answers=last_run_summary.original_answers,
        new_case_names=set(last_run_summary.new_case_names),
        unused_answer_case_names=set(last_run_summary.unused_answer_case_names),
    )
