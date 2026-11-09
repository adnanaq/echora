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

from collections.abc import Mapping, Sequence

from pydantic import JsonValue
from review_page.case_panel_tables import (
    build_comparison_rows,
    build_entry_views,
    build_linked_page_case_links,
    build_page_rows,
    list_evidence_pages,
    list_shown_columns,
)
from review_page.models import (
    CaseEditState,
    CaseRow,
    CaseView,
    InvalidNamedPageKeyReason,
    PageRowView,
    RunSummary,
    SavedChange,
)
from review_page.page_state import (
    build_case_buttons,
    build_suggested_answer,
    is_answer_empty,
    resolve_case_edit_state,
    resolve_saved_change,
)
from work_matching import (
    ReviewCase,
    ReviewCaseKind,
    WorkPartialSources,
    combine_group_entries,
    describe_case_reasons,
    list_linked_page_options,
    list_work_partial_sources,
)
from work_matching.answer_words import PLACEMENT_WORD_MEANINGS


def find_original_answer(
    case: ReviewCase, last_run_summary: RunSummary | None
) -> JsonValue:
    """Find a case's original answer, which is the answer the last run wrote.

    Before the first run, the saved answer counts as the original answer.

    Args:
        case: The review case with its saved answer.
        last_run_summary: Summary of the last run, or None before the first run.

    Returns:
        The original answer, or None when the last run wrote no answer for the case.
    """
    if last_run_summary is None:
        return case.answer
    return last_run_summary.original_answers.get(case.case_name)


def build_case_rows(
    cases: Sequence[ReviewCase], last_run_summary: RunSummary | None
) -> list[CaseRow]:
    """Build the case list rows, marking each case against the last run.

    Args:
        cases: The review cases to list.
        last_run_summary: Summary of the last run, or None before the first run.

    Returns:
        One CaseRow per case, with its change, open, new, unused-answer,
        applied-answer and previous-answer marks.
    """
    new_case_names = set(last_run_summary.new_case_names) if last_run_summary else set()
    unused_answer_case_names = (
        set(last_run_summary.unused_answer_case_names) if last_run_summary else set()
    )
    rejected_answer_case_names = (
        set(last_run_summary.rejected_answer_case_names) if last_run_summary else set()
    )
    rows: list[CaseRow] = []
    for case in cases:
        saved_change = resolve_saved_change(
            find_original_answer(case, last_run_summary), case.answer
        )
        is_open = is_answer_empty(case.answer)
        is_answer_unused = case.case_name in unused_answer_case_names
        rows.append(
            CaseRow(
                case_name=case.case_name,
                kind=case.kind,
                title=find_case_title(case),
                saved_change=saved_change,
                is_open=is_open,
                is_new=case.case_name in new_case_names,
                has_suggestion=bool(case.suggested_answer),
                is_answer_unused=is_answer_unused,
                has_previous_answer=is_open and case.previous_answer is not None,
                is_answer_applied=last_run_summary is not None
                and not is_open
                and saved_change is SavedChange.UNCHANGED
                and not is_answer_unused
                and case.case_name not in rejected_answer_case_names,
            )
        )
    return rows


def build_case_view(
    case: ReviewCase,
    *,
    original_answer: JsonValue,
    unsaved_answer: JsonValue,
    is_rejected: bool,
    is_run_going: bool,
    show_only_differences: bool,
    invalid_named_page_key_reasons: Mapping[str, InvalidNamedPageKeyReason],
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
        invalid_named_page_key_reasons: Each page key named after ``other work`` that
            cannot be used, mapped to the reason it is invalid, so its row is marked.

    Returns:
        A CaseView model containing assembled tables, candidate rows, and action button states.
    """
    chosen_answer = resolve_shown_answer(case, unsaved_answer)
    is_linked_page = case.kind is ReviewCaseKind.LINKED_PAGE
    linking_entry_link, linked_page_link = build_linked_page_case_links(case)
    page_rows, work_partial_sources = _build_group_case_tables(
        case,
        chosen_answer,
        original_answer,
        is_rejected=is_rejected,
        invalid_named_page_key_reasons=invalid_named_page_key_reasons,
    )
    return CaseView(
        case=case,
        title=find_case_title(case),
        edit_state=_resolve_edit_state(
            case, original_answer, unsaved_answer, is_rejected=is_rejected
        ),
        chosen_answer=chosen_answer,
        is_suggestion_shown=_is_suggestion_shown(case, unsaved_answer),
        reason_sentences=describe_case_reasons(case),
        entries=build_entry_views(case),
        page_rows=page_rows,
        work_partial_sources=work_partial_sources,
        shown_columns=list_shown_columns(page_rows, show_only_differences),
        evidence_pages=list_evidence_pages(case, page_rows),
        linked_page_options=list_linked_page_options(case) if is_linked_page else (),
        linking_entry_link=linking_entry_link,
        linked_page_link=linked_page_link,
        choice_meanings=() if is_linked_page else PLACEMENT_WORD_MEANINGS,
        comparison_rows=build_comparison_rows(case, show_only_differences)
        if is_linked_page
        else (),
        buttons=build_case_buttons(
            case,
            original_answer=original_answer,
            saved_answer=case.answer,
            chosen_answer=chosen_answer,
            is_run_going=is_run_going,
        ),
        show_only_differences=show_only_differences,
    )


def resolve_shown_answer(case: ReviewCase, unsaved_answer: JsonValue) -> JsonValue:
    """Resolve the answer the case panel shows and a choice starts from.

    The panel shows the first of these answers that is present.

    - The unsaved browser edit.
    - The saved answer, when the saved answer places a page.
    - An editable copy of the suggested placements.

    Args:
        case: The review case with its saved answer and suggestion.
        unsaved_answer: The unsaved browser edit, or None when the case is unedited.

    Returns:
        The first answer of the list above that is present.
    """
    if unsaved_answer is not None:
        return unsaved_answer
    if not is_answer_empty(case.answer):
        return case.answer
    return build_suggested_answer(case)


def find_case_title(case: ReviewCase) -> str:
    """Derive the primary display title for a review case.

    Selects the linking entry title for linked page cases and the combined
    canonical title across member entries for split work or entry group cases.

    Args:
        case: The review case whose display title is needed.

    Returns:
        The resolved title string for header and navigation display.
    """
    decision = case.linked_page_decision
    if decision is not None:
        return decision.linking_entries[0].title
    return combine_group_entries(list(case.entries.values())).title


def list_original_answers(
    cases: Sequence[ReviewCase], last_run_summary: RunSummary | None
) -> dict[str, JsonValue]:
    """List the original answer of every case that has one.

    Args:
        cases: The review cases to check for original answers.
        last_run_summary: Summary of the last run, or None before the first run.

    Returns:
        Each case name mapped to its original answer; cases without one are left out.
    """
    return {
        case.case_name: original_answer
        for case in cases
        if (original_answer := find_original_answer(case, last_run_summary))
    }


def _is_suggestion_shown(case: ReviewCase, unsaved_answer: JsonValue) -> bool:
    """Report whether the case panel shows the run's suggestion as the answer.

    Args:
        case: The review case with its saved answer and suggestion.
        unsaved_answer: The staged browser edit for this case, or None.

    Returns:
        True when the case has a suggestion and neither an edit nor a saved
        answer places a page.
    """
    return (
        unsaved_answer is None
        and is_answer_empty(case.answer)
        and bool(case.suggested_answer)
    )


def _resolve_edit_state(
    case: ReviewCase,
    original_answer: JsonValue,
    unsaved_answer: JsonValue,
    *,
    is_rejected: bool,
) -> CaseEditState:
    """Resolve whether a case is unchanged, edited, saved or rejected.

    Args:
        case: The review case with its saved answer.
        original_answer: The baseline answer recorded by the previous pipeline run.
        unsaved_answer: The staged browser edit for this case, or None.
        is_rejected: Whether the last save attempt failed validation rules.

    Returns:
        The case's edit state.
    """
    return resolve_case_edit_state(
        original_answer=original_answer,
        saved_answer=case.answer,
        unsaved_answer=unsaved_answer,
        has_unsaved_edit=unsaved_answer is not None,
        is_rejected=is_rejected,
    )


def _build_group_case_tables(
    case: ReviewCase,
    chosen_answer: JsonValue,
    original_answer: JsonValue,
    *,
    is_rejected: bool,
    invalid_named_page_key_reasons: Mapping[str, InvalidNamedPageKeyReason],
) -> tuple[tuple[PageRowView, ...], tuple[WorkPartialSources, ...]]:
    """Build the pages table and the Works table of an entry group case.

    A linked-page case has neither table, since the case compares one linked
    page with one entry instead of placing pages.

    Args:
        case: The review case to present.
        chosen_answer: The active placement selections shown in the panel.
        original_answer: The baseline answer recorded by the previous pipeline run.
        is_rejected: Whether the last save attempt broke validation rules.
        invalid_named_page_key_reasons: Each invalid page key named after
            ``other work``, mapped to the reason it is invalid.

    Returns:
        The page rows and each work's partial sources; both empty for a
        linked-page case.
    """
    if case.kind is ReviewCaseKind.LINKED_PAGE:
        return (), ()
    page_rows = _build_marked_page_rows(
        case,
        chosen_answer,
        original_answer,
        is_rejected=is_rejected,
        invalid_named_page_key_reasons=invalid_named_page_key_reasons,
    )
    return page_rows, list_work_partial_sources(
        case, chosen_answer if isinstance(chosen_answer, dict) else {}
    )


def _build_marked_page_rows(
    case: ReviewCase,
    chosen_answer: JsonValue,
    original_answer: JsonValue,
    *,
    is_rejected: bool,
    invalid_named_page_key_reasons: Mapping[str, InvalidNamedPageKeyReason],
) -> tuple[PageRowView, ...]:
    """Build candidate page rows for an entry group case, marking altered placements.

    Unanswered cases display suggested placements by default, comparing choices
    against the suggestion rather than an empty baseline.

    Args:
        case: The entry group review case containing candidates to display.
        chosen_answer: The active placement selections shown in the panel.
        original_answer: The baseline answer recorded by the previous pipeline run.
        is_rejected: Whether the last save attempt broke validation rules.
        invalid_named_page_key_reasons: Each invalid page key named after
            ``other work``, mapped to the reason it is invalid.

    Returns:
        A tuple of PageRowView models for placeable candidate pages.
    """
    starting_answer = (
        original_answer
        if not is_answer_empty(original_answer) or not case.suggested_answer
        else build_suggested_answer(case)
    )
    return build_page_rows(
        case,
        chosen_answer,
        starting_answer,
        is_rejected=is_rejected,
        invalid_named_page_key_reasons=invalid_named_page_key_reasons,
    )
