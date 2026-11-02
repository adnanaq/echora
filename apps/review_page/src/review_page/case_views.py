"""Turn review cases into what the page's templates show.

The templates only lay out what these views hold. Every decision is made
here or in the libraries, so every decision is tested without a browser. Each
item gives the rule, then the reason for it.

- The case panel's tables (entries, page rows, comparison rows, evidence)
  are built by ``case_panel_tables``; this module assembles them.
- A case starts on its unsaved edit, else its saved answer, else its
  suggestion, which is then marked as not yet answered.
- A page whose place differs from the case's original value is marked
  changed, so the reviewer sees what a save would change.
- Show only differences never hides a page or its choices; the tables hide
  only details every page shares.
- A list row's title is the combined entry's title for an entry group (the
  most trusted entry's), and the linking entry's for a linked page.
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
    find_case_buttons,
    find_case_edit_state,
    find_saved_change,
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
    """Return a case's original value: its answer as the last run wrote it.

    Args:
        case_name: The case.
        saved_answer: The case's answer in the file now.
        original_answers: The last run's answers, or None before any run wrote
            a result, when the file's answers stand as originals.

    Returns:
        The case's original value, or None when the case had no answer.
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
    """Build the case list's rows, in the cases' order.

    Args:
        cases: The review cases.
        original_answers: The last run's answers, or None before any run.
        new_case_names: The cases the last run brought up for the first time.
        unused_answer_case_names: The kept answers whose case the last run no
            longer brought up.

    Returns:
        One row per case, with the case's states and tags.
    """
    return [
        CaseRow(
            case_name=case.case_name,
            kind=case.kind,
            title=find_case_title(case),
            saved_change=find_saved_change(
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
    original: JsonValue,
    unsaved: JsonValue,
    is_rejected: bool,
    is_run_going: bool,
    show_only_differences: bool,
) -> CaseView:
    """Build the case panel's view, starting on the edit, saved answer or suggestion.

    Args:
        case: The review case, with its saved answer.
        original: The case's answer as the last run wrote it.
        unsaved: The browser's unsaved answer, or None.
        is_rejected: Whether the case's last save broke a rule.
        is_run_going: Whether a run is rewriting the cases.
        show_only_differences: Whether to hide details every page shares.

    Returns:
        What the panel shows: the case's pages with their details and choices,
        its buttons and its notes.
    """
    is_suggestion_shown = unsaved is None and case.answer in (None, {})
    chosen_answer = _choose_shown_answer(case, unsaved, is_suggestion_shown)
    is_linked_page = case.kind is ReviewCaseKind.LINKED_PAGE
    page_rows = (
        ()
        if is_linked_page
        else _build_marked_page_rows(case, chosen_answer, original, is_rejected)
    )
    comparison_rows = (
        build_comparison_rows(case, show_only_differences) if is_linked_page else ()
    )
    return CaseView(
        case=case,
        title=find_case_title(case),
        edit_state=find_case_edit_state(
            original=original,
            saved=case.answer,
            unsaved=unsaved,
            has_unsaved_edit=unsaved is not None,
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
        buttons=find_case_buttons(
            case,
            original=original,
            saved=case.answer,
            chosen=chosen_answer,
            is_run_going=is_run_going,
        ),
        show_only_differences=show_only_differences,
    )


def _choose_shown_answer(
    case: ReviewCase, unsaved: JsonValue, is_suggestion_shown: bool
) -> JsonValue:
    """Return the answer the case panel starts on: the edit, else the saved answer or suggestion.

    Args:
        case: The review case, with its saved answer and suggestion.
        unsaved: The browser's unsaved answer, or None.
        is_suggestion_shown: Whether the case has neither an edit nor a saved
            answer, so the panel starts on the suggestion.

    Returns:
        The unsaved answer, a copy of the suggestion, or the saved answer.
    """
    if unsaved is not None:
        return unsaved
    return _copy_suggested_answer(case) if is_suggestion_shown else case.answer


def _build_marked_page_rows(
    case: ReviewCase, chosen_answer: JsonValue, original: JsonValue, is_rejected: bool
) -> tuple[PageRowView, ...]:
    """Build an entry group's page rows, marking places that differ from the start.

    An unanswered case starts from its suggestion, so rows are marked where
    they differ from what the reviewer was first shown, not from no answer.

    Args:
        case: An entry group's case.
        chosen_answer: The answer the case's choices show now.
        original: The case's answer as the last run wrote it.
        is_rejected: Whether the case's last save broke a rule.

    Returns:
        One row per placeable page.
    """
    starting_answer = (
        original
        if original not in (None, {}) or not case.suggested_answer
        else _copy_suggested_answer(case)
    )
    return build_page_rows(
        case, chosen_answer, starting_answer, is_rejected=is_rejected
    )


def find_case_title(case: ReviewCase) -> str:
    """Return the title a case is listed under.

    Args:
        case: The review case.

    Returns:
        The linking entry's title for a linked-page case; the combined entry's
        title for an entry group's case.
    """
    if case.kind is ReviewCaseKind.LINKED_PAGE:
        [decision] = case.decisions
        if isinstance(decision, LinkedPageDecision):
            return decision.linking_entries[0].title
    return combine_group_entries(list(case.entries.values())).title


def _copy_suggested_answer(case: ReviewCase) -> dict[str, JsonValue]:
    """Return the case's suggestion as an answer the reviewer can change.

    Args:
        case: The review case.

    Returns:
        A new dictionary holding the suggestion's places.
    """
    return {page_key: place for page_key, place in case.suggested_answer.items()}


def list_original_answers(
    cases: list[ReviewCase], run_result: RunSummary | None
) -> dict[str, Any]:
    """Return each case's original value, leaving out cases without one.

    Args:
        cases: Every case, with its saved answer.
        run_result: The last run's result, or None before the first run.

    Returns:
        Each case's original value, by case name, for cases that have one.
    """
    original_answers = run_result.original_answers if run_result else None
    return {
        case.case_name: original
        for case in cases
        if (
            original := find_original_answer(
                case.case_name, case.answer, original_answers
            )
        )
    }


def build_all_case_rows(
    cases: Sequence[ReviewCase],
    run_result: RunSummary | None,
) -> list[CaseRow]:
    """Build every case's list row, marked from the last run's result.

    Args:
        cases: Every case.
        run_result: The last run's result, or None before the first run.

    Returns:
        One list row per case, with its new and unused-answer tags.
    """
    if run_result is None:
        return build_case_rows(cases, original_answers=None, new_case_names=())
    return build_case_rows(
        cases,
        original_answers=run_result.original_answers,
        new_case_names=set(run_result.new_case_names),
        unused_answer_case_names=set(run_result.unused_answer_case_names),
    )
