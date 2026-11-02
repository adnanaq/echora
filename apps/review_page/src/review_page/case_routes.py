"""Serve the Cases tab: the toolbar, the case list, the open case, choices and saves.

The browser asks for each region as an HTML fragment with htmx. Each item
gives the rule, then the reason for it.

- Every request carries the browser's state (``BrowserState``) and reads the
  cases from the source again, so the server holds nothing between requests
  and a restart loses nothing.
- Changing a choice posts it, and the server answers with the redrawn case and
  an ``answerchanged`` event carrying the new answer, which the browser keeps
  as an unsaved edit. The rules decide the choices; the browser never does.
- A save answers with an ``answerssaved`` event naming the saved and rejected
  cases, with the broken rules, so the browser updates its own edits and
  redraws. For Save & next, the event also names the next case.
- A save while a run holds the review file answers 409 with a notice, and
  changes nothing.
- A save during which a run removes one of its cases, between the source's
  check and its write, also answers 409 and changes nothing. A case removed
  before the check is rejected on its own, like a broken answer.
"""

import json
from collections.abc import Sequence
from typing import Annotated, Any, Final

from fastapi import APIRouter, Form, Request, Response
from fastapi.templating import Jinja2Templates
from pydantic import JsonValue
from review_page.case_views import (
    build_all_case_rows,
    build_case_view,
    find_original_answer,
    list_original_answers,
)
from review_page.correction_views import list_case_correction_links
from review_page.htmx_responses import build_event_response, build_notice_response
from review_page.models import (
    BrowserState,
    BrowserStateField,
    CaseEditState,
    CaseState,
    RunSummary,
)
from review_page.page_state import (
    count_case_rows,
    filter_case_rows,
    find_next_case_name,
    find_page_buttons,
    find_row_edit_state,
    page_list_rows,
)
from review_page.review_case_messages import (
    build_case_answer,
    convert_message_to_case,
)
from review_page.review_case_source import (
    ANSWER_SAVED_ELSEWHERE_RULE_NAME,
    ReviewCaseSource,
)
from review_page.run_summary import build_run_summary
from shared_proto.v1 import review_cases_pb2
from work_matching import (
    ReviewCase,
    ReviewCaseKind,
    ReviewFileLockedError,
    SavedAnswerChangedError,
    UnknownCaseError,
    choose_group_page_place,
)

# The notice each save refusal shows; every refusal saves nothing.
_SAVE_REFUSAL_NOTICES: Final[dict[type[Exception], str]] = {
    ReviewFileLockedError: "A run is writing the cases; saves wait until it ends.",
    SavedAnswerChangedError: (
        "Another tab saved one of these cases while this save ran; nothing was "
        "saved. Save again."
    ),
    UnknownCaseError: (
        "A run changed the cases since the page loaded them; nothing was saved. "
        "Reload the page."
    ),
}


class CaseRoutes:
    """The Cases tab's routes over one source; the instance keeps no request state."""

    def __init__(self, source: ReviewCaseSource, templates: Jinja2Templates) -> None:
        """Register the Cases tab's routes on a router.

        Args:
            source: Where cases are read, answers saved and runs started.
            templates: The page's Jinja templates.
        """
        self._source = source
        self._templates = templates
        self.router = APIRouter()
        for path, endpoint in (
            ("/fragments/toolbar", self.render_toolbar),
            ("/fragments/case-list", self.render_case_list),
            ("/fragments/case-panel", self.render_case_panel),
            ("/fragments/choice", self.choose_answer),
            ("/answers", self.save_answers),
        ):
            self.router.add_api_route(path, endpoint, methods=["POST"])

    def render_toolbar(
        self, request: Request, browser_state: BrowserStateField = None
    ) -> Response:
        """Return the toolbar: Save all and Run, with their counts.

        Args:
            request: The request being answered.
            browser_state: The browser's state, as its JSON form field.

        Returns:
            The toolbar fragment.
        """
        state = BrowserState.from_form_field(browser_state)
        cases, run_result = self._load_cases(), self._load_run_result()
        case_names = {case.case_name for case in cases}
        is_run_going = self._source.get_run_state().is_running
        page_buttons = find_page_buttons(
            unsaved_case_names=set(state.edits) & case_names,
            original_answers=list_original_answers(cases, run_result),
            saved_answers={
                case.case_name: case.answer for case in cases if case.answer
            },
            is_run_going=is_run_going,
        )
        return self._templates.TemplateResponse(
            request,
            "toolbar.html",
            {
                "buttons": page_buttons,
                "is_run_going": is_run_going,
                "run_result": run_result,
            },
        )

    def render_case_list(
        self, request: Request, browser_state: BrowserStateField = None
    ) -> Response:
        """Return the filter chips with their counts, and one page of the filtered list.

        Args:
            request: The request being answered.
            browser_state: The browser's state, as its JSON form field.

        Returns:
            The case list fragment.
        """
        state = BrowserState.from_form_field(browser_state)
        cases, run_result = self._load_cases(), self._load_run_result()
        case_filter = state.to_case_filter()
        all_rows = build_all_case_rows(cases, run_result)
        case_page = page_list_rows(
            filter_case_rows(all_rows, case_filter), state.page_number
        )
        return self._templates.TemplateResponse(
            request,
            "case_list.html",
            {
                "case_page": case_page,
                "chip_counts": count_case_rows(all_rows, case_filter),
                "kinds": list(ReviewCaseKind),
                "case_states": list(CaseState),
                "edit_states": list(CaseEditState),
                "row_edit_states": {
                    row.case_name: find_row_edit_state(row, case_filter)
                    for row in case_page.rows
                },
                "open_case_name": state.open_case_name,
                "state": state,
            },
        )

    def render_case_panel(
        self, request: Request, browser_state: BrowserStateField = None
    ) -> Response:
        """Return the open case, starting on its edit, saved answer or suggestion.

        Args:
            request: The request being answered.
            browser_state: The browser's state, as its JSON form field.

        Returns:
            The case panel fragment, or its empty state when no case is open.
        """
        state = BrowserState.from_form_field(browser_state)
        return self._render_case(request, state, state.open_case_name, None)

    def choose_answer(
        self,
        request: Request,
        browser_state: BrowserStateField = None,
        page_key: Annotated[str, Form()] = "",
        work: Annotated[str, Form()] = "",
        mark: Annotated[str, Form()] = "",
        linked_answer: Annotated[str, Form()] = "",
    ) -> Response:
        """Apply one choice to the open case and return the redrawn case with its new answer.

        Args:
            request: The request being answered.
            browser_state: The browser's state, as its JSON form field.
            page_key: The page whose place changes, for an entry group's case.
            work: The work chosen for the page, or empty for no place.
            mark: The mark chosen for the page, or empty for no mark.
            linked_answer: The answer chosen for a linked-page case.

        Returns:
            The redrawn case panel, with an ``answerchanged`` event carrying the
            new answer, whether it is unsaved, and the saved answer the edit
            started from.
        """
        state = BrowserState.from_form_field(browser_state)
        case = self._load_case(state.open_case_name)
        if case is None:
            return self._render_case(request, state, None, None)
        current_answer = state.edits.get(case.case_name)
        if current_answer is None:
            current_answer = _get_shown_answer(case)
        if case.kind is ReviewCaseKind.LINKED_PAGE:
            new_answer: dict[str, JsonValue] = (
                {page_key: linked_answer} if linked_answer else {}
            )
        else:
            new_answer = choose_group_page_place(
                current_answer if isinstance(current_answer, dict) else {},
                page_key,
                work or None,
                mark,
            )
        # An unanswered case starts on its suggestion, so choosing back to it
        # leaves nothing unsaved; Accept is the button that saves it.
        is_unsaved = (new_answer or None) != (_get_shown_answer(case) or None)
        if is_unsaved:
            state.edits[case.case_name] = new_answer
        else:
            state.edits.pop(case.case_name, None)
        state.rejected.pop(case.case_name, None)
        response = self._render_case(request, state, case.case_name, case)
        response.headers["HX-Trigger"] = json.dumps(
            {
                "answerchanged": {
                    "case_name": case.case_name,
                    "answer": new_answer,
                    "is_unsaved": is_unsaved,
                    "saved_answer": case.answer,
                }
            }
        )
        return response

    def save_answers(
        self,
        browser_state: BrowserStateField = None,
        action: Annotated[str, Form()] = "save",
        case_name: Annotated[str, Form()] = "",
    ) -> Response:
        """Save, accept, reset or save all, and report what was saved or rejected.

        Each item reads ``<action>: <answers it saves>``.

        - ``save`` and ``save_next``: the case's shown answer.
        - ``accept``: the case's suggestion.
        - ``reset``: the case's original value.
        - ``save_all``: every unsaved edit.

        Args:
            browser_state: The browser's state, as its JSON form field.
            action: Which answers to save, from the list above.
            case_name: The case the action is for; unused by ``save_all``.

        Returns:
            An ``answerssaved`` event naming the saved, rejected and
            changed-elsewhere cases; a 409 notice when a run holds the review
            file or changed the cases during the save.
        """
        state = BrowserState.from_form_field(browser_state)
        cases_by_name = {case.case_name: case for case in self._load_cases()}
        run_result = self._load_run_result()
        answers_by_case_name = _choose_answers_to_save(
            action, case_name, state, cases_by_name, run_result
        )
        try:
            results = self._source.save_answers(
                [
                    build_case_answer(
                        name, answer, seen_saved_answers=state.seen_saved_answers
                    )
                    for name, answer in answers_by_case_name.items()
                ]
            )
        except (
            ReviewFileLockedError,
            SavedAnswerChangedError,
            UnknownCaseError,
        ) as error:
            return build_notice_response(409, _SAVE_REFUSAL_NOTICES[type(error)])
        next_case_name = None
        if action == "save_next" and all(result.is_saved for result in results):
            rows = filter_case_rows(
                build_all_case_rows(list(cases_by_name.values()), run_result),
                state.to_case_filter(),
            )
            next_case_name = find_next_case_name(rows, case_name)
        return build_event_response(
            {"answerssaved": _build_answers_saved_event(results, next_case_name)}
        )

    def _render_case(
        self,
        request: Request,
        state: BrowserState,
        case_name: str | None,
        case: ReviewCase | None,
    ) -> Response:
        """Return the case panel for a case, or the empty panel when there is none.

        Args:
            request: The request being answered.
            state: The browser's state.
            case_name: The case to show, when ``case`` is not given.
            case: The case already read, or None to read it by name.

        Returns:
            The case panel fragment, or its empty state.
        """
        case = case or self._load_case(case_name)
        if case is None:
            return self._templates.TemplateResponse(
                request, "case_panel.html", {"view": None}
            )
        run_result = self._load_run_result()
        view = build_case_view(
            case,
            original=find_original_answer(
                case.case_name,
                case.answer,
                run_result.original_answers if run_result else None,
            ),
            unsaved=state.edits.get(case.case_name),
            is_rejected=case.case_name in state.rejected,
            is_run_going=self._source.get_run_state().is_running,
            show_only_differences=state.show_only_differences,
        )
        return self._templates.TemplateResponse(
            request,
            "case_panel.html",
            {
                "view": view,
                "rejected_sentences": state.rejected.get(case.case_name, []),
                "correction_links": list_case_correction_links(
                    case, self._source.list_corrections()
                ),
                "is_answer_unused": run_result is not None
                and case.case_name in run_result.unused_answer_case_names,
            },
        )

    def _load_cases(self) -> list[ReviewCase]:
        """Read every case from the source.

        Returns:
            Every case, converted from its message to the library model.
        """
        return [
            convert_message_to_case(message) for message in self._source.list_cases()
        ]

    def _load_case(self, case_name: str | None) -> ReviewCase | None:
        """Read one case from the source, or None when it is not there.

        Args:
            case_name: The case to read, or None.

        Returns:
            The case, or None when no name is given or the source lacks the case.
        """
        message = self._source.get_case(case_name) if case_name else None
        return None if message is None else convert_message_to_case(message)

    def _load_run_result(self) -> RunSummary | None:
        """Read the last run's result from the source.

        Returns:
            The last run's summary, or None before the first run.
        """
        message = self._source.get_last_run_result()
        return None if message is None else build_run_summary(message)


def _get_shown_answer(case: ReviewCase) -> dict[str, Any] | Any:
    """Return the answer the case shows before any edit: saved, else the suggestion.

    Args:
        case: The review case, with its saved answer and suggestion.

    Returns:
        The saved answer, or a copy of the suggestion for an unanswered case.
    """
    if case.answer not in (None, {}):
        return case.answer
    return {page_key: place for page_key, place in case.suggested_answer.items()}


def _choose_answers_to_save(
    action: str,
    case_name: str,
    state: BrowserState,
    cases_by_name: dict[str, ReviewCase],
    run_result: RunSummary | None,
) -> dict[str, Any]:
    """Return the answers an action saves, by case name; unknown cases are kept to report.

    Args:
        action: ``save``, ``save_next``, ``accept``, ``reset`` or ``save_all``.
        case_name: The case the action is for; unused by ``save_all``.
        state: The browser's state, holding the unsaved edits.
        cases_by_name: Every case, by case name.
        run_result: The last run's result, or None before the first run.

    Returns:
        The answers to save, by case name. A case the file no longer holds is
        kept, so the source reports it as rejected.
    """
    if action == "save_all":
        return dict(state.edits)
    case = cases_by_name.get(case_name)
    if case is None:
        return {case_name: state.edits.get(case_name)}
    if action == "accept":
        return {case_name: dict(case.suggested_answer)}
    if action == "reset":
        return {
            case_name: find_original_answer(
                case_name,
                case.answer,
                run_result.original_answers if run_result else None,
            )
        }
    answer = state.edits.get(case_name)
    return {case_name: answer if answer is not None else _get_shown_answer(case)}


def _build_answers_saved_event(
    results: Sequence[review_cases_pb2.RuleCheckResult], next_case_name: str | None
) -> dict[str, Any]:
    """Build the ``answerssaved`` event the browser updates its edits from.

    Args:
        results: The source's result for each answer saved.
        next_case_name: The case Save & next opens, or None.

    Returns:
        The saved cases, the rejected cases with their rule sentences, the
        cases another tab changed, the warnings, and the next case.
    """
    return {
        "saved": [result.case_name for result in results if result.is_saved],
        "rejected": {
            result.case_name: list(result.broken_rule_sentences)
            for result in results
            if not result.is_saved
        },
        "saved_elsewhere": [
            result.case_name
            for result in results
            if ANSWER_SAVED_ELSEWHERE_RULE_NAME in result.broken_rule_names
        ],
        "warnings": [warning for result in results for warning in result.warnings],
        "next_case_name": next_case_name,
    }
