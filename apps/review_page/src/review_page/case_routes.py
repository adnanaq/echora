"""Serve HTTP routes for the Cases tab toolbar, case list, case panel, choices, and saves.

Renders regional HTML fragments for htmx requests and processes interactive
choice updates and answer saves.

Each route rule reads `<rule>: <reason>`:
- Stateless request handling: Every request supplies the client's ``BrowserState``
  and reloads cases from the ``ReviewCaseSource`` so the server retains no
  session state across requests.
- Server-evaluated choices: Selecting a placement option posts to ``/fragments/choice``,
  returning the redrawn case panel alongside an ``answerchanged`` event so
  placement rules are evaluated exclusively on the server.
- Batch and single-case saves: Saving dispatches an ``answerssaved`` event
  listing saved cases, rejected cases with broken rule explanations, and the
  next case identifier when ``save_next`` succeeds.
- Lock contention handling: Save requests attempted while a pipeline run holds
  the review file lock return HTTP 409 with a notice without modifying storage.
- Concurrent case removal handling: Cases removed by a run between validation
  and write return HTTP 409 and leave storage unchanged, whereas cases absent
  before validation are reported as rejected in ``answerssaved``.
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
    build_page_buttons,
    count_applied_answer_rows_by_kind,
    count_case_rows,
    filter_case_rows,
    find_next_case_name,
    paginate_rows,
    resolve_row_edit_state,
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
    OTHER_WORK,
    ReviewCase,
    ReviewCaseKind,
    ReviewFileLockedError,
    SavedAnswerChangedError,
    UnknownCaseError,
    choose_group_page_place,
    resolve_page_key,
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
    """Register and serve stateless HTTP routes for the Cases tab."""

    def __init__(self, source: ReviewCaseSource, templates: Jinja2Templates) -> None:
        """Register Cases tab endpoints on a new router instance.

        Args:
            source: The review case source used to read cases and persist answers.
            templates: The Jinja2 templates collection for rendering HTML fragments.
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
        """Render the top toolbar fragment with Save All and Run button states.

        Args:
            request: The incoming HTTP request.
            browser_state: The serialized ``BrowserState`` JSON form field.

        Returns:
            The rendered toolbar HTML fragment.
        """
        state = BrowserState.parse_form_field(browser_state)
        cases, last_run_summary = self._load_cases(), self._load_last_run_summary()
        case_names = {case.case_name for case in cases}
        is_run_going = self._source.get_run_state().is_running
        page_buttons = build_page_buttons(
            unsaved_case_names=set(state.edits) & case_names,
            original_answers=list_original_answers(cases, last_run_summary),
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
                "run_summary": last_run_summary,
            },
        )

    def render_case_list(
        self, request: Request, browser_state: BrowserStateField = None
    ) -> Response:
        """Render the filter chips with counts and one paginated page of case rows.

        Args:
            request: The incoming HTTP request.
            browser_state: The serialized ``BrowserState`` JSON form field.

        Returns:
            The rendered case list HTML fragment.
        """
        state = BrowserState.parse_form_field(browser_state)
        cases, last_run_summary = self._load_cases(), self._load_last_run_summary()
        case_filter = state.build_case_filter()
        all_rows = build_all_case_rows(cases, last_run_summary)
        case_page = paginate_rows(
            filter_case_rows(all_rows, case_filter), state.page_number
        )
        return self._templates.TemplateResponse(
            request,
            "case_list.html",
            {
                "case_page": case_page,
                "chip_counts": count_case_rows(all_rows, case_filter),
                "applied_answer_counts": count_applied_answer_rows_by_kind(all_rows),
                "kinds": list(ReviewCaseKind),
                "case_states": list(CaseState),
                "edit_states": list(CaseEditState),
                "row_edit_states": {
                    row.case_name: resolve_row_edit_state(row, case_filter)
                    for row in case_page.rows
                },
                "open_case_name": state.open_case_name,
                "state": state,
            },
        )

    def render_case_panel(
        self, request: Request, browser_state: BrowserStateField = None
    ) -> Response:
        """Render the inspection panel for the currently selected review case.

        Args:
            request: The incoming HTTP request.
            browser_state: The serialized ``BrowserState`` JSON form field.

        Returns:
            The rendered case panel HTML fragment, or its empty state when no
            case is selected.
        """
        state = BrowserState.parse_form_field(browser_state)
        return self._render_case(request, state, state.open_case_name, None)

    def choose_answer(
        self,
        request: Request,
        browser_state: BrowserStateField = None,
        page_key: Annotated[str, Form()] = "",
        work: Annotated[str, Form()] = "",
        mark: Annotated[str, Form()] = "",
        linked_answer: Annotated[str, Form()] = "",
        named_entry: Annotated[str, Form()] = "",
    ) -> Response:
        """Apply a placement choice to the open case and return the updated panel.

        Args:
            request: The incoming HTTP request.
            browser_state: The serialized ``BrowserState`` JSON form field.
            page_key: The provider page key whose placement is being updated.
            work: The selected work placement word, or empty for no placement.
            mark: The selected page mark (``main`` or ``part``), or empty.
            linked_answer: The selected decision word for a linked-page case.
            named_entry: The target seed entry page key or address for ``other work``.

        Returns:
            The redrawn case panel HTML fragment with an ``answerchanged`` event
            header carrying the updated answer and baseline saved answer.
        """
        state = BrowserState.parse_form_field(browser_state)
        case = self._load_case(state.open_case_name)
        if case is None:
            return self._render_case(request, state, None, None)
        current_answer = state.edits.get(case.case_name)
        if current_answer is None:
            current_answer = _choose_shown_answer(case)
        if case.kind is ReviewCaseKind.LINKED_PAGE:
            new_answer: dict[str, JsonValue] = (
                {page_key: linked_answer} if linked_answer else {}
            )
        else:
            new_answer = choose_group_page_place(
                current_answer if isinstance(current_answer, dict) else {},
                page_key,
                _write_work(work, named_entry),
                mark,
            )
        # An unanswered case starts on its suggestion, so choosing back to it
        # leaves nothing unsaved; Accept is the button that saves it.
        is_unsaved = (new_answer or None) != (_choose_shown_answer(case) or None)
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
        """Persist answers for a save action and report saved or rejected cases.

        Each action reads ``<action>: <answers persisted>``:
        - ``save`` and ``save_next``: The active answer shown for ``case_name``.
        - ``accept``: The rule-generated suggested answer for ``case_name``.
        - ``reset``: The baseline answer recorded by the last run for ``case_name``.
        - ``save_all``: All staged unsaved edits across all cases in ``browser_state``.

        Args:
            browser_state: The serialized ``BrowserState`` JSON form field.
            action: The save operation name from the list above.
            case_name: The target case identifier for single-case actions.

        Returns:
            A 204 event response carrying ``answerssaved``, or a 409 notice
            response when a concurrent run or save conflict prevents writing.
        """
        state = BrowserState.parse_form_field(browser_state)
        cases_by_name = {case.case_name: case for case in self._load_cases()}
        last_run_summary = self._load_last_run_summary()
        answers_by_case_name = _choose_answers_to_save(
            action, case_name, state, cases_by_name, last_run_summary
        )
        try:
            check_results = self._source.save_answers(
                [
                    build_case_answer(
                        target_case_name,
                        answer,
                        seen_saved_answers=state.seen_saved_answers,
                    )
                    for target_case_name, answer in answers_by_case_name.items()
                ]
            )
        except (
            ReviewFileLockedError,
            SavedAnswerChangedError,
            UnknownCaseError,
        ) as error:
            return build_notice_response(409, _SAVE_REFUSAL_NOTICES[type(error)])
        next_case_name = None
        if action == "save_next" and all(
            check_result.is_saved for check_result in check_results
        ):
            rows = filter_case_rows(
                build_all_case_rows(list(cases_by_name.values()), last_run_summary),
                state.build_case_filter(),
            )
            next_case_name = find_next_case_name(rows, case_name)
        return build_event_response(
            {"answerssaved": _build_answers_saved_event(check_results, next_case_name)}
        )

    def _render_case(
        self,
        request: Request,
        state: BrowserState,
        case_name: str | None,
        case: ReviewCase | None,
    ) -> Response:
        """Render the case panel fragment for a given case or its empty state.

        Args:
            request: The incoming HTTP request.
            state: The parsed ``BrowserState`` from the request.
            case_name: The case identifier to load when ``case`` is None.
            case: A pre-loaded ``ReviewCase`` instance, or None to load by name.

        Returns:
            The rendered ``case_panel.html`` response.
        """
        case = case or self._load_case(case_name)
        if case is None:
            return self._templates.TemplateResponse(
                request, "case_panel.html", {"view": None}
            )
        last_run_summary = self._load_last_run_summary()
        view = build_case_view(
            case,
            original_answer=find_original_answer(
                case.case_name,
                case.answer,
                last_run_summary.original_answers if last_run_summary else None,
            ),
            unsaved_answer=state.edits.get(case.case_name),
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
                "is_answer_unused": last_run_summary is not None
                and case.case_name in last_run_summary.unused_answer_case_names,
            },
        )

    def _load_cases(self) -> list[ReviewCase]:
        """Load all review cases from the source as domain models.

        Returns:
            All review cases converted from protobuf messages to domain models.
        """
        return [
            convert_message_to_case(message) for message in self._source.list_cases()
        ]

    def _load_case(self, case_name: str | None) -> ReviewCase | None:
        """Load a single review case by name from the source.

        Args:
            case_name: The case identifier to look up, or None.

        Returns:
            The converted ``ReviewCase`` domain model, or None when absent.
        """
        message = self._source.get_case(case_name) if case_name else None
        return None if message is None else convert_message_to_case(message)

    def _load_last_run_summary(self) -> RunSummary | None:
        """Read the last pipeline run summary from the source.

        Returns:
            The last run's summary, or None before the first run.
        """
        last_run_result = self._source.get_last_run_result()
        return None if last_run_result is None else build_run_summary(last_run_result)


def _choose_shown_answer(case: ReviewCase) -> dict[str, Any] | Any:
    """Select the initial answer displayed for a case before any browser edits.

    Args:
        case: The review case containing its saved answer and suggested answer.

    Returns:
        The persisted answer when present, or a mutable copy of the suggested
        answer for an unanswered case.
    """
    if case.answer not in (None, {}):
        return case.answer
    return {page_key: place for page_key, place in case.suggested_answer.items()}


def _write_work(work: str, named_entry: str) -> str | None:
    """Format a page's selected work placement string, resolving ``other work`` targets.

    Normalizes pasted provider URLs to their canonical page keys while preserving
    unrecognized text so validation rules can report invalid entry keys.

    Args:
        work: The selected work placement word, or empty for no placement.
        named_entry: The target seed entry page key or provider URL, or empty.

    Returns:
        ``"other work <page_key>"`` when ``other work`` is paired with a target
        entry, the selected work string, or None when ``work`` is empty.
    """
    named_entry_text = named_entry.strip()
    if work == OTHER_WORK and named_entry_text:
        return f"{OTHER_WORK} {resolve_page_key(named_entry_text)}"
    return work or None


def _choose_answers_to_save(
    action: str,
    case_name: str,
    state: BrowserState,
    cases_by_name: dict[str, ReviewCase],
    last_run_summary: RunSummary | None,
) -> dict[str, Any]:
    """Select the mapping of case names to answer payloads for a save action.

    Args:
        action: One of ``save``, ``save_next``, ``accept``, ``reset``, or ``save_all``.
        case_name: The target case identifier for single-case actions.
        state: The parsed ``BrowserState`` holding staged unsaved edits.
        cases_by_name: All stored review cases indexed by case name.
        last_run_summary: The last pipeline run summary, or None before the first run.

    Returns:
        A dictionary mapping target case names to the answer payloads to persist.
        Unknown case names are retained so the source reports them as rejected.
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
                last_run_summary.original_answers if last_run_summary else None,
            )
        }
    answer = state.edits.get(case_name)
    return {case_name: answer if answer is not None else _choose_shown_answer(case)}


def _build_answers_saved_event(
    check_results: Sequence[review_cases_pb2.RuleCheckResult],
    next_case_name: str | None,
) -> dict[str, Any]:
    """Construct the ``answerssaved`` htmx event payload from rule check results.

    Args:
        check_results: Validation results returned by ``ReviewCaseSource.save_answers``.
        next_case_name: The next case identifier to open for ``save_next``, or None.

    Returns:
        A dictionary containing ``saved`` case names, ``rejected`` rule sentences,
        ``saved_elsewhere`` case names, ``warnings``, and ``next_case_name``.
    """
    return {
        "saved": [
            check_result.case_name
            for check_result in check_results
            if check_result.is_saved
        ],
        "rejected": {
            check_result.case_name: list(check_result.broken_rule_sentences)
            for check_result in check_results
            if not check_result.is_saved
        },
        "saved_elsewhere": [
            check_result.case_name
            for check_result in check_results
            if ANSWER_SAVED_ELSEWHERE_RULE_NAME in check_result.broken_rule_names
        ],
        "warnings": [
            warning
            for check_result in check_results
            for warning in check_result.warnings
        ],
        "next_case_name": next_case_name,
    }
