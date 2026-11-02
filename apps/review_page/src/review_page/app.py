"""The review page's web routes: the page, its fragments, saves and Run.

The browser loads one page, then asks for parts of it (the toolbar, the case
list, the open case, the run panel) as HTML fragments with htmx. Each item
gives the rule, then the reason for it.

- Every request carries the browser's state (``BrowserState``) and reads the
  cases from the source again, so the server holds nothing between requests
  and a restart loses nothing.
- Changing a choice posts it, and the server answers with the redrawn case and
  an ``answerchanged`` event carrying the new answer, which the browser keeps
  as an unsaved edit. The rules decide the choices; the browser never does.
- Saves answer with an ``answerssaved`` event naming the saved and rejected
  cases (with the broken rules) and, for Save & next, the next case, so the
  browser updates its own edits and redraws.
- A save while a run holds the review file answers 409 with a notice, and
  changes nothing. So does a save during which a run removes one of its cases,
  between the source's check and its write; a case removed before the check
  is rejected on its own, like a broken answer.
- Requests from anywhere but the page itself are refused (``request_guards``).
- The server listens where ``ReviewPageConfig`` says, on localhost by default.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, FastAPI, Form, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import JsonValue
from review_page.browser_state import BrowserState
from review_page.case_views import (
    build_case_rows,
    build_case_view,
    find_original_answer,
)
from review_page.correction_routes import CorrectionRoutes
from review_page.correction_views import list_case_correction_links
from review_page.htmx_responses import build_event_response, build_notice_response
from review_page.page_state import (
    CaseEditState,
    CaseRow,
    CaseState,
    count_case_rows,
    filter_case_rows,
    find_next_case_name,
    find_page_buttons,
    find_row_edit_state,
    page_list_rows,
)
from review_page.request_guards import add_request_guards
from review_page.review_case_messages import (
    build_case_answer,
    convert_message_to_case,
)
from review_page.review_case_source import (
    ANSWER_SAVED_ELSEWHERE_RULE_NAME,
    ReviewCaseSource,
)
from review_page.run_routes import RunRoutes
from review_page.run_summary import RunSummary, build_run_summary
from review_page.template_functions import STATIC_DIR, add_template_functions
from work_matching import (
    ReviewCase,
    ReviewCaseKind,
    ReviewFileLockedError,
    SavedAnswerChangedError,
    UnknownCaseError,
    choose_group_page_place,
)

_PACKAGE_DIR = Path(__file__).parent
BrowserStateField = Annotated[str | None, Form()]


def create_app(source: ReviewCaseSource) -> FastAPI:
    """Build the review page's web app over a review case source.

    Args:
        source: Where cases are read, answers saved and runs started.

    Returns:
        The app, with its routes, templates and static files.
    """
    app = FastAPI(title="Review page", docs_url=None, redoc_url=None, openapi_url=None)
    add_request_guards(app)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    templates = Jinja2Templates(
        directory=_PACKAGE_DIR / "templates",
        context_processors=[add_template_functions],
    )
    app.include_router(ReviewPageRoutes(source, templates).router)
    app.include_router(CorrectionRoutes(source, templates).router)
    app.include_router(RunRoutes(source, templates).router)
    return app


class ReviewPageRoutes:
    """The page's routes over one source; the instance keeps no request state."""

    def __init__(self, source: ReviewCaseSource, templates: Jinja2Templates) -> None:
        """Register every route on a router.

        Args:
            source: Where cases are read, answers saved and runs started.
            templates: The page's Jinja templates.
        """
        self._source = source
        self._templates = templates
        self.router = APIRouter()
        self.router.add_api_route("/health", self.get_health, methods=["GET"])
        self.router.add_api_route(
            "/", self.get_page, methods=["GET"], response_class=HTMLResponse
        )
        for path, endpoint in (
            ("/fragments/toolbar", self.render_toolbar),
            ("/fragments/case-list", self.render_case_list),
            ("/fragments/case-panel", self.render_case_panel),
            ("/fragments/choice", self.choose_answer),
            ("/answers", self.save_answers),
        ):
            self.router.add_api_route(path, endpoint, methods=["POST"])

    def get_health(self) -> dict[str, str]:
        """Answer the health check used by Docker now and Kubernetes later."""
        return {"status": "ok"}

    def get_page(self, request: Request) -> Response:
        """Return the page shell; its parts load as fragments."""
        return self._templates.TemplateResponse(
            request,
            "page.html",
            {
                "kinds": list(ReviewCaseKind),
                "case_states": list(CaseState),
                "edit_states": list(CaseEditState),
            },
        )

    def render_toolbar(
        self, request: Request, browser_state: BrowserStateField = None
    ) -> Response:
        """Return the toolbar: Save all and Run, with their counts."""
        state = BrowserState.from_form_field(browser_state)
        cases, run_result = self._load_cases(), self._load_run_result()
        case_names = {case.case_name for case in cases}
        is_run_going = self._source.get_run_state().is_running
        page_buttons = find_page_buttons(
            unsaved_case_names=set(state.edits) & case_names,
            original_answers=_list_original_answers(cases, run_result),
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
        """Return the filter chips with their counts, and one page of the filtered list."""
        state = BrowserState.from_form_field(browser_state)
        cases, run_result = self._load_cases(), self._load_run_result()
        case_filter = state.to_case_filter()
        all_rows = _build_all_case_rows(cases, run_result)
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
        """Return the open case, starting on its edit, saved answer or suggestion."""
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
        """Apply one choice to the open case and return the redrawn case with its new answer."""
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

        Each action names its answers: ``save`` and ``save_next`` the case's
        shown answer, ``accept`` its suggestion, ``reset`` its original value,
        ``save_all`` every unsaved edit.
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
        except ReviewFileLockedError:
            return build_notice_response(
                409, "A run is writing the cases; saves wait until it ends."
            )
        except SavedAnswerChangedError:
            return build_notice_response(
                409,
                "Another tab saved one of these cases while this save ran; "
                "nothing was saved. Save again.",
            )
        except UnknownCaseError:
            return build_notice_response(
                409,
                "A run changed the cases since the page loaded them; "
                "nothing was saved. Reload the page.",
            )
        next_case_name = None
        if action == "save_next" and all(result.is_saved for result in results):
            rows = filter_case_rows(
                _build_all_case_rows(list(cases_by_name.values()), run_result),
                state.to_case_filter(),
            )
            next_case_name = find_next_case_name(rows, case_name)
        return build_event_response(
            {
                "answerssaved": {
                    "saved": [
                        result.case_name for result in results if result.is_saved
                    ],
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
                    "warnings": [
                        warning for result in results for warning in result.warnings
                    ],
                    "next_case_name": next_case_name,
                }
            }
        )

    def _render_case(
        self,
        request: Request,
        state: BrowserState,
        case_name: str | None,
        case: ReviewCase | None,
    ) -> Response:
        """Return the case panel for a case, or the empty panel when there is none."""
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
        """Read every case from the source."""
        return [
            convert_message_to_case(message) for message in self._source.list_cases()
        ]

    def _load_case(self, case_name: str | None) -> ReviewCase | None:
        """Read one case from the source, or None when it is not there."""
        message = self._source.get_case(case_name) if case_name else None
        return None if message is None else convert_message_to_case(message)

    def _load_run_result(self) -> RunSummary | None:
        """Read the last run's result from the source."""
        message = self._source.get_last_run_result()
        return None if message is None else build_run_summary(message)


def _get_shown_answer(case: ReviewCase) -> dict[str, Any] | Any:
    """Return the answer the case shows before any edit: saved, else the suggestion."""
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
    """Return the answers an action saves, by case name; unknown cases are kept to report."""
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


def _list_original_answers(
    cases: list[ReviewCase], run_result: RunSummary | None
) -> dict[str, Any]:
    """Return each case's original value, leaving out cases without one."""
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


def _build_all_case_rows(
    cases: Sequence[ReviewCase],
    run_result: RunSummary | None,
) -> list[CaseRow]:
    """Build every case's list row, marked from the last run's result."""
    if run_result is None:
        return build_case_rows(cases, original_answers=None, new_case_names=())
    return build_case_rows(
        cases,
        original_answers=run_result.original_answers,
        new_case_names=set(run_result.new_case_names),
        unused_answer_case_names=set(run_result.unused_answer_case_names),
    )
