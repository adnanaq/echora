"""Serve Run: start it, show its panel, and tell every open page whether a run is going.

Each item gives the rule, then the reason for it.

- Run starts the source's run and answers with a ``runstarted`` event, so the
  page follows it at once.
- The run panel shows the run going or the last run's results, and sends a
  ``runstate`` event with whether a run is going.
- ``/run-state`` sends only that event, with no HTML. Every open page asks
  for it every few seconds, so a run started elsewhere, such as from the
  terminal or another tab, disables the page's buttons too. Redrawing the
  panel on a timer instead would close its summary while it is read.
"""

from typing import Annotated

from fastapi import APIRouter, Form, Request, Response
from fastapi.templating import Jinja2Templates
from review_page.browser_state import BrowserState
from review_page.file_review_case_source import RunSeedFileUnknownError
from review_page.htmx_responses import (
    build_event_headers,
    build_event_response,
    build_notice_response,
)
from review_page.review_case_source import ReviewCaseSource
from review_page.run_summary import build_run_summary

BrowserStateField = Annotated[str | None, Form()]


class RunRoutes:
    """Run's routes over one source; the instance keeps no request state."""

    def __init__(self, source: ReviewCaseSource, templates: Jinja2Templates) -> None:
        """Register Run's routes on a router.

        Args:
            source: Where runs are started and their state and results read.
            templates: The page's Jinja templates.
        """
        self._source = source
        self._templates = templates
        self.router = APIRouter()
        for path, endpoint in (
            ("/run", self.start_run),
            ("/run-state", self.report_run_state),
            ("/fragments/run-panel", self.render_run_panel),
        ):
            self.router.add_api_route(path, endpoint, methods=["POST"])

    def start_run(self) -> Response:
        """Start a run, unless one is going, and tell the browser to follow it."""
        try:
            self._source.start_run()
        except RunSeedFileUnknownError as error:
            return build_notice_response(409, str(error))
        return build_event_response({"runstarted": True})

    def report_run_state(self) -> Response:
        """Send whether a run is going, as a ``runstate`` event and no content."""
        is_running = self._source.get_run_state().is_running
        return build_event_response({"runstate": {"is_running": is_running}})

    def render_run_panel(
        self, request: Request, browser_state: BrowserStateField = None
    ) -> Response:
        """Return the run panel: the run going, or the last run's results."""
        state = BrowserState.from_form_field(browser_state)
        run_state = self._source.get_run_state()
        run_result_message = self._source.get_last_run_result()
        case_names = {message.case_name for message in self._source.list_cases()}
        response = self._templates.TemplateResponse(
            request,
            "run_panel.html",
            {
                "run_state": run_state,
                "run_result": (
                    None
                    if run_result_message is None
                    else build_run_summary(run_result_message)
                ),
                "gone_edit_case_names": sorted(set(state.edits) - case_names),
            },
        )
        response.headers.update(
            build_event_headers({"runstate": {"is_running": run_state.is_running}})
        )
        return response
