"""Serve pipeline run routes to start runs, render run summaries, and poll run status.

Each route rule reads `<rule>: <reason>`:
- Immediate run tracking: Starting a run triggers the review case source and
  emits a ``runstarted`` event so the browser begins polling run status immediately.
- Run panel rendering: The run panel renders either the in-progress status or
  the completed run summary alongside a ``runstate`` event.
- Lightweight status polling: ``/run-state`` returns only the ``runstate`` event
  without HTML so open browser tabs detect externally started runs and disable
  buttons without collapsing an open summary details element.
"""

from fastapi import APIRouter, Request, Response
from fastapi.templating import Jinja2Templates
from review_page.file_review_case_source import RunSeedFileUnknownError
from review_page.models import BrowserState, BrowserStateField
from review_page.request_guards import (
    build_event_headers,
    build_event_response,
    build_notice_response,
)
from review_page.review_case_source import ReviewCaseSource, load_last_run_summary


class RunRoutes:
    """Register and serve HTTP routes for starting and monitoring pipeline runs."""

    def __init__(self, source: ReviewCaseSource, templates: Jinja2Templates) -> None:
        """Register pipeline run endpoints on a new router instance.

        Args:
            source: The review case source used to start runs and read run summaries.
            templates: The Jinja2 templates collection for rendering HTML fragments.
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
        """Start a seed-cleaning pipeline run if none is active and notify the browser.

        Returns:
            A 204 event response emitting ``runstarted``, or a 409 notice response
            when no seed file path is known.
        """
        try:
            self._source.start_run()
        except RunSeedFileUnknownError as error:
            return build_notice_response(409, str(error))
        return build_event_response({"runstarted": True})

    def report_run_state(self) -> Response:
        """Report whether a pipeline run is currently active via an htmx event header.

        Returns:
            A 204 response carrying the ``runstate`` event payload.
        """
        is_running = self._source.get_run_state().is_running
        return build_event_response({"runstate": {"is_running": is_running}})

    def render_run_panel(
        self, request: Request, browser_state: BrowserStateField = None
    ) -> Response:
        """Render the run status and summary panel fragment with a ``runstate`` header.

        Args:
            request: The incoming HTTP request.
            browser_state: The serialized ``BrowserState`` JSON form field.

        Returns:
            The rendered run panel HTML fragment with a ``runstate`` event header.
        """
        state = BrowserState.parse_form_field(browser_state)
        run_state = self._source.get_run_state()
        case_names = {case.case_name for case in self._source.list_cases()}
        response = self._templates.TemplateResponse(
            request,
            "run_panel.html",
            {
                "run_state": run_state,
                "run_summary": load_last_run_summary(self._source),
                "gone_edit_case_names": sorted(set(state.edits) - case_names),
            },
        )
        response.headers.update(
            build_event_headers({"runstate": {"is_running": run_state.is_running}})
        )
        return response
