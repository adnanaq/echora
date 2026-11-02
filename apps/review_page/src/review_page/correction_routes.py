"""Serve the Corrections tab: the corrected entries, and one entry's pages.

The routes work like the Cases tab's routes. Each route reads the source on
every request and the browser's state from its form field, and returns an
htmx fragment. The Corrections tab's routes only read, so they never take the
review file's lock.
"""

from typing import Annotated

from fastapi import APIRouter, Form, Request, Response
from fastapi.templating import Jinja2Templates
from review_page.browser_state import BrowserState
from review_page.correction_views import (
    ANSWER_SOURCE,
    RULES_SOURCE,
    UNCHANGED,
    build_corrected_entry_rows,
    build_correction_panel,
    count_corrected_entry_rows,
    filter_corrected_entry_rows,
)
from review_page.page_state import page_list_rows
from review_page.review_case_messages import convert_message_to_case
from review_page.review_case_source import ReviewCaseSource
from review_page.run_summary import build_run_summary
from work_matching import ReviewCase

BrowserStateField = Annotated[str | None, Form()]


class CorrectionRoutes:
    """The Corrections tab's routes over one source; the instance keeps no request state."""

    def __init__(self, source: ReviewCaseSource, templates: Jinja2Templates) -> None:
        """Register the tab's routes on a router.

        Args:
            source: Where corrections, checked entries and cases are read.
            templates: The page's Jinja templates.
        """
        self._source = source
        self._templates = templates
        self.router = APIRouter()
        self.router.add_api_route(
            "/fragments/correction-list", self.render_correction_list, methods=["POST"]
        )
        self.router.add_api_route(
            "/fragments/correction-panel",
            self.render_correction_panel,
            methods=["POST"],
        )

    def render_correction_list(
        self, request: Request, browser_state: BrowserStateField = None
    ) -> Response:
        """Return the change and source chips with their counts, and one page of entries."""
        state = BrowserState.from_form_field(browser_state)
        all_rows = build_corrected_entry_rows(
            self._source.list_corrections(),
            self._source.list_checked_entries(),
            self._load_cases(),
        )
        chip_counts = count_corrected_entry_rows(all_rows)
        return self._templates.TemplateResponse(
            request,
            "correction_list.html",
            {
                "entry_page": page_list_rows(
                    filter_corrected_entry_rows(all_rows, state.to_correction_filter()),
                    state.correction_page_number,
                ),
                "has_corrections": bool(all_rows),
                "chip_counts": chip_counts,
                "change_values": sorted(
                    value
                    for value in chip_counts
                    if value not in (UNCHANGED, ANSWER_SOURCE, RULES_SOURCE)
                )
                + [UNCHANGED],
                "source_values": [ANSWER_SOURCE, RULES_SOURCE],
                "open_entry": state.open_correction_entry,
                "state": state,
            },
        )

    def render_correction_panel(
        self, request: Request, browser_state: BrowserStateField = None
    ) -> Response:
        """Return the open entry's corrected pages, and the cases holding it."""
        state = BrowserState.from_form_field(browser_state)
        run_message = self._source.get_last_run_result()
        panel = (
            build_correction_panel(
                state.open_correction_entry,
                self._source.list_corrections(),
                self._source.list_checked_entries(),
                self._load_cases(),
            )
            if state.open_correction_entry
            else None
        )
        return self._templates.TemplateResponse(
            request,
            "correction_panel.html",
            {
                "panel": panel,
                "run_summary": None
                if run_message is None
                else build_run_summary(run_message),
            },
        )

    def _load_cases(self) -> list[ReviewCase]:
        """Read every case from the source, for titles and links."""
        return [
            convert_message_to_case(message) for message in self._source.list_cases()
        ]
