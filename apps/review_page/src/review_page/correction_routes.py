"""Serve HTTP routes for the Corrections tab list and entry detail panel.

Reads seed corrections, checked entries, and review cases from the
``ReviewCaseSource`` on each request and renders htmx HTML fragments using the
submitted ``BrowserState`` form field. Because these routes only read data,
they never acquire the review file lock.
"""

from fastapi import APIRouter, Request, Response
from fastapi.templating import Jinja2Templates
from review_page.correction_views import (
    ANSWER_SOURCE,
    RULES_SOURCE,
    UNCHANGED,
    build_corrected_entry_rows,
    build_correction_panel,
    count_corrected_entry_rows,
    filter_corrected_entry_rows,
)
from review_page.models import BrowserState, BrowserStateField
from review_page.page_state import paginate_rows
from review_page.review_case_source import (
    ReviewCaseSource,
    load_cases,
    load_last_run_summary,
)


class CorrectionRoutes:
    """Register and serve stateless HTTP routes for the Corrections tab."""

    def __init__(self, source: ReviewCaseSource, templates: Jinja2Templates) -> None:
        """Register Corrections tab endpoints on a new router instance.

        Args:
            source: The review case source providing corrections, checked entries,
                and review cases.
            templates: The Jinja2 templates collection for rendering HTML fragments.
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
        """Render the Corrections tab filter chips, counts, and paginated entry list.

        Args:
            request: The incoming HTTP request.
            browser_state: The serialized ``BrowserState`` JSON form field.

        Returns:
            The rendered correction list HTML fragment.
        """
        state = BrowserState.parse_form_field(browser_state)
        all_rows = build_corrected_entry_rows(
            self._source.list_corrections(),
            self._source.list_checked_entries(),
            load_cases(self._source),
        )
        chip_counts = count_corrected_entry_rows(all_rows)
        return self._templates.TemplateResponse(
            request,
            "correction_list.html",
            {
                "entry_page": paginate_rows(
                    filter_corrected_entry_rows(
                        all_rows, state.build_correction_filter()
                    ),
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
        """Render the detail panel for the selected corrected entry and its case links.

        Args:
            request: The incoming HTTP request.
            browser_state: The serialized ``BrowserState`` JSON form field.

        Returns:
            The rendered correction panel HTML fragment, or its empty state when
            no entry is selected.
        """
        state = BrowserState.parse_form_field(browser_state)
        panel = (
            build_correction_panel(
                state.open_correction_entry,
                self._source.list_corrections(),
                self._source.list_checked_entries(),
                load_cases(self._source),
            )
            if state.open_correction_entry
            else None
        )
        return self._templates.TemplateResponse(
            request,
            "correction_panel.html",
            {
                "panel": panel,
                "run_summary": load_last_run_summary(self._source),
            },
        )
