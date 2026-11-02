"""What the browser sends with each request: its unsaved edits, filters and open case.

The server keeps no state between requests, so everything that is the
reviewer's alone lives in the browser (kept in localStorage by Alpine's
persist plugin) and comes with each request as one JSON form field. Each
item gives the rule, then the reason for it.

- Unsaved edits are answers by case name; an edit equal to the saved answer is
  dropped by the browser, so every edit here differs from the file.
- Rejected cases carry the sentences of the rules their last save broke, so
  the list and the panel can show them until the case is edited or saved.
- A malformed field is read as an empty state rather than failing the page,
  since localStorage can hold values an older page version wrote.
"""

import json

from pydantic import BaseModel, JsonValue, ValidationError
from review_page.correction_views import CorrectionFilter
from review_page.page_state import CaseEditState, CaseFilter, CaseState
from work_matching import ReviewCaseKind


class BrowserState(BaseModel):
    """The reviewer's own state, as the browser keeps it."""

    edits: dict[str, JsonValue] = {}
    rejected: dict[str, list[str]] = {}
    kinds: list[ReviewCaseKind] = []
    case_states: list[CaseState] = []
    edit_states: list[CaseEditState] = []
    search_text: str = ""
    page_number: int = 1
    open_case_name: str | None = None
    show_only_differences: bool = False
    # The Corrections tab's own filters, search, page and open entry.
    correction_changes: list[str] = []
    correction_sources: list[str] = []
    correction_search_text: str = ""
    correction_page_number: int = 1
    open_correction_entry: str | None = None

    @classmethod
    def from_form_field(cls, browser_state_json: str | None) -> BrowserState:
        """Read the browser's state from its JSON form field.

        Args:
            browser_state_json: The ``browser_state`` field, or None when absent.

        Returns:
            The state; an empty one when the field is missing or malformed.
        """
        if not browser_state_json:
            return cls()
        try:
            return cls.model_validate(json.loads(browser_state_json))
        except json.JSONDecodeError, ValidationError:
            return cls()

    def to_correction_filter(self) -> CorrectionFilter:
        """Return the Corrections tab's filter from the chosen chips and search."""
        return CorrectionFilter(
            changes=frozenset(self.correction_changes),
            sources=frozenset(self.correction_sources),
            search_text=self.correction_search_text,
        )

    def to_case_filter(self) -> CaseFilter:
        """Return the list filter the state's chosen filters and edits make."""
        return CaseFilter(
            kinds=frozenset(self.kinds),
            case_states=frozenset(state.value for state in self.case_states),
            edit_states=frozenset(self.edit_states),
            search_text=self.search_text,
            unsaved_case_names=frozenset(self.edits),
            rejected_case_names=frozenset(self.rejected),
        )
