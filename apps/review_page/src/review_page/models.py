"""Hold data models exchanged across the review page modules and templates.

The functions constructing these models reside within the modules that govern
their business rules: ``page_state``, ``case_views``, ``correction_views``,
and ``run_summary``. Consolidating models in this module exposes the schema
and field contracts in a single location.

Each section groups related models:
- Case states: lifecycle states, persisted states, and answer diff statuses.
- Case list: list row projections, filters, pagination models, and button states.
- Case panel: active case views, entry headers, page rows, and comparison rows.
- Corrections tab: corrected entry rows, correction filters, and page panels.
- Run: pipeline execution summaries and diff baselines.
- Browser state: client-side staged edits and UI filter state.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Annotated

from fastapi import Form
from pydantic import BaseModel, JsonValue, ValidationError
from review_page.tooltips import CASE_STATE_TOOLTIPS, EDIT_STATE_TOOLTIPS
from work_matching import (
    ChoiceOption,
    GroupPageChoices,
    ProviderPage,
    ReviewCase,
    ReviewCaseKind,
    TargetEntrySuggestion,
)

# Case states.


class SavedChange(StrEnum):
    """Track differences between a case's saved answer and its baseline run answer.

    Values denote whether a decision was newly recorded, modified from baseline,
    cleared to empty, or left unchanged relative to the last pipeline run.
    """

    UNCHANGED = "unchanged"
    ANSWERED = "answered"
    CHANGED = "changed"
    CLEARED = "cleared"


class CaseEditState(StrEnum):
    """Represent an answer's lifecycle between browser edits and the next pipeline run.

    Values indicate whether the answer failed rule validation, holds unsaved
    in-memory edits, is saved to disk waiting for the next run, or remains
    identical to the baseline answer.
    """

    REJECTED = "rejected"
    EDITED_NOT_SAVED = "edited, not saved"
    SAVED_APPLIES_ON_RUN = "saved, applies on Run"
    UNCHANGED = "unchanged"

    @property
    def description(self) -> str:
        """Return the explanation text shown in review page tooltips.

        Returns:
            The human-readable description for the edit state.
        """
        return EDIT_STATE_TOOLTIPS[self.value]


class CaseState(StrEnum):
    """Represent the persistence and resolution state of a review case in storage.

    Values indicate whether a case currently awaits an answer, holds a saved
    answer, has pages modified by another review decision, or is newly surfaced.
    """

    OPEN = "open"
    ANSWERED = "answered"
    CHANGED = "changed"
    NEW = "new"

    @property
    def description(self) -> str:
        """Return the explanation text shown in review page tooltips.

        Returns:
            The human-readable description for the case state.
        """
        return CASE_STATE_TOOLTIPS[self.value]


# Case list.


@dataclass(frozen=True, slots=True)
class CaseRow:
    """Represent a summary projection of a review case for the list view.

    Carries unique identifiers, display titles, diff indicators, and state flags
    used for filtering and rendering case list badges.
    """

    case_name: str
    kind: ReviewCaseKind
    title: str
    saved_change: SavedChange
    # Whether the review file currently holds no saved answer for the case.
    is_open: bool
    # Whether the last pipeline run surfaced the case for the first time.
    is_new: bool
    # Whether the case carries an automated suggested answer.
    has_suggestion: bool
    # Whether the review file retains a saved answer even though the last run
    # no longer produced the case (for example, after an upstream answer merged its pages).
    is_answer_unused: bool = False


@dataclass(frozen=True, slots=True)
class CaseFilter:
    """Hold criteria selected in the review page to filter visible review cases.

    Empty filter sets permit every case through without exclusion.
    """

    kinds: frozenset[ReviewCaseKind] = frozenset()
    case_states: frozenset[str] = frozenset()
    edit_states: frozenset[CaseEditState] = frozenset()
    search_text: str = ""
    # The browser's cases with unsaved edits, and with rejected saves.
    unsaved_case_names: frozenset[str] = frozenset()
    rejected_case_names: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class ListPage[Row]:
    """Hold a paginated slice of filtered rows alongside navigation metadata.

    Tracks the active page index, total page count, starting row offset,
    and total matching row count across the active filter.
    """

    rows: list[Row]
    page_number: int
    page_count: int
    # The position of the page's first case in the filtered list, from 1.
    first_row_number: int
    filtered_count: int


@dataclass(frozen=True, slots=True)
class CaseButtons:
    """Track enabled states for action buttons in the active case panel.

    Controls whether Save, Accept suggestion, Reset to baseline, and
    Save & Next buttons are enabled for the current review case.
    """

    is_save_enabled: bool
    is_accept_enabled: bool
    is_reset_enabled: bool
    is_save_and_next_enabled: bool


@dataclass(frozen=True, slots=True)
class PageButtons:
    """Track toolbar execution buttons alongside staged edit and change counts.

    Controls whether the Save All staged edits button and Run pipeline button
    are enabled based on pending answer modifications.
    """

    unsaved_count: int
    run_count: int
    is_save_all_enabled: bool
    is_run_enabled: bool
    run_changes: dict[SavedChange, int] = field(default_factory=dict)


# Case panel.


class DetailColumn(StrEnum):
    """Identify metadata attribute columns shown in candidate page tables.

    Values correspond to extracted anime attributes compared across pages.
    """

    TITLES = "Titles"
    TYPE = "Type"
    EPISODE_COUNT = "Eps"
    AIRED_FROM = "Aired from"
    AIRED_TO = "Aired to"
    LINKS = "Links"


@dataclass(frozen=True, slots=True)
class PageLink:
    """Represent an external hyperlink from a provider page to a related page.

    Holds the target page key and resolved provider web URL when recognized.
    """

    page_key: str
    # None for a key that names no known provider's page; shown as text.
    address: str | None


@dataclass(frozen=True, slots=True)
class PageRowView:
    """Represent a candidate provider page to be placed within a review case.

    Contains extracted anime details, available work placement choices,
    disabling constraint notes, and external navigation links.
    """

    page_key: str
    address: str
    entry_label: str
    is_reference_page: bool
    details: dict[DetailColumn, str]
    choices: GroupPageChoices
    # Whether the selected placement differs from the baseline run answer.
    is_changed: bool
    # Explanations for placement options disabled by multi-page rules on other rows.
    disabled_choice_notes: tuple[str, ...] = ()
    # Whether a rejected save left this required page without a selected placement.
    is_place_missing: bool = False
    # External links to pages referenced by this provider page.
    linked_pages: tuple[PageLink, ...] = ()
    # Candidate target seed entries available when selecting "other work".
    target_entry_options: tuple[TargetEntrySuggestion, ...] = ()


@dataclass(frozen=True, slots=True)
class ComparisonRowView:
    """Represent a side-by-side metadata field comparison for a linked page.

    Compares an attribute value from the existing work entry against the
    corresponding value from the candidate linked page to assist review.
    """

    label: str
    entry_value: str
    page_value: str
    is_different: bool


@dataclass(frozen=True, slots=True)
class EntryView:
    """Represent a work entry header within an entry group case panel.

    Displays work label designations (e.g. Work A, Work B), canonical titles,
    synopsis summaries, and reference page keys.
    """

    label: str
    title: str
    summary: str
    reference_page: str


@dataclass(frozen=True, slots=True)
class CaseView:
    """Aggregate all presentation data required to render an active case panel.

    Combines case metadata, work entries, candidate provider page rows,
    side-by-side attribute comparisons, evidence tables, and action buttons.
    """

    case: ReviewCase
    # The title the case list shows, so the heading matches the list.
    title: str
    edit_state: CaseEditState
    chosen_answer: JsonValue
    is_suggestion_shown: bool
    reason_sentences: tuple[str, ...]
    entries: tuple[EntryView, ...]
    page_rows: tuple[PageRowView, ...]
    shown_columns: tuple[DetailColumn, ...]
    evidence_pages: tuple[ProviderPage, ...]
    linked_page_options: tuple[ChoiceOption, ...]
    comparison_rows: tuple[ComparisonRowView, ...]
    # A linked-page case's entry reference page and linked page, so each key
    # opens its provider page; None for entry group cases.
    linking_entry_link: PageLink | None
    linked_page_link: PageLink | None
    # Each placement word and its meaning, listed under an entry group's pages.
    choice_meanings: tuple[tuple[str, str], ...]
    buttons: CaseButtons
    show_only_differences: bool


# Corrections tab.


@dataclass(frozen=True, slots=True)
class CorrectedEntryRow:
    """Represent an entry modified during the last pipeline run in the list view.

    Holds the entry reference page, resolved title, distinct page change types,
    and indicators of whether corrections stemmed from answers or rules.
    """

    reference_page: str
    title: str
    # Every change the entry's pages make, UNCHANGED included.
    changes: frozenset[str]
    page_count: int
    # The pages that change, leaving out pages that stay as they were.
    changed_page_count: int
    has_answer_correction: bool
    has_rule_correction: bool
    page_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CorrectionFilter:
    """Hold filtering criteria for visible corrected entries in the Corrections tab.

    Filters entries by change types, decision sources, and free text search.
    """

    changes: frozenset[str] = frozenset()
    sources: frozenset[str] = frozenset()
    search_text: str = ""


@dataclass(frozen=True, slots=True)
class CorrectionRowView:
    """Represent one provider page placement within a corrected entry panel.

    Shows page keys, URLs, change descriptions, work destinations, rule reasons,
    and provenance flags indicating whether a change originated from an answer.
    """

    page_key: str
    address: str
    # The change as the run summary names it, or "Unchanged".
    change_label: str
    # The page's work and mark, as written in the answer: "work 1, part".
    place: str
    destination_label: str
    destination_address: str | None
    reason_sentence: str
    is_from_answer: bool


@dataclass(frozen=True, slots=True)
class CorrectionPanelView:
    """Aggregate data to render the detail panel of a corrected work entry.

    Contains entry reference details, individual provider page placement rows,
    and cross-links back to originating review cases.
    """

    reference_page: str
    reference_address: str | None
    title: str
    rows: tuple[CorrectionRowView, ...]
    # Each case holding the entry: its name and its list title.
    case_links: tuple[tuple[str, str], ...] = field(default=())


# Run.


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Aggregate execution statistics and outcome data from the last pipeline run.

    Includes execution timestamps, correction category counts, rejected or
    unused answer identifiers, and baseline answer snapshots for diffing.
    """

    seed_file: str
    started_at: datetime
    finished_at: datetime
    # Counts by name, in the order the run reports them.
    correction_counts: dict[str, int]
    corrected_entry_counts: dict[str, int]
    linked_page_counts: dict[str, int]
    rejected_answer_case_names: tuple[str, ...]
    conflicting_linked_page_keys: tuple[str, ...]
    unused_answer_case_names: tuple[str, ...]
    case_names: tuple[str, ...]
    new_case_names: tuple[str, ...]
    # Every answer as the run wrote it: each case's original value.
    original_answers: dict[str, JsonValue]


# Browser state.

# A route parameter receiving the browser's state as one JSON form field.
BrowserStateField = Annotated[str | None, Form()]


class BrowserState(BaseModel):
    """Represent the reviewer's staged browser state sent with each request.

    The server remains stateless across requests. Session state is retained in
    the browser localStorage via Alpine.js and transmitted with each request
    inside a JSON form field.

    The state tracks:
    - Staged unsaved edits mapped by case name.
    - Base saved answers mapped by case name for optimistic concurrency checks.
    - Rule rejection messages for cases that failed validation on last save.
    - Active UI filters, pagination indices, and open case designations.
    """

    edits: dict[str, JsonValue] = {}
    # The saved answer each edit started from (None for none), by case name.
    seen_saved_answers: dict[str, JsonValue] = {}
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
    def parse_form_field(cls, browser_state_json: str | None) -> BrowserState:
        """Parse the browser state from a submitted JSON form field.

        Args:
            browser_state_json: The submitted ``browser_state`` string, or None.

        Returns:
            The parsed BrowserState model, or an empty default state when absent
            or malformed.
        """
        if not browser_state_json:
            return cls()
        try:
            return cls.model_validate(json.loads(browser_state_json))
        except json.JSONDecodeError, ValidationError:
            return cls()

    def build_correction_filter(self) -> CorrectionFilter:
        """Build the filter criteria for the Corrections tab from staged choices.

        Returns:
            The active CorrectionFilter criteria.
        """
        return CorrectionFilter(
            changes=frozenset(self.correction_changes),
            sources=frozenset(self.correction_sources),
            search_text=self.correction_search_text,
        )

    def build_case_filter(self) -> CaseFilter:
        """Build the filter criteria for the Cases tab from staged choices.

        Returns:
            The active CaseFilter criteria combining kind, case, and edit filters
            alongside staged edit names and search terms.
        """
        return CaseFilter(
            kinds=frozenset(self.kinds),
            case_states=frozenset(state.value for state in self.case_states),
            edit_states=frozenset(self.edit_states),
            search_text=self.search_text,
            unsaved_case_names=frozenset(self.edits),
            rejected_case_names=frozenset(self.rejected),
        )
