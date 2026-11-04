"""Hold every data shape the review page passes between its modules and templates.

The functions that build these models stay in the modules that own the
rules: ``page_state``, ``case_views``, ``correction_views`` and
``run_summary``. Keeping the shapes in one file shows every field the page
reads in one place. Each section reads ``<area>: <what its models hold>``.

- Case states: the edit and case states, and the change a saved answer makes.
- Case list: list rows, filters, paging and the button states.
- Case panel: the open case's entries, page rows and comparison rows.
- Corrections tab: corrected entries, their filter and one entry's pages.
- Run: the last run's summary as the page reads it.
- Browser state: what the browser sends with each request.
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
    """How a case's saved answer differs from its original value."""

    UNCHANGED = "unchanged"
    ANSWERED = "answered"
    CHANGED = "changed"
    CLEARED = "cleared"


class CaseEditState(StrEnum):
    """Where a case's answer stands between the reviewer's edits and the next Run."""

    REJECTED = "rejected"
    EDITED_NOT_SAVED = "edited, not saved"
    SAVED_APPLIES_ON_RUN = "saved, applies on Run"
    UNCHANGED = "unchanged"

    @property
    def description(self) -> str:
        """Return what the state means, as the page's tooltips show it."""
        return EDIT_STATE_TOOLTIPS[self.value]


class CaseState(StrEnum):
    """A case's answer as the file holds it, as the state filter names it."""

    OPEN = "open"
    ANSWERED = "answered"
    CHANGED = "changed"
    NEW = "new"

    @property
    def description(self) -> str:
        """Return what the state means, as the page's tooltips show it."""
        return CASE_STATE_TOOLTIPS[self.value]


# Case list.


@dataclass(frozen=True, slots=True)
class CaseRow:
    """One case as the case list shows it."""

    case_name: str
    kind: ReviewCaseKind
    title: str
    saved_change: SavedChange
    # Whether the file holds no answer for the case.
    is_open: bool
    # Whether the last run brought the case up for the first time.
    is_new: bool
    # Whether the case carries a suggested answer.
    has_suggestion: bool
    # Whether the file keeps the case's answer although the last run no longer
    # brought the case up, as when another answer changed its pages.
    is_answer_unused: bool = False


@dataclass(frozen=True, slots=True)
class CaseFilter:
    """The filters chosen on the page; an empty group lets every case through."""

    kinds: frozenset[ReviewCaseKind] = frozenset()
    case_states: frozenset[str] = frozenset()
    edit_states: frozenset[CaseEditState] = frozenset()
    search_text: str = ""
    # The browser's cases with unsaved edits, and with rejected saves.
    unsaved_case_names: frozenset[str] = frozenset()
    rejected_case_names: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class ListPage[Row]:
    """One page of a filtered list: cases, or corrected entries."""

    rows: list[Row]
    page_number: int
    page_count: int
    # The position of the page's first case in the filtered list, from 1.
    first_row_number: int
    filtered_count: int


@dataclass(frozen=True, slots=True)
class CaseButtons:
    """Which of one case's buttons are enabled."""

    is_save_enabled: bool
    is_accept_enabled: bool
    is_reset_enabled: bool
    is_save_and_next_enabled: bool


@dataclass(frozen=True, slots=True)
class PageButtons:
    """The page's Save all and Run buttons, with the counts they show."""

    unsaved_count: int
    run_count: int
    is_save_all_enabled: bool
    is_run_enabled: bool
    run_changes: dict[SavedChange, int] = field(default_factory=dict)


# Case panel.


class DetailColumn(StrEnum):
    """A page detail shown in an entry group's page table, by its heading."""

    TITLES = "Titles"
    TYPE = "Type"
    EPISODE_COUNT = "Eps"
    AIRED_FROM = "Aired from"
    AIRED_TO = "Aired to"
    LINKS = "Links"


@dataclass(frozen=True, slots=True)
class PageLink:
    """A page key a page links to, with the provider address it opens."""

    page_key: str
    # None for a key that names no known provider's page; shown as text.
    address: str | None


@dataclass(frozen=True, slots=True)
class PageRowView:
    """One page of an entry group's case, with its details and choices."""

    page_key: str
    address: str
    entry_label: str
    is_reference_page: bool
    details: dict[DetailColumn, str]
    choices: GroupPageChoices
    # Whether the page's place differs from the case's original value.
    is_changed: bool
    # Why choices that other pages' places disable are not offered.
    disabled_choice_notes: tuple[str, ...] = ()
    # Whether a rejected save left this page, which the answer must place,
    # without a place, so the reviewer sees which page the rule is about.
    is_place_missing: bool = False
    # The pages this page links to, so the Links column opens each one.
    linked_pages: tuple[PageLink, ...] = ()
    # The suggested seed entries ``other work`` can name for this page.
    target_entry_options: tuple[TargetEntrySuggestion, ...] = ()


@dataclass(frozen=True, slots=True)
class ComparisonRowView:
    """One detail of a linked-page case: the entry's value against the page's."""

    label: str
    entry_value: str
    page_value: str
    is_different: bool


@dataclass(frozen=True, slots=True)
class EntryView:
    """One entry of a case, as its heading shows it."""

    label: str
    title: str
    summary: str
    reference_page: str


@dataclass(frozen=True, slots=True)
class CaseView:
    """Everything the case panel shows for one case."""

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
    # The linked page's address for a linked-page case, so its key opens it.
    linked_page_address: str
    # Each placement word and its meaning, listed under an entry group's pages.
    choice_meanings: tuple[tuple[str, str], ...]
    buttons: CaseButtons
    show_only_differences: bool


# Corrections tab.


@dataclass(frozen=True, slots=True)
class CorrectedEntryRow:
    """One corrected entry as the Corrections list shows it."""

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
    """The Corrections tab's filters; an empty group lets every entry through."""

    changes: frozenset[str] = frozenset()
    sources: frozenset[str] = frozenset()
    search_text: str = ""


@dataclass(frozen=True, slots=True)
class CorrectionRowView:
    """One page of a corrected entry, as the entry's table shows it."""

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
    """One corrected entry, as the Corrections panel shows it."""

    reference_page: str
    reference_address: str | None
    title: str
    rows: tuple[CorrectionRowView, ...]
    # Each case holding the entry: its name and its list title.
    case_links: tuple[tuple[str, str], ...] = field(default=())


# Run.


@dataclass(frozen=True, slots=True)
class RunSummary:
    """What the last run did, as the run panel and the case states read it."""

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
    """The reviewer's own state, as the browser keeps it and sends it.

    The server keeps no state between requests. Everything that is the
    reviewer's alone therefore lives in the browser, kept in localStorage by
    Alpine's persist plugin, and comes with each request as one JSON form
    field. Each item gives the rule, then the reason for it.

    - Unsaved edits are answers by case name. The browser drops an edit equal
      to the saved answer, so every edit here differs from the file.
    - Rejected cases carry the sentences of the rules their last save broke,
      so the list and the panel can show them until the case is edited or
      saved.
    - Each unsaved edit keeps the saved answer it started from, so a save made
      meanwhile from another tab is noticed instead of replaced.
    - A malformed field is read as an empty state rather than failing the
      page, since localStorage can hold values an older page version wrote.
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
        """Return the Corrections tab's filter from the chosen chips and search.

        Returns:
            The filter the chosen change and source chips and the search make.
        """
        return CorrectionFilter(
            changes=frozenset(self.correction_changes),
            sources=frozenset(self.correction_sources),
            search_text=self.correction_search_text,
        )

    def to_case_filter(self) -> CaseFilter:
        """Return the list filter the state's chosen filters and edits make.

        Returns:
            The filter the chosen kind, case and edit chips, the search, and the
            browser's unsaved and rejected cases make.
        """
        return CaseFilter(
            kinds=frozenset(self.kinds),
            case_states=frozenset(state.value for state in self.case_states),
            edit_states=frozenset(self.edit_states),
            search_text=self.search_text,
            unsaved_case_names=frozenset(self.edits),
            rejected_case_names=frozenset(self.rejected),
        )
