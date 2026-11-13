"""Define centralized tooltip strings and descriptive text for the review interface.

Centralizes all user interface copy, button hints, filter descriptions, and
status messages so that display text is decoupled from HTML templates.

Each item reads `<category>: <purpose>`:
- Interface text: Button, column, filter, state, and tab descriptions are
  maintained in this module and consumed by templates and state builders.
- Domain rule explanations: Explanations of placement rules, disabled options,
  and case classifications reside in ``work_matching`` alongside the logic.

All tooltip values are written in plain language. Solitary sentence tooltips
omit trailing periods, while multi-sentence tooltips use standard punctuation.
Text mappings for dynamic values (states, changes, themes) are keyed by their
respective domain enum values, verified by automated unit tests.
"""

from enum import StrEnum
from typing import Final


class Tooltip(StrEnum):
    """Static tooltip text constants for buttons, columns, tabs, and navigation controls."""

    # Tabs and page layout.
    CASES_TAB = "Review and answer open cases, then run seed cleaning"
    CORRECTIONS_TAB = "View changes made to seed entries by the previous run"
    HIDE_CASE_LIST = "Hide the case list"
    SHOW_CASE_LIST = "Show the case list"
    RESIZE_CASE_LIST = (
        "Drag or use the arrow keys to resize the case list. "
        "Double-click to restore the default width."
    )
    FIRST_PAGE = "First page"
    PREVIOUS_PAGE = "Previous page"
    NEXT_PAGE = "Next page"
    LAST_PAGE = "Last page"

    # Toolbar.
    DISCARD_EDITS = "Discard all unsaved edits in this browser"
    SAVE_ALL_ANSWERS = (
        "Save all staged answer edits to the review file. "
        "Edits that violate placement rules remain unsaved and are marked rejected."
    )
    RUN_PIPELINE = (
        "Apply all saved answers to the seed entries using cached provider pages. "
        "Enabled once all unsaved edits are saved."
    )
    RUN_CHANGES_PREFIX = "Saved changes:"

    # Case list and its filters.
    RESET_CASE_FILTERS = (
        "Clear all active filters and search terms to display every review case"
    )
    APPLIED_ANSWER_ICON = (
        "Applied: the previous run applied this case's saved answer to the seed "
        "entries. Every subsequent run reapplies the saved answer."
    )
    UNRESOLVED_CASE_KIND_COUNT = "The count shows resolved / total cases."
    RESOLVED_CASE_KIND_COUNT = "All cases are resolved; the count shows the total."
    SUGGESTED_ANSWER_PILL = (
        "Matching rules suggested an answer for this case. "
        "Select Accept to save the suggestion."
    )
    PREVIOUS_ANSWER_PILL = (
        "A seed release changed this case's pages after it was answered. "
        "Answer the case again to place the added pages."
    )
    PREVIOUS_ANSWER_APPLIED = (
        "The previous answer still places the pages it covered, and the case "
        "opens with those placements. Place the added pages, then save."
    )
    PREVIOUS_ANSWER_NOT_APPLIED = (
        "The seed release changed which entries this case holds, so the previous "
        "answer no longer applies. The previous placements are shown below for "
        "reference."
    )
    ADDED_PAGE_PILL = "The seed release added this page since the previous answer"
    UNUSED_ANSWER_PILL = (
        "The previous run did not apply this saved answer because a seed release "
        "or another answer already settled its pages. The answer remains saved in "
        "the review file."
    )
    UNUSED_ANSWER_ACTION = "No action is needed unless a new case includes these pages."

    # Case panel.
    SHOW_CORRECTIONS = "Open this entry on the Corrections tab"
    WORK_COLUMN = (
        "Select which work the provider page belongs to. "
        "All pages assigned the same work number form one work."
    )
    NO_LINKED_PAGE_ANSWER = (
        "Leave the case unanswered for now. The review case remains open."
    )
    CHANGED_CHOICE_MARK = (
        "This placement differs from the baseline answer recorded by the previous run"
    )
    TARGET_ENTRY_DROPDOWN = (
        "Select the seed entry that the page moves to. Choose New entry to create "
        "a separate entry, or select an existing entry by title or page key."
    )
    TARGET_ENTRY_KEY_INPUT = (
        "Type or paste a seed entry's title, page key, or provider address"
    )
    WORKS_SECTION = (
        "A page that covers multiple works is a full source of one work and a "
        "partial source of the others. Enrichment never fetches a partial source."
    )
    PARTIAL_SOURCE_PICKER = (
        "Select pages from other works that also cover this work as partial sources"
    )
    PARTIAL_SOURCE_ADD = "Add the selected pages as partial sources of this work"
    PARTIAL_SOURCE_NEEDS_WORK = (
        "Assign a page in this case to a work first; then it can be added here."
    )
    OUTSIDE_THIS_CASE = (
        "This seed entry is outside the current case, and a page in this case "
        "also covers it"
    )
    PARTIAL_SOURCE_SEARCH = "Filter candidate pages by page key or title"
    PARTIAL_SOURCE_REMOVE = "Remove this page as a partial source of the work"
    NOT_OFFERED_PREFIX = "Not offered:"
    RESET_ANSWER = (
        "Restore the baseline answer from the previous run and save it to the "
        "review file"
    )
    ACCEPT_SUGGESTION = "Save the suggested answer to the review file"
    SAVE_ANSWER = (
        "Validate the current answer against placement rules and save it to the "
        "review file"
    )
    SAVE_ANSWER_AND_NEXT = (
        "Save the current answer and open the next review case in the filtered list"
    )

    # Corrections tab.
    RESET_CORRECTION_FILTERS = (
        "Clear all active change and source filters and search terms to display "
        "every corrected entry"
    )
    ENTRY_FROM_ANSWER_PILL = (
        "A reviewer's answer placed at least one page in this entry"
    )
    PAGE_FROM_ANSWER_PILL = "A reviewer's answer placed this page"
    SHOW_CASE = "Open this review case on the Cases tab"
    CHANGE_COLUMN = (
        "Shows the change applied to the page, matching the categories in the "
        "run summary"
    )
    DESTINATION_COLUMN = (
        "Shows where the page is placed (this entry, another seed entry, a new "
        "entry, or its current address)"
    )
    PLACE_COLUMN = (
        "Shows the placement recorded for the page, including any works it "
        "covers as a partial source"
    )
    REASON_COLUMN = "Shows the matching rule or decision that determined this placement"
    SOURCE_COLUMN = (
        "Shows whether the placement came from a reviewer's answer or from "
        "matching rules"
    )


# Keyed by CaseEditState values.
EDIT_STATE_TOOLTIPS: Final[dict[str, str]] = {
    "rejected": (
        "The last save attempt violated placement rules. "
        "Staged edits remain in this browser and are not saved."
    ),
    "edited, not saved": (
        "The answer has unsaved edits in this browser. "
        "Select Save or Save all to write them to the review file."
    ),
    "saved, applies on Run": (
        "The saved answer differs from the previous run. "
        "Select Run to apply it to the seed entries."
    ),
    "unchanged": (
        "The answer matches the baseline from the previous run with no pending changes"
    ),
}

# Keyed by CaseState values.
CASE_STATE_TOOLTIPS: Final[dict[str, str]] = {
    "open": "Show cases that are open and not resolved yet",
    "answered": "Show cases that are resolved",
    "changed": "Show cases whose saved answer changed since the last run",
    "new": "Show cases that appeared for the first time in the latest run",
    "previous answer": (
        "Show open cases that replace a previously answered case after a seed "
        "release changed its pages"
    ),
}

# Keyed by the change names the run summary counts, and "unchanged".
CORRECTION_CHANGE_TOOLTIPS: Final[dict[str, str]] = {
    "partial sources added": (
        "Show entries where a page also joined other entries as a partial source"
    ),
    "duplicate pages dropped": (
        "Show entries where an answer dropped a duplicate page"
    ),
    "old addresses dropped": (
        "Show entries where the run dropped an outdated page address because "
        "the entry already holds the current address"
    ),
    "dead pages dropped": (
        "Show entries where the run dropped a page that no longer exists on its "
        "provider"
    ),
    "pages moved to a seed entry": (
        "Show entries where the run moved a page to another seed entry"
    ),
    "pages moved to a new entry": (
        "Show entries where the run moved a page to a new entry"
    ),
    "pages added": ("Show entries where the run added a discovered linked page"),
    "pages added as partial sources": (
        "Show entries where the run added a linked page as a partial source"
    ),
    "unchanged": (
        "Show entries where the run left at least one provider page unchanged"
    ),
}

# Keyed by the theme setting's values.
# Keyed by the Corrections tab's source chip values.
CORRECTION_SOURCE_TOOLTIPS: Final[dict[str, str]] = {
    "your answer": (
        "Show entries containing at least one page placed by a reviewer's answer"
    ),
    "rules": (
        "Show entries containing pages placed automatically by matching rules "
        "without an answer"
    ),
}

THEME_TOOLTIPS: Final[dict[str, str]] = {
    "system": "Follow the operating system's light or dark mode setting",
    "light": "Always use the light color theme",
    "dark": "Always use the dark color theme",
}
