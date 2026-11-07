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
    CASES_TAB = "Answer the review cases, then Run"
    CORRECTIONS_TAB = "View changes made to seed entries by the previous run"
    HIDE_CASE_LIST = "Hide the case list"
    SHOW_CASE_LIST = "Show the case list"
    RESIZE_CASE_LIST = (
        "Drag, or use the arrow keys, to change the case list's width. "
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
        "Apply every saved answer to the seed database using cached provider pages "
        "without network fetching. Run is enabled once all staged edits are saved."
    )
    RUN_CHANGES_PREFIX = "Saved changes:"

    # Case list and its filters.
    RESET_CASE_FILTERS = (
        "Clear all active filters and search terms to display every review case"
    )
    APPLIED_ANSWER_ICON = (
        "Applied: the last run applied this case's saved answer, so the case is "
        "corrected. Every later run applies the saved answer again."
    )
    APPLIED_ANSWER_KIND_COUNT = (
        "While some cases of this kind are unresolved, the count reads cases whose "
        "answer the last run applied / all cases. Once every case is resolved, the "
        "count shows the total alone."
    )
    SUGGESTED_ANSWER_PILL = (
        "Matching rules have suggested an answer. "
        "The case opens on this suggestion, and Accept saves it."
    )
    UNUSED_ANSWER_PILL = (
        "The previous pipeline run resolved this case through other decisions. "
        "This saved answer was not applied."
    )

    # Case panel.
    SHOW_CORRECTIONS = "Open this entry on the Corrections tab"
    WORK_COLUMN = (
        "Select which work the provider page belongs to. "
        "All pages assigned the same number form one work."
    )
    MARK_COLUMN = (
        "Select how the page stands beside the provider's other pages in the work "
        "(unmarked, main page, or part page)"
    )
    NO_LINKED_PAGE_ANSWER = "No answer selected yet. The review case remains open."
    CHANGED_CHOICE_MARK = (
        "This placement differs from the baseline answer recorded by the previous run"
    )
    TARGET_ENTRY_DROPDOWN = (
        "Select the seed entry the page moves to. New entry assigns the page to a "
        "new work entry; other entry takes a page key or provider address."
    )
    TARGET_ENTRY_KEY_INPUT = (
        "Paste the target seed entry's page key or provider address, then press Enter."
    )
    NOT_OFFERED_PREFIX = "Not offered:"
    RESET_ANSWER = (
        "Revert choices to the baseline answer recorded by the previous run, and "
        "save that answer"
    )
    ACCEPT_SUGGESTION = "Save the complete suggested answer to the review file"
    SAVE_ANSWER = (
        "Validate the selected answer against placement rules and save it to the "
        "review file. The next Run applies it."
    )
    SAVE_ANSWER_AND_NEXT = (
        "Save the active answer and open the next review case in the filtered list"
    )

    # Corrections tab.
    RESET_CORRECTION_FILTERS = (
        "Clear all active change and source filters and search terms to display "
        "every corrected entry"
    )
    ENTRY_FROM_ANSWER_PILL = (
        "A reviewer's answer placed at least one of this entry's pages"
    )
    PAGE_FROM_ANSWER_PILL = "A reviewer's answer placed this page"
    SHOW_CASE = "Open this case on the Cases tab"
    CHANGE_COLUMN = (
        "The change applied to the page, matching the categories in the run summary"
    )
    DESTINATION_COLUMN = (
        "Where the page is placed (this entry, another seed entry, a new entry, "
        "or its current address)"
    )
    WORK_AND_MARK_COLUMN = (
        "The assigned work number and mark written in the placement answer"
    )
    REASON_COLUMN = "The matching rule or decision that determined this placement"
    SOURCE_COLUMN = (
        "Whether the placement was determined by a reviewer's answer or by rules alone"
    )


# Keyed by CaseEditState values.
EDIT_STATE_TOOLTIPS: Final[dict[str, str]] = {
    "rejected": (
        "The last save attempt violated placement rules. "
        "Staged edits remain in this browser and are not saved."
    ),
    "edited, not saved": (
        "The answer was changed in this browser and not saved yet. "
        "Save or Save all writes the edits to the review file."
    ),
    "saved, applies on Run": (
        "The answer is saved to the review file. "
        "The next Run applies the saved answer to the seed database."
    ),
    "unchanged": (
        "The answer matches the baseline recorded by the previous run with no "
        "pending changes"
    ),
}

# Keyed by CaseState values.
CASE_STATE_TOOLTIPS: Final[dict[str, str]] = {
    "open": "The review file holds no answer for this case yet",
    "answered": "The review file holds a saved answer for this case",
    "changed": (
        "The saved answer differs from the baseline answer recorded by the "
        "previous run. The next Run applies it."
    ),
    "new": "The latest pipeline run identified this review case for the first time",
}

# Keyed by the change names the run summary counts, and "unchanged".
CORRECTION_CHANGE_TOOLTIPS: Final[dict[str, str]] = {
    "pages kept as main page": (
        "Show entries where the run kept a page as its provider's main page"
    ),
    "pages kept as part page": (
        "Show entries where the run kept a page as a part page"
    ),
    "old addresses dropped": (
        "Show entries where the run dropped an old address because "
        "the entry already lists the current address"
    ),
    "pages moved to a seed entry": (
        "Show entries where the run moved a page to another seed entry"
    ),
    "pages moved to a new entry": (
        "Show entries where the run moved a page to a new entry"
    ),
    "pages added": ("Show entries where the run added a discovered linked page"),
    "pages added as parts": (
        "Show entries where the run added a linked page as a part page"
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
