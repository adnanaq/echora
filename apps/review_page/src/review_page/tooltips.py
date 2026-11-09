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
    RESOLVED_CASE_KIND_COUNT = (
        "While some cases of this kind are unresolved, the count reads resolved "
        "cases / all cases. A case is resolved when the last run applied its "
        "answer, or when its answer is unused because the pages are already "
        "settled. Once every case is resolved, the count shows the total alone."
    )
    SUGGESTED_ANSWER_PILL = (
        "Matching rules have suggested an answer. "
        "The case opens on this suggestion, and Accept saves it."
    )
    PREVIOUS_ANSWER_PILL = (
        "A seed release changed this case's pages, so the case replaces one "
        "answered before. Answer it again with the added pages."
    )
    PREVIOUS_ANSWER_APPLIED = (
        "The previous answer still places the pages it saw, and the case opens on "
        "it. Place the added pages, then save."
    )
    PREVIOUS_ANSWER_NOT_APPLIED = (
        "The release changed which entries this case holds, so the previous "
        "answer no longer applies. It is shown below for reference."
    )
    ADDED_PAGE_PILL = "The seed release added this page since the previous answer"
    UNUSED_ANSWER_PILL = (
        "The last run did not use this saved answer, because a seed release or "
        "another answer already settled these pages and no case needs it. The "
        "answer stays in the review file."
    )
    UNUSED_ANSWER_ACTION = (
        "No action is needed. If a new case holds these pages, answer that case."
    )

    # Case panel.
    SHOW_CORRECTIONS = "Open this entry on the Corrections tab"
    WORK_COLUMN = (
        "Select which work the provider page belongs to. "
        "All pages assigned the same number form one work."
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
    WORKS_SECTION = (
        "A page covering several works is a source of one of them and a partial "
        "source of the others. Enrichment never fetches a partial source."
    )
    PARTIAL_SOURCE_PICKER = (
        "Add pages that are a source of another work and also cover this one"
    )
    PARTIAL_SOURCE_ADD = "Add the ticked pages as partial sources of this work"
    PARTIAL_SOURCE_NEEDS_WORK = (
        "Give a page of this case a work first; then it can be added here."
    )
    OUTSIDE_THIS_CASE = (
        "A seed entry outside this case that a page of the case also covers"
    )
    PARTIAL_SOURCE_SEARCH = "Type part of a page key or title to filter the pages"
    PARTIAL_SOURCE_REMOVE = "Stop listing this page as a partial source of the work"
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
    PLACE_COLUMN = (
        "The page's place as the answer writes it, with the works it is a "
        "partial source of"
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
    "previous answer": (
        "Open cases that replace a case answered before a seed release changed "
        "its pages"
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
        "Show entries where the run dropped an old address because "
        "the entry already lists the current address"
    ),
    "dead pages dropped": (
        "Show entries where the run dropped a page its provider says no longer exists"
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
