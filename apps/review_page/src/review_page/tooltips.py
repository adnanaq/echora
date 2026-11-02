"""Hold every tooltip the review page writes itself, so all its wording is in one place.

The page shows two kinds of tooltip. Each item gives where the text lives,
then the reason for it.

- Page wording (buttons, columns, filters, states, tabs) lives here. The
  templates and ``page_state`` read it from this module, so changing a text
  never means searching the templates.
- Rule wording (what an answer word does, why an option is not offered, what
  a case kind is) lives in ``work_matching``, next to the rules it explains.
  The run's logs and rejections use the same sentences, so they stay true to
  the code that applies them.

Every text is a full sentence with one idea, in the reviewer's words. Texts
that vary by a value (a state, a change, a theme) are mappings keyed by that
value as the page writes it. The tooltips module imports nothing from the
page, so any module can use the tooltips. A test checks the keys against
their enums.
"""

from enum import StrEnum
from typing import Final


class Tooltip(StrEnum):
    """One fixed tooltip, named for the control or column that shows it."""

    # Tabs and page layout.
    CASES_TAB = "Answer the review cases, then Run."
    CORRECTIONS_TAB = "See what the last run changed in the seed, entry by entry."
    HIDE_CASE_LIST = "Hide the case list"
    SHOW_CASE_LIST = "Show the case list"
    RESIZE_CASE_LIST = (
        "Drag, or use the arrow keys, to change the case list's width. "
        "Double-click for the default width."
    )
    FIRST_PAGE = "First page"
    PREVIOUS_PAGE = "Previous page"
    NEXT_PAGE = "Next page"
    LAST_PAGE = "Last page"

    # Toolbar.
    DISCARD = "Drop every unsaved edit in this browser."
    SAVE_ALL = (
        "Save every unsaved edit. Edits that break a rule stay unsaved and are "
        "marked rejected."
    )
    RUN = (
        "Apply every saved answer to the seed. Run uses the pages already "
        "fetched and fetches none. Run is enabled once every edit is saved."
    )
    RUN_CHANGES_PREFIX = "Saved changes:"

    # Case list and its filters.
    RESET_CASE_FILTERS = (
        "Show every case: clear the kind, case and edit filters and the search."
    )
    SUGGESTED = (
        "The rules suggest an answer. The case opens on the suggestion. "
        "Accept saves the suggestion."
    )
    ANSWER_UNUSED = (
        "The last run did not bring this case up, so its saved answer changed "
        "nothing. Another answer changed the case's pages."
    )

    # Case panel.
    SHOW_CORRECTIONS = "Open this entry on the Corrections tab."
    WORK_COLUMN = (
        "Which work the page belongs to. All pages set to the same number form "
        "one work."
    )
    MARK_COLUMN = "How the page stands beside the provider's other pages in its work."
    NO_LINKED_PAGE_ANSWER = "No answer yet. The case stays open."
    CHANGED_CHOICE = "Changed since the last run."
    NOT_OFFERED_PREFIX = "Not offered:"
    RESET = "Go back to the answer the last run used, and save that answer."
    ACCEPT = "Save the suggested answer."
    SAVE = (
        "Check the shown answer against the rules, then save it. The next Run "
        "applies the saved answer."
    )
    SAVE_AND_NEXT = "Save, then open the next case in the list."

    # Corrections tab.
    ANSWER_SOURCE_CHIP = "Entries with a page a reviewer's answer placed."
    RULES_SOURCE_CHIP = "Entries with a page the rules placed without an answer."
    RESET_CORRECTION_FILTERS = (
        "Show every corrected entry: clear the change and source filters and "
        "the search."
    )
    ENTRY_FROM_ANSWER = "A reviewer's answer placed at least one of this entry's pages."
    PAGE_FROM_ANSWER = "A reviewer's answer placed the page."
    SHOW_CASE = "Open this case on the Cases tab."
    CHANGE_COLUMN = "What the run changes for the page, named as in the run summary."
    DESTINATION_COLUMN = "Where the page ends up."
    WORK_AND_MARK_COLUMN = "The page's work and mark, as written in the answer."
    REASON_COLUMN = "Why the page goes to that place."
    SOURCE_COLUMN = "Whether a reviewer's answer or the rules placed the page."


# Keyed by CaseEditState values.
EDIT_STATE_TOOLTIPS: Final[dict[str, str]] = {
    "rejected": (
        "The last save broke a rule. The edit is kept in this browser, not saved."
    ),
    "edited, not saved": (
        "Changed in this browser and not saved yet. Save or Save all writes the "
        "edit to the review file."
    ),
    "saved, applies on Run": (
        "Saved to the review file. The next Run applies the saved answer to the seed."
    ),
    "unchanged": "Nothing edited or saved since the last run.",
}

# Keyed by CaseState values.
CASE_STATE_TOOLTIPS: Final[dict[str, str]] = {
    "open": "The review file holds no answer for the case yet.",
    "answered": "The review file holds an answer for the case.",
    "changed": (
        "The saved answer differs from the answer the last run used. The next "
        "Run changes the seed."
    ),
    "new": "The last run brought the case up for the first time.",
}

# Keyed by the change names the run summary counts, and "unchanged".
CORRECTION_CHANGE_TOOLTIPS: Final[dict[str, str]] = {
    "pages kept as main page": (
        "Entries where the run kept a page as its provider's main page."
    ),
    "pages kept as part page": "Entries where the run kept a page as a part page.",
    "old addresses dropped": (
        "Entries where the run dropped a page's old address, because the entry "
        "also lists the page's current address."
    ),
    "pages moved to a seed entry": (
        "Entries where the run moved a page to another seed entry."
    ),
    "pages moved to a new entry": (
        "Entries where the run moved a page to a new entry."
    ),
    "pages added": "Entries where the run added a linked page.",
    "pages added as parts": "Entries where the run added a linked page as a part.",
    "unchanged": "Entries with a page the run left as it was.",
}

# Keyed by the theme setting's values.
THEME_TOOLTIPS: Final[dict[str, str]] = {
    "system": "Follow the computer's light or dark setting.",
    "light": "Always use the light theme.",
    "dark": "Always use the dark theme.",
}
