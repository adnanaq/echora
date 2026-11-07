"""Provide Jinja2 template helper functions and tooltip constants via context processing.

Registers ``add_template_functions`` as a Starlette template context processor
so full pages and htmx fragments share the same helper functions and tooltip
mappings.

Each helper function reads `<helper>: <purpose>`:
- ``static_url``: Appends a 12-character SHA-256 content hash query parameter
  to static asset URLs so browsers fetch updated stylesheets or scripts while
  caching unchanged files.
- ``element_id``: Builds deterministic HTML element IDs for controls inside
  htmx-swapped regions so htmx preserves keyboard focus and scroll position
  across redraws (https://htmx.org/attributes/hx-swap/).
- ``capitalize_first_letter``: Turns a filter or change value into its label,
  so a chip and a table cell show the same value the same way.
- ``kinds``, ``case_states`` and ``edit_states``: The filter values every
  template lists, in enum order, so no route passes them itself.
"""

import hashlib
import re
from functools import cache
from pathlib import Path
from typing import Any

from fastapi import Request
from review_page.models import CaseEditState, CaseState
from review_page.tooltips import (
    CASE_STATE_TOOLTIPS,
    CORRECTION_CHANGE_TOOLTIPS,
    CORRECTION_SOURCE_TOOLTIPS,
    EDIT_STATE_TOOLTIPS,
    THEME_TOOLTIPS,
    Tooltip,
)
from work_matching import ReviewCaseKind

STATIC_DIR = Path(__file__).parent / "static"
_NOT_ID_CHARACTERS = re.compile(r"[^A-Za-z0-9_-]+")


@cache
def build_static_url(file_path: str) -> str:
    """Build a cache-busted static asset URL carrying a SHA-256 content hash parameter.

    Computes the hash once per file path so unchanged assets remain cached in
    the browser while modified assets receive a new query string.

    Args:
        file_path: Relative path to the asset inside the ``static`` directory,
            such as ``review_page.css``.

    Returns:
        The URL path ``/static/<file_path>?v=<first 12 SHA-256 hex digits>``.
    """
    content_hash = hashlib.sha256((STATIC_DIR / file_path).read_bytes()).hexdigest()
    return f"/static/{file_path}?v={content_hash[:12]}"


def build_element_id(*id_segments: object) -> str:
    """Build a deterministic, HTML-safe element ID from its segments.

    Args:
        id_segments: Segments naming the control kind and its target, such
            as ``("choice", "work", "Work of mal:anime:1")``.

    Returns:
        The segments joined by hyphens with any disallowed characters collapsed
        into single hyphens.
    """
    joined_segments = "-".join(str(id_segment) for id_segment in id_segments)
    return _NOT_ID_CHARACTERS.sub("-", joined_segments).strip("-")


def capitalize_first_letter(text: str) -> str:
    """Capitalize the first letter of a text and keep the rest as written.

    ``str.capitalize`` is not used because it lowercases the rest, which
    would change words such as ``Run`` inside a value.

    Args:
        text: The text to label, such as ``"edited, not saved"``.

    Returns:
        The text with its first character in upper case.
    """
    return text[:1].upper() + text[1:]


def add_template_functions(request: Request) -> dict[str, Any]:
    """Provide template helper functions and tooltip dictionaries to every template context.

    Exposes centralized tooltip constants from ``tooltips`` so templates contain
    no inline tooltip prose.

    Args:
        request: The active HTTP request being rendered.

    Returns:
        A dictionary containing the helper functions, the filter values, and
        the tooltip mappings, keyed by the names templates use.
    """
    del request  # Every template gets the same functions.
    return {
        "static_url": build_static_url,
        "element_id": build_element_id,
        "capitalize_first_letter": capitalize_first_letter,
        "tooltips": Tooltip,
        "correction_change_tooltips": CORRECTION_CHANGE_TOOLTIPS,
        "correction_source_tooltips": CORRECTION_SOURCE_TOOLTIPS,
        "case_state_tooltips": CASE_STATE_TOOLTIPS,
        "edit_state_tooltips": EDIT_STATE_TOOLTIPS,
        "kinds": list(ReviewCaseKind),
        "case_states": list(CaseState),
        "edit_states": list(CaseEditState),
        "theme_tooltips": THEME_TOOLTIPS,
    }
