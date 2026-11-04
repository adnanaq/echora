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
"""

import hashlib
import re
from functools import cache
from pathlib import Path
from typing import Any

from fastapi import Request
from review_page.tooltips import CORRECTION_CHANGE_TOOLTIPS, THEME_TOOLTIPS, Tooltip

STATIC_DIR = Path(__file__).parent / "static"
_NOT_ID_CHARACTERS = re.compile(r"[^A-Za-z0-9_-]+")


@cache
def build_static_url(file_path: str) -> str:
    """Build a cache-busted static asset URL carrying a SHA-256 content hash parameter.

    Computes the hash once per file path so unchanged assets remain cached in
    the browser while modified assets receive a new query string.

    Args:
        file_path: Relative path to the asset inside the ``static`` directory,
            such as ``parts.css``.

    Returns:
        The URL path ``/static/<file_path>?v=<first 12 SHA-256 hex digits>``.
    """
    content_hash = hashlib.sha256((STATIC_DIR / file_path).read_bytes()).hexdigest()
    return f"/static/{file_path}?v={content_hash[:12]}"


def build_element_id(*parts: object) -> str:
    """Build a deterministic, HTML-safe element ID from descriptive component parts.

    Args:
        parts: Component segments identifying the control kind and target, such
            as ``("choice", "work", "Work of mal:anime:1")``.

    Returns:
        The segments joined by hyphens with any disallowed characters collapsed
        into single hyphens.
    """
    joined_parts = "-".join(str(part) for part in parts)
    return _NOT_ID_CHARACTERS.sub("-", joined_parts).strip("-")


def add_template_functions(request: Request) -> dict[str, Any]:
    """Provide template helper functions and tooltip dictionaries to every template context.

    Exposes centralized tooltip constants from ``tooltips`` so templates contain
    no inline tooltip prose.

    Args:
        request: The active HTTP request being rendered.

    Returns:
        A dictionary containing ``static_url``, ``element_id``, ``tooltips``,
        ``correction_change_tooltips``, and ``theme_tooltips``.
    """
    del request  # Every template gets the same functions.
    return {
        "static_url": build_static_url,
        "element_id": build_element_id,
        "tooltips": Tooltip,
        "correction_change_tooltips": CORRECTION_CHANGE_TOOLTIPS,
        "theme_tooltips": THEME_TOOLTIPS,
    }
