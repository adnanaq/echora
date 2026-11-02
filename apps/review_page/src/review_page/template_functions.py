"""Give every template the functions it calls and the page's tooltips.

The functions and tooltips are registered once as a Starlette context
processor (``add_template_functions``), so every template, fragment or page,
calls them the same way. Each item gives the function, then the reason for
it.

- ``static_url``: a static file's address carries its content hash, so a
  browser fetches a changed stylesheet or script again instead of its copy.
- ``element_id``: every control inside a region htmx redraws carries an id
  built from what it controls. htmx keeps focus on an element with the same
  id across a redraw, and by default does not scroll to that element
  (https://htmx.org/attributes/hx-swap/, "Focus scroll"). Without an id, the
  focused control is removed, focus falls back to the page, and the browser
  jumps.
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
    """Return a static file's address with its content hash, so a changed file is fetched again.

    The hash changes only when the file does, so an unchanged file stays cached.

    Args:
        file_path: The file's path inside the static folder, such as
            ``parts.css``.

    Returns:
        ``/static/<file_path>?v=<first 12 hex digits of its SHA-256>``.
    """
    content_hash = hashlib.sha256((STATIC_DIR / file_path).read_bytes()).hexdigest()
    return f"/static/{file_path}?v={content_hash[:12]}"


def build_element_id(*parts: object) -> str:
    """Return an element id built from what the element controls, the same on every redraw.

    Args:
        parts: The element's kind and what it controls, such as
            ``"choice", "work", "Work of mal:anime:1"``.

    Returns:
        The parts joined by hyphens, every run of characters an id should not
        hold replaced by one hyphen.
    """
    joined = "-".join(str(part) for part in parts)
    return _NOT_ID_CHARACTERS.sub("-", joined).strip("-")


def add_template_functions(request: Request) -> dict[str, Any]:
    """Give every template its helpers and tooltips, as a context processor.

    The tooltips come from ``tooltips``, so no template writes its own text.

    Args:
        request: The request being rendered; every template gets the same values.

    Returns:
        ``static_url``, ``element_id``, ``tooltips`` and the tooltip mappings
        for changes and themes, by the name templates use.
    """
    del request  # Every template gets the same functions.
    return {
        "static_url": build_static_url,
        "element_id": build_element_id,
        "tooltips": Tooltip,
        "correction_change_tooltips": CORRECTION_CHANGE_TOOLTIPS,
        "theme_tooltips": THEME_TOOLTIPS,
    }
