"""Build the responses that hand the browser htmx events instead of HTML.

htmx reads the ``HX-Trigger`` header and fires each event it names on the
page, where ``review_page.js`` handles it
(https://htmx.org/headers/hx-trigger/). Each item gives the rule, then the
reason for it.

- An event response has no content (204), so htmx swaps nothing.
- A refusal also carries its notice in a JSON body, because htmx does not
  handle response headers on error statuses the same way everywhere.
"""

import json
from typing import Any

from fastapi import Response
from fastapi.responses import JSONResponse


def build_event_response(events: dict[str, Any]) -> Response:
    """Return an empty response that fires these htmx events in the browser.

    Args:
        events: Each event's name, mapped to the detail the browser handler reads.

    Returns:
        A 204 response whose ``HX-Trigger`` header fires the events.
    """
    return Response(status_code=204, headers=build_event_headers(events))


def build_notice_response(status_code: int, notice: str) -> Response:
    """Return a refusal the browser shows as a notice.

    Args:
        status_code: The refusal's HTTP status, such as 409.
        notice: The sentence the page shows.

    Returns:
        A JSON response holding the notice, whose ``HX-Trigger`` header also
        fires a ``notice`` event.
    """
    return JSONResponse(
        {"notice": notice},
        status_code=status_code,
        headers=build_event_headers({"notice": {"text": notice}}),
    )


def build_event_headers(events: dict[str, Any]) -> dict[str, str]:
    """Return the header that hands htmx the events to fire in the browser.

    Args:
        events: Each event's name, mapped to the detail the browser handler reads.

    Returns:
        The ``HX-Trigger`` header, its value the events as JSON.
    """
    return {"HX-Trigger": json.dumps(events)}
