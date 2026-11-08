"""Construct HTTP responses that trigger client-side htmx events.

Populates the ``HX-Trigger`` response header so ``review_page.js`` can handle
events in the browser (https://htmx.org/headers/hx-trigger/).

Each response rule reads `<rule>: <reason>`:
- Empty event responses: Successful event-only responses use HTTP 204 so htmx
  dispatches the event without swapping any DOM region.
- Refusal notices: Error responses include the notice text in both the JSON
  response body and the ``HX-Trigger`` header so client error handlers can
  read the message consistently across browsers.
"""

import json
from typing import Any

from fastapi import Response
from fastapi.responses import JSONResponse


def build_event_response(events: dict[str, Any]) -> Response:
    """Build an empty 204 response that triggers htmx events in the browser.

    Args:
        events: Mapping of event names to detail payloads read by browser handlers.

    Returns:
        A 204 ``Response`` whose ``HX-Trigger`` header encodes the events.
    """
    return Response(status_code=204, headers=build_event_headers(events))


def build_notice_response(status_code: int, notice: str) -> Response:
    """Build an error response that displays a user-facing notice in the browser.

    Args:
        status_code: The HTTP error status code, such as 409 or 500.
        notice: The human-readable error message displayed in the notice banner.

    Returns:
        A ``JSONResponse`` containing the notice payload and an ``HX-Trigger``
        header that dispatches a ``notice`` event.
    """
    return JSONResponse(
        {"notice": notice},
        status_code=status_code,
        headers=build_event_headers({"notice": {"text": notice}}),
    )


def build_event_headers(events: dict[str, Any]) -> dict[str, str]:
    """Serialize a mapping of htmx event payloads into an ``HX-Trigger`` header dictionary.

    Args:
        events: Mapping of event names to detail payloads read by browser handlers.

    Returns:
        A single-entry dictionary mapping ``HX-Trigger`` to the JSON-encoded events.
    """
    return {"HX-Trigger": json.dumps(events)}
