"""Validate incoming HTTP requests and build htmx response headers.

Although the review page binds to localhost by default, arbitrary websites open
in the same browser can still issue cross-origin requests to localhost. Three
layered checks block external sites from mutating review answers or starting
runs. Helper builders populate ``HX-Trigger`` headers so ``review_page.js``
can handle events in the browser (https://htmx.org/headers/hx-trigger/).

Each security check or response rule reads `<rule>: <reason>`:
- Host header validation: Requests whose ``Host`` header is not in
  ``LOCAL_HOST_NAMES`` are rejected with HTTP 400 to defeat DNS rebinding attacks.
- Custom htmx header enforcement: POST requests lacking ``HX-Request: true``
  are rejected with HTTP 403 because cross-origin pages cannot attach custom
  headers without a CORS preflight, which the server never grants.
- Origin header verification: POST requests carrying an ``Origin`` header whose
  hostname is not local are rejected with HTTP 403 as defense-in-depth.
- Read-only GET exemption: GET requests for the HTML shell, static assets, and
  ``/health`` require only the host header check because they do not mutate state.
- Empty event responses: Successful event-only responses use HTTP 204 so htmx
  dispatches the event without swapping any DOM region.
- Refusal notices: Error responses include the notice text in both the JSON
  response body and the ``HX-Trigger`` header so client error handlers can
  read the message consistently across browsers.

Starlette executes the last registered middleware first, so
``TrustedHostMiddleware`` is registered after the POST guard to run before it.
"""

import json
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

# Allowed Host header values without ports; IPv6 loopback uses brackets.
LOCAL_HOST_NAMES = ("127.0.0.1", "localhost", "[::1]")
HTMX_REQUEST_HEADER = "HX-Request"
# Allowed Origin hostnames as parsed by urlsplit (without IPv6 brackets).
_LOCAL_ORIGIN_HOST_NAMES = frozenset(name.strip("[]") for name in LOCAL_HOST_NAMES)


def add_request_guards(app: FastAPI) -> None:
    """Register host validation and cross-origin POST guard middleware on the app.

    Args:
        app: The FastAPI application instance to protect.
    """
    app.middleware("http")(refuse_post_from_other_sites)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(LOCAL_HOST_NAMES))


async def refuse_post_from_other_sites(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Reject POST requests lacking the htmx header or originating from external hosts.

    Args:
        request: The incoming HTTP request.
        call_next: The next middleware or route handler in the ASGI chain.

    Returns:
        An HTTP 403 refusal response for unauthorized POST requests, or the
        downstream handler response.
    """
    if request.method == "POST" and not _is_post_from_page(request):
        return Response("Requests must come from the review page.", status_code=403)
    return await call_next(request)


def _is_post_from_page(request: Request) -> bool:
    """Report whether a POST request carries ``HX-Request: true`` and a local ``Origin``.

    Args:
        request: The incoming HTTP POST request to inspect.

    Returns:
        True when ``HX-Request`` is ``"true"`` and the ``Origin`` header is
        either absent or matches a local hostname.
    """
    if request.headers.get(HTMX_REQUEST_HEADER) != "true":
        return False
    origin = request.headers.get("origin")
    return origin is None or urlsplit(origin).hostname in _LOCAL_ORIGIN_HOST_NAMES


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
