"""Refuse requests to the review page that do not come from the page itself.

The page listens on localhost, but any website open in the same browser can
still send requests to it. Three checks keep other websites from saving
answers or starting runs. The checks run in the order listed. Each item gives
the rule, then the reason for it.

- A request naming any host other than ``LOCAL_HOST_NAMES`` gets 400. A site
  that points its own name at 127.0.0.1 (DNS rebinding) sends its own name as
  the host, so the request is refused.
- A POST without the ``HX-Request: true`` header gets 403. htmx sends that
  header on every request, and the page makes every POST through htmx. A
  website on another origin can set the header only after a CORS preflight,
  which the server never answers with permission, so a cross-site form post
  or ``fetch`` is refused.
- A POST whose ``Origin`` header names another host gets 403. Browsers send
  ``Origin`` on every POST, so this second check refuses another site's
  request even if the header check ever let one through.
- GET requests (the page, the static files and ``/health``) need only the
  host check, because GET requests change nothing.

Starlette runs the middleware added last first
(``Starlette.add_middleware`` inserts at the front), so the host check is
added after the POST checks.
"""

from collections.abc import Awaitable, Callable
from urllib.parse import urlsplit

from fastapi import FastAPI, Request, Response
from starlette.middleware.trustedhost import TrustedHostMiddleware

# The names the page is opened under, as Starlette reads the Host header:
# without the port, and an IPv6 address in brackets.
LOCAL_HOST_NAMES = ("127.0.0.1", "localhost", "[::1]")
HTMX_REQUEST_HEADER = "HX-Request"
# The same names as an Origin's host name gives them, without brackets.
_LOCAL_ORIGIN_HOST_NAMES = frozenset(name.strip("[]") for name in LOCAL_HOST_NAMES)


def add_request_guards(app: FastAPI) -> None:
    """Add the host check and the POST checks to the app, the host check running first.

    Args:
        app: The review page's web app.
    """
    app.middleware("http")(refuse_post_from_other_sites)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(LOCAL_HOST_NAMES))


async def refuse_post_from_other_sites(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Answer 403 to a POST htmx did not send or another site sent; pass every other request on.

    Args:
        request: The incoming request.
        call_next: The rest of the app.

    Returns:
        The 403 refusal, or the app's own response.
    """
    if request.method == "POST" and not _is_post_from_page(request):
        return Response("Requests must come from the review page.", status_code=403)
    return await call_next(request)


def _is_post_from_page(request: Request) -> bool:
    """Report whether a POST carries htmx's header and, when present, a local Origin.

    Args:
        request: The incoming POST.

    Returns:
        True when the POST carries ``HX-Request: true`` and either no Origin or
        an Origin naming a local host.
    """
    if request.headers.get(HTMX_REQUEST_HEADER) != "true":
        return False
    origin = request.headers.get("origin")
    return origin is None or urlsplit(origin).hostname in _LOCAL_ORIGIN_HOST_NAMES
