"""Refuse requests to the review page that do not come from the page itself.

The page listens on localhost, but any website open in the same browser can
still send requests to it. Two checks keep other websites from saving answers
or starting runs. Each item gives the rule, then the reason for it.

- A request naming any host other than ``LOCAL_HOST_NAMES`` gets 400. A site
  that points its own name at 127.0.0.1 (DNS rebinding) sends its own name as
  the host, so it is refused.
- A POST without the ``HX-Request: true`` header gets 403. htmx sends that
  header on every request, and the page makes every POST through htmx. A
  website on another origin can set the header only after a CORS preflight,
  which the server never answers with permission, so a cross-site form post
  or ``fetch`` is refused.
- GET requests (the page, the static files and ``/health``) need only the
  host check: they change nothing.
"""

from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from starlette.middleware.trustedhost import TrustedHostMiddleware

# The names the page is opened under; Starlette compares them without the port.
LOCAL_HOST_NAMES = ("127.0.0.1", "localhost")
HTMX_REQUEST_HEADER = "HX-Request"


def add_request_guards(app: FastAPI) -> None:
    """Add the host check and the htmx header check to the app.

    Args:
        app: The review page's web app.
    """
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(LOCAL_HOST_NAMES))
    app.middleware("http")(refuse_post_without_htmx_header)


async def refuse_post_without_htmx_header(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Answer 403 to a POST that htmx did not send; pass every other request on.

    Args:
        request: The incoming request.
        call_next: The rest of the app.

    Returns:
        The 403 refusal, or the app's own response.
    """
    is_htmx_request = request.headers.get(HTMX_REQUEST_HEADER) == "true"
    if request.method == "POST" and not is_htmx_request:
        return Response("Requests must come from the review page.", status_code=403)
    return await call_next(request)
