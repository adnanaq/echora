"""Construct the FastAPI web application, route handlers, and template processors.

Configures application middleware, static asset routing, Jinja template context
processors, exception handlers, and modular routers for the review interface.

Module components include:
- Shell routes: Serves the base single-page HTML layout and health endpoints.
- Case routes: Handles case review interactions, pagination, choices, and saves.
- Named entry routes: Checks a page key typed in the named entry picker.
- Correction routes: Serves pipeline correction reports and entry filters.
- Run routes: Triggers pipeline runs and streams background execution states.
- Security guards: Validates host headers and enforces same-origin POST requests.
- Concurrency error handling: Translates storage conflicts and unreadable review
  records into clear error notifications.
"""

from pathlib import Path

from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from review_page.case_routes import CaseRoutes
from review_page.correction_routes import CorrectionRoutes
from review_page.named_entry_routes import NamedEntryRoutes
from review_page.request_guards import add_request_guards, build_notice_response
from review_page.review_case_loader import UnreadableReviewFileError
from review_page.review_case_source import (
    ReviewCaseSource,
)
from review_page.run_routes import RunRoutes
from review_page.template_functions import STATIC_DIR, add_template_functions

_PACKAGE_DIR = Path(__file__).parent


class PageRoutes:
    """Register and serve routes for the main HTML shell and health check endpoint."""

    def __init__(self, templates: Jinja2Templates) -> None:
        """Register the root page shell and ``/health`` routes on a new router.

        Args:
            templates: The Jinja2 templates collection for rendering the page shell.
        """
        self._templates = templates
        self.router = APIRouter()
        self.router.add_api_route("/health", self.report_health, methods=["GET"])
        self.router.add_api_route(
            "/", self.render_page, methods=["GET"], response_class=HTMLResponse
        )

    def report_health(self) -> dict[str, str]:
        """Report service readiness for container health probes.

        Returns:
            A status dictionary ``{"status": "ok"}``.
        """
        return {"status": "ok"}

    def render_page(self, request: Request) -> Response:
        """Render the top-level HTML shell that loads regional fragments via htmx.

        Args:
            request: The incoming HTTP GET request.

        Returns:
            The rendered ``page.html`` response.
        """
        return self._templates.TemplateResponse(request, "page.html")


def create_app(source: ReviewCaseSource) -> FastAPI:
    """Construct the FastAPI application wired to the given review case source.

    Args:
        source: The review case source used to read cases, save answers, and
            manage pipeline runs.

    Returns:
        The configured ``FastAPI`` application with routes, guards, templates,
        and static asset mounts.
    """
    app = FastAPI(title="Review page", docs_url=None, redoc_url=None, openapi_url=None)
    add_request_guards(app)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    templates = Jinja2Templates(
        directory=_PACKAGE_DIR / "templates",
        context_processors=[add_template_functions],
    )
    app.include_router(PageRoutes(templates).router)
    app.include_router(CaseRoutes(source, templates).router)
    app.include_router(NamedEntryRoutes(source, templates).router)
    app.include_router(CorrectionRoutes(source, templates).router)
    app.include_router(RunRoutes(source, templates).router)
    app.add_exception_handler(UnreadableReviewFileError, _report_unreadable_review_file)
    return app


async def _report_unreadable_review_file(
    request: Request, error: Exception
) -> Response:
    """Convert an ``UnreadableReviewFileError`` into an HTTP 500 notice response.

    Surfaces the specific corrupted line number in the browser notice banner
    rather than returning a generic server error.

    Args:
        request: The incoming HTTP request that triggered the file read.
        error: The raised ``UnreadableReviewFileError`` instance.

    Returns:
        An HTTP 500 response carrying the error message as an htmx notice event.
    """
    del request  # Every request gets the same notice.
    return build_notice_response(500, str(error))
