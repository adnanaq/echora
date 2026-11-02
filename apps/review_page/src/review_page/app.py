"""Build the review page's web app: its routes, guards, templates and error notices.

The browser loads one page, then asks for parts of it (the toolbar, the case
list, the open case, the run panel, the Corrections tab) as HTML fragments
with htmx. Each item gives the part, then the module serving it.

- The page shell and the health check: ``PageRoutes``, here.
- The Cases tab's regions, choices and saves: ``case_routes``.
- The Corrections tab: ``correction_routes``.
- Run and the run panel: ``run_routes``.
- Refusing requests from anywhere but the page itself: ``request_guards``.
- A review file the page cannot read answers every request with a notice
  naming the broken line, rather than a bare server error.

The server listens where ``ReviewPageConfig`` says, on localhost by default.
"""

from pathlib import Path

from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from review_page.case_routes import CaseRoutes
from review_page.correction_routes import CorrectionRoutes
from review_page.file_review_case_source import UnreadableReviewFileError
from review_page.htmx_responses import build_notice_response
from review_page.models import (
    CaseEditState,
    CaseState,
)
from review_page.request_guards import add_request_guards
from review_page.review_case_source import (
    ReviewCaseSource,
)
from review_page.run_routes import RunRoutes
from review_page.template_functions import STATIC_DIR, add_template_functions
from work_matching import (
    ReviewCaseKind,
)

_PACKAGE_DIR = Path(__file__).parent


def create_app(source: ReviewCaseSource) -> FastAPI:
    """Build the review page's web app over a review case source.

    Args:
        source: Where cases are read, answers saved and runs started.

    Returns:
        The app, with its routes, templates and static files.
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
    app.include_router(CorrectionRoutes(source, templates).router)
    app.include_router(RunRoutes(source, templates).router)
    app.add_exception_handler(UnreadableReviewFileError, _report_unreadable_review_file)
    return app


async def _report_unreadable_review_file(
    request: Request, error: Exception
) -> Response:
    """Answer a request the review file cannot serve with a notice naming the broken line.

    Without the notice, every region would fail with the browser's general
    "the server could not do that", and the reviewer would not know which
    line to fix.

    Args:
        request: The request that read the review file.
        error: The ``UnreadableReviewFileError`` naming the line.

    Returns:
        A 500 response carrying the error's sentence as the page's notice.
    """
    del request  # Every request gets the same notice.
    return build_notice_response(500, str(error))


class PageRoutes:
    """The page shell and the health check; the instance keeps no request state."""

    def __init__(self, templates: Jinja2Templates) -> None:
        """Register the page and the health check on a router.

        Args:
            templates: The page's Jinja templates.
        """
        self._templates = templates
        self.router = APIRouter()
        self.router.add_api_route("/health", self.get_health, methods=["GET"])
        self.router.add_api_route(
            "/", self.get_page, methods=["GET"], response_class=HTMLResponse
        )

    def get_health(self) -> dict[str, str]:
        """Answer the health check used by Docker now and Kubernetes later.

        Returns:
            ``{"status": "ok"}`` while the server answers.
        """
        return {"status": "ok"}

    def get_page(self, request: Request) -> Response:
        """Return the page shell; its parts load as fragments.

        Args:
            request: The request being answered.

        Returns:
            The page shell, with the kinds and states its filters list.
        """
        return self._templates.TemplateResponse(
            request,
            "page.html",
            {
                "kinds": list(ReviewCaseKind),
                "case_states": list(CaseState),
                "edit_states": list(CaseEditState),
            },
        )
