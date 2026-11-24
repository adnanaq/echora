"""Launch the review page web server with configured file storage and uvicorn settings.

Run this entry point from the repository root so default relative paths for the
seed-cleaning directory (``data/seed_cleaning``) and script
(``scripts/clean_seed.py``) resolve accurately. In the development Docker stack,
this module runs as the ``review-page`` container service.
"""

import uvicorn
from common.config import get_settings
from review_page.agent_suggestions_client import GrpcAgentSuggestionsClient
from review_page.app import create_app
from review_page.enrichment_run_client import GrpcSeedCleaningRunClient
from review_page.file_review_case_source import FileReviewCaseSource


def main() -> None:
    """Load review page settings, initialize the file source, and start the server."""
    settings = get_settings()
    config = settings.review_page
    service = getattr(settings, "service", None)
    if service is None:
        source = FileReviewCaseSource(
            config.review_page_data_dir,
            config.review_page_clean_seed_script,
            seed_file=config.review_page_seed_file,
        )
        app = create_app(source)
    else:
        enrichment_host = (
            "127.0.0.1"
            if service.enrichment_service_host == "0.0.0.0"  # noqa: S104
            else service.enrichment_service_host
        )
        run_client = GrpcSeedCleaningRunClient(
            f"{enrichment_host}:{service.enrichment_service_port}"
        )
        source = FileReviewCaseSource(
            config.review_page_data_dir,
            config.review_page_clean_seed_script,
            seed_file=config.review_page_seed_file,
            run_client=run_client,
        )
        agent_host = (
            "127.0.0.1"
            if service.agent_service_host == "0.0.0.0"  # noqa: S104
            else service.agent_service_host
        )
        suggestion_client = GrpcAgentSuggestionsClient(
            f"{agent_host}:{service.agent_service_port}"
        )
        app = create_app(source, suggestion_client=suggestion_client)
    uvicorn.run(
        app,
        host=config.review_page_host,
        port=config.review_page_port,
    )


if __name__ == "__main__":
    main()
