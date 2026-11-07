"""Launch the review page web server with configured file storage and uvicorn settings.

Run this entry point from the repository root so default relative paths for the
seed-cleaning directory (``data/seed_cleaning``) and script
(``scripts/clean_seed.py``) resolve accurately. In the development Docker stack,
this module runs as the ``review-page`` container service.
"""

import uvicorn
from common.config import get_settings
from review_page.app import create_app
from review_page.file_review_case_source import FileReviewCaseSource


def main() -> None:
    """Load review page settings, initialize the file source, and start the server."""
    config = get_settings().review_page
    source = FileReviewCaseSource(
        config.review_page_data_dir,
        config.review_page_clean_seed_script,
        seed_file=config.review_page_seed_file,
    )
    uvicorn.run(
        create_app(source),
        host=config.review_page_host,
        port=config.review_page_port,
    )


if __name__ == "__main__":
    main()
