"""Start the review page: read the settings, build the file source, serve the app.

Run it from the repository root, so the default data folder
(``data/seed_cleaning``) and seed-cleaning script (``scripts/clean_seed.py``)
are found:

    uv run python -m review_page.main      (with the libraries on PYTHONPATH)

In the dev Docker stack it runs as the ``review-page`` service instead.
"""

import uvicorn
from common.config import get_settings
from review_page.app import create_app
from review_page.file_review_case_source import FileReviewCaseSource


def main() -> None:
    """Serve the review page where ``ReviewPageConfig`` says, on localhost by default."""
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
