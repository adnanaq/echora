"""Start the review page: read the settings, build the file source, serve the app.

Start the page from the repository root, so the default data folder
(``data/seed_cleaning``) and the seed-cleaning script
(``scripts/clean_seed.py``) are found. The README gives the commands, with
Pants or with uv. In the dev Docker stack, the page runs as the
``review-page`` service instead.
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
