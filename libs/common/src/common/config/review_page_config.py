"""Configuration of the review page, where a person answers seed-cleaning review cases."""

from pathlib import Path

from pydantic import Field

from .env_settings import EnvSettings


class ReviewPageConfig(EnvSettings):
    """Where the review page listens, and the files its Run reads and writes.

    The page listens on localhost only by default: it reads and writes local
    files and starts a seed-cleaning run, so nothing outside this machine may
    reach it. A container sets ``REVIEW_PAGE_HOST=0.0.0.0`` and publishes the
    port on ``127.0.0.1`` only, which keeps that rule.
    """

    review_page_host: str = Field(
        default="127.0.0.1", description="Address the review page listens on"
    )
    review_page_port: int = Field(
        default=8003, ge=1, le=65535, description="Port the review page listens on"
    )
    review_page_data_dir: Path = Field(
        default=Path("data/seed_cleaning"),
        description=(
            "Seed cleaning's data folder: the review file, the page store and "
            "the last run's result"
        ),
    )
    review_page_seed_file: Path | None = Field(
        default=None,
        description=(
            "The seed release file Run cleans; unset uses the file of the last "
            "run, so a Run never switches releases by itself"
        ),
    )
    review_page_clean_seed_script: Path = Field(
        default=Path("scripts/clean_seed.py"),
        description="The seed-cleaning script Run starts with --no-fetch",
    )
