from pathlib import Path
from unittest.mock import patch

import pytest
from common.config import ReviewPageConfig, Settings
from pydantic import ValidationError


def test_review_page_config_defaults_listen_on_localhost_only() -> None:
    with patch.dict("os.environ", {}, clear=True):
        config = ReviewPageConfig()
    assert (
        config.review_page_host,
        config.review_page_port,
        config.review_page_data_dir,
        config.review_page_seed_file,
        config.review_page_clean_seed_script,
    ) == (
        "127.0.0.1",
        8003,
        Path("data/seed_cleaning"),
        None,
        Path("scripts/clean_seed.py"),
    )


def test_review_page_config_reads_environment_variables() -> None:
    with patch.dict(
        "os.environ",
        {
            "REVIEW_PAGE_HOST": "127.0.0.2",
            "REVIEW_PAGE_PORT": "9003",
            "REVIEW_PAGE_DATA_DIR": "/app/data/seed_cleaning",
            "REVIEW_PAGE_SEED_FILE": "/app/assets/seed_data/seed.json",
        },
        clear=True,
    ):
        config = ReviewPageConfig()
    assert (
        config.review_page_host,
        config.review_page_port,
        config.review_page_data_dir,
        config.review_page_seed_file,
    ) == (
        "127.0.0.2",
        9003,
        Path("/app/data/seed_cleaning"),
        Path("/app/assets/seed_data/seed.json"),
    )


def test_review_page_config_port_out_of_range_rejected() -> None:
    with (
        patch.dict("os.environ", {"REVIEW_PAGE_PORT": "70000"}, clear=True),
        pytest.raises(ValidationError),
    ):
        ReviewPageConfig()


def test_settings_holds_review_page_config() -> None:
    with patch.dict(
        "os.environ",
        {"ENVIRONMENT": "development", "REVIEW_PAGE_PORT": "9004"},
        clear=True,
    ):
        settings = Settings()
    assert settings.review_page.review_page_port == 9004
