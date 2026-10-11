import logging
from unittest.mock import patch

import pytest
from common.config import EnrichmentConfig
from pydantic import ValidationError


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("enrichment_max_concurrent_browsers", 4),
        ("enrichment_batch_size", 10),
        ("enrichment_temp_dir", "temp"),
        ("enrichment_skip_failed_apis", True),
        ("enrichment_verbose_logging", False),
        ("anidb_client", "animeenrichment"),
        ("anidb_min_request_interval", 2.0),
        ("weighted_score_baseline", 6.2),
        ("weighted_score_baseline_votes", 1000),
    ],
)
def test_enrichment_config_returns_documented_default(
    field: str, expected: object
) -> None:
    assert getattr(EnrichmentConfig(), field) == expected


@pytest.mark.parametrize(
    ("variable", "field", "value", "expected"),
    [
        (
            "ENRICHMENT_MAX_CONCURRENT_BROWSERS",
            "enrichment_max_concurrent_browsers",
            "2",
            2,
        ),
        ("ENRICHMENT_BATCH_SIZE", "enrichment_batch_size", "25", 25),
        ("ANIDB_CLIENT", "anidb_client", "registered", "registered"),
        ("ANIDB_BAN_COOLDOWN", "anidb_ban_cooldown", "60", 60.0),
        ("WEIGHTED_SCORE_BASELINE", "weighted_score_baseline", "6.5", 6.5),
    ],
)
def test_enrichment_config_environment_variable_sets_field(
    variable: str, field: str, value: str, expected: object
) -> None:
    with patch.dict("os.environ", {variable: value}):
        assert getattr(EnrichmentConfig(), field) == expected


def test_enrichment_config_max_concurrent_browsers_below_one_raises_validation_error() -> (
    None
):
    with pytest.raises(ValidationError, match="enrichment_max_concurrent_browsers"):
        EnrichmentConfig(enrichment_max_concurrent_browsers=0)


@pytest.mark.parametrize("batch_size", [1, 100])
def test_enrichment_config_batch_size_at_limit_is_accepted(batch_size: int) -> None:
    config = EnrichmentConfig(enrichment_batch_size=batch_size)

    assert config.enrichment_batch_size == batch_size


@pytest.mark.parametrize("batch_size", [0, 101])
def test_enrichment_config_batch_size_out_of_range_raises_validation_error(
    batch_size: int,
) -> None:
    with pytest.raises(ValidationError, match="enrichment_batch_size"):
        EnrichmentConfig(enrichment_batch_size=batch_size)


def test_log_configuration_logs_browsers_batch_size_and_failure_handling(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="common.config.enrichment_config"):
        EnrichmentConfig().log_configuration()

    assert "4 browsers" in caplog.text
    assert "batches of 10" in caplog.text
    assert "failed providers skipped" in caplog.text


def test_log_configuration_failed_providers_not_skipped_logs_fatal(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="common.config.enrichment_config"):
        EnrichmentConfig(enrichment_skip_failed_apis=False).log_configuration()

    assert "failed providers fatal" in caplog.text
