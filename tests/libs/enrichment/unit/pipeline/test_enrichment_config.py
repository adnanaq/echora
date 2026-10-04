import logging
from unittest.mock import patch

import pytest
from enrichment.pipeline.config import EnrichmentConfig
from pydantic import ValidationError


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("api_timeout", 200),
        ("batch_size", 10),
        ("cache_ttl", 86400),
        ("skip_failed_apis", True),
        ("verbose_logging", False),
        ("max_concurrent_browsers", 4),
    ],
)
def test_enrichment_config_returns_documented_default(
    field: str, expected: object
) -> None:
    assert getattr(EnrichmentConfig(), field) == expected


def test_enrichment_config_max_concurrent_browsers_read_from_environment() -> None:
    with patch.dict("os.environ", {"ENRICHMENT_MAX_CONCURRENT_BROWSERS": "2"}):
        assert EnrichmentConfig().max_concurrent_browsers == 2


def test_enrichment_config_max_concurrent_browsers_below_one_raises_validation_error() -> (
    None
):
    with pytest.raises(ValidationError, match="max_concurrent_browsers"):
        EnrichmentConfig(max_concurrent_browsers=0)


@pytest.mark.parametrize("api_timeout", [1, 3600])
def test_enrichment_config_api_timeout_at_limit_is_accepted(api_timeout: int) -> None:
    assert EnrichmentConfig(api_timeout=api_timeout).api_timeout == api_timeout


@pytest.mark.parametrize("api_timeout", [0, 3601])
def test_enrichment_config_api_timeout_out_of_range_raises_validation_error(
    api_timeout: int,
) -> None:
    with pytest.raises(ValidationError, match="between 1 and 3600"):
        EnrichmentConfig(api_timeout=api_timeout)


@pytest.mark.parametrize("batch_size", [1, 100])
def test_enrichment_config_batch_size_at_limit_is_accepted(batch_size: int) -> None:
    assert EnrichmentConfig(batch_size=batch_size).batch_size == batch_size


@pytest.mark.parametrize("batch_size", [0, 101])
def test_enrichment_config_batch_size_out_of_range_raises_validation_error(
    batch_size: int,
) -> None:
    with pytest.raises(ValidationError, match="between 1 and 100"):
        EnrichmentConfig(batch_size=batch_size)


@pytest.mark.parametrize("cache_ttl", [0, 3600])
def test_enrichment_config_cache_ttl_zero_or_positive_is_accepted(
    cache_ttl: int,
) -> None:
    assert EnrichmentConfig(cache_ttl=cache_ttl).cache_ttl == cache_ttl


def test_enrichment_config_negative_cache_ttl_raises_validation_error() -> None:
    with pytest.raises(ValidationError, match="non-negative"):
        EnrichmentConfig(cache_ttl=-1)


def test_log_configuration_logs_each_setting(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="enrichment.pipeline.config"):
        EnrichmentConfig().log_configuration()

    for label in (
        "API Timeout",
        "Max Concurrent Browsers: 4",
        "Batch Size",
        "Caching",
        "Graceful Degradation",
    ):
        assert label in caplog.text


def test_log_configuration_caching_disabled_logs_disabled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="enrichment.pipeline.config"):
        EnrichmentConfig(enable_caching=False).log_configuration()

    assert "Caching: Disabled" in caplog.text
