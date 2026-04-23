"""
Tests for EnrichmentConfig validation and logging.
"""

import pytest
from pydantic import ValidationError

from enrichment.programmatic.config import EnrichmentConfig


class TestEnrichmentConfigDefaults:
    """Default values are valid and reflect documented settings."""

    def test_default_api_timeout(self):
        assert EnrichmentConfig().api_timeout == 200

    def test_default_batch_size(self):
        assert EnrichmentConfig().batch_size == 10

    def test_default_cache_ttl(self):
        assert EnrichmentConfig().cache_ttl == 86400

    def test_default_skip_failed_apis(self):
        assert EnrichmentConfig().skip_failed_apis is True

    def test_default_verbose_logging(self):
        assert EnrichmentConfig().verbose_logging is False


class TestValidateTimeout:
    """validate_timeout rejects out-of-range values."""

    def test_valid_minimum(self):
        assert EnrichmentConfig(api_timeout=1).api_timeout == 1

    def test_valid_maximum(self):
        assert EnrichmentConfig(api_timeout=3600).api_timeout == 3600

    def test_too_low_raises(self):
        with pytest.raises(ValidationError, match="between 1 and 3600"):
            EnrichmentConfig(api_timeout=0)

    def test_too_high_raises(self):
        with pytest.raises(ValidationError, match="between 1 and 3600"):
            EnrichmentConfig(api_timeout=3601)


class TestValidateBatchSize:
    """validate_batch_size rejects out-of-range values."""

    def test_valid_minimum(self):
        assert EnrichmentConfig(batch_size=1).batch_size == 1

    def test_valid_maximum(self):
        assert EnrichmentConfig(batch_size=100).batch_size == 100

    def test_too_low_raises(self):
        with pytest.raises(ValidationError, match="between 1 and 100"):
            EnrichmentConfig(batch_size=0)

    def test_too_high_raises(self):
        with pytest.raises(ValidationError, match="between 1 and 100"):
            EnrichmentConfig(batch_size=101)


class TestValidateCacheTtl:
    """validate_cache_ttl rejects negative values."""

    def test_zero_is_valid(self):
        assert EnrichmentConfig(cache_ttl=0).cache_ttl == 0

    def test_positive_is_valid(self):
        assert EnrichmentConfig(cache_ttl=3600).cache_ttl == 3600

    def test_negative_raises(self):
        with pytest.raises(ValidationError, match="non-negative"):
            EnrichmentConfig(cache_ttl=-1)


class TestLogConfiguration:
    """log_configuration emits structured log lines without raising."""

    def test_log_configuration_does_not_raise(self, caplog):
        import logging

        config = EnrichmentConfig()
        with caplog.at_level(logging.INFO, logger="enrichment.programmatic.config"):
            config.log_configuration()

        assert any("API Timeout" in r.message for r in caplog.records)
        assert any("Max Concurrent APIs" in r.message for r in caplog.records)
        assert any("Batch Size" in r.message for r in caplog.records)
        assert any("Caching" in r.message for r in caplog.records)
        assert any("Graceful Degradation" in r.message for r in caplog.records)

    def test_log_configuration_reflects_caching_disabled(self, caplog):
        import logging

        config = EnrichmentConfig(enable_caching=False)
        with caplog.at_level(logging.INFO, logger="enrichment.programmatic.config"):
            config.log_configuration()

        assert any("Disabled" in r.message for r in caplog.records)
