"""Integration tests for environment configuration."""

import os
from unittest.mock import patch

import pytest

from common.config.settings import Environment, Settings

pytestmark = pytest.mark.integration


class TestEnvironmentIntegration:
    """Integration tests for all environment modes."""

    def test_development_mode_complete(self):
        """Verify complete development mode configuration."""
        with patch.dict(os.environ, {"APP_ENV": "development"}, clear=True):
            settings = Settings()

            assert settings.environment == Environment.DEVELOPMENT
            assert settings.debug is True
            assert settings.log_level == "DEBUG"
            # Other settings should use defaults
            assert settings.vector_service_port == 8002
            assert settings.qdrant_collection_name == "anime_database"

    def test_staging_mode_complete(self):
        """Verify complete staging mode configuration."""
        with patch.dict(os.environ, {"APP_ENV": "staging"}, clear=True):
            settings = Settings()

            assert settings.environment == Environment.STAGING
            assert settings.debug is True
            assert settings.log_level == "INFO"
            assert settings.qdrant_enable_wal is True
            # Other settings should use defaults
            assert settings.vector_service_port == 8002

    def test_production_mode_complete(self):
        """Verify complete production mode configuration with all enforcements."""
        with patch.dict(os.environ, {"APP_ENV": "production"}, clear=True):
            settings = Settings()

            assert settings.environment == Environment.PRODUCTION
            assert settings.debug is False
            assert settings.log_level == "WARNING"
            assert settings.qdrant_enable_wal is True
            assert settings.model_warm_up is True

    def test_production_override_enforcement(self):
        """Verify production mode CANNOT be bypassed by .env misconfiguration."""
        with patch.dict(
            os.environ,
            {
                "APP_ENV": "production",
                "DEBUG": "true",  # Try to enable debug
                "LOG_LEVEL": "DEBUG",  # Try to use debug logging
                "QDRANT_ENABLE_WAL": "false",  # Try to disable WAL
                "MODEL_WARM_UP": "false",  # Try to disable warmup
            },
            clear=True,
        ):
            settings = Settings()

            # All production safety settings MUST be enforced
            assert settings.debug is False, "Production MUST override DEBUG=true"
            assert settings.log_level == "WARNING", "Production MUST override LOG_LEVEL"
            assert settings.qdrant_enable_wal is True, "Production MUST enable WAL"
            assert settings.model_warm_up is True, "Production MUST enable model warmup"

    def test_case_insensitive_environment_names(self):
        """Verify environment names are case-insensitive."""
        test_cases = [
            ("PRODUCTION", Environment.PRODUCTION),
            ("Production", Environment.PRODUCTION),
            ("STAGING", Environment.STAGING),
            ("Staging", Environment.STAGING),
            ("DEVELOPMENT", Environment.DEVELOPMENT),
        ]

        for env_name, expected_env in test_cases:
            with patch.dict(os.environ, {"APP_ENV": env_name}, clear=True):
                settings = Settings()
                assert settings.environment == expected_env
