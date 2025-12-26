"""Tests for environment detection and settings overrides."""

import os
from unittest.mock import patch

from common.config.settings import Environment, get_environment


class TestEnvironmentEnum:
    """Test Environment enum values."""

    def test_has_development(self):
        assert Environment.DEVELOPMENT == "development"

    def test_has_staging(self):
        assert Environment.STAGING == "staging"

    def test_has_production(self):
        assert Environment.PRODUCTION == "production"


class TestGetEnvironment:
    """Test environment detection function."""

    def test_defaults_to_development(self):
        with patch.dict(os.environ, {}, clear=True):
            assert get_environment() == Environment.DEVELOPMENT

    def test_detects_production(self):
        with patch.dict(os.environ, {"APP_ENV": "production"}):
            assert get_environment() == Environment.PRODUCTION

    def test_detects_staging(self):
        with patch.dict(os.environ, {"APP_ENV": "staging"}):
            assert get_environment() == Environment.STAGING

    def test_handles_prod_alias(self):
        with patch.dict(os.environ, {"APP_ENV": "prod"}):
            assert get_environment() == Environment.PRODUCTION

    def test_handles_stage_alias(self):
        with patch.dict(os.environ, {"APP_ENV": "stage"}):
            assert get_environment() == Environment.STAGING

    def test_case_insensitive(self):
        with patch.dict(os.environ, {"APP_ENV": "PRODUCTION"}):
            assert get_environment() == Environment.PRODUCTION


from common.config.settings import Settings


class TestSettingsEnvironmentField:
    """Test Settings.environment field."""

    def test_has_environment_field(self):
        settings = Settings()
        assert hasattr(settings, "environment")

    def test_defaults_to_development(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.environment == Environment.DEVELOPMENT

    def test_respects_app_env(self):
        with patch.dict(os.environ, {"APP_ENV": "production"}):
            settings = Settings()
            assert settings.environment == Environment.PRODUCTION


class TestProductionSafety:
    """Test production safety enforcement."""

    def test_production_forces_debug_false(self):
        with patch.dict(os.environ, {"APP_ENV": "production", "DEBUG": "true"}):
            settings = Settings()
            assert settings.debug is False

    def test_production_forces_warning_log_level(self):
        with patch.dict(os.environ, {"APP_ENV": "production", "LOG_LEVEL": "DEBUG"}):
            settings = Settings()
            assert settings.log_level == "WARNING"

    def test_production_enables_wal(self):
        with patch.dict(os.environ, {"APP_ENV": "production"}):
            settings = Settings()
            assert settings.qdrant_enable_wal is True

    def test_production_enables_model_warmup(self):
        with patch.dict(os.environ, {"APP_ENV": "production"}):
            settings = Settings()
            assert settings.model_warm_up is True

    def test_staging_enables_debug(self):
        with patch.dict(os.environ, {"APP_ENV": "staging"}):
            settings = Settings()
            assert settings.debug is True
            assert settings.log_level == "INFO"
            assert settings.qdrant_enable_wal is True

    def test_development_enables_debug(self):
        with patch.dict(os.environ, {"APP_ENV": "development"}):
            settings = Settings()
            assert settings.debug is True
            assert settings.log_level == "DEBUG"
