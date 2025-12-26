"""Tests for environment detection and settings overrides."""

import os
from unittest.mock import patch

import pytest  # Added import for pytest
from common.config.settings import Environment, Settings, get_environment


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

    def test_raises_error_when_app_env_not_set(self):
        """Test that missing APP_ENV raises ValueError for production safety."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="APP_ENV environment variable must be set"):
                get_environment()

    def test_detects_development(self):
        with patch.dict(os.environ, {"APP_ENV": "development"}):
            assert get_environment() == Environment.DEVELOPMENT

    def test_detects_production(self):
        with patch.dict(os.environ, {"APP_ENV": "production"}):
            assert get_environment() == Environment.PRODUCTION

    def test_detects_staging(self):
        with patch.dict(os.environ, {"APP_ENV": "staging"}):
            assert get_environment() == Environment.STAGING

    def test_raises_error_for_invalid_value(self):
        """Test that invalid APP_ENV values raise ValueError."""
        with patch.dict(os.environ, {"APP_ENV": "invalid"}):
            with pytest.raises(ValueError, match="Invalid APP_ENV value 'invalid'"):
                get_environment()

    def test_case_insensitive(self):
        with patch.dict(os.environ, {"APP_ENV": "PRODUCTION"}):
            assert get_environment() == Environment.PRODUCTION


class TestSettingsEnvironmentField:
    """Test Settings.environment field."""

    def test_has_environment_field(self):
        with patch.dict(os.environ, {"APP_ENV": "development"}):
            settings = Settings()
            assert hasattr(settings, "environment")

    def test_requires_app_env_to_be_set(self):
        """Test that Settings raises error when APP_ENV is not set."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="APP_ENV environment variable must be set"):
                Settings()

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


class TestUserProvidedValues:
    """Test that user-provided values are respected in dev/staging but not production."""

    def test_development_respects_user_debug_false(self):
        """Development should respect user DEBUG=false."""
        with patch.dict(os.environ, {"APP_ENV": "development", "DEBUG": "false"}):
            settings = Settings()
            assert settings.debug is False

    def test_development_respects_user_log_level(self):
        """Development should respect user LOG_LEVEL=ERROR."""
        with patch.dict(os.environ, {"APP_ENV": "development", "LOG_LEVEL": "ERROR"}):
            settings = Settings()
            assert settings.log_level == "ERROR"

    def test_staging_respects_user_debug_false(self):
        """Staging should respect user DEBUG=false."""
        with patch.dict(os.environ, {"APP_ENV": "staging", "DEBUG": "false"}):
            settings = Settings()
            assert settings.debug is False

    def test_staging_respects_user_log_level(self):
        """Staging should respect user LOG_LEVEL=ERROR."""
        with patch.dict(os.environ, {"APP_ENV": "staging", "LOG_LEVEL": "ERROR"}):
            settings = Settings()
            assert settings.log_level == "ERROR"

    def test_staging_respects_user_wal_disabled(self):
        """Staging should respect user QDRANT_ENABLE_WAL=false."""
        with patch.dict(os.environ, {"APP_ENV": "staging", "QDRANT_ENABLE_WAL": "false"}):
            settings = Settings()
            assert settings.qdrant_enable_wal is False

    def test_production_still_enforces_debug_false(self):
        """Production MUST enforce debug=False even if user sets DEBUG=true."""
        with patch.dict(os.environ, {"APP_ENV": "production", "DEBUG": "true"}):
            settings = Settings()
            assert settings.debug is False, "Production MUST enforce debug=False"

    def test_production_still_enforces_log_level(self):
        """Production MUST enforce log_level=WARNING even if user sets DEBUG."""
        with patch.dict(os.environ, {"APP_ENV": "production", "LOG_LEVEL": "DEBUG"}):
            settings = Settings()
            assert settings.log_level == "WARNING", "Production MUST enforce WARNING log level"

    def test_production_still_enforces_wal(self):
        """Production MUST enforce WAL even if user disables it."""
        with patch.dict(os.environ, {"APP_ENV": "production", "QDRANT_ENABLE_WAL": "false"}):
            settings = Settings()
            assert settings.qdrant_enable_wal is True, "Production MUST enforce WAL"

    def test_production_still_enforces_warmup(self):
        """Production MUST enforce model warmup even if user disables it."""
        with patch.dict(os.environ, {"APP_ENV": "production", "MODEL_WARM_UP": "false"}):
            settings = Settings()
            assert settings.model_warm_up is True, "Production MUST enforce model warmup"
