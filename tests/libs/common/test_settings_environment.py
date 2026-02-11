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

    def test_raises_error_when_environment_not_set(self):
        """Test that missing ENVIRONMENT raises ValueError for production safety."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError, match="ENVIRONMENT environment variable must be set"
            ):
                get_environment()

    def test_detects_development(self):
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            assert get_environment() == Environment.DEVELOPMENT

    def test_detects_production(self):
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            assert get_environment() == Environment.PRODUCTION

    def test_detects_staging(self):
        with patch.dict(os.environ, {"ENVIRONMENT": "staging"}):
            assert get_environment() == Environment.STAGING

    def test_raises_error_for_invalid_value(self):
        """Test that invalid ENVIRONMENT values raise ValueError."""
        with patch.dict(os.environ, {"ENVIRONMENT": "invalid"}):
            with pytest.raises(ValueError, match="Invalid ENVIRONMENT value 'invalid'"):
                get_environment()

    def test_case_insensitive(self):
        with patch.dict(os.environ, {"ENVIRONMENT": "PRODUCTION"}):
            assert get_environment() == Environment.PRODUCTION


class TestSettingsEnvironmentField:
    """Test Settings.environment field."""

    def test_has_environment_field(self):
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            settings = Settings()
            assert hasattr(settings, "environment")

    def test_requires_environment_to_be_set(self):
        """Test that Settings raises error when ENVIRONMENT is not set anywhere."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError, match="ENVIRONMENT environment variable must be set"
            ):
                Settings()

    def test_respects_environment(self):
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            settings = Settings()
            assert settings.environment == Environment.PRODUCTION


class TestProductionSafety:
    """Test production safety enforcement."""

    def test_production_forces_debug_false(self):
        with patch.dict(os.environ, {"ENVIRONMENT": "production", "DEBUG": "true"}):
            settings = Settings()
            assert settings.debug is False

    def test_production_forces_warning_log_level(self):
        with patch.dict(
            os.environ, {"ENVIRONMENT": "production", "LOG_LEVEL": "DEBUG"}
        ):
            settings = Settings()
            assert settings.service.log_level == "WARNING"

    def test_production_enables_wal(self):
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            settings = Settings()
            assert settings.qdrant.qdrant_enable_wal is True

    def test_production_enables_model_warmup(self):
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            settings = Settings()
            assert settings.embedding.model_warm_up is True

    def test_staging_enables_debug(self):
        with patch.dict(os.environ, {"ENVIRONMENT": "staging"}, clear=True):
            settings = Settings()
            assert settings.debug is True
            assert settings.service.log_level == "INFO"
            assert settings.qdrant.qdrant_enable_wal is True

    def test_development_enables_debug(self):
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=True):
            settings = Settings()
            assert settings.debug is True
            assert settings.service.log_level == "DEBUG"


class TestUserProvidedValues:
    """Test that user-provided values are respected in dev/staging but not production."""

    def test_development_respects_user_debug_false(self):
        """Development should respect user DEBUG=false."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development", "DEBUG": "false"}):
            settings = Settings()
            assert settings.debug is False

    def test_development_respects_user_log_level(self):
        """Development should respect user LOG_LEVEL=ERROR."""
        with patch.dict(
            os.environ, {"ENVIRONMENT": "development", "LOG_LEVEL": "ERROR"}
        ):
            settings = Settings()
            assert settings.service.log_level == "ERROR"

    def test_staging_respects_user_debug_false(self):
        """Staging should respect user DEBUG=false."""
        with patch.dict(os.environ, {"ENVIRONMENT": "staging", "DEBUG": "false"}):
            settings = Settings()
            assert settings.debug is False

    def test_staging_respects_user_log_level(self):
        """Staging should respect user LOG_LEVEL=ERROR."""
        with patch.dict(os.environ, {"ENVIRONMENT": "staging", "LOG_LEVEL": "ERROR"}):
            settings = Settings()
            assert settings.service.log_level == "ERROR"

    def test_staging_respects_user_wal_disabled(self):
        """Staging should respect user QDRANT_ENABLE_WAL=false."""
        with patch.dict(
            os.environ, {"ENVIRONMENT": "staging", "QDRANT_ENABLE_WAL": "false"}
        ):
            settings = Settings()
            assert settings.qdrant.qdrant_enable_wal is False

    def test_production_still_enforces_debug_false(self):
        """Production MUST enforce debug=False even if user sets DEBUG=true."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production", "DEBUG": "true"}):
            settings = Settings()
            assert settings.debug is False, "Production MUST enforce debug=False"

    def test_production_still_enforces_log_level(self):
        """Production MUST enforce log_level=WARNING even if user sets DEBUG."""
        with patch.dict(
            os.environ, {"ENVIRONMENT": "production", "LOG_LEVEL": "DEBUG"}
        ):
            settings = Settings()
            assert settings.service.log_level == "WARNING", (
                "Production MUST enforce WARNING log level"
            )

    def test_production_still_enforces_wal(self):
        """Production MUST enforce WAL even if user disables it."""
        with patch.dict(
            os.environ, {"ENVIRONMENT": "production", "QDRANT_ENABLE_WAL": "false"}
        ):
            settings = Settings()
            assert settings.qdrant.qdrant_enable_wal is True, (
                "Production MUST enforce WAL"
            )

    def test_production_still_enforces_warmup(self):
        """Production MUST enforce model warmup even if user disables it."""
        with patch.dict(
            os.environ, {"ENVIRONMENT": "production", "MODEL_WARM_UP": "false"}
        ):
            settings = Settings()
            assert settings.embedding.model_warm_up is True, (
                "Production MUST enforce model warmup"
            )

    def test_development_respects_explicit_default_value(self):
        """Development should respect LOG_LEVEL=INFO even though INFO is the default.

        REGRESSION TEST: Previously, the code compared current value to default,
        so setting LOG_LEVEL=INFO (which equals the default) would be overridden
        to DEBUG. Now we check os.environ directly, so explicit defaults are respected.
        """
        with patch.dict(
            os.environ, {"ENVIRONMENT": "development", "LOG_LEVEL": "INFO"}
        ):
            settings = Settings()
            # User explicitly set LOG_LEVEL=INFO, should NOT be overridden to DEBUG
            assert settings.service.log_level == "INFO"

    def test_staging_respects_explicit_default_debug(self):
        """Staging should respect DEBUG=true even though true is staging's target default.

        REGRESSION TEST: Ensures explicit env vars are respected even when they
        match the environment-specific default we would otherwise apply.
        """
        with patch.dict(os.environ, {"ENVIRONMENT": "staging", "DEBUG": "true"}):
            settings = Settings()
            # User explicitly set DEBUG=true, should be respected
            assert settings.debug is True


class TestMultivectorConfiguration:
    """Test multivector settings configuration."""

    def test_multivector_vectors_default(self):
        """Test default multivector configuration."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            settings = Settings()
            assert "image_vector" in settings.qdrant.multivector_vectors
            assert settings.qdrant.multivector_vectors == ["image_vector"]

    def test_vector_names_unchanged(self):
        """Test vector_names still contains both vectors."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            settings = Settings()
            assert settings.qdrant.vector_names == {
                "text_vector": 1024,
                "image_vector": 768,
            }

    def test_invalid_multivector_vectors_raises_error(self):
        """Test that unknown multivector vectors raise ValueError."""
        with patch.dict(
            os.environ,
            {
                "ENVIRONMENT": "development",
                "MULTIVECTOR_VECTORS": '["nonexistent_vector"]',
            },
        ):
            with pytest.raises(ValueError, match="Unknown multivector vectors"):
                Settings()


class TestDistributeEnvVarsEdgeCases:
    """Test edge cases in distribute_env_vars validator."""

    def test_env_var_overrides_pre_built_sub_config(self):
        """Test that env vars override pre-built sub-config objects.

        When Settings receives a pre-built sub-config object (e.g.,
        Settings(qdrant=QdrantConfig(qdrant_url="x"))) and a corresponding
        env var is set, the env var should take precedence.

        This tests the behavior on line 196 where the entire object is
        replaced with a dict from env vars rather than merging.
        """
        from common.config.qdrant_config import QdrantConfig

        # Pre-build a QdrantConfig with a specific URL
        custom_qdrant = QdrantConfig(qdrant_url="http://custom:6333")

        # Set env var that should override it
        with patch.dict(
            os.environ,
            {"ENVIRONMENT": "development", "QDRANT_URL": "http://envvar:6333"},
        ):
            settings = Settings(qdrant=custom_qdrant)

            # Env var should win
            assert settings.qdrant.qdrant_url == "http://envvar:6333"
            assert settings.qdrant.qdrant_url != "http://custom:6333"

    def test_pre_built_sub_config_respected_without_env_var(self):
        """Test that pre-built sub-config objects are respected when no env var set."""
        from common.config.qdrant_config import QdrantConfig

        # Pre-build a QdrantConfig with a specific URL
        custom_qdrant = QdrantConfig(qdrant_url="http://custom:6333")

        with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=True):
            settings = Settings(qdrant=custom_qdrant)

            # Custom value should be preserved
            assert settings.qdrant.qdrant_url == "http://custom:6333"

    def test_env_var_merges_with_pre_built_sub_config(self):
        """Test that env vars MERGE with pre-built sub-config, preserving non-overridden fields.

        When a pre-built Pydantic model (e.g., QdrantConfig) is provided and
        env vars are set for specific fields, the env vars override those fields
        while preserving all other fields from the pre-built object.

        This implements proper precedence: env var > pre-built field > default
        """
        from common.config.qdrant_config import QdrantConfig

        # Pre-build a QdrantConfig with multiple custom values
        custom_qdrant = QdrantConfig(
            qdrant_url="http://custom:6333",
            qdrant_collection_name="custom_collection",
        )

        # Set env var for only ONE field
        with patch.dict(
            os.environ,
            {"ENVIRONMENT": "development", "QDRANT_URL": "http://envvar:6333"},
        ):
            settings = Settings(qdrant=custom_qdrant)

            # Env var field wins (overrides pre-built)
            assert settings.qdrant.qdrant_url == "http://envvar:6333"

            # Other pre-built field is PRESERVED (not lost)
            assert settings.qdrant.qdrant_collection_name == "custom_collection"

    def test_multiple_env_vars_preserve_pre_built_fields(self):
        """Test that multiple env vars merge with pre-built config correctly."""
        from common.config.qdrant_config import QdrantConfig

        # Pre-build with many custom values
        custom_qdrant = QdrantConfig(
            qdrant_url="http://custom:6333",
            qdrant_collection_name="custom_collection",
            qdrant_enable_quantization=True,
            qdrant_quantization_type="binary",
        )

        # Set env vars for TWO fields
        with patch.dict(
            os.environ,
            {
                "ENVIRONMENT": "development",
                "QDRANT_URL": "http://envvar:6333",
                "QDRANT_ENABLE_QUANTIZATION": "false",
            },
        ):
            settings = Settings(qdrant=custom_qdrant)

            # Env vars override their specific fields
            assert settings.qdrant.qdrant_url == "http://envvar:6333"
            assert settings.qdrant.qdrant_enable_quantization is False

            # Non-overridden pre-built fields are preserved
            assert settings.qdrant.qdrant_collection_name == "custom_collection"
            assert settings.qdrant.qdrant_quantization_type == "binary"

    def test_invalid_json_env_var_logs_warning(self, caplog):
        """Test that invalid JSON in env vars logs a warning.

        When a user sets an env var with invalid JSON for a complex field,
        the parser should log a warning to help diagnose misconfiguration.
        """
        import logging

        from pydantic import ValidationError

        with caplog.at_level(logging.WARNING):
            with patch.dict(
                os.environ,
                {
                    "ENVIRONMENT": "development",
                    "VECTOR_NAMES": "not valid json",  # Invalid JSON
                },
            ):
                try:
                    Settings()
                except ValidationError:
                    # Pydantic rejects the invalid value, which is expected
                    pass

            # Check that a warning was logged about the parse failure
            assert any(
                "Failed to parse JSON for field 'vector_names'" in record.message
                for record in caplog.records
            ), "Expected warning about JSON parse failure"


class TestAgentSettings:
    """Test agent-specific settings distributed via common Settings."""

    def test_agent_prefixed_env_vars_are_loaded(self):
        """AGENT_* environment variables should map into settings.agent.* fields."""
        with patch.dict(
            os.environ,
            {
                "ENVIRONMENT": "development",
                "AGENT_SERVICE_PORT": "50123",
                "AGENT_LLM_ENABLED": "false",
                "AGENT_OPENAI_MODEL": "gpt-5",
            },
            clear=True,
        ):
            settings = Settings()
            assert settings.agent.service_port == 50123
            assert settings.agent.llm_enabled is False
            assert settings.agent.openai_model == "gpt-5"

    def test_agent_constructor_values_preserved_without_env_override(self):
        """Explicit agent sub-config values should be preserved if env vars are unset."""
        from common.config.agent_config import AgentConfig

        custom_agent = AgentConfig(
            service_port=51111,
            llm_enabled=False,
            openai_model="local-model",
        )
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=True):
            settings = Settings(agent=custom_agent)
            assert settings.agent.service_port == 51111
            assert settings.agent.llm_enabled is False
            assert settings.agent.openai_model == "local-model"
