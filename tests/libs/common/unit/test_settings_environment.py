import os
from unittest.mock import patch

import pytest
from common.config import QdrantConfig, ServiceConfig
from common.config.settings import Environment, Settings, get_settings
from pydantic import ValidationError
from pydantic_settings import SettingsError


@pytest.mark.parametrize(
    ("member", "text"),
    [
        (Environment.DEVELOPMENT, "development"),
        (Environment.STAGING, "staging"),
        (Environment.PRODUCTION, "production"),
    ],
)
def test_environment_member_reads_as_its_value(member, text):
    assert member == text
    assert str(member) == text
    assert f"{member}" == text


def test_settings_without_environment_raises_validation_error():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValidationError, match="environment"):
            Settings()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("development", Environment.DEVELOPMENT),
        ("staging", Environment.STAGING),
        ("production", Environment.PRODUCTION),
        ("PRODUCTION", Environment.PRODUCTION),
    ],
)
def test_settings_environment_variable_sets_environment(value, expected):
    with patch.dict(os.environ, {"ENVIRONMENT": value}, clear=True):
        assert Settings().environment == expected


def test_settings_unknown_environment_raises_validation_error():
    with patch.dict(os.environ, {"ENVIRONMENT": "invalid"}, clear=True):
        with pytest.raises(ValidationError, match="environment"):
            Settings()


def test_settings_production_overrides_user_values():
    with patch.dict(
        os.environ,
        {
            "ENVIRONMENT": "production",
            "DEBUG": "true",
            "LOG_LEVEL": "DEBUG",
            "QDRANT_ENABLE_WAL": "false",
            "MODEL_WARM_UP": "false",
        },
        clear=True,
    ):
        settings = Settings()

    assert settings.debug is False
    assert settings.service.log_level == "WARNING"
    assert settings.qdrant.qdrant_enable_wal is True
    assert settings.embedding.model_warm_up is True


def test_settings_development_unset_values_get_development_defaults():
    with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=True):
        settings = Settings()

    assert settings.debug is True
    assert settings.service.log_level == "DEBUG"


def test_settings_staging_unset_values_get_staging_defaults():
    with patch.dict(os.environ, {"ENVIRONMENT": "staging"}, clear=True):
        settings = Settings()

    assert settings.debug is True
    assert settings.service.log_level == "INFO"
    assert settings.qdrant.qdrant_enable_wal is True


@pytest.mark.parametrize("environment", ["development", "staging"])
def test_settings_user_environment_variables_kept_outside_production(environment):
    with patch.dict(
        os.environ,
        {
            "ENVIRONMENT": environment,
            "DEBUG": "false",
            "LOG_LEVEL": "ERROR",
            "QDRANT_ENABLE_WAL": "false",
        },
        clear=True,
    ):
        settings = Settings()

    assert settings.debug is False
    assert settings.service.log_level == "ERROR"
    assert settings.qdrant.qdrant_enable_wal is False


def test_settings_development_log_level_equal_to_code_default_kept():
    with patch.dict(
        os.environ, {"ENVIRONMENT": "development", "LOG_LEVEL": "INFO"}, clear=True
    ):
        assert Settings().service.log_level == "INFO"


def test_settings_staging_debug_equal_to_staging_default_kept():
    with patch.dict(
        os.environ, {"ENVIRONMENT": "staging", "DEBUG": "true"}, clear=True
    ):
        assert Settings().debug is True


def test_settings_development_log_level_given_in_code_kept():
    with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=True):
        settings = Settings(service=ServiceConfig(log_level="ERROR"))

    assert settings.service.log_level == "ERROR"


def test_settings_sub_config_given_in_code_wins_over_environment_variable():
    with patch.dict(
        os.environ,
        {"ENVIRONMENT": "development", "QDRANT_URL": "http://envvar:6333"},
        clear=True,
    ):
        settings = Settings(qdrant=QdrantConfig(qdrant_url="http://custom:6333"))

    assert settings.qdrant.qdrant_url == "http://custom:6333"


def test_settings_sub_config_environment_variable_sets_field():
    with patch.dict(
        os.environ,
        {"ENVIRONMENT": "development", "QDRANT_URL": "http://envvar:6333"},
        clear=True,
    ):
        assert Settings().qdrant.qdrant_url == "http://envvar:6333"


def test_settings_multivector_vectors_default_is_image_vector():
    with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=True):
        settings = Settings()

    assert settings.qdrant.multivector_vectors == ["image_vector"]
    assert settings.qdrant.vector_names == {"text_vector": 1024, "image_vector": 768}


def test_settings_unknown_multivector_vector_raises_validation_error():
    with patch.dict(
        os.environ,
        {"ENVIRONMENT": "development", "MULTIVECTOR_VECTORS": '["nonexistent_vector"]'},
        clear=True,
    ):
        with pytest.raises(ValidationError, match="Unknown multivector vectors"):
            Settings()


def test_settings_sparse_vectors_default_is_text_sparse_vector():
    with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=True):
        settings = Settings()

    assert settings.qdrant.sparse_vector_names == ["text_sparse_vector"]
    assert settings.qdrant.primary_sparse_vector_name == "text_sparse_vector"


def test_settings_primary_sparse_name_missing_from_sparse_names_raises_validation_error():
    with patch.dict(
        os.environ,
        {
            "ENVIRONMENT": "development",
            "SPARSE_VECTOR_NAMES": '["sparse_a"]',
            "PRIMARY_SPARSE_VECTOR_NAME": "sparse_b",
        },
        clear=True,
    ):
        with pytest.raises(
            ValidationError,
            match="primary_sparse_vector_name must be a key in sparse_vector_names",
        ):
            Settings()


def test_settings_invalid_json_environment_variable_raises_settings_error():
    with patch.dict(
        os.environ,
        {"ENVIRONMENT": "development", "VECTOR_NAMES": "not valid json"},
        clear=True,
    ):
        with pytest.raises(SettingsError, match="vector_names"):
            Settings()


def test_get_settings_repeated_call_returns_same_instance():
    get_settings.cache_clear()
    with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=True):
        first = get_settings()
        second = get_settings()
    get_settings.cache_clear()

    assert first is second
