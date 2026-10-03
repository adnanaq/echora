from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from common.config.settings import Settings


def test_settings_otel_env_vars_fill_observability_config() -> None:
    with patch.dict(
        "os.environ",
        {
            "ENVIRONMENT": "development",
            "OTEL_ENABLED": "false",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector:4317",
            "OTEL_ENABLE_AIOHTTP_CLIENT_INSTRUMENTATION": "true",
        },
        clear=True,
    ):
        settings = Settings()

    assert settings.observability.otel_enabled is False
    assert settings.observability.otel_exporter_otlp_endpoint == "http://collector:4317"
    assert settings.observability.otel_enable_aiohttp_client_instrumentation is True


def test_settings_record_query_text_unset_defaults_to_false() -> None:
    with patch.dict("os.environ", {"ENVIRONMENT": "development"}, clear=True):
        settings = Settings()

    assert settings.observability.otel_record_query_text is False


def test_settings_record_query_text_env_true_turns_recording_on() -> None:
    with patch.dict(
        "os.environ",
        {"ENVIRONMENT": "development", "OTEL_RECORD_QUERY_TEXT": "true"},
        clear=True,
    ):
        settings = Settings()

    assert settings.observability.otel_record_query_text is True


def test_settings_local_env_file_present_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text("OTEL_RECORD_QUERY_TEXT=true\n")
    monkeypatch.chdir(tmp_path)

    with patch.dict("os.environ", {"ENVIRONMENT": "development"}, clear=True):
        settings = Settings()

    assert settings.observability.otel_record_query_text is False
