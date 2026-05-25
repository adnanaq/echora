from __future__ import annotations

from unittest.mock import patch

from common.config.settings import Settings


def test_settings_routes_otel_env_vars_to_observability_config() -> None:
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
