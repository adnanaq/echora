from __future__ import annotations

import asyncio
import signal
from unittest.mock import AsyncMock

import pytest
from enrichment_service import main


def test_setup_observability_calls_telemetry_bootstrap(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_setup_telemetry(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(main, "setup_telemetry", _fake_setup_telemetry)

    settings = type(
        "Settings",
        (),
        {
            "service": type(
                "Service",
                (),
                {
                    "api_version": "1.0.0",
                    "log_level": "DEBUG",
                },
            )(),
            "observability": type(
                "Observability",
                (),
                {
                    "otel_enabled": True,
                    "otel_exporter_otlp_endpoint": "http://otel:4317",
                    "otel_enable_metrics": True,
                    "otel_enable_tracing": True,
                    "otel_enable_logging": True,
                    "otel_enable_grpc_server_instrumentation": True,
                    "otel_enable_grpc_client_instrumentation": True,
                    "otel_enable_aiohttp_client_instrumentation": True,
                    "otel_enable_redis_instrumentation": False,
                },
            )(),
            "environment": type("Env", (), {"value": "staging"})(),
        },
    )()

    main._setup_observability(settings)

    assert captured["service_name"] == "echora-enrichment-service"
    assert captured["environment"] == "staging"
    assert captured["endpoint"] == "http://otel:4317"
    assert captured["enable_aiohttp_client_instrumentation"] is True


class _FakeLoop:
    def __init__(self) -> None:
        self.handlers: dict[signal.Signals, object] = {}

    def add_signal_handler(self, sig: signal.Signals, callback: object) -> None:
        self.handlers[sig] = callback


@pytest.mark.asyncio
async def test_register_sigterm_shutdown_stops_server_with_grace() -> None:
    server = AsyncMock()
    loop = _FakeLoop()

    main._register_sigterm_shutdown(server, grace=5, loop=loop)

    assert signal.SIGTERM in loop.handlers

    callback = loop.handlers[signal.SIGTERM]
    callback()
    await asyncio.sleep(0)

    server.stop.assert_awaited_once_with(grace=5)
