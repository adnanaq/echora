"""Unit tests verifying enrichment_service browser pool startup configuration."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_serve_configures_browser_pool_from_settings() -> None:
    """Confirm serve() applies enrichment_max_concurrent_browsers at startup."""
    from enrichment_service import main as service_main

    fake_settings = SimpleNamespace(
        environment=SimpleNamespace(value="development"),
        observability=SimpleNamespace(otel_enabled=False),
        service=SimpleNamespace(
            enrichment_service_host="127.0.0.1",
            enrichment_service_port=0,
            max_grpc_message_size_mb=64,
            enrichment_grpc_max_concurrent_rpcs=8,
            log_level="INFO",
            api_version="0.1.0",
        ),
        enrichment=SimpleNamespace(enrichment_max_concurrent_browsers=3),
    )
    fake_server = MagicMock()
    fake_server.add_insecure_port = MagicMock()
    fake_server.start = AsyncMock()
    fake_server.wait_for_termination = AsyncMock()
    fake_server.stop = AsyncMock()
    fake_runtime = SimpleNamespace(cache=MagicMock())

    with (
        patch.object(service_main, "get_settings", return_value=fake_settings),
        patch.object(service_main, "configure_browser_pool") as mock_pool,
        patch.object(
            service_main, "build_runtime", new=AsyncMock(return_value=fake_runtime)
        ),
        patch.object(service_main.grpc.aio, "server", return_value=fake_server),
        patch.object(service_main, "close_cache", new=AsyncMock()),
    ):
        await service_main.serve()

    mock_pool.assert_called_once_with(3)
