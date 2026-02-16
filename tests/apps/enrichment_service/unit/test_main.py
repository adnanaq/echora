from __future__ import annotations

import asyncio
import signal
from unittest.mock import AsyncMock

import pytest
from enrichment_service import main


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
