from collections.abc import Iterator
from unittest.mock import AsyncMock, patch

import pytest
from http_cache import network_request_pacer
from http_cache.network_request_pacer import NetworkRequestPacer


@pytest.fixture
def waits() -> Iterator[AsyncMock]:
    with patch.object(network_request_pacer.asyncio, "sleep", autospec=True) as sleep:
        yield sleep


def _waited_seconds(waits: AsyncMock) -> list[float]:
    return [call.args[0] for call in waits.await_args_list]


async def test_wait_for_network_turn_second_request_waits_gap(waits) -> None:
    pacer = NetworkRequestPacer()
    pacer.set_seconds_between_requests(2)
    await pacer.wait_for_network_turn("kitsu")
    await pacer.wait_for_network_turn("kitsu")
    assert _waited_seconds(waits) == [pytest.approx(2, abs=0.05)]


async def test_wait_for_network_turn_other_provider_not_delayed(waits) -> None:
    pacer = NetworkRequestPacer()
    pacer.set_seconds_between_requests(2)
    await pacer.wait_for_network_turn("kitsu")
    await pacer.wait_for_network_turn("anilist")
    assert _waited_seconds(waits) == []


async def test_wait_for_network_turn_provider_minimum_above_default_used(
    waits,
) -> None:
    pacer = NetworkRequestPacer()
    pacer.set_seconds_between_requests(2)
    pacer.set_minimum_seconds_between_requests("anidb", 4)
    await pacer.wait_for_network_turn("anidb")
    await pacer.wait_for_network_turn("anidb")
    assert _waited_seconds(waits) == [pytest.approx(4, abs=0.05)]


async def test_wait_for_network_turn_without_gap_never_waits(waits) -> None:
    pacer = NetworkRequestPacer()
    await pacer.wait_for_network_turn("kitsu")
    await pacer.wait_for_network_turn("kitsu")
    assert _waited_seconds(waits) == []
