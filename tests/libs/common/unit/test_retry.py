from collections.abc import Iterator
from unittest.mock import patch

import pytest
from common.utils import retry as retry_module
from common.utils.retry import retry_with_backoff


def _is_connection_error(error: Exception) -> bool:
    return isinstance(error, (ConnectionError, TimeoutError))


def _build_operation(outcomes: list[object]):
    remaining_outcomes: Iterator[object] = iter(outcomes)
    calls: list[int] = []

    async def operation() -> object:
        calls.append(1)
        outcome = next(remaining_outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return operation, calls


@pytest.mark.asyncio
async def test_retry_with_backoff_negative_max_retries_raises_value_error() -> None:
    operation, _ = _build_operation(["ok"])
    with pytest.raises(ValueError, match="max_retries must be >= 0"):
        await retry_with_backoff(
            operation, max_retries=-1, is_transient_error=_is_connection_error
        )


@pytest.mark.asyncio
async def test_retry_with_backoff_negative_retry_delay_raises_value_error() -> None:
    operation, _ = _build_operation(["ok"])
    with pytest.raises(ValueError, match="retry_delay must be >= 0"):
        await retry_with_backoff(
            operation, retry_delay=-0.1, is_transient_error=_is_connection_error
        )


@pytest.mark.asyncio
async def test_retry_with_backoff_forwards_operation_args_and_kwargs() -> None:
    async def add_numbers(first: int, second: int, *, third: int) -> int:
        return first + second + third

    total = await retry_with_backoff(
        add_numbers,
        operation_args=(2, 3),
        operation_kwargs={"third": 5},
        is_transient_error=_is_connection_error,
    )
    assert total == 10


@pytest.mark.asyncio
async def test_retry_with_backoff_transient_errors_retried_with_doubling_delay() -> (
    None
):
    operation, calls = _build_operation(
        [ConnectionError("network"), TimeoutError("timeout"), "ok"]
    )
    retry_reports: list[dict[str, object]] = []
    with patch.object(retry_module.asyncio, "sleep", autospec=True) as sleep:
        outcome = await retry_with_backoff(
            operation,
            max_retries=3,
            retry_delay=0.5,
            is_transient_error=_is_connection_error,
            on_retry=lambda **retry_report: retry_reports.append(retry_report),
        )
    assert (
        outcome,
        len(calls),
        [call.args[0] for call in sleep.await_args_list],
        [(report["attempt"], report["delay"]) for report in retry_reports],
    ) == ("ok", 3, [0.5, 1.0], [(1, 0.5), (2, 1.0)])


@pytest.mark.asyncio
async def test_retry_with_backoff_error_check_refuses_raises_at_once() -> None:
    operation, calls = _build_operation([ValueError("bad input")])
    with patch.object(retry_module.asyncio, "sleep", autospec=True) as sleep:
        with pytest.raises(ValueError, match="bad input"):
            await retry_with_backoff(
                operation, max_retries=5, is_transient_error=_is_connection_error
            )
    assert (len(calls), sleep.await_count) == (1, 0)


@pytest.mark.asyncio
async def test_retry_with_backoff_retries_used_up_raises_last_error() -> None:
    operation, calls = _build_operation(
        [ConnectionError("still down"), ConnectionError("still down")]
    )
    with patch.object(retry_module.asyncio, "sleep", autospec=True) as sleep:
        with pytest.raises(ConnectionError, match="still down"):
            await retry_with_backoff(
                operation,
                max_retries=1,
                retry_delay=0.25,
                is_transient_error=_is_connection_error,
            )
    assert (len(calls), [call.args[0] for call in sleep.await_args_list]) == (
        2,
        [0.25],
    )
