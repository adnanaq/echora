"""Unit tests for retry utility helpers."""

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from qdrant_db.utils import retry as retry_module
from qdrant_db.utils.retry import (
    _extract_status_code,
    default_is_transient_error,
    retry_with_backoff,
)


class _StatusError(Exception):
    """Exception carrying optional status-like fields."""

    def __init__(
        self,
        *,
        status: int | None = None,
        status_code: int | None = None,
        code: int | None = None,
        message: str = "status error",
    ) -> None:
        super().__init__(message)
        if status is not None:
            self.status = status
        if status_code is not None:
            self.status_code = status_code
        if code is not None:
            self.code = code


@pytest.mark.asyncio
async def test_retry_with_backoff_rejects_negative_max_retries() -> None:
    """Negative max_retries must be rejected."""
    with pytest.raises(ValueError, match="max_retries must be >= 0"):
        await retry_with_backoff(operation=AsyncMock(return_value="ok"), max_retries=-1)


@pytest.mark.asyncio
async def test_retry_with_backoff_rejects_negative_retry_delay() -> None:
    """Negative retry_delay must be rejected."""
    with pytest.raises(ValueError, match="retry_delay must be >= 0"):
        await retry_with_backoff(operation=AsyncMock(return_value="ok"), retry_delay=-0.1)


@pytest.mark.asyncio
async def test_retry_with_backoff_passes_operation_args_and_kwargs() -> None:
    """Operation args and kwargs should be forwarded to the callable."""

    async def operation(a: int, b: int, *, c: int) -> int:
        return a + b + c

    result = await retry_with_backoff(
        operation=operation,
        operation_args=(2, 3),
        operation_kwargs={"c": 5},
    )
    assert result == 10


@pytest.mark.asyncio
async def test_retry_with_backoff_retries_transient_error_until_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transient errors should trigger retries and exponential delays."""
    sleep_calls: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr(retry_module.asyncio, "sleep", fake_sleep)

    operation = AsyncMock(
        side_effect=[ConnectionError("network"), TimeoutError("timeout"), "ok"]
    )
    on_retry = Mock()

    result = await retry_with_backoff(
        operation=operation,
        max_retries=3,
        retry_delay=0.5,
        on_retry=on_retry,
    )

    assert result == "ok"
    assert operation.call_count == 3
    assert sleep_calls == [0.5, 1.0]
    assert on_retry.call_count == 2
    first_retry = on_retry.call_args_list[0].kwargs
    second_retry = on_retry.call_args_list[1].kwargs
    assert first_retry["attempt"] == 1
    assert second_retry["attempt"] == 2
    assert first_retry["max_retries"] == 3
    assert second_retry["max_retries"] == 3
    assert first_retry["delay"] == 0.5
    assert second_retry["delay"] == 1.0


@pytest.mark.asyncio
async def test_retry_with_backoff_non_transient_error_fails_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-transient errors should fail immediately without sleeping."""
    sleep = AsyncMock()
    monkeypatch.setattr(retry_module.asyncio, "sleep", sleep)
    operation = AsyncMock(side_effect=ValueError("bad input"))

    with pytest.raises(ValueError, match="bad input"):
        await retry_with_backoff(operation=operation, max_retries=5, retry_delay=0.1)

    assert operation.call_count == 1
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_retry_with_backoff_honors_custom_transient_checker() -> None:
    """Custom transient checker should control retry behavior."""
    operation = AsyncMock(side_effect=RuntimeError("temporary issue"))
    checker = Mock(return_value=False)

    with pytest.raises(RuntimeError, match="temporary issue"):
        await retry_with_backoff(
            operation=operation,
            max_retries=3,
            is_transient_error=checker,
        )

    assert operation.call_count == 1
    checker.assert_called_once()


@pytest.mark.asyncio
async def test_retry_with_backoff_raises_after_max_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry loop should re-raise after max retry attempts are exhausted."""
    sleep_calls: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr(retry_module.asyncio, "sleep", fake_sleep)
    exception_logger = Mock()
    monkeypatch.setattr(retry_module.logger, "exception", exception_logger)

    operation = AsyncMock(side_effect=ConnectionError("still down"))

    with pytest.raises(ConnectionError, match="still down"):
        await retry_with_backoff(operation=operation, max_retries=1, retry_delay=0.25)

    assert operation.call_count == 2
    assert sleep_calls == [0.25]
    exception_logger.assert_called_once()


def test_extract_status_code_prefers_status_field() -> None:
    """Status extraction should prefer status before status_code/code."""
    error = _StatusError(status=503, status_code=400, code=401)
    assert _extract_status_code(error) == 503


def test_extract_status_code_uses_status_code_then_code() -> None:
    """Status extraction should fall back to status_code and then code."""
    assert _extract_status_code(_StatusError(status_code=429)) == 429
    assert _extract_status_code(_StatusError(code=504)) == 504


def test_extract_status_code_returns_none_when_missing() -> None:
    """Status extraction should return None if no recognized fields exist."""
    assert _extract_status_code(Exception("no status")) is None


def test_default_is_transient_error_for_timeout_and_connection() -> None:
    """Built-in timeout/connection exceptions should be transient."""
    assert default_is_transient_error(TimeoutError("timeout"))
    assert default_is_transient_error(ConnectionError("connection error"))


def test_default_is_transient_error_for_response_handling_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Response handling exceptions should be considered transient."""

    class _FakeResponseHandlingException(Exception):
        pass

    monkeypatch.setattr(
        retry_module,
        "ResponseHandlingException",
        _FakeResponseHandlingException,
    )

    assert default_is_transient_error(_FakeResponseHandlingException("decode error"))


def test_default_is_transient_error_for_api_exception_retryable_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ApiException with retryable status should be classified as transient."""

    class _FakeApiException(Exception):
        def __init__(self, status: int) -> None:
            super().__init__(f"status {status}")
            self.status = status

    monkeypatch.setattr(retry_module, "ApiException", _FakeApiException)

    assert default_is_transient_error(_FakeApiException(429))
    assert default_is_transient_error(_FakeApiException(503))


def test_default_is_transient_error_for_api_exception_non_retryable_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ApiException with non-retryable status should not be transient."""

    class _FakeApiException(Exception):
        def __init__(self, status: int) -> None:
            super().__init__(f"status {status}")
            self.status = status

    monkeypatch.setattr(retry_module, "ApiException", _FakeApiException)

    assert not default_is_transient_error(_FakeApiException(400))


def test_default_is_transient_error_for_generic_status_code() -> None:
    """Generic exceptions with retryable status_code should be transient."""
    assert default_is_transient_error(_StatusError(status_code=504))
    assert not default_is_transient_error(_StatusError(status_code=422))


def test_default_is_transient_error_keyword_fallback() -> None:
    """Keyword fallback should classify clearly transient error messages."""
    assert default_is_transient_error(Exception("temporary network unavailable"))
    assert not default_is_transient_error(Exception("invalid payload schema"))


def test_retry_module_import_guard_fallback_creates_placeholder_exception_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Import-guard fallback should define placeholder ApiException classes."""
    retry_path = Path(retry_module.__file__).resolve()
    spec = importlib.util.spec_from_file_location("qdrant_retry_fallback_test", retry_path)
    assert spec is not None and spec.loader is not None

    loaded_module = ModuleType("qdrant_retry_fallback_test")

    import builtins

    original_import = builtins.__import__

    def fake_import(
        name: str,
        globals_dict: dict[str, Any] | None = None,
        locals_dict: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "qdrant_client.http.exceptions":
            raise ImportError("forced import failure for fallback branch")
        return original_import(name, globals_dict, locals_dict, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    spec.loader.exec_module(loaded_module)

    assert isinstance(loaded_module.ApiException, type)
    assert isinstance(loaded_module.ResponseHandlingException, type)
