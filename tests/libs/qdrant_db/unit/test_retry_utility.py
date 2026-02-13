"""Unit tests for retry utility function."""

import asyncio
from unittest.mock import AsyncMock

import pytest
from qdrant_db.utils.retry import default_is_transient_error, retry_with_backoff


class FakeApiException(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.status = status_code


class TestRetryWithBackoff:
    @pytest.mark.asyncio
    async def test_success_after_transient_errors(self) -> None:
        mock_operation = AsyncMock()
        mock_operation.side_effect = [Exception("Connection timeout"), "success"]

        result = await retry_with_backoff(
            operation=mock_operation,
            max_retries=3,
            retry_delay=0.01,
        )

        assert result == "success"
        assert mock_operation.call_count == 2

    @pytest.mark.asyncio
    async def test_non_transient_error_fails_immediately(self) -> None:
        mock_operation = AsyncMock(side_effect=ValueError("Invalid data"))

        with pytest.raises(ValueError, match="Invalid data"):
            await retry_with_backoff(
                operation=mock_operation,
                max_retries=3,
                retry_delay=0.01,
            )

        assert mock_operation.call_count == 1

    @pytest.mark.asyncio
    async def test_exponential_backoff(self) -> None:
        mock_operation = AsyncMock()
        mock_operation.side_effect = [Exception("timeout"), Exception("timeout"), "ok"]

        start_time = asyncio.get_running_loop().time()
        result = await retry_with_backoff(
            operation=mock_operation,
            max_retries=3,
            retry_delay=0.05,
        )
        elapsed = asyncio.get_running_loop().time() - start_time

        assert result == "ok"
        assert elapsed >= 0.15


class TestDefaultIsTransientError:
    def test_timeout_and_connection_errors_are_transient(self) -> None:
        assert default_is_transient_error(TimeoutError("timeout"))
        assert default_is_transient_error(ConnectionError("connection"))

    def test_retryable_http_status_is_transient(self) -> None:
        assert default_is_transient_error(FakeApiException(429))
        assert default_is_transient_error(FakeApiException(503))

    def test_non_retryable_http_status_is_not_transient(self) -> None:
        assert not default_is_transient_error(FakeApiException(400))

    def test_generic_keyword_fallback_still_works(self) -> None:
        assert default_is_transient_error(Exception("temporary network unavailable"))
