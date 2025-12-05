"""Unit tests for retry utility function."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from src.vector.utils.retry import retry_with_backoff


class TestRetryWithBackoff:
    """Test suite for retry_with_backoff utility function."""

    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self):
        """Test that successful operation on first attempt doesn't retry."""
        mock_operation = AsyncMock(return_value="success")

        result = await retry_with_backoff(
            operation=mock_operation,
            max_retries=3,
            retry_delay=0.1,
        )

        assert result == "success"
        assert mock_operation.call_count == 1

    @pytest.mark.asyncio
    async def test_success_after_transient_errors(self):
        """Test that operation retries on transient errors and eventually succeeds."""
        mock_operation = AsyncMock()
        # Fail twice with transient errors, then succeed
        mock_operation.side_effect = [
            Exception("Connection timeout"),
            Exception("Network error"),
            "success"
        ]

        result = await retry_with_backoff(
            operation=mock_operation,
            max_retries=3,
            retry_delay=0.01,  # Small delay for fast tests
        )

        assert result == "success"
        assert mock_operation.call_count == 3

    @pytest.mark.asyncio
    async def test_non_transient_error_fails_immediately(self):
        """Test that non-transient errors don't trigger retries."""
        mock_operation = AsyncMock(side_effect=ValueError("Invalid data"))

        with pytest.raises(ValueError, match="Invalid data"):
            await retry_with_backoff(
                operation=mock_operation,
                max_retries=3,
                retry_delay=0.01,
            )

        # Should only be called once since non-transient error
        assert mock_operation.call_count == 1

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Test that operation fails after max retries are exceeded."""
        mock_operation = AsyncMock(side_effect=Exception("Connection timeout"))

        with pytest.raises(Exception, match="Connection timeout"):
            await retry_with_backoff(
                operation=mock_operation,
                max_retries=2,
                retry_delay=0.01,
            )

        # Should be called max_retries + 1 times (initial + retries)
        assert mock_operation.call_count == 3

    @pytest.mark.asyncio
    async def test_exponential_backoff(self):
        """Test that retry delays follow exponential backoff pattern."""
        mock_operation = AsyncMock()
        mock_operation.side_effect = [
            Exception("timeout"),
            Exception("timeout"),
            "success"
        ]

        start_time = asyncio.get_running_loop().time()

        await retry_with_backoff(
            operation=mock_operation,
            max_retries=3,
            retry_delay=0.1,
        )

        end_time = asyncio.get_running_loop().time()
        elapsed = end_time - start_time

        # First retry: 0.1s, second retry: 0.2s
        # Total should be at least 0.3s
        assert elapsed >= 0.3
        assert mock_operation.call_count == 3

    @pytest.mark.asyncio
    async def test_transient_error_detection(self):
        """Test that transient errors are correctly identified."""
        transient_errors = [
            Exception("Connection timeout occurred"),
            Exception("Network unreachable"),
            Exception("Service temporarily unavailable"),
            Exception("Connection refused"),
        ]

        for error in transient_errors:
            mock_operation = AsyncMock()
            mock_operation.side_effect = [error, "success"]

            result = await retry_with_backoff(
                operation=mock_operation,
                max_retries=2,
                retry_delay=0.01,
            )

            assert result == "success"
            assert mock_operation.call_count == 2

    @pytest.mark.asyncio
    async def test_operation_with_args_and_kwargs(self):
        """Test that operation receives correct args and kwargs."""
        mock_operation = AsyncMock(return_value="success")

        result = await retry_with_backoff(
            operation=mock_operation,
            max_retries=3,
            retry_delay=0.01,
            operation_args=("arg1", "arg2"),
            operation_kwargs={"key1": "value1", "key2": "value2"},
        )

        assert result == "success"
        mock_operation.assert_called_once_with("arg1", "arg2", key1="value1", key2="value2")

    @pytest.mark.asyncio
    async def test_custom_error_checker(self):
        """Test that custom error checker can override default transient detection."""
        def is_retryable(error: Exception) -> bool:
            return "RETRY_ME" in str(error)

        mock_operation = AsyncMock()
        mock_operation.side_effect = [
            Exception("RETRY_ME please"),
            "success"
        ]

        result = await retry_with_backoff(
            operation=mock_operation,
            max_retries=2,
            retry_delay=0.01,
            is_transient_error=is_retryable,
        )

        assert result == "success"
        assert mock_operation.call_count == 2

    @pytest.mark.asyncio
    async def test_zero_retries(self):
        """Test that max_retries=0 means no retries."""
        mock_operation = AsyncMock(side_effect=Exception("timeout"))

        with pytest.raises(Exception, match="timeout"):
            await retry_with_backoff(
                operation=mock_operation,
                max_retries=0,
                retry_delay=0.01,
            )

        assert mock_operation.call_count == 1

    @pytest.mark.asyncio
    async def test_callback_on_retry(self):
        """Test that callback is called on each retry attempt."""
        retry_callback = Mock()
        mock_operation = AsyncMock()
        mock_operation.side_effect = [
            Exception("timeout"),
            Exception("timeout"),
            "success"
        ]

        await retry_with_backoff(
            operation=mock_operation,
            max_retries=3,
            retry_delay=0.01,
            on_retry=retry_callback,
        )

        # Callback should be called twice (for 2 retries)
        assert retry_callback.call_count == 2

        # Check callback was called with correct arguments
        first_call = retry_callback.call_args_list[0]
        assert first_call[1]["attempt"] == 1
        assert first_call[1]["max_retries"] == 3
        assert isinstance(first_call[1]["error"], Exception)
