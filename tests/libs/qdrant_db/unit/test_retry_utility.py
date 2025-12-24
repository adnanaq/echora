"""Unit tests for retry utility function."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from qdrant_db.utils.retry import retry_with_backoff, default_is_transient_error


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

    @pytest.mark.asyncio
    async def test_negative_max_retries_validation(self):
        """Test that negative max_retries raises ValueError."""
        mock_operation = AsyncMock(return_value="success")

        with pytest.raises(ValueError, match="max_retries must be >= 0"):
            await retry_with_backoff(
                operation=mock_operation,
                max_retries=-1,
                retry_delay=0.01,
            )

        # Operation should never be called due to validation failure
        assert mock_operation.call_count == 0

    @pytest.mark.asyncio
    async def test_negative_retry_delay_validation(self):
        """Test that negative retry_delay raises ValueError."""
        mock_operation = AsyncMock(return_value="success")

        with pytest.raises(ValueError, match="retry_delay must be >= 0"):
            await retry_with_backoff(
                operation=mock_operation,
                max_retries=3,
                retry_delay=-0.5,
            )

        # Operation should never be called due to validation failure
        assert mock_operation.call_count == 0

    @pytest.mark.asyncio
    async def test_type_based_transient_error_detection(self):
        """Test that common transient exception types are detected."""
        # Test asyncio.TimeoutError
        mock_operation = AsyncMock()
        mock_operation.side_effect = [asyncio.TimeoutError(), "success"]

        result = await retry_with_backoff(
            operation=mock_operation,
            max_retries=2,
            retry_delay=0.01,
        )

        assert result == "success"
        assert mock_operation.call_count == 2

        # Test ConnectionError
        mock_operation = AsyncMock()
        mock_operation.side_effect = [ConnectionError("Connection failed"), "success"]

        result = await retry_with_backoff(
            operation=mock_operation,
            max_retries=2,
            retry_delay=0.01,
        )

        assert result == "success"
        assert mock_operation.call_count == 2

        # Test TimeoutError
        mock_operation = AsyncMock()
        mock_operation.side_effect = [TimeoutError("Timeout occurred"), "success"]

        result = await retry_with_backoff(
            operation=mock_operation,
            max_retries=2,
            retry_delay=0.01,
        )

        assert result == "success"
        assert mock_operation.call_count == 2

    @pytest.mark.asyncio
    async def test_none_args_and_kwargs_normalization(self):
        """Test that None args and kwargs are properly normalized."""
        mock_operation = AsyncMock(return_value="success")

        result = await retry_with_backoff(
            operation=mock_operation,
            max_retries=1,
            retry_delay=0.01,
            operation_args=None,
            operation_kwargs=None,
        )

        assert result == "success"
        mock_operation.assert_called_once_with()


class TestDefaultIsTransientError:
    """Test suite for default_is_transient_error function."""

    def test_type_based_detection_asyncio_timeout(self):
        """Test that asyncio.TimeoutError is detected as transient."""
        error = asyncio.TimeoutError()
        assert default_is_transient_error(error) is True

    def test_type_based_detection_connection_error(self):
        """Test that ConnectionError is detected as transient."""
        error = ConnectionError("Connection failed")
        assert default_is_transient_error(error) is True

    def test_type_based_detection_timeout_error(self):
        """Test that TimeoutError is detected as transient."""
        error = TimeoutError("Timeout occurred")
        assert default_is_transient_error(error) is True

    def test_keyword_based_detection_timeout(self):
        """Test that errors with 'timeout' keyword are detected as transient."""
        error = Exception("Connection timeout occurred")
        assert default_is_transient_error(error) is True

    def test_keyword_based_detection_connection(self):
        """Test that errors with 'connection' keyword are detected as transient."""
        error = Exception("Connection refused")
        assert default_is_transient_error(error) is True

    def test_keyword_based_detection_network(self):
        """Test that errors with 'network' keyword are detected as transient."""
        error = Exception("Network unreachable")
        assert default_is_transient_error(error) is True

    def test_keyword_based_detection_temporary(self):
        """Test that errors with 'temporary' keyword are detected as transient."""
        error = Exception("Service temporarily unavailable")
        assert default_is_transient_error(error) is True

    def test_keyword_based_detection_unavailable(self):
        """Test that errors with 'unavailable' keyword are detected as transient."""
        error = Exception("Resource unavailable")
        assert default_is_transient_error(error) is True

    def test_case_insensitive_keyword_matching(self):
        """Test that keyword matching is case-insensitive."""
        error = Exception("CONNECTION TIMEOUT")
        assert default_is_transient_error(error) is True

    def test_non_transient_error_detection(self):
        """Test that non-transient errors are not detected as transient."""
        non_transient_errors = [
            ValueError("Invalid value"),
            TypeError("Wrong type"),
            KeyError("Missing key"),
            Exception("Something went wrong"),
        ]

        for error in non_transient_errors:
            assert default_is_transient_error(error) is False
