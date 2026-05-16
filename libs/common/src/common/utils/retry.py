"""Retry utility for handling transient errors with exponential backoff."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)


def default_is_transient_error(error: Exception) -> bool:
    """Check if an error is transient and worth retrying.

    Detects common transient errors using type checking and keyword matching.
    Type-based detection covers standard Python exceptions (asyncio.TimeoutError,
    ConnectionError, TimeoutError). Keyword matching handles library-specific errors
    from HTTP clients, database drivers, etc.

    For library-specific exceptions (e.g., qdrant_client.http.exceptions, httpx errors),
    consider passing a custom is_transient_error function to retry_with_backoff that
    checks for specific exception types relevant to your use case.

    Args:
        error: Exception to check

    Returns:
        True if error is transient, False otherwise
    """
    # Check common transient exception types
    transient_types = (
        asyncio.TimeoutError,
        ConnectionError,
        TimeoutError,
    )
    if isinstance(error, transient_types):
        return True

    # Fallback to keyword matching for library-specific errors
    error_str = str(error).lower()
    transient_keywords = [
        "timeout",
        "connection",
        "network",
        "temporary",
        "unavailable",
    ]
    return any(keyword in error_str for keyword in transient_keywords)


async def retry_with_backoff(
    operation: Callable[..., Awaitable[T]],
    max_retries: int = 3,
    retry_delay: float = 1.0,
    operation_args: tuple[Any, ...] | None = None,
    operation_kwargs: dict[str, Any] | None = None,
    is_transient_error: Callable[[Exception], bool] | None = None,
    on_retry: Callable[..., None] | None = None,
) -> T:
    """Execute an async operation with retry logic and exponential backoff.

    This utility function handles transient errors by retrying the operation
    with exponential backoff. It's designed to be reusable across different
    parts of the codebase that need retry functionality.

    Args:
        operation: Async function to execute
        max_retries: Maximum number of retry attempts (default: 3)
        retry_delay: Initial delay in seconds between retries, doubles each retry (default: 1.0)
        operation_args: Tuple of positional arguments to pass to operation
        operation_kwargs: Dictionary of keyword arguments to pass to operation
        is_transient_error: Optional custom function to determine if error is transient
        on_retry: Optional callback called on each retry with (attempt, max_retries, error, delay)

    Returns:
        Result of the operation if successful

    Raises:
        ValueError: If max_retries or retry_delay are negative
        Exception: The last exception if all retries are exhausted or if non-transient error

    Example:
        >>> async def update_db(id: str, value: int):
        ...     await db.update(id, value)
        ...
        >>> result = await retry_with_backoff(
        ...     operation=update_db,
        ...     max_retries=3,
        ...     retry_delay=1.0,
        ...     operation_kwargs={"id": "123", "value": 456}
        ... )
    """
    # Validate input parameters
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")  # noqa: TRY003
    if retry_delay < 0:
        raise ValueError("retry_delay must be >= 0")  # noqa: TRY003

    # Normalize optional arguments for type checker
    if operation_args is None:
        operation_args = ()
    if operation_kwargs is None:
        operation_kwargs = {}

    retry_count = 0

    while retry_count <= max_retries:
        try:
            result = await operation(*operation_args, **operation_kwargs)
        except Exception as e:
            retry_count += 1

            # Check if this is a transient error worth retrying
            # Use provided error checker or default
            if is_transient_error is not None:
                is_transient = is_transient_error(e)
            else:
                is_transient = default_is_transient_error(e)

            if is_transient and retry_count <= max_retries:
                # Exponential backoff
                delay = retry_delay * (2 ** (retry_count - 1))

                logger.warning(
                    f"Transient error on attempt {retry_count}/{max_retries + 1}: {e}. "
                    f"Retrying in {delay}s..."
                )

                # Call retry callback if provided
                if on_retry:
                    on_retry(
                        attempt=retry_count,
                        max_retries=max_retries,
                        error=e,
                        delay=delay,
                    )

                await asyncio.sleep(delay)
            else:
                # Non-transient error or max retries exceeded
                if retry_count > max_retries:
                    logger.exception(f"Max retries ({max_retries}) exceeded")
                else:
                    logger.exception("Non-transient error")

                raise
        else:
            # Success - return result
            return result

    # pragma: no cover - This is truly unreachable with validation in place
    # The loop always returns on success or raises on error
    raise RuntimeError
