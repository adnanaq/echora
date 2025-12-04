"""Retry utility for handling transient errors with exponential backoff."""

import asyncio
import logging
from typing import Any, Callable, Optional, Tuple

logger = logging.getLogger(__name__)


def default_is_transient_error(error: Exception) -> bool:
    """Check if an error is transient and worth retrying.

    Args:
        error: Exception to check

    Returns:
        True if error is transient, False otherwise
    """
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
    operation: Callable[..., Any],
    max_retries: int = 3,
    retry_delay: float = 1.0,
    operation_args: Optional[Tuple[Any, ...]] = None,
    operation_kwargs: Optional[dict] = None,
    is_transient_error: Optional[Callable[[Exception], bool]] = None,
    on_retry: Optional[Callable[..., None]] = None,
) -> Any:
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
        on_retry: Optional callback function called on each retry attempt

    Returns:
        Result of the operation if successful

    Raises:
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
    operation_args = operation_args or ()
    operation_kwargs = operation_kwargs or {}
    error_checker = is_transient_error or default_is_transient_error

    retry_count = 0
    last_error = None

    while retry_count <= max_retries:
        try:
            result = await operation(*operation_args, **operation_kwargs)
            return result

        except Exception as e:
            last_error = e
            retry_count += 1

            # Check if this is a transient error worth retrying
            is_transient = error_checker(e)

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
                    logger.error(
                        f"Max retries ({max_retries}) exceeded. Last error: {last_error}"
                    )
                else:
                    logger.error(f"Non-transient error: {e}")

                raise

    # This should never be reached, but raise last error as fallback
    raise last_error  # type: ignore
