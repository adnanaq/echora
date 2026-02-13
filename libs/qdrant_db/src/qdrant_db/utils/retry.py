"""Retry helpers for transient Qdrant-related failures."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)

_RETRYABLE_HTTP_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}

try:
    from qdrant_client.http.exceptions import (  # type: ignore[attr-defined]
        ApiException,
        ResponseHandlingException,
    )
except Exception:  # pragma: no cover - import guard for optional runtime shapes
    # Fallback to empty tuple for isinstance checks when qdrant_client unavailable
    ApiException = type("ApiException", (), {})  # type: ignore[misc,assignment]
    ResponseHandlingException = type("ResponseHandlingException", (), {})  # type: ignore[misc,assignment]


def _extract_status_code(error: Exception) -> int | None:
    """Extract best-effort HTTP-like status code from an exception.

    Args:
        error: Exception instance to inspect.

    Returns:
        Integer status code when available, otherwise ``None``.
    """
    for attr in ("status", "status_code", "code"):
        value = getattr(error, attr, None)
        if isinstance(value, int):
            return value
    return None


def default_is_transient_error(error: Exception) -> bool:
    """Check if an exception is retryable.

    Args:
        error: Exception raised by an operation.

    Returns:
        ``True`` when the error is likely transient/retryable, otherwise
        ``False``.
    """
    if isinstance(error, (asyncio.TimeoutError, ConnectionError, TimeoutError)):
        return True

    if isinstance(error, ResponseHandlingException):
        return True

    if isinstance(error, ApiException):
        status_code = _extract_status_code(error)
        if status_code in _RETRYABLE_HTTP_STATUS_CODES:
            return True

    status_code = _extract_status_code(error)
    if status_code in _RETRYABLE_HTTP_STATUS_CODES:
        return True

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
    """Execute an async operation with exponential backoff retries.

    Args:
        operation: Async callable to execute.
        max_retries: Number of retry attempts after initial call.
        retry_delay: Initial backoff delay in seconds.
        operation_args: Optional positional args passed to operation.
        operation_kwargs: Optional keyword args passed to operation.
        is_transient_error: Optional predicate to classify retryable errors.
        on_retry: Optional callback invoked before each retry.

    Returns:
        Result returned by ``operation``.

    Raises:
        ValueError: If retry configuration is invalid.
        Exception: Re-raises terminal operation exception.
    """
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    if retry_delay < 0:
        raise ValueError("retry_delay must be >= 0")

    if operation_args is None:
        operation_args = ()
    if operation_kwargs is None:
        operation_kwargs = {}

    retry_count = 0

    while retry_count <= max_retries:
        try:
            result = await operation(*operation_args, **operation_kwargs)
        except Exception as error:
            retry_count += 1
            checker = is_transient_error or default_is_transient_error
            if checker(error) and retry_count <= max_retries:
                delay = retry_delay * (2 ** (retry_count - 1))
                logger.warning(
                    "Transient error on attempt %s/%s: %s. Retrying in %ss...",
                    retry_count,
                    max_retries + 1,
                    error,
                    delay,
                )
                if on_retry:
                    on_retry(
                        attempt=retry_count,
                        max_retries=max_retries,
                        error=error,
                        delay=delay,
                    )
                await asyncio.sleep(delay)
            else:
                if retry_count > max_retries:
                    logger.exception("Max retries (%s) exceeded", max_retries)
                raise
        else:
            return result

    raise RuntimeError  # pragma: no cover - unreachable by construction
