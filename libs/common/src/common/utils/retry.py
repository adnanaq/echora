"""Retry async operations with exponential backoff.

The caller decides which errors are worth another attempt, because each
service raises its own error types. Each rule reads ``<rule>: <reason>``.

- Required error check: the caller passes ``is_transient_error``, such as
  ``common.utils.transient_errors.is_transient_service_error``, so this
  shared library imports no service client, such as Qdrant's.
- Doubling delay: each retry waits twice as long as the one before, so a
  struggling service gets more time to recover.
- Last error raised: once the retries are used up, or the error check refuses
  an error, that error is raised unchanged, so the caller handles it as usual.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)


class NegativeRetrySettingError(ValueError):
    """A retry count or delay is below zero."""

    def __init__(self, setting_name: str) -> None:
        """Name the setting that must not be negative.

        Args:
            setting_name: ``max_retries`` or ``retry_delay``.
        """
        super().__init__(f"{setting_name} must be >= 0")


async def retry_with_backoff[OperationResult](
    operation: Callable[..., Awaitable[OperationResult]],
    *,
    is_transient_error: Callable[[Exception], bool],
    max_retries: int = 3,
    retry_delay: float = 1.0,
    operation_args: tuple[Any, ...] | None = None,
    operation_kwargs: dict[str, Any] | None = None,
    on_retry: Callable[..., None] | None = None,
) -> OperationResult:
    """Run an async operation, retrying transient errors with exponential backoff.

    Args:
        operation: Async callable to execute.
        is_transient_error: Decides whether an error is worth another attempt.
        max_retries: Number of retry attempts after the first call.
        retry_delay: Delay before the first retry, in seconds.
        operation_args: Positional arguments passed to ``operation``.
        operation_kwargs: Keyword arguments passed to ``operation``.
        on_retry: Callback run before each retry, with ``attempt``,
            ``max_retries``, ``error`` and ``delay``.

    Returns:
        The value ``operation`` returns.

    Raises:
        NegativeRetrySettingError: ``max_retries`` or ``retry_delay`` is negative.
        Exception: The operation's last error, once retrying stops.
    """
    if max_retries < 0:
        raise NegativeRetrySettingError("max_retries")
    if retry_delay < 0:
        raise NegativeRetrySettingError("retry_delay")
    positional_arguments = operation_args or ()
    keyword_arguments = operation_kwargs or {}
    attempt = 0
    while True:
        try:
            return await operation(*positional_arguments, **keyword_arguments)
        except Exception as error:
            attempt += 1
            if not is_transient_error(error) or attempt > max_retries:
                if attempt > max_retries:
                    logger.exception(f"Max retries ({max_retries}) exceeded")
                raise
            delay = retry_delay * (2 ** (attempt - 1))
            logger.warning(
                f"Transient error on attempt {attempt}/{max_retries + 1}: {error}. "
                f"Retrying in {delay}s..."
            )
            if on_retry:
                on_retry(
                    attempt=attempt,
                    max_retries=max_retries,
                    error=error,
                    delay=delay,
                )
            await asyncio.sleep(delay)
