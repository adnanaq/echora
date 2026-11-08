"""Decide which errors from a remote service are worth retrying.

Callers pass ``is_transient_service_error`` to
``common.utils.retry.retry_with_backoff``, adding the error types their own
client raises for transient faults, so this shared library imports no service
client. Each rule reads ``<error>: <outcome>. <reason>``.

- Timeout or dropped connection: retried. The next attempt usually reaches
  the server.
- One of the caller's ``transient_error_types``: retried. The caller knows its
  client raises these for transient faults, such as a broken response.
- HTTP status 408, 425, 429 or 5xx (``RETRYABLE_HTTP_STATUS_CODES``):
  retried. The server is busy, rate limiting, or briefly failing.
- Any other status, such as 400, 409 or 422: not retried. The request itself
  is wrong, or conflicts with the stored state, such as a Qdrant collection
  that already exists, so sending it again gives the same answer.
- Message naming a timeout, connection, network, temporary or unavailable
  fault: retried. Some client errors carry no status, only this text.
"""

from typing import Final

RETRYABLE_HTTP_STATUS_CODES: Final = frozenset({408, 425, 429, 500, 502, 503, 504})
_TRANSIENT_MESSAGE_WORDS: Final = (
    "timeout",
    "connection",
    "network",
    "temporary",
    "unavailable",
)


def is_transient_service_error(
    error: Exception,
    *,
    transient_error_types: tuple[type[Exception], ...] = (),
) -> bool:
    """Check whether an error from a remote service is worth another attempt.

    Args:
        error: The error the service operation raised.
        transient_error_types: Error types of the caller's client that always
            mean a transient fault.

    Returns:
        True for the transient errors the module docstring lists.
    """
    if isinstance(
        error,
        (ConnectionError, TimeoutError, *transient_error_types),
    ):
        return True
    status_code = _find_status_code(error)
    if status_code is not None:
        return status_code in RETRYABLE_HTTP_STATUS_CODES
    error_message = str(error).lower()
    return any(word in error_message for word in _TRANSIENT_MESSAGE_WORDS)


def _find_status_code(error: Exception) -> int | None:
    """Find the HTTP status code an error carries, under either of its usual names.

    A ``code`` attribute is not read, because errors also use it for exit
    codes and other numbers that are not HTTP statuses.

    Args:
        error: The error to inspect.

    Returns:
        The value of ``status`` or ``status_code``, in that order,
        when one is an integer; otherwise None.
    """
    for attribute_name in ("status", "status_code"):
        status_code = getattr(error, attribute_name, None)
        if isinstance(status_code, int):
            return status_code
    return None
