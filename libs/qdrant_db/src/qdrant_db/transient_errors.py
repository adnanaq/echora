"""Decide which Qdrant client errors are worth retrying.

``qdrant_db.client`` passes ``is_transient_qdrant_error`` to
``common.utils.retry.retry_with_backoff``, so the Qdrant exception types stay
in this library. Each rule reads ``<error>: <outcome>. <reason>``.

- Timeout or dropped connection: retried. The next attempt usually reaches
  the server.
- ``ResponseHandlingException``: retried. Qdrant's client raises it when a
  response arrives broken, which is usually a network fault.
- HTTP status 408, 409, 425, 429 or 5xx (``RETRYABLE_HTTP_STATUS_CODES``):
  retried. The server is busy, rate limiting, or briefly failing.
- Any other status, such as 400 or 422: not retried. The request itself is
  wrong, so sending it again gives the same answer.
- Message naming a timeout, connection, network, temporary or unavailable
  fault: retried. Some client errors carry no status, only this text.
"""

import asyncio
from typing import Final

from qdrant_client.http.exceptions import ResponseHandlingException

RETRYABLE_HTTP_STATUS_CODES: Final = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_TRANSIENT_MESSAGE_WORDS: Final = (
    "timeout",
    "connection",
    "network",
    "temporary",
    "unavailable",
)


def is_transient_qdrant_error(error: Exception) -> bool:
    """Check whether a Qdrant client error is worth another attempt.

    Args:
        error: The error the Qdrant operation raised.

    Returns:
        True for the transient errors the module docstring lists.
    """
    if isinstance(
        error,
        (
            asyncio.TimeoutError,
            ConnectionError,
            TimeoutError,
            ResponseHandlingException,
        ),
    ):
        return True
    status_code = _find_status_code(error)
    if status_code is not None:
        return status_code in RETRYABLE_HTTP_STATUS_CODES
    error_message = str(error).lower()
    return any(word in error_message for word in _TRANSIENT_MESSAGE_WORDS)


def _find_status_code(error: Exception) -> int | None:
    """Find the HTTP status code an error carries, under any of its usual names.

    Args:
        error: The error to inspect.

    Returns:
        The value of ``status``, ``status_code`` or ``code``, in that order,
        when one is an integer; otherwise None.
    """
    for attribute_name in ("status", "status_code", "code"):
        status_code = getattr(error, attribute_name, None)
        if isinstance(status_code, int):
            return status_code
    return None
