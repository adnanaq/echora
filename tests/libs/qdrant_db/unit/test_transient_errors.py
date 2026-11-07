import httpx
import pytest
from qdrant_client.http.exceptions import (
    ResponseHandlingException,
    UnexpectedResponse,
)
from qdrant_db.transient_errors import is_transient_qdrant_error


def _unexpected_response(status_code: int) -> UnexpectedResponse:
    return UnexpectedResponse(status_code, "reason", b"", httpx.Headers())


class _StatusCodeError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__("status error")
        self.status_code = status_code


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("timeout"),
        ConnectionError("refused"),
        ResponseHandlingException(ValueError("decode error")),
        _unexpected_response(429),
        _unexpected_response(503),
        _StatusCodeError(504),
        Exception("temporary network unavailable"),
    ],
)
def test_is_transient_qdrant_error_retryable_error_returns_true(error) -> None:
    assert is_transient_qdrant_error(error)


@pytest.mark.parametrize(
    "error",
    [
        _unexpected_response(400),
        _StatusCodeError(422),
        Exception("invalid payload schema"),
    ],
)
def test_is_transient_qdrant_error_lasting_error_returns_false(error) -> None:
    assert not is_transient_qdrant_error(error)
