import pytest
from common.utils.transient_errors import is_transient_service_error


class _StatusCodeError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__("status error")
        self.status_code = status_code


class _StatusError(Exception):
    def __init__(self, status: int) -> None:
        super().__init__("status error")
        self.status = status


class _BrokenResponseError(Exception):
    pass


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("timeout"),
        ConnectionError("refused"),
        _StatusCodeError(504),
        _StatusError(429),
        Exception("temporary network unavailable"),
    ],
)
def test_is_transient_service_error_retryable_error_returns_true(error) -> None:
    assert is_transient_service_error(error)


@pytest.mark.parametrize(
    "error",
    [
        _StatusCodeError(422),
        _StatusError(400),
        Exception("invalid payload schema"),
    ],
)
def test_is_transient_service_error_lasting_error_returns_false(error) -> None:
    assert not is_transient_service_error(error)


def test_is_transient_service_error_status_decides_over_message_words() -> None:
    error = _StatusCodeError(400)
    error.args = ("connection header invalid",)

    assert not is_transient_service_error(error)


def test_is_transient_service_error_caller_error_type_returns_true() -> None:
    assert is_transient_service_error(
        _BrokenResponseError("bad"), transient_error_types=(_BrokenResponseError,)
    )


def test_is_transient_service_error_caller_error_type_not_given_returns_false() -> None:
    assert not is_transient_service_error(_BrokenResponseError("bad"))
