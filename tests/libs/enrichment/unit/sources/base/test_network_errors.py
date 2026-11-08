import aiohttp
import pytest
from enrichment.sources.base.network_errors import is_aiohttp_network_error


@pytest.mark.parametrize(
    ("error", "is_network_error"),
    [
        (aiohttp.ClientConnectionError("connection refused"), True),
        (aiohttp.ServerTimeoutError("timed out"), True),
        (aiohttp.ClientPayloadError("cut off"), True),
        (aiohttp.ClientResponseError(None, (), status=503), False),
        (ValueError("connection string"), False),
    ],
    ids=["connection", "server_timeout", "payload", "http_answer", "other"],
)
def test_is_aiohttp_network_error_tells_connection_from_http_answer(
    error: Exception, is_network_error: bool
) -> None:
    assert is_aiohttp_network_error(error) is is_network_error
