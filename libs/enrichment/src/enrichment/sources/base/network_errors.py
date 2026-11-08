"""Tell an aiohttp network error from an answer the server gave.

API helpers retry a request when the connection failed on the way, and not
when the server answered with an HTTP error. A connection that drops, refuses
or times out usually works on the next attempt. An HTTP error status is the
server's answer, which each helper handles itself, such as waiting for a rate
limit to pass.
"""

import aiohttp


def is_aiohttp_network_error(error: Exception) -> bool:
    """Report whether an aiohttp error came from the connection, not from an HTTP answer.

    Args:
        error: The error a request raised.

    Returns:
        True for an ``aiohttp.ClientError`` other than
        ``aiohttp.ClientResponseError``, which carries the server's HTTP status.
    """
    return isinstance(error, aiohttp.ClientError) and not isinstance(
        error, aiohttp.ClientResponseError
    )
