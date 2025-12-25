"""Type stubs for redis.exceptions module."""

class RedisError(Exception):
    """Base exception for redis errors."""
    ...

class ConnectionError(RedisError):
    """Redis connection error."""
    ...

class TimeoutError(RedisError):
    """Redis timeout error."""
    ...

class ResponseError(RedisError):
    """Redis response error."""
    ...
