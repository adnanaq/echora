"""Custom exceptions for HTTP cache library."""


class CacheError(Exception):
    """Base exception for cache-related errors."""


class CacheStorageError(CacheError):
    """Raised when cache storage operations fail."""


class InvalidTTLError(CacheError, ValueError):
    """Raised when TTL value is invalid (negative or incorrect type)."""

    def __init__(self, ttl_value: float | None = None, field_name: str = "TTL"):
        if ttl_value is not None:
            super().__init__(f"{field_name} must be non-negative, got {ttl_value}")
        else:
            super().__init__(f"{field_name} must be non-negative")


class EntryMismatchError(CacheStorageError, ValueError):
    """Raised when cache entry ID does not match expected value."""

    def __init__(self):
        super().__init__("Entry ID mismatch")


class InvalidStreamTypeError(CacheStorageError, TypeError):
    """Raised when response stream is not an AsyncIterator."""

    def __init__(self, actual_type: str):
        super().__init__(
            f"Expected AsyncIterator for response.stream, got {actual_type}"
        )
