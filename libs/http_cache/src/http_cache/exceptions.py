"""Custom exceptions for HTTP cache library."""


class CacheError(Exception):
    """Base exception for cache-related errors."""


class StorageConfigurationError(CacheError):
    """Raised when cache storage is misconfigured."""

    def __init__(self, storage_type: str | None = None):
        if storage_type:
            super().__init__(f"Unknown storage type: {storage_type}")
        else:
            super().__init__("Storage configuration error")


class RedisConfigurationError(StorageConfigurationError):
    """Raised when Redis storage is misconfigured."""

    def __init__(self):
        super().__init__()
        self.args = ("redis_url required for Redis storage",)


class CacheStorageError(CacheError):
    """Raised when cache storage operations fail."""


class RedisInitializationError(CacheStorageError):
    """Raised when Redis client initialization fails."""

    def __init__(self):
        super().__init__("Failed to initialize Redis client for result cache")


class InvalidTTLError(CacheError):
    """Raised when TTL value is invalid (negative or incorrect type)."""

    def __init__(self, ttl_value: float | None = None, field_name: str = "TTL"):
        if ttl_value is not None:
            super().__init__(f"{field_name} must be non-negative, got {ttl_value}")
        else:
            super().__init__(f"{field_name} must be non-negative")


class EntryMismatchError(CacheStorageError):
    """Raised when cache entry ID does not match expected value."""

    def __init__(self):
        super().__init__("Entry ID mismatch")


class InvalidStreamTypeError(CacheStorageError):
    """Raised when response stream is not an AsyncIterator."""

    def __init__(self, actual_type: str):
        super().__init__(
            f"Expected AsyncIterator for response.stream, got {actual_type}"
        )
