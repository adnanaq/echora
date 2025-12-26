"""Type stubs for redis.asyncio.client module.

Extended to accept bytes for hset method parameters, matching actual runtime behavior.
"""

from typing import Any

class Redis:
    """Redis async client with extended type hints for bytes support."""

    @classmethod
    def from_url(
        cls,
        url: str,
        **kwargs: Any,
    ) -> Redis:
        """Create Redis client from URL."""
        ...

    async def hset(
        self,
        name: str | bytes,
        key: str | bytes | None = None,
        value: str | bytes | None = None,
        mapping: dict[str | bytes, str | bytes] | None = None,
        items: list[Any] | None = None,
    ) -> int:
        """
        Set field in hash to value.

        Accepts both str and bytes for all parameters to match runtime behavior.
        The official redis-py type hints are conservative (str only), but the
        implementation accepts bytes and converts internally.

        Parameters:
            name: Hash name (str or bytes)
            key: Field name (str or bytes)
            value: Field value (str or bytes)
            mapping: Dict of field-value pairs (keys and values can be str or bytes)
            items: List of alternating field-value pairs

        Returns:
            Number of fields that were added
        """
        ...

    def pipeline(self, transaction: bool = True, shard_hint: Any | None = None) -> Any:
        """Create a pipeline for atomic operations."""
        ...

    async def hgetall(self, name: str | bytes) -> dict[bytes, bytes]:
        """Return all fields and values in a hash as dict of bytes."""
        ...

    async def smembers(self, name: str | bytes) -> set[bytes]:
        """Return all members of a set."""
        ...

    async def srem(self, name: str | bytes, *values: str | bytes) -> int:
        """Remove members from a set."""
        ...

    async def sadd(self, name: str | bytes, *values: str | bytes) -> int:
        """Add members to a set."""
        ...

    async def ttl(self, name: str | bytes) -> int:
        """Return time to live in seconds, -1 if persistent, -2 if not exists."""
        ...

    async def expire(self, name: str | bytes, time: int) -> bool:
        """Set a timeout on key."""
        ...

    async def rpush(self, name: str | bytes, *values: str | bytes) -> int:
        """Append one or multiple values to a list."""
        ...

    async def lrange(
        self, name: str | bytes, start: int, end: int
    ) -> list[bytes]:
        """Return a range of elements from a list."""
        ...

    async def scan(
        self,
        cursor: int = 0,
        match: str | bytes | None = None,
        count: int | None = None,
    ) -> tuple[int, list[bytes]]:
        """Incrementally iterate the keys space."""
        ...

    async def get(self, name: str | bytes) -> bytes | None:
        """Get the value of a key."""
        ...

    async def setex(
        self, name: str | bytes, time: int, value: str | bytes
    ) -> bool:
        """Set the value and expiration of a key."""
        ...

    async def delete(self, *names: str | bytes) -> int:
        """Delete one or more keys."""
        ...

    async def close(self, close_connection_pool: bool = True) -> None:
        """Close the connection."""
        ...

    async def aclose(self) -> None:
        """Async context manager close."""
        ...
