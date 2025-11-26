"""
Async Redis storage backend for Hishel HTTP caching.

Implements AsyncBaseStorage interface for Redis-backed async HTTP response caching
with support for multi-agent concurrent access, service-specific TTLs, and
streaming responses.

This is the async parallel to SyncRedisStorage, using redis.asyncio for
aiohttp-based API helpers (AniList, Kitsu, AniDB).
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import (
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Union,
    cast,
)

# Import Hishel core types
from hishel._core._storages._async_base import AsyncBaseStorage
from hishel._core._storages._packing import pack, unpack
from hishel._core.models import Entry, EntryMeta, Request, Response
from redis.asyncio import Redis


class AsyncRedisStorage(AsyncBaseStorage):
    """
    Async Redis-backed storage for Hishel HTTP caching.

    Supports:
    - Multi-agent concurrent access
    - TTL-based expiration per service
    - Streaming response storage
    - Soft deletion with cleanup

    Redis Key Structure:
        cache:entry:{uuid}          → Hash with serialized Entry metadata
        cache:stream:{uuid}         → List of response stream chunks
        cache:key_index:{cache_key} → Set of entry UUIDs
    """

    _COMPLETE_CHUNK_MARKER = b"__STREAM_COMPLETE__"

    def __init__(
        self,
        *,
        client: Optional[Redis] = None,
        redis_url: str = "redis://localhost:6379/0",
        default_ttl: Optional[float] = None,
        refresh_ttl_on_access: bool = True,
        key_prefix: str = "hishel_cache",
    ) -> None:
        """
        Initialize async Redis storage.

        Args:
            client: Existing async Redis client (optional)
            redis_url: Redis connection URL (used if client not provided)
            default_ttl: Default TTL in seconds for cache entries
            refresh_ttl_on_access: Whether to refresh TTL on cache hits
            key_prefix: Prefix for all Redis keys (default: "hishel_cache")
        """
        self._owns_client = client is None
        if client:
            self.client = client
        else:
            # Import here to avoid circular dependency
            from .config import get_cache_config

            config = get_cache_config()
            # Configure connection pool for multi-agent concurrency and reliability
            self.client = Redis.from_url(
                redis_url,
                decode_responses=False,
                max_connections=config.redis_max_connections,
                socket_keepalive=config.redis_socket_keepalive,
                socket_connect_timeout=config.redis_socket_connect_timeout,
                socket_timeout=config.redis_socket_timeout,
                retry_on_timeout=config.redis_retry_on_timeout,
                health_check_interval=config.redis_health_check_interval,
            )
        if default_ttl is not None and default_ttl < 0:
            raise ValueError("default_ttl must be non-negative")
        self.default_ttl = default_ttl
        self.refresh_ttl_on_access = refresh_ttl_on_access
        self.key_prefix = key_prefix

        # Note: Connection test done lazily on first use

    def _make_key(self, key_type: str, identifier: str) -> str:
        """
        Build a namespaced Redis key by joining the instance `key_prefix`, a key type, and an identifier.
        
        Parameters:
            key_type (str): Segment indicating the key category (e.g., "entry", "stream", "index").
            identifier (str): Unique identifier to append to the key.
        
        Returns:
            str: Redis key in the form "<key_prefix>:<key_type>:<identifier>".
        """
        return f"{self.key_prefix}:{key_type}:{identifier}"

    def _entry_key(self, entry_id: uuid.UUID) -> str:
        """
        Return the Redis key used to store an entry's metadata and data for the given entry UUID.
        
        Parameters:
            entry_id (uuid.UUID): UUID of the entry.
        
        Returns:
            str: Redis key for the entry.
        """
        return self._make_key("entry", str(entry_id))

    def _stream_key(self, entry_id: uuid.UUID) -> str:
        """
        Return the Redis key for the response stream associated with the given entry ID.
        
        Parameters:
            entry_id (uuid.UUID): Entry identifier.
        
        Returns:
            str: Redis key for the stream chunks for the specified entry.
        """
        return self._make_key("stream", str(entry_id))

    def _index_key(self, cache_key: str | bytes) -> str:
        """
        Redis key for the index that maps a cache key to its entry IDs.
        
        Parameters:
            cache_key (str | bytes): Cache key to index; strings are UTF-8 encoded before hex-encoding.
        
        Returns:
            str: Redis key for the set that contains entry UUIDs associated with the given cache key.
        """
        # Use hex encoding for cache key (bytes)
        cache_key_hex = (
            cache_key.encode("utf-8").hex()
            if isinstance(cache_key, str)
            else cache_key.hex()
        )
        return self._make_key("key_index", cache_key_hex)

    async def create_entry(
        self,
        request: Request,
        response: Response,
        key: str,
        id: uuid.UUID | None = None,
    ) -> Entry:
        """
        Create and store a cache Entry for the given request/response and cache key.

        The response's stream is replaced with a wrapper that saves streamed chunks to Redis so the stored entry can be replayed later; the entry metadata and an index mapping from the cache key to the entry ID are persisted, and optional TTLs are applied.

        Parameters:
            request: Original HTTP request associated with the entry.
            response: HTTP response whose `stream` will be wrapped to persist streamed chunks.
            key: Cache key used to index the entry.
            id: Optional UUID to use for the entry; a new UUID is generated if omitted.

        Returns:
            The persisted Entry with its response.stream wrapped to save chunks to Redis.

        Raises:
            TypeError: If `response.stream` is not an AsyncIterator.
        """
        entry_id = id if id is not None else uuid.uuid4()
        entry_meta = EntryMeta(created_at=time.time())
        ttl = self._get_entry_ttl(request)

        # Replace response stream with saving wrapper
        if not isinstance(response.stream, AsyncIterator):
            raise TypeError(
                f"Expected AsyncIterator for response.stream, got {type(response.stream).__name__}"
            )
        response_with_stream = Response(
            status_code=response.status_code,
            headers=response.headers,
            stream=self._save_stream(response.stream, entry_id, ttl),
            metadata=response.metadata,
        )

        # Create complete entry
        complete_entry = Entry(
            id=entry_id,
            request=request,
            response=response_with_stream,
            meta=entry_meta,
            cache_key=key.encode("utf-8"),
        )

        # Serialize entry
        entry_data = pack(complete_entry, kind="pair")

        # Store in Redis
        entry_key = self._entry_key(entry_id)
        index_key = self._index_key(key)

        # Use pipeline for atomic operations
        pipe = self.client.pipeline()

        current_index_ttl: int | None = None
        if ttl is not None:
            current_index_ttl = await self.client.ttl(index_key)

        # Store entry data
        pipe.hset(
            entry_key,
            mapping={
                b"data": entry_data,
                b"created_at": str(entry_meta.created_at).encode("utf-8"),
                b"cache_key": key.encode("utf-8"),
            },
        )

        # Add to cache key index
        pipe.sadd(index_key, str(entry_id).encode("utf-8"))

        # Set TTL if configured (stream TTL is handled in _save_stream)
        # Redis TTL return values:
        # -2: Key doesn't exist (should create with TTL)
        # -1: Key is persistent (should NOT apply TTL to preserve persistence)
        # >= 0: Key has finite TTL (should extend if shorter than new TTL)
        if ttl is not None:
            ttl_seconds = int(ttl)
            pipe.expire(entry_key, ttl_seconds)
            # Only apply expire to index if:
            # 1. Key doesn't exist (current_index_ttl == -2), OR
            # 2. Key has finite TTL shorter than new TTL (0 <= current_index_ttl < ttl_seconds)
            # Never apply if key is persistent (current_index_ttl == -1)
            if current_index_ttl == -2 or (
                current_index_ttl is not None
                and current_index_ttl >= 0
                and current_index_ttl < ttl_seconds
            ):
                pipe.expire(index_key, ttl_seconds)

        await pipe.execute()

        return complete_entry

    async def get_entries(self, key: str) -> List[Entry]:
        """
        Retrieve cached entries associated with the provided cache key.
        
        Only entries that exist, are deserializable, have a response, and are not soft-deleted are returned. Each returned Entry has its Response.stream reconstructed to read stored chunks from Redis. If `refresh_ttl_on_access` is enabled and the entry has a TTL, the entry and its stream TTLs are refreshed.
        
        Returns:
            List[Entry]: A list of matching entries with their response streams restored from cache.
        """
        index_key = self._index_key(key)
        entry_ids_bytes = await cast(
            "Awaitable[Set[bytes]]", self.client.smembers(index_key)
        )

        if not entry_ids_bytes:
            return []

        final_entries: List[Entry] = []

        for entry_id_bytes in entry_ids_bytes:
            entry_id = uuid.UUID(entry_id_bytes.decode("utf-8"))
            entry_key = self._entry_key(entry_id)

            # Get entry data
            entry_hash = await cast(
                "Awaitable[Dict[bytes, bytes]]", self.client.hgetall(entry_key)
            )
            if not entry_hash or b"data" not in entry_hash:
                continue

            # Deserialize entry
            entry = unpack(entry_hash[b"data"], kind="pair")
            if not isinstance(entry, Entry) or entry.response is None:
                continue

            # Check if soft deleted
            if self.is_soft_deleted(entry):
                continue

            # Restore stream from Redis
            entry_with_stream = Entry(
                id=entry.id,
                request=entry.request,
                response=Response(
                    status_code=entry.response.status_code,
                    headers=entry.response.headers,
                    stream=self._stream_data_from_cache(entry_id),
                    metadata=entry.response.metadata,
                ),
                meta=entry.meta,
                cache_key=entry.cache_key,
            )

            # Refresh TTL on access if configured
            if self.refresh_ttl_on_access:
                ttl = self._get_entry_ttl(entry.request)
                if ttl is not None:
                    ttl_seconds = int(ttl)
                    await self.client.expire(entry_key, ttl_seconds)
                    await self.client.expire(self._stream_key(entry_id), ttl_seconds)
                    # Also refresh index key TTL to keep all three keys aging consistently
                    # Only extend if current TTL is shorter (same logic as create_entry)
                    current_index_ttl = await self.client.ttl(index_key)
                    if current_index_ttl == -2 or (
                        current_index_ttl >= 0 and current_index_ttl < ttl_seconds
                    ):
                        await self.client.expire(index_key, ttl_seconds)

            final_entries.append(entry_with_stream)

        return final_entries

    async def update_entry(
        self,
        id: uuid.UUID,
        new_entry: Union[Entry, Callable[[Entry], Entry]],
    ) -> Optional[Entry]:
        """
        Update an existing cache entry by replacing it or applying a transformer to the current entry.
        
        Parameters:
            id (uuid.UUID): UUID of the entry to update.
            new_entry (Entry | Callable[[Entry], Entry]): Either a complete Entry to store or a callable that receives the current Entry and returns the updated Entry.
        
        Returns:
            Optional[Entry]: The updated Entry if the entry existed and was stored, `None` if the entry was not found or could not be deserialized.
        
        Raises:
            ValueError: If the provided update changes the entry's UUID.
        """
        entry_key = self._entry_key(id)
        entry_hash = await cast(
            "Awaitable[Dict[bytes, bytes]]", self.client.hgetall(entry_key)
        )

        if not entry_hash or b"data" not in entry_hash:
            return None

        # Deserialize current entry
        current_entry = unpack(entry_hash[b"data"], kind="pair")
        if not isinstance(current_entry, Entry) or current_entry.response is None:
            return None

        # Apply update
        if isinstance(new_entry, Entry):
            complete_entry = new_entry
        else:
            complete_entry = new_entry(current_entry)

        if current_entry.id != complete_entry.id:
            raise ValueError("Entry ID mismatch")

        # Serialize and store
        entry_data = pack(complete_entry, kind="pair")

        pipe = self.client.pipeline()
        pipe.hset(entry_key, b"data", entry_data)  # type: ignore[arg-type]

        # Update cache key index if changed
        if current_entry.cache_key != complete_entry.cache_key:
            old_index_key = self._index_key(current_entry.cache_key.decode("utf-8"))
            new_index_key = self._index_key(complete_entry.cache_key.decode("utf-8"))

            pipe.srem(old_index_key, str(id).encode("utf-8"))
            pipe.sadd(new_index_key, str(id).encode("utf-8"))
            pipe.hset(entry_key, b"cache_key", complete_entry.cache_key)  # type: ignore[arg-type]

        await pipe.execute()

        return complete_entry

    async def remove_entry(self, id: uuid.UUID) -> None:
        """
        Mark the cache entry identified by `id` as soft-deleted by setting its `deleted_at` timestamp.
        
        This is a no-op if the entry does not exist or the stored data cannot be deserialized into an Entry.
        
        Parameters:
            id (uuid.UUID): UUID of the entry to mark as deleted.
        """
        entry_key = self._entry_key(id)
        entry_hash = await cast(
            "Awaitable[Dict[bytes, bytes]]", self.client.hgetall(entry_key)
        )

        if not entry_hash or b"data" not in entry_hash:
            return

        # Deserialize entry
        entry = unpack(entry_hash[b"data"], kind="pair")
        if not isinstance(entry, Entry):
            return

        # Mark as deleted
        deleted_entry = self.mark_pair_as_deleted(entry)
        entry_data = pack(deleted_entry, kind="pair")

        # Update entry with deleted_at timestamp
        await cast(
            "Awaitable[int]",
            self.client.hset(
                entry_key,
                mapping={
                    b"data": entry_data,
                    b"deleted_at": str(deleted_entry.meta.deleted_at).encode("utf-8"),
                },
            ),
        )

    async def close(self) -> None:
        """
        Close the Redis client when this storage instance owns it.
        
        If this storage created/owns the Redis client, attempts to asynchronously close it; on failure a warning is logged.
        """
        if not self._owns_client:
            return
        try:
            await self.client.aclose()
        except Exception as e:
            # Intentionally broad: best-effort cleanup during shutdown
            # Log and continue rather than propagating errors during teardown
            logging.warning(f"Error closing Redis client: {e}")

    async def _save_stream(
        self, stream: AsyncIterator[bytes], entry_id: uuid.UUID, ttl: Optional[float]
    ) -> AsyncIterator[bytes]:
        """
        Save each chunk from `stream` into Redis under the entry's stream key and yield the same chunks.
        
        Parameters:
            stream (AsyncIterator[bytes]): Source async iterator of response body chunks.
            entry_id (uuid.UUID): UUID of the cache entry used to derive the Redis stream key.
            ttl (Optional[float]): Time-to-live in seconds to apply to the Redis stream key after the stream is completed; if None, no TTL is set.
        
        Returns:
            AsyncIterator[bytes]: Yields each byte chunk produced by `stream`. The function also appends a completion marker to the Redis list for the stream and sets the TTL (if provided).
        """
        stream_key = self._stream_key(entry_id)

        async for chunk in stream:
            # Save chunk to Redis list
            await cast("Awaitable[int]", self.client.rpush(stream_key, chunk))
            yield chunk

        # Mark stream as complete
        await cast(
            "Awaitable[int]", self.client.rpush(stream_key, self._COMPLETE_CHUNK_MARKER)
        )

        # Apply TTL now that the stream key exists
        if ttl is not None:
            await self.client.expire(stream_key, int(ttl))

    async def _stream_data_from_cache(
        self, entry_id: uuid.UUID
    ) -> AsyncIterator[bytes]:
        """
        Yield stored response stream chunks for the given entry ID read from Redis.
        
        Parameters:
            entry_id (uuid.UUID): Identifier of the cached entry whose stream is stored in Redis.
        
        Returns:
            AsyncIterator[bytes]: An async iterator yielding each stored stream chunk; stops when the stream completion marker is encountered.
        """
        stream_key = self._stream_key(entry_id)

        # Get all chunks at once (could optimize with cursor for large streams)
        chunks = await cast(
            "Awaitable[List[bytes]]", self.client.lrange(stream_key, 0, -1)
        )

        for chunk in chunks:
            # Skip completion marker
            if chunk == self._COMPLETE_CHUNK_MARKER:
                break
            yield chunk

    def _get_entry_ttl(self, request: Request) -> Optional[float]:
        """
        Determine the time-to-live (TTL) for a cache entry by checking the request's metadata and falling back to the storage's default.
        
        Parameters:
            request (Request): The HTTP request whose `metadata` may include a numeric `hishel_ttl` value (seconds).
        
        Returns:
            Optional[float]: TTL in seconds as a float if configured, or `None` when no TTL is set.
        
        Raises:
            ValueError: If `hishel_ttl` is present and is a negative number.
        """
        # Check for per-request TTL
        if "hishel_ttl" in request.metadata:
            ttl_value = request.metadata["hishel_ttl"]
            if isinstance(ttl_value, (int, float)):
                ttl_float = float(ttl_value)
                if ttl_float < 0:
                    raise ValueError(
                        f"TTL must be non-negative, got {ttl_float}"
                    )
                return ttl_float

        # Use default TTL
        return self.default_ttl

    async def cleanup_expired(self) -> int:
        """
        Remove soft-deleted cache entries that are safe for permanent removal.
        
        Scans stored entry records in Redis and hard-deletes entries that are marked as soft-deleted and considered safe to remove.
        
        Returns:
            cleaned (int): Number of entries that were hard-deleted.
        """
        cleaned = 0

        # Scan for all entry keys
        pattern = f"{self.key_prefix}:entry:*"
        cursor = 0

        while True:
            cursor, keys = await self.client.scan(cursor, match=pattern, count=100)

            for key in keys:
                entry_hash = await cast(
                    "Awaitable[Dict[bytes, bytes]]", self.client.hgetall(key)
                )
                if not entry_hash or b"data" not in entry_hash:
                    continue

                entry = unpack(entry_hash[b"data"], kind="pair")
                if not isinstance(entry, Entry):
                    continue

                # Hard delete if safe to do so
                if self.is_soft_deleted(entry) and self.is_safe_to_hard_delete(entry):
                    await self._hard_delete_entry(entry.id)
                    cleaned += 1

            if cursor == 0:
                break

        return cleaned

    async def _hard_delete_entry(self, entry_id: uuid.UUID) -> None:
        """
        Permanently remove an entry and its stream from Redis and unlink it from the cache-key index.
        
        If the entry hash contains a `cache_key` field, the entry ID is removed from that index set; afterwards the entry metadata key and its stream key are deleted.
        
        Parameters:
            entry_id (uuid.UUID): UUID of the entry to hard-delete.
        """
        entry_key = self._entry_key(entry_id)
        stream_key = self._stream_key(entry_id)

        # Get cache key before deleting
        entry_hash = await cast(
            "Awaitable[Dict[bytes, bytes]]", self.client.hgetall(entry_key)
        )
        if entry_hash and b"cache_key" in entry_hash:
            cache_key = entry_hash[b"cache_key"].decode("utf-8")
            index_key = self._index_key(cache_key)

            # Remove from index
            await cast(
                "Awaitable[int]", self.client.srem(index_key, str(entry_id).encode("utf-8"))
            )

        # Delete entry and stream
        pipe = self.client.pipeline()
        pipe.delete(entry_key)
        pipe.delete(stream_key)
        await pipe.execute()
