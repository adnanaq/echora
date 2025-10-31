"""
Async Redis storage backend for Hishel HTTP caching.

Implements AsyncBaseStorage interface for Redis-backed async HTTP response caching
with support for multi-agent concurrent access, service-specific TTLs, and
streaming responses.

This is the async parallel to SyncRedisStorage, using redis.asyncio for
aiohttp-based API helpers (AniList, Kitsu, AniDB).
"""

from __future__ import annotations

import time
import uuid
from typing import AsyncIterator, Callable, List, Optional, Union

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
        client: Optional[Redis[bytes]] = None,
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
        self.client = client or Redis.from_url(redis_url, decode_responses=False)
        self.default_ttl = default_ttl
        self.refresh_ttl_on_access = refresh_ttl_on_access
        self.key_prefix = key_prefix

        # Note: Connection test done lazily on first use

    def _make_key(self, key_type: str, identifier: str) -> str:
        """Generate Redis key with prefix."""
        return f"{self.key_prefix}:{key_type}:{identifier}"

    def _entry_key(self, entry_id: uuid.UUID) -> str:
        """Key for entry metadata and data."""
        return self._make_key("entry", str(entry_id))

    def _stream_key(self, entry_id: uuid.UUID) -> str:
        """Key for response stream chunks."""
        return self._make_key("stream", str(entry_id))

    def _index_key(self, cache_key: str) -> str:
        """Key for cache_key → entry_ids index."""
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
        id_: uuid.UUID | None = None,
    ) -> Entry:
        """
        Create and store a new cache entry.

        Args:
            request: HTTP request
            response: HTTP response
            key: Cache key (string)
            id_: Optional entry UUID (generated if not provided)

        Returns:
            Created Entry object
        """
        entry_id = id_ if id_ is not None else uuid.uuid4()
        entry_meta = EntryMeta(created_at=time.time())

        # Replace response stream with saving wrapper
        assert isinstance(response.stream, AsyncIterator)
        response_with_stream = Response(
            status_code=response.status_code,
            headers=response.headers,
            stream=self._save_stream(response.stream, entry_id),
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

        # Set TTL if configured
        ttl = self._get_entry_ttl(request)
        if ttl is not None:
            pipe.expire(entry_key, int(ttl))
            pipe.expire(self._stream_key(entry_id), int(ttl))
            pipe.expire(index_key, int(ttl))

        await pipe.execute()

        return complete_entry

    async def get_entries(self, key: str) -> List[Entry]:
        """
        Retrieve all entries for a given cache key.

        Args:
            key: Cache key (string)

        Returns:
            List of Entry objects
        """
        index_key = self._index_key(key)
        entry_ids_bytes = await self.client.smembers(index_key)

        if not entry_ids_bytes:
            return []

        final_entries: List[Entry] = []

        for entry_id_bytes in entry_ids_bytes:
            entry_id = uuid.UUID(entry_id_bytes.decode("utf-8"))
            entry_key = self._entry_key(entry_id)

            # Get entry data
            entry_hash = await self.client.hgetall(entry_key)
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
                    await self.client.expire(entry_key, int(ttl))
                    await self.client.expire(self._stream_key(entry_id), int(ttl))

            final_entries.append(entry_with_stream)

        return final_entries

    async def update_entry(
        self,
        id: uuid.UUID,
        new_entry: Union[Entry, Callable[[Entry], Entry]],
    ) -> Optional[Entry]:
        """
        Update an existing cache entry.

        Args:
            id: Entry UUID
            new_entry: New Entry object or callable that transforms existing entry

        Returns:
            Updated Entry or None if not found
        """
        entry_key = self._entry_key(id)
        entry_hash = await self.client.hgetall(entry_key)

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
        pipe.hset(entry_key, b"data", entry_data)

        # Update cache key index if changed
        if current_entry.cache_key != complete_entry.cache_key:
            old_index_key = self._index_key(current_entry.cache_key.decode("utf-8"))
            new_index_key = self._index_key(complete_entry.cache_key.decode("utf-8"))

            pipe.srem(old_index_key, str(id).encode("utf-8"))
            pipe.sadd(new_index_key, str(id).encode("utf-8"))
            pipe.hset(entry_key, b"cache_key", complete_entry.cache_key)

        await pipe.execute()

        return complete_entry

    async def remove_entry(self, id: uuid.UUID) -> None:
        """
        Soft delete an entry (sets deleted_at timestamp).

        Args:
            id: Entry UUID
        """
        entry_key = self._entry_key(id)
        entry_hash = await self.client.hgetall(entry_key)

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
        await self.client.hset(
            entry_key,
            mapping={
                b"data": entry_data,
                b"deleted_at": str(deleted_entry.meta.deleted_at).encode("utf-8"),
            },
        )

    async def close(self) -> None:
        """Close Redis connection."""
        try:
            await self.client.close()
        except Exception:
            pass

    async def _save_stream(
        self, stream: AsyncIterator[bytes], entry_id: uuid.UUID
    ) -> AsyncIterator[bytes]:
        """
        Wrapper around async iterator that saves response stream to Redis in chunks.

        Args:
            stream: Original response stream
            entry_id: Entry UUID

        Yields:
            Stream chunks
        """
        stream_key = self._stream_key(entry_id)

        async for chunk in stream:
            # Save chunk to Redis list
            await self.client.rpush(stream_key, chunk)
            yield chunk

        # Mark stream as complete
        await self.client.rpush(stream_key, self._COMPLETE_CHUNK_MARKER)

    async def _stream_data_from_cache(
        self, entry_id: uuid.UUID
    ) -> AsyncIterator[bytes]:
        """
        Get async iterator that yields response stream from Redis.

        Args:
            entry_id: Entry UUID

        Yields:
            Stream chunks
        """
        stream_key = self._stream_key(entry_id)

        # Get all chunks at once (could optimize with cursor for large streams)
        chunks = await self.client.lrange(stream_key, 0, -1)

        for chunk in chunks:
            # Skip completion marker
            if chunk == self._COMPLETE_CHUNK_MARKER:
                break
            yield chunk

    def _get_entry_ttl(self, request: Request) -> Optional[float]:
        """
        Get TTL for an entry from request metadata or default.

        Args:
            request: HTTP request

        Returns:
            TTL in seconds or None
        """
        # Check for per-request TTL
        if "hishel_ttl" in request.metadata:
            ttl_value = request.metadata["hishel_ttl"]
            if isinstance(ttl_value, (int, float)):
                return float(ttl_value)

        # Use default TTL
        return self.default_ttl

    async def cleanup_expired(self) -> int:
        """
        Cleanup expired and soft-deleted entries.

        Redis handles TTL-based expiration automatically via EXPIRE,
        but we still need to cleanup soft-deleted entries.

        Returns:
            Number of entries cleaned up
        """
        cleaned = 0

        # Scan for all entry keys
        pattern = f"{self.key_prefix}:entry:*"
        cursor = 0

        while True:
            cursor, keys = await self.client.scan(cursor, match=pattern, count=100)

            for key in keys:
                entry_hash = await self.client.hgetall(key)
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
        Permanently delete an entry from Redis.

        Args:
            entry_id: Entry UUID
        """
        entry_key = self._entry_key(entry_id)
        stream_key = self._stream_key(entry_id)

        # Get cache key before deleting
        entry_hash = await self.client.hgetall(entry_key)
        if entry_hash and b"cache_key" in entry_hash:
            cache_key = entry_hash[b"cache_key"].decode("utf-8")
            index_key = self._index_key(cache_key)

            # Remove from index
            await self.client.srem(index_key, str(entry_id).encode("utf-8"))

        # Delete entry and stream
        pipe = self.client.pipeline()
        pipe.delete(entry_key)
        pipe.delete(stream_key)
        await pipe.execute()
