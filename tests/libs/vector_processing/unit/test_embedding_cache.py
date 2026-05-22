"""Unit tests for EmbeddingCache.

Tests cover get/set (single and batch), fail-open behavior on Redis
errors, close, and edge cases like empty batches.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from vector_processing.cache import EmbeddingCache


@pytest.fixture
def mock_redis():
    """Create a mock async Redis client."""
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock()
    redis.mget = AsyncMock(return_value=[])
    redis.pipeline = MagicMock()
    redis.aclose = AsyncMock()
    return redis


@pytest.fixture
def cache(mock_redis):
    """Create an EmbeddingCache with a mock Redis client."""
    return EmbeddingCache(mock_redis, ttl=3600)


class TestGet:
    """Tests for single-key get."""

    @pytest.mark.asyncio
    async def test_cache_miss_returns_none(self, cache, mock_redis):
        """Test that a cache miss returns None."""
        mock_redis.get.return_value = None

        result = await cache.get("bge-m3", "abc123")

        assert result is None
        mock_redis.get.assert_awaited_once_with("emb:bge-m3:abc123")

    @pytest.mark.asyncio
    async def test_cache_hit_returns_embedding(self, cache, mock_redis):
        """Test that a cache hit deserializes and returns the embedding."""
        embedding = [0.1, 0.2, 0.3]
        mock_redis.get.return_value = json.dumps(embedding)

        result = await cache.get("bge-m3", "abc123")

        assert result == embedding

    @pytest.mark.asyncio
    async def test_redis_error_returns_none(self, cache, mock_redis, caplog):
        """Test fail-open: Redis error returns None without raising."""
        mock_redis.get.side_effect = ConnectionError("Redis down")

        with caplog.at_level("WARNING"):
            result = await cache.get("bge-m3", "abc123")

        assert result is None
        assert "embedding cache get failed" in caplog.text


class TestSet:
    """Tests for single-key set."""

    @pytest.mark.asyncio
    async def test_set_stores_embedding_with_ttl(self, cache, mock_redis):
        """Test that set stores JSON-serialized embedding with TTL."""
        embedding = [0.1, 0.2, 0.3]

        await cache.set("bge-m3", "abc123", embedding)

        mock_redis.set.assert_awaited_once_with(
            "emb:bge-m3:abc123", json.dumps(embedding), ex=3600
        )

    @pytest.mark.asyncio
    async def test_redis_error_silently_ignored(self, cache, mock_redis, caplog):
        """Test fail-open: Redis error on set is logged but not raised."""
        mock_redis.set.side_effect = ConnectionError("Redis down")

        with caplog.at_level("WARNING"):
            await cache.set("bge-m3", "abc123", [0.1])

        assert "embedding cache set failed" in caplog.text


class TestGetBatch:
    """Tests for batch get (MGET)."""

    @pytest.mark.asyncio
    async def test_empty_hashes_returns_empty_list(self, cache):
        """Test that empty input returns empty list without hitting Redis."""
        result = await cache.get_batch("bge-m3", [])

        assert result == []

    @pytest.mark.asyncio
    async def test_all_hits(self, cache, mock_redis):
        """Test batch get with all cache hits."""
        emb1, emb2 = [0.1, 0.2], [0.3, 0.4]
        mock_redis.mget.return_value = [json.dumps(emb1), json.dumps(emb2)]

        result = await cache.get_batch("bge-m3", ["hash1", "hash2"])

        assert result == [emb1, emb2]
        mock_redis.mget.assert_awaited_once_with(
            ["emb:bge-m3:hash1", "emb:bge-m3:hash2"]
        )

    @pytest.mark.asyncio
    async def test_partial_hits(self, cache, mock_redis):
        """Test batch get with mix of hits and misses."""
        emb1 = [0.1, 0.2]
        mock_redis.mget.return_value = [json.dumps(emb1), None]

        result = await cache.get_batch("bge-m3", ["hash1", "hash2"])

        assert result == [emb1, None]

    @pytest.mark.asyncio
    async def test_redis_error_returns_all_none(self, cache, mock_redis, caplog):
        """Test fail-open: Redis error returns list of None."""
        mock_redis.mget.side_effect = ConnectionError("Redis down")

        with caplog.at_level("WARNING"):
            result = await cache.get_batch("bge-m3", ["h1", "h2", "h3"])

        assert result == [None, None, None]
        assert "embedding cache get_batch failed" in caplog.text


class TestSetBatch:
    """Tests for batch set (pipeline)."""

    @pytest.mark.asyncio
    async def test_empty_entries_skips_redis(self, cache, mock_redis):
        """Test that empty entries dict doesn't touch Redis."""
        await cache.set_batch("bge-m3", {})

        mock_redis.pipeline.assert_not_called()

    @pytest.mark.asyncio
    async def test_set_batch_uses_pipeline(self, cache, mock_redis):
        """Test that set_batch uses a Redis pipeline with TTL per key."""
        mock_pipe = AsyncMock()
        mock_redis.pipeline.return_value = mock_pipe

        entries = {"h1": [0.1], "h2": [0.2]}
        await cache.set_batch("bge-m3", entries)

        mock_redis.pipeline.assert_called_once_with(transaction=False)
        assert mock_pipe.set.call_count == 2
        mock_pipe.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_redis_error_silently_ignored(self, cache, mock_redis, caplog):
        """Test fail-open: pipeline error is logged but not raised."""
        mock_redis.pipeline.side_effect = ConnectionError("Redis down")

        with caplog.at_level("WARNING"):
            await cache.set_batch("bge-m3", {"h1": [0.1]})

        assert "embedding cache set_batch failed" in caplog.text


class TestClose:
    """Tests for close."""

    @pytest.mark.asyncio
    async def test_close_calls_aclose(self, cache, mock_redis):
        """Test that close delegates to Redis aclose."""
        await cache.close()

        mock_redis.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_error_silently_ignored(self, cache, mock_redis, caplog):
        """Test fail-open: close error is logged but not raised."""
        mock_redis.aclose.side_effect = ConnectionError("Redis gone")

        with caplog.at_level("WARNING"):
            await cache.close()

        assert "embedding cache close failed" in caplog.text


class TestKeyFormat:
    """Tests for key generation."""

    def test_key_format(self, cache):
        """Test that keys follow the expected pattern."""
        key = cache._key("bge-m3", "abc123def456")

        assert key == "emb:bge-m3:abc123def456"

    def test_different_models_produce_different_keys(self, cache):
        """Test that different model names produce different keys."""
        key1 = cache._key("bge-m3", "same_hash")
        key2 = cache._key("openclip-vit-l14", "same_hash")

        assert key1 != key2
