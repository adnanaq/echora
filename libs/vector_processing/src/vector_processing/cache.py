"""Embedding cache backed by Redis for deterministic model inference results.

Provides a thin async wrapper around Redis that caches embedding vectors
keyed by model name and input content hash. All Redis errors are caught
and logged — cache failures never block inference (fail-open).
"""

from __future__ import annotations

import json
import logging
import time

from opentelemetry import metrics as otel_metrics
from redis.asyncio import Redis

logger = logging.getLogger(__name__)

# Default TTL: 7 days (embeddings are deterministic per model version)
_DEFAULT_TTL_SECONDS = 7 * 24 * 60 * 60

_meter = otel_metrics.get_meter("echora.embedding_cache")
_cache_op_duration = _meter.create_histogram(
    "echora_cache_operation_duration_seconds",
    unit="s",
    description="Redis cache operation duration in seconds",
)


class EmbeddingCache:
    """Async Redis cache for embedding vectors.

    Key pattern: ``emb:{model_name}:{sha256_hex}``
    Values are JSON-encoded ``list[float]``.
    """

    def __init__(self, redis: Redis, ttl: int = _DEFAULT_TTL_SECONDS) -> None:
        self._redis = redis
        self._ttl = ttl

    @staticmethod
    def _key(model_name: str, input_hash: str) -> str:
        return f"emb:{model_name}:{input_hash}"

    async def get(self, model_name: str, input_hash: str) -> list[float] | None:
        """Fetch a cached embedding. Returns None on miss or Redis error."""
        try:
            _start = time.perf_counter()
            raw = await self._redis.get(self._key(model_name, input_hash))
            _cache_op_duration.record(
                time.perf_counter() - _start, {"operation": "get"}
            )
            if raw is None:
                return None
            return json.loads(raw)
        except Exception:
            logger.warning("embedding cache get failed", exc_info=True)
            return None

    async def set(
        self, model_name: str, input_hash: str, embedding: list[float]
    ) -> None:
        """Store an embedding. Silently ignores Redis errors."""
        try:
            key = self._key(model_name, input_hash)
            _start = time.perf_counter()
            await self._redis.set(key, json.dumps(embedding), ex=self._ttl)  # ty: ignore[possibly-missing-attribute]
            _cache_op_duration.record(
                time.perf_counter() - _start, {"operation": "set"}
            )
        except Exception:
            logger.warning("embedding cache set failed", exc_info=True)

    async def get_batch(
        self, model_name: str, input_hashes: list[str]
    ) -> list[list[float] | None]:
        """MGET multiple embeddings. Returns list aligned with input_hashes."""
        if not input_hashes:
            return []
        try:
            keys = [self._key(model_name, h) for h in input_hashes]
            _start = time.perf_counter()
            raw_values = await self._redis.mget(keys)  # ty: ignore[possibly-missing-attribute]
            _cache_op_duration.record(
                time.perf_counter() - _start, {"operation": "get_batch"}
            )
            return [json.loads(v) if v is not None else None for v in raw_values]
        except Exception:
            logger.warning("embedding cache get_batch failed", exc_info=True)
            return [None] * len(input_hashes)

    async def set_batch(self, model_name: str, entries: dict[str, list[float]]) -> None:
        """MSET multiple embeddings with per-key TTL."""
        if not entries:
            return
        try:
            pipe = self._redis.pipeline(transaction=False)
            for input_hash, embedding in entries.items():
                key = self._key(model_name, input_hash)
                pipe.set(key, json.dumps(embedding), ex=self._ttl)  # ty: ignore[possibly-missing-attribute]
            _start = time.perf_counter()
            await pipe.execute()
            _cache_op_duration.record(
                time.perf_counter() - _start, {"operation": "set_batch"}
            )
        except Exception:
            logger.warning("embedding cache set_batch failed", exc_info=True)

    async def close(self) -> None:
        """Close the underlying Redis connection."""
        try:
            await self._redis.aclose()
        except Exception:
            logger.warning("embedding cache close failed", exc_info=True)
