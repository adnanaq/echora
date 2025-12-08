# Cache Improvements Implementation Plan

## Overview

This document outlines planned improvements to the HTTP cache system based on code review feedback. These enhancements will provide better observability, performance optimization, and production-ready monitoring capabilities.

**Status**: Planned for future PR
**Related PR**: feat/avs-29-implement-redis-based-http-caching-for-enrichment-pipeline
**Created**: 2025-11-20

---

## Code Review Feedback Summary

### 1. Missing Cache Statistics ⚠️

**Current State** (`src/cache_manager/manager.py:214-228`):
```python
def get_stats(self) -> Dict[str, Any]:
    if not self.config.enabled:
        return {"enabled": False}

    return {
        "enabled": True,
        "storage_type": self.config.storage_type,
        "redis_url": self.config.redis_url,
    }
```

**Issues**:
- Only returns basic configuration info
- No performance metrics (hit/miss ratios)
- No visibility into cache effectiveness
- No capacity monitoring (total entries, cache size)
- No TTL information

**Impact**: Cannot assess cache performance or debug cache-related issues in production.

---

### 2. No Cache Warming Strategy ⚠️

**Current State**: No cache warming functionality exists

**Issues**:
- Cold cache on service startup
- Slow initial requests after deployment
- Predictable workloads (nightly enrichment runs) don't benefit from preloading

**Suggested Implementation**:
```python
async def warm_cache(anime_ids: List[str]):
    """Pre-populate cache for known anime IDs"""
    for anime_id in anime_ids:
        await fetch_anime_metadata(anime_id)
```

**Impact**: Every deployment/restart results in cache misses until warmed up naturally.

---

### 3. No Monitoring & Observability ⚠️

**Current State**: Basic logging only (INFO for config, DEBUG for client creation, WARNING for errors)

**Issues**:
- No structured logging for cache operations
- Cannot build dashboards showing cache performance per service
- No visibility into cache hit/miss patterns
- Cannot track performance degradation over time

**Suggested Implementation**:
```python
logger.info(
    "cache_hit",
    extra={
        "service": service,
        "cache_key": key,
        "ttl_remaining": ttl
    }
)
```

**Impact**: Blind to cache performance in production; difficult to optimize or debug.

---

## Validation Results

✅ **All three concerns are VALID and should be addressed**

| Concern | Validated | Priority | Complexity |
|---------|-----------|----------|------------|
| Enhanced Cache Statistics | ✅ | **HIGH** | Medium |
| Cache Warming Strategy | ✅ | Medium | Low |
| Structured Logging | ✅ | **HIGH** | Low |

---

## Design Decisions

### 1. Cache Statistics Tracking

**Decision**: Support both in-memory and Redis-backed statistics with configuration flag

**Rationale**:
- **In-memory**: Fast, lightweight, good for development/debugging
- **Redis-backed**: Persistent across restarts, survives deployments, production-ready
- **Config flag**: Flexibility for different environments

**Configuration**:
```python
class CacheConfig:
    # ... existing fields ...
    statistics_storage: Literal["memory", "redis"] = "memory"
    statistics_enabled: bool = True
```

---

### 2. Structured Logging Levels

**Decision**: Implement all four logging levels (configurable per environment)

**Logging Categories**:

1. **Cache Hits/Misses Per Service** (DEBUG level)
   - Log every cache operation
   - Includes: service, cache_key, operation (hit/miss)
   - Good for: Development, debugging specific issues

2. **Aggregate Metrics** (INFO level)
   - Log periodic summaries (every 100 requests)
   - Includes: hit_rate, total_requests, service breakdown
   - Good for: Production monitoring without log spam

3. **TTL and Expiry Info** (DEBUG level)
   - Includes: ttl_remaining, expires_at, refresh_status
   - Good for: Understanding cache freshness patterns

4. **Request Timing Metrics** (INFO level)
   - Includes: cache_lookup_time, backend_request_time, cache_save_time
   - Good for: Performance optimization, SLA monitoring

**Configuration**:
```python
class CacheConfig:
    # ... existing fields ...
    logging_level: Literal["detailed", "aggregate", "minimal"] = "aggregate"
    log_cache_hits: bool = True
    log_cache_misses: bool = True
    log_ttl_info: bool = False  # Only in DEBUG mode
    log_timing_metrics: bool = True
```

---

### 3. Cache Warming Strategy

**Decision**: Defer to future PR, but document the approach

**Recommended Approach**: Manual API endpoint

**Rationale**:
- **Explicit control**: Admin can trigger warming when needed
- **No startup delay**: Service starts quickly
- **Flexible**: Can warm specific anime IDs or ranges
- **Testable**: Easy to test and validate

**Deferred Options**:
- ❌ Automatic on startup (slows down deployment)
- ❌ Scheduled background task (adds complexity, requires job scheduler)

---

## Implementation Plan (TDD Approach)

### Phase 1: Enhanced Cache Statistics

**Test-Driven Development Sequence**:

#### Step 1: Write Failing Tests

```python
# tests/cache_manager/test_cache_statistics.py

@pytest.mark.asyncio
async def test_get_stats_includes_hit_miss_ratio():
    """Test that get_stats() returns hit/miss ratio."""
    manager = HTTPCacheManager(config)

    # Simulate 10 hits, 5 misses
    # ... perform cache operations ...

    stats = manager.get_stats()
    assert "hit_rate" in stats
    assert stats["hit_rate"] == 0.667  # 10/(10+5)
    assert stats["total_hits"] == 10
    assert stats["total_misses"] == 5
    assert stats["total_requests"] == 15

@pytest.mark.asyncio
async def test_get_stats_includes_cache_size_metrics():
    """Test that get_stats() returns cache size metrics."""
    manager = HTTPCacheManager(config)

    # Add entries to cache
    # ... perform cache operations ...

    stats = manager.get_stats()
    assert "total_entries" in stats
    assert "cache_size_bytes" in stats
    assert stats["total_entries"] > 0

@pytest.mark.asyncio
async def test_get_stats_includes_average_ttl():
    """Test that get_stats() returns average TTL remaining."""
    manager = HTTPCacheManager(config)

    # Add entries with known TTLs
    # ... perform cache operations ...

    stats = manager.get_stats()
    assert "average_ttl_remaining" in stats
    assert stats["average_ttl_remaining"] > 0
```

#### Step 2: Run Tests (Watch Them Fail)

```bash
$ uv run pytest tests/cache_manager/test_cache_statistics.py -v

FAIL: test_get_stats_includes_hit_miss_ratio
FAIL: test_get_stats_includes_cache_size_metrics
FAIL: test_get_stats_includes_average_ttl
```

**Expected Failures**:
- `KeyError: 'hit_rate'` - Field not in stats dict
- `KeyError: 'total_entries'` - Field not in stats dict
- `KeyError: 'average_ttl_remaining'` - Field not in stats dict

#### Step 3: Implement Minimal Code

**3a. Add Statistics Tracking to HTTPCacheManager**

```python
# src/cache_manager/manager.py

from dataclasses import dataclass
from typing import Optional

@dataclass
class CacheStatistics:
    """Cache statistics tracker."""
    total_hits: int = 0
    total_misses: int = 0
    total_entries: int = 0
    cache_size_bytes: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.total_hits + self.total_misses
        return self.total_hits / total if total > 0 else 0.0

    @property
    def total_requests(self) -> int:
        """Total cache requests."""
        return self.total_hits + self.total_misses


class HTTPCacheManager:
    def __init__(self, config: CacheConfig):
        # ... existing init ...
        self._stats = CacheStatistics()
        self._stats_storage_type = config.statistics_storage

    def _record_cache_hit(self, key: str, size_bytes: int = 0) -> None:
        """Record a cache hit."""
        self._stats.total_hits += 1
        # If Redis-backed, also increment Redis counter
        if self._stats_storage_type == "redis":
            # ... Redis INCR commands ...
            pass

    def _record_cache_miss(self, key: str) -> None:
        """Record a cache miss."""
        self._stats.total_misses += 1
        if self._stats_storage_type == "redis":
            # ... Redis INCR commands ...
            pass

    async def _get_average_ttl(self) -> Optional[float]:
        """Calculate average TTL across cached entries."""
        if not self._async_redis_client:
            return None

        # Scan cache keys and get TTLs
        total_ttl = 0
        count = 0

        async for key in self._async_redis_client.scan_iter(
            match="hishel_cache:*"
        ):
            ttl = await self._async_redis_client.ttl(key)
            if ttl > 0:
                total_ttl += ttl
                count += 1

        return total_ttl / count if count > 0 else None

    async def get_cache_size_metrics(self) -> dict:
        """Get cache size metrics from Redis."""
        if not self._async_redis_client:
            return {"total_entries": 0, "cache_size_bytes": 0}

        # Count keys matching cache prefix
        total_entries = 0
        async for _ in self._async_redis_client.scan_iter(
            match="hishel_cache:*"
        ):
            total_entries += 1

        # Get Redis memory usage for our keys
        info = await self._async_redis_client.info("memory")
        cache_size_bytes = info.get("used_memory", 0)

        return {
            "total_entries": total_entries,
            "cache_size_bytes": cache_size_bytes
        }

    async def get_stats_async(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics (async version).

        Returns:
            Dictionary with cache statistics including:
            - enabled: bool
            - storage_type: str
            - redis_url: str
            - hit_rate: float (0.0-1.0)
            - total_hits: int
            - total_misses: int
            - total_requests: int
            - total_entries: int
            - cache_size_bytes: int
            - average_ttl_remaining: float | None
        """
        if not self.config.enabled:
            return {"enabled": False}

        # Get cache size metrics
        size_metrics = await self.get_cache_size_metrics()

        # Get average TTL
        avg_ttl = await self._get_average_ttl()

        return {
            # Basic config
            "enabled": True,
            "storage_type": self.config.storage_type,
            "redis_url": self.config.redis_url,

            # Hit/Miss metrics
            "hit_rate": self._stats.hit_rate,
            "total_hits": self._stats.total_hits,
            "total_misses": self._stats.total_misses,
            "total_requests": self._stats.total_requests,

            # Capacity metrics
            "total_entries": size_metrics["total_entries"],
            "cache_size_bytes": size_metrics["cache_size_bytes"],

            # TTL metrics
            "average_ttl_remaining": avg_ttl,
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics (sync version - deprecated).

        Note: This is the old sync version. Use get_stats_async() instead.
        This version only returns basic config info for backwards compatibility.
        """
        if not self.config.enabled:
            return {"enabled": False}

        return {
            "enabled": True,
            "storage_type": self.config.storage_type,
            "redis_url": self.config.redis_url,
            # Note: Hit/miss stats require async Redis calls
            # Use get_stats_async() for complete statistics
        }
```

**3b. Integrate Statistics Recording**

```python
# src/cache_manager/aiohttp_adapter.py (or wherever cache hits/misses occur)

class CachedAiohttpSession:
    async def _request(self, method, url, **kwargs):
        # ... existing cache lookup logic ...

        if cached_response:
            # Record cache hit
            if hasattr(self._manager, '_record_cache_hit'):
                self._manager._record_cache_hit(
                    key=cache_key,
                    size_bytes=len(cached_response.content)
                )
            return cached_response

        # Cache miss - fetch from backend
        if hasattr(self._manager, '_record_cache_miss'):
            self._manager._record_cache_miss(key=cache_key)

        # ... existing backend fetch logic ...
```

#### Step 4: Run Tests (Watch Them Pass)

```bash
$ uv run pytest tests/cache_manager/test_cache_statistics.py -v

PASS: test_get_stats_includes_hit_miss_ratio
PASS: test_get_stats_includes_cache_size_metrics
PASS: test_get_stats_includes_average_ttl
```

#### Step 5: Refactor (Keep Tests Green)

- Extract statistics tracking to separate class
- Add Redis-backed statistics option
- Improve efficiency of TTL calculation (sampling instead of full scan)

---

### Phase 2: Structured Logging

**Test-Driven Development Sequence**:

#### Step 1: Write Failing Tests

```python
# tests/cache_manager/test_cache_logging.py

import logging
from unittest.mock import patch

@pytest.mark.asyncio
async def test_cache_hit_logs_structured_data(caplog):
    """Test that cache hits log structured data."""
    caplog.set_level(logging.INFO)

    manager = HTTPCacheManager(config)
    # ... simulate cache hit ...

    # Check log contains structured data
    assert any(
        record.extra.get("service") == "anilist" and
        record.extra.get("cache_operation") == "hit" and
        "cache_key" in record.extra and
        "ttl_remaining" in record.extra
        for record in caplog.records
    )

@pytest.mark.asyncio
async def test_aggregate_metrics_logged_periodically(caplog):
    """Test that aggregate metrics are logged every N requests."""
    caplog.set_level(logging.INFO)

    manager = HTTPCacheManager(
        config_with_aggregate_logging_every_100_requests
    )

    # Simulate 150 requests
    for i in range(150):
        # ... perform cache operations ...
        pass

    # Should have 1 aggregate log (after 100 requests)
    aggregate_logs = [
        r for r in caplog.records
        if r.extra.get("log_type") == "aggregate_cache_stats"
    ]
    assert len(aggregate_logs) == 1
    assert aggregate_logs[0].extra["total_requests"] == 100
    assert "hit_rate" in aggregate_logs[0].extra

@pytest.mark.asyncio
async def test_timing_metrics_logged_for_cache_operations(caplog):
    """Test that cache operations log timing metrics."""
    caplog.set_level(logging.INFO)

    manager = HTTPCacheManager(config)
    # ... perform cache operation ...

    # Check timing metrics in logs
    assert any(
        "cache_lookup_time_ms" in record.extra and
        "backend_request_time_ms" in record.extra
        for record in caplog.records
    )
```

#### Step 2: Implement Logging Infrastructure

```python
# src/cache_manager/logging_utils.py

import logging
import time
from contextlib import contextmanager
from typing import Dict, Any

logger = logging.getLogger(__name__)

class CacheLogger:
    """Structured logging for cache operations."""

    def __init__(self, config: CacheConfig):
        self.config = config
        self._aggregate_counter = 0
        self._aggregate_interval = 100  # Log every 100 requests

    def log_cache_hit(
        self,
        service: str,
        cache_key: str,
        ttl_remaining: int,
        lookup_time_ms: float
    ) -> None:
        """Log a cache hit with structured data."""
        if not self.config.log_cache_hits:
            return

        logger.info(
            f"Cache hit for {service}",
            extra={
                "service": service,
                "cache_operation": "hit",
                "cache_key": cache_key,
                "ttl_remaining": ttl_remaining,
                "cache_lookup_time_ms": lookup_time_ms,
            }
        )

        self._maybe_log_aggregate()

    def log_cache_miss(
        self,
        service: str,
        cache_key: str,
        lookup_time_ms: float,
        backend_request_time_ms: float
    ) -> None:
        """Log a cache miss with structured data."""
        if not self.config.log_cache_misses:
            return

        logger.info(
            f"Cache miss for {service}",
            extra={
                "service": service,
                "cache_operation": "miss",
                "cache_key": cache_key,
                "cache_lookup_time_ms": lookup_time_ms,
                "backend_request_time_ms": backend_request_time_ms,
            }
        )

        self._maybe_log_aggregate()

    def _maybe_log_aggregate(self) -> None:
        """Log aggregate metrics every N requests."""
        self._aggregate_counter += 1

        if self._aggregate_counter >= self._aggregate_interval:
            # Get stats from manager
            stats = self._manager.get_stats()

            logger.info(
                f"Aggregate cache stats: {stats['hit_rate']:.2%} hit rate",
                extra={
                    "log_type": "aggregate_cache_stats",
                    "total_requests": stats["total_requests"],
                    "hit_rate": stats["hit_rate"],
                    "total_hits": stats["total_hits"],
                    "total_misses": stats["total_misses"],
                }
            )

            # Reset counter
            self._aggregate_counter = 0

    @contextmanager
    def time_cache_operation(self, operation: str):
        """Context manager to time cache operations."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.debug(
                f"Cache operation '{operation}' took {elapsed_ms:.2f}ms",
                extra={
                    "cache_operation_type": operation,
                    "duration_ms": elapsed_ms,
                }
            )
```

---

### Phase 3: Cache Warming (Deferred)

**To be implemented in future PR**

**Proposed API Endpoint**:

```python
# src/api/routers/admin.py

@router.post("/warm-cache")
async def warm_cache(
    request: WarmCacheRequest,
    manager: HTTPCacheManager = Depends(get_cache_manager)
) -> WarmCacheResponse:
    """
    Warm cache for specified anime IDs.

    Args:
        request: WarmCacheRequest with anime_ids list

    Returns:
        WarmCacheResponse with success/failure counts
    """
    warmed = 0
    failed = 0

    for anime_id in request.anime_ids:
        try:
            # Fetch anime metadata to populate cache
            await fetch_anime_metadata(anime_id, manager)
            warmed += 1
        except Exception as e:
            logger.warning(f"Failed to warm cache for {anime_id}: {e}")
            failed += 1

    return WarmCacheResponse(
        warmed_count=warmed,
        failed_count=failed,
        total_requested=len(request.anime_ids)
    )
```

**Test Plan**:
1. Test warming with valid anime IDs
2. Test warming with invalid anime IDs (should fail gracefully)
3. Test warming with empty list (should return 0 warmed)
4. Test warming with duplicate IDs (should handle idempotently)
5. Test performance impact on service startup

---

## Technical Specifications

### New Configuration Parameters

```python
# src/cache_manager/config.py

class CacheConfig(BaseSettings):
    # ... existing fields ...

    # Statistics Configuration
    statistics_enabled: bool = Field(
        default=True,
        description="Enable cache statistics tracking"
    )
    statistics_storage: Literal["memory", "redis"] = Field(
        default="memory",
        description="Storage backend for cache statistics"
    )

    # Logging Configuration
    logging_level: Literal["detailed", "aggregate", "minimal"] = Field(
        default="aggregate",
        description="Cache logging verbosity level"
    )
    log_cache_hits: bool = Field(
        default=True,
        description="Log cache hit operations"
    )
    log_cache_misses: bool = Field(
        default=True,
        description="Log cache miss operations"
    )
    log_ttl_info: bool = Field(
        default=False,
        description="Log TTL and expiry information (DEBUG mode)"
    )
    log_timing_metrics: bool = Field(
        default=True,
        description="Log cache operation timing metrics"
    )
    aggregate_log_interval: int = Field(
        default=100,
        description="Log aggregate stats every N requests"
    )
```

### Enhanced API Response Schema

```python
# Enhanced get_stats() response

{
    # Existing fields
    "enabled": true,
    "storage_type": "redis",
    "redis_url": "redis://localhost:6379/0",

    # NEW: Hit/Miss Metrics
    "hit_rate": 0.847,  # 84.7% hit rate
    "total_hits": 1234,
    "total_misses": 221,
    "total_requests": 1455,

    # NEW: Capacity Metrics
    "total_entries": 5678,
    "cache_size_bytes": 123456789,
    "cache_size_mb": 117.7,

    # NEW: TTL Metrics
    "average_ttl_remaining": 43200.5,  # seconds
    "average_ttl_remaining_hours": 12.0,

    # NEW: Per-Service Breakdown (optional, expensive)
    "per_service": {
        "anilist": {
            "hit_rate": 0.92,
            "total_hits": 450,
            "total_misses": 39
        },
        "jikan": {
            "hit_rate": 0.78,
            "total_hits": 784,
            "total_misses": 182
        }
    }
}
```

---

## Performance Considerations

### Statistics Tracking Overhead

**In-Memory Statistics**:
- Overhead: ~0.1μs per cache operation (negligible)
- Memory: ~1KB for counters
- Thread-safe using `asyncio.Lock`

**Redis-Backed Statistics**:
- Overhead: ~1-2ms per cache operation (Redis INCR)
- Can be optimized with pipelining
- Persistent across restarts

**Recommendation**: Use in-memory for development, Redis for production

---

### TTL Calculation Optimization

**Naive Approach** (expensive):
```python
# Scan ALL cache keys - O(n) where n = total entries
async for key in redis.scan_iter("hishel_cache:*"):
    ttl = await redis.ttl(key)
    total += ttl
```

**Optimized Approach** (sampling):
```python
# Sample 100 random keys - O(1) with high confidence
sample_size = min(100, total_entries)
sampled_ttls = []

for _ in range(sample_size):
    key = await redis.randomkey()
    if key and key.startswith("hishel_cache:"):
        ttl = await redis.ttl(key)
        sampled_ttls.append(ttl)

average_ttl = sum(sampled_ttls) / len(sampled_ttls)
```

**Trade-off**: Sampling gives ~95% accuracy with 100x speedup

---

### Logging Performance Impact

**Detailed Logging** (every operation):
- Overhead: ~0.5-1ms per cache operation
- Log volume: High (100-1000 entries/minute)
- Good for: Development, debugging

**Aggregate Logging** (periodic summaries):
- Overhead: ~0.01ms per cache operation (counter increment)
- Log volume: Low (1 entry per 100 requests)
- Good for: Production monitoring

**Recommendation**: Use aggregate logging in production

---

## Migration Strategy

### Backwards Compatibility

**Existing `get_stats()` method**: Keep for backwards compatibility

```python
def get_stats(self) -> Dict[str, Any]:
    """Legacy sync version - returns basic config only."""
    # ... existing implementation ...

async def get_stats_async(self) -> Dict[str, Any]:
    """New async version with comprehensive statistics."""
    # ... new implementation ...
```

**Deprecation Path**:
1. **v1.0**: Add `get_stats_async()`, keep `get_stats()` unchanged
2. **v1.1**: Deprecate `get_stats()` with warning
3. **v2.0**: Remove `get_stats()`, rename `get_stats_async()` to `get_stats()`

---

### Deployment Rollout

**Phase 1: Statistics Tracking** (Week 1)
- Deploy in-memory statistics
- Monitor for performance regressions
- Validate accuracy of metrics

**Phase 2: Redis-Backed Stats** (Week 2)
- Enable Redis statistics in staging
- Compare with in-memory stats for accuracy
- Deploy to production with feature flag

**Phase 3: Structured Logging** (Week 3)
- Enable aggregate logging in production
- Build Grafana dashboards
- Set up alerts for low hit rates

**Phase 4: Cache Warming** (Week 4+)
- Add admin API endpoint
- Test with small batches
- Roll out to nightly enrichment jobs

---

## Testing Strategy

### Unit Tests

- `test_cache_statistics.py`: Statistics tracking and calculation
- `test_cache_logging.py`: Structured logging output
- `test_cache_warming.py`: Cache warming logic

### Integration Tests

- Redis-backed statistics with real Redis instance
- Multi-agent concurrent statistics tracking
- Cache warming with real API endpoints

### Performance Tests

- Benchmark statistics tracking overhead
- Measure logging performance impact
- Load test cache warming endpoint

---

## Success Metrics

### KPIs to Track

1. **Cache Hit Rate**: Target 70-80% after cache is warmed
2. **Average TTL**: Should be > 12 hours on average
3. **Cache Size**: Monitor growth, set alerts for capacity limits
4. **Logging Overhead**: < 1ms per request for aggregate logging

### Monitoring Dashboard

**Grafana Panels**:
- Cache hit rate over time (per service)
- Total cache entries and size
- Average TTL remaining
- Request latency (cache vs. backend)

**Alerts**:
- Cache hit rate drops below 50%
- Cache size exceeds 80% of Redis memory
- Average TTL drops below 6 hours

---

## Open Questions

1. **Should we track per-service statistics by default, or make it opt-in?**
   - Pro: More granular visibility
   - Con: Higher memory overhead (N services × stats)

2. **What sampling rate for TTL calculation?**
   - 100 keys = 95% confidence
   - 1000 keys = 99.9% confidence
   - Trade-off: Accuracy vs. performance

3. **Should cache warming be synchronous or async (background task)?**
   - Sync: Blocks API response until cache is warmed
   - Async: Returns immediately, warms in background
   - Recommendation: Async with progress tracking

4. **Redis SCAN pattern for cache key enumeration?**
   - Pattern: `hishel_cache:*`
   - Consider using dedicated Redis DB for cache to avoid scanning non-cache keys

---

## References

- **Code Review PR**: feat/avs-29-implement-redis-based-http-caching-for-enrichment-pipeline
- **Related Files**:
  - `src/cache_manager/manager.py` (HTTPCacheManager)
  - `src/cache_manager/config.py` (CacheConfig)
  - `src/cache_manager/aiohttp_adapter.py` (CachedAiohttpSession)
- **External Documentation**:
  - [Redis INFO command](https://redis.io/commands/info/)
  - [Redis SCAN command](https://redis.io/commands/scan/)
  - [Python structlog](https://www.structlog.org/) (alternative to stdlib logging)

---

## Next Steps

1. **Create GitHub Issue** for cache improvements tracking
2. **Review this plan** with team for feedback
3. **Schedule implementation** for next sprint
4. **Set up monitoring infrastructure** (Grafana dashboards) in advance

---

**Document Version**: 1.0
**Last Updated**: 2025-11-20
**Author**: Backend Engineering Team
**Reviewers**: [To be added]
