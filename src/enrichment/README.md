# Anime Enrichment Pipeline

Multi-source data enrichment pipeline for anime metadata with intelligent caching and multi-agent support.

## Architecture

```mermaid
flowchart TB
    subgraph Pipeline["Enrichment Pipeline"]
        subgraph API["API Helpers (Async)"]
            Jikan["Jikan<br/>(episodes, characters)"]
            AniList["AniList<br/>(GraphQL)"]
            Kitsu["Kitsu"]
            AniDB["AniDB"]
            AnimSchedule["AnimSchedule"]
        end

        subgraph Crawlers["Browser-Based Crawlers"]
            AniSearch["AniSearch<br/>(crawl4ai)"]
            AnimePlanet["Anime-Planet<br/>(crawl4ai)"]
        end

        subgraph CacheLayer["Cache Layer"]
            HTTPMgr["HTTPCacheManager<br/>(HTTP-level caching)"]
            ResultCache["@cached_result<br/>(Result-level caching)"]
        end

        Redis["Redis Server<br/>localhost:6379/0"]
    end

    API -->|get_aiohttp_session| HTTPMgr
    Crawlers -->|@cached_result| ResultCache
    HTTPMgr -->|hishel_cache:* keys| Redis
    ResultCache -->|result_cache:* keys| Redis

    style Redis fill:#ff6b6b
    style HTTPMgr fill:#4ecdc4
    style ResultCache fill:#4ecdc4
    style API fill:#95e1d3
    style Crawlers fill:#95e1d3
```

## Data Sources

### API Helpers (5 sources)
- **Jikan** - MyAnimeList API wrapper for episodes and characters
- **AniList** - GraphQL API for comprehensive anime metadata
- **Kitsu** - Anime metadata with community ratings
- **AniDB** - Detailed technical metadata
- **AnimSchedule** - Broadcast schedules and airing times

### Crawlers (2 sources)
- **AniSearch** - German anime database (browser automation via crawl4ai)
- **Anime-Planet** - Community-driven anime database (browser automation via crawl4ai)

## Caching Strategy

All enrichment sources use intelligent caching via Redis. See [cache_manager/README.md](../cache_manager/README.md) for cache manager implementation details.

### Two Caching Approaches

**API Helpers**: HTTP-level caching (transparent, RFC 9111 compliant)
- Cache key: URL + headers + body (for GraphQL)
- TTL: 24 hours
- Keys: `hishel_cache:*`
- Implementation: `HTTPCacheManager` from `src.cache_manager`

**Crawlers**: Result-level caching (function output)
- Cache key: Function name + arguments
- TTL: 24 hours
- Keys: `result_cache:*`
- Implementation: `@cached_result` decorator from `src.cache_manager`

**Benefits**:
- 100-7000x speedup on cache hits
- Multi-agent concurrent processing support
- Automatic cache invalidation on schema changes

### Service-Specific Cache Configuration

Cache expiration times are optimized per service (all set to **24 hours** for consistency):

| Service          | Type        | TTL      | Caching Strategy              |
| ---------------- | ----------- | -------- | ----------------------------- |
| **Jikan**        | API (async) | 24 hours | HTTP-level (aiohttp + Hishel) |
| **AniList**      | API (async) | 24 hours | HTTP-level (aiohttp + Hishel) |
| **AniDB**        | API (async) | 24 hours | HTTP-level (aiohttp + Hishel) |
| **Kitsu**        | API (async) | 24 hours | HTTP-level (aiohttp + Hishel) |
| **AnimSchedule** | API (async) | 24 hours | HTTP-level (aiohttp + Hishel) |
| **Anime-Planet** | Crawler     | 24 hours | Result-level (JSON caching)   |
| **AniSearch**    | Crawler     | 24 hours | Result-level (JSON caching)   |

### Helper Implementation Patterns

API helpers use three different patterns for cache integration:

#### Pattern A: Simple Function (animeschedule_fetcher)

For one-off fetches, create session and close immediately:

```python
from src.cache_manager.instance import http_cache_manager

async def fetch_data(search_term: str):
    session = http_cache_manager.get_aiohttp_session("animeschedule")
    try:
        async with session.get(url) as response:
            return await response.json()
    finally:
        await session.close()  # Always cleanup
```

**Use when**: Single function call, no state to maintain

#### Pattern B: Class with Ownership Tracking (jikan_helper)

For batch operations, optionally accept external session or create owned one:

```python
from src.cache_manager.instance import http_cache_manager

class JikanDetailedFetcher:
    def __init__(self, anime_id: str, session: Optional[Any] = None):
        self._owns_session = session is None
        self.session = session or http_cache_manager.get_aiohttp_session("jikan")

    async def fetch_episode_detail(self, episode_id: int):
        async with self.session.get(url) as response:
            return await response.json()

    async def close(self) -> None:
        if self._owns_session and self.session:
            await self.session.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        return False
```

**Use when**: Batch operations, reusable class, shared sessions

#### Pattern C: Per-Event-Loop Session (anilist_helper)

For concurrent multi-agent processing, create new session per event loop:

```python
from src.cache_manager.instance import http_cache_manager
import asyncio

class AniListEnrichmentHelper:
    def __init__(self) -> None:
        self.session: Optional[aiohttp.ClientSession] = None
        self._session_event_loop: Optional[asyncio.AbstractEventLoop] = None

    async def _make_request(self, query: str):
        # Recreate session if event loop changed
        current_loop = asyncio.get_running_loop()
        if self.session is None or self._session_event_loop != current_loop:
            if self.session is not None:
                await self.session.close()
            self.session = http_cache_manager.get_aiohttp_session("anilist")
            self._session_event_loop = current_loop

        async with self.session.post(url, json=payload) as response:
            return await response.json()

    async def close(self) -> None:
        if self.session:
            await self.session.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        return False
```

**Use when**: Multi-agent concurrent processing (multiple event loops)

**Why**: aiohttp sessions are bound to event loops. When the same helper instance is used across different event loops (e.g., concurrent enrichment agents), a new session must be created for each loop to avoid `RuntimeError: Event loop is closed`.

## Directory Structure

```
src/enrichment/
├── api_helpers/          # Async API integrations
│   ├── jikan_helper.py
│   ├── anilist_helper.py
│   ├── kitsu_helper.py
│   ├── anidb_helper.py
│   └── animeschedule_fetcher.py
├── crawlers/             # Browser-based crawlers
│   ├── anisearch_anime_crawler.py
│   └── anime_planet_character_crawler.py
└── programmatic/         # Pipeline orchestration
```

## Usage

### Running Enrichment

```bash
# Enrich by anime title
python run_enrichment.py --title "One Piece"

# Enrich by database index
python run_enrichment.py --index 0

# Skip specific services
python run_enrichment.py --title "Dandadan" --skip animeschedule anidb

# Only fetch from specific services
python run_enrichment.py --title "Dandadan" --only anime_planet anisearch
```

### Multi-Agent Concurrency

The pipeline supports concurrent multi-agent processing via Redis-based caching. Multiple enrichment processes can run simultaneously and share the same cache:

```bash
# Terminal 1
ENABLE_HTTP_CACHE=true python run_enrichment.py --title "One Piece" &

# Terminal 2
ENABLE_HTTP_CACHE=true python run_enrichment.py --title "Naruto" &

# Terminal 3
ENABLE_HTTP_CACHE=true python run_enrichment.py --title "Dandadan" &

# All 3 processes share the same Redis cache
# If One Piece and Naruto hit same API endpoints, cache is reused
```

**How it works:**
- All processes connect to the same Redis server (default: `localhost:6379/0`)
- Cache keys are generated from request parameters, not process ID
- Redis atomic operations prevent race conditions
- Each process uses event-loop-aware session management (Pattern C)

## Performance

**First Run** (cache miss):
- API helpers: 20-60 seconds
- Crawlers: 10-20 seconds per crawler
- Total: ~40-80 seconds

**Second Run** (cache hit):
- API helpers: 0.1-0.5 seconds (100-1000x faster)
- Crawlers: 0.001-0.004 seconds (1000-7000x faster)
- Total: ~0.5-2 seconds

## Cache Management

### Quick Start

```bash
# 1. Start Redis (required for caching)
docker compose up -d redis

# 2. Verify Redis is running
docker ps --filter name=redis
```

### Clear Cache

```bash
# Clear all cached data (HTTP + result caches)
docker exec anime-vector-redis redis-cli FLUSHALL

# Alternative: Clear specific database (DB 0)
docker exec anime-vector-redis redis-cli FLUSHDB
```

### View Cache Statistics

```bash
# Connect to Redis CLI
docker exec -it anime-vector-redis redis-cli

# Check total cache size
> DBSIZE

# View HTTP cache keys (from Hishel)
> KEYS hishel_cache:*

# View result cache keys (from crawlers)
> KEYS result_cache:*

# Check TTL on any key
> TTL result_cache:anisearch_anime:18878

# Exit Redis CLI
> exit
```

## Testing

```bash
# Unit tests (mocked, fast)
pytest tests/enrichment/

# Integration tests (requires Redis + ENABLE_LIVE_API_TESTS=1)
ENABLE_LIVE_API_TESTS=1 pytest tests/enrichment/api_helpers/test_jikan_helper_integration.py

# Multi-agent concurrency tests (requires ENABLE_LIVE_CONCURRENCY_TESTS=1)
ENABLE_LIVE_CONCURRENCY_TESTS=1 pytest tests/enrichment/programmatic/test_multi_agent_concurrency_integration.py
```

## Environment Variables

```bash
# Enable HTTP caching (recommended)
ENABLE_HTTP_CACHE=true

# Redis connection
REDIS_CACHE_URL=redis://localhost:6379/0

# Storage backend (redis or sqlite)
HTTP_CACHE_STORAGE=redis
```

See [cache_manager/README.md](../cache_manager/README.md) for complete configuration options.
