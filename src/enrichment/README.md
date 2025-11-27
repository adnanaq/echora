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

```text
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

## Dual-Usage Pattern: CLI Scripts vs Function Imports

**All helpers and crawlers support two usage modes:**

### Mode 1: Standalone CLI Scripts

**Every helper/crawler has a `main()` function** and can be executed directly from command line.

#### API Helpers CLI Examples

```bash
# Jikan Helper - Fetch ONLY episodes OR ONLY characters (specify data_type)
python -m src.enrichment.api_helpers.jikan_helper episodes 21 input.json output.json
python -m src.enrichment.api_helpers.jikan_helper characters 21 input.json output.json

# AniList Helper - Fetch by AniList ID
python -m src.enrichment.api_helpers.anilist_helper --anilist-id 21 --output anilist.json

# Kitsu Helper - Fetch by Kitsu anime ID
python -m src.enrichment.api_helpers.kitsu_helper 1234 kitsu_output.json

# AniDB Helper - Fetch by AniDB ID or search
python -m src.enrichment.api_helpers.anidb_helper --anidb-id 4563 --output anidb.json
python -m src.enrichment.api_helpers.anidb_helper --search-name "One Piece" --output anidb.json

# Anime-Planet Helper - Fetch by slug
python -m src.enrichment.api_helpers.animeplanet_helper one-piece output.json

# AnimSchedule Fetcher - Search by title
python -m src.enrichment.api_helpers.animeschedule_fetcher "One Piece" --output schedule.json
```

#### Crawlers CLI Examples

```bash
# AniSearch Anime Crawler - Accepts ID, path, or full URL
python -m src.enrichment.crawlers.anisearch_anime_crawler "18878,dan-da-dan" --output anisearch.json
python -m src.enrichment.crawlers.anisearch_anime_crawler "/18878,dan-da-dan" --output anisearch.json
python -m src.enrichment.crawlers.anisearch_anime_crawler "https://www.anisearch.com/anime/18878,dan-da-dan" --output anisearch.json

# AniSearch Character Crawler
python -m src.enrichment.crawlers.anisearch_character_crawler 18878 --output characters.json

# AniSearch Episode Crawler
python -m src.enrichment.crawlers.anisearch_episode_crawler 18878 --output episodes.json

# Anime-Planet Anime Crawler
python -m src.enrichment.crawlers.anime_planet_anime_crawler "dandadan" --output anime.json

# Anime-Planet Character Crawler
python -m src.enrichment.crawlers.anime_planet_character_crawler "dandadan" --output characters.json
```

**CLI Pattern Structure:**

```python
# Every helper/crawler follows this pattern:

async def main() -> int:
    """CLI entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument(...)
    args = parser.parse_args()

    try:
        # Fetch data
        data = await fetch_function(args.id, ...)

        # Write to file
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2)

        return 0  # Success
    except Exception:
        logger.exception("Error")
        return 1  # Failure

if __name__ == "__main__":  # pragma: no cover
    sys.exit(asyncio.run(main()))
```

### Mode 2: Programmatic Function/Class Import

**All helpers/crawlers export classes or functions** for import and use in Python code.

#### API Helpers Programmatic Usage

```python
# Jikan Helper - ONLY for detailed episodes OR characters (NOT anime metadata)
from src.enrichment.api_helpers.jikan_helper import JikanDetailedFetcher

# Fetch ONLY detailed episodes
async with JikanDetailedFetcher("21", "episodes") as fetcher:
    await fetcher.fetch_detailed_data("episodes_input.json", "episodes_output.json")

# Fetch ONLY detailed characters
async with JikanDetailedFetcher("21", "characters") as fetcher:
    await fetcher.fetch_detailed_data("characters_input.json", "characters_output.json")

# Or use individual methods
fetcher = JikanDetailedFetcher("21", "episodes")
episode_detail = await fetcher.fetch_episode_detail(episode_num=1)
character_detail = await fetcher.fetch_character_detail(character_data)

# NOTE: To fetch anime metadata + episodes + characters together, use ParallelAPIFetcher
# JikanDetailedFetcher is a low-level helper for batch fetching episode/character details only

# AniList Helper - Class with methods
from src.enrichment.api_helpers.anilist_helper import AniListEnrichmentHelper

async with AniListEnrichmentHelper() as helper:
    anime_data = await helper.fetch_all_data_by_anilist_id(21)
    characters = await helper.fetch_all_characters(21)

# Kitsu Helper - Class with methods
from src.enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper

helper = KitsuEnrichmentHelper()
anime_data = await helper.fetch_all_data(anime_id=1234)

# AniDB Helper - Class with methods
from src.enrichment.api_helpers.anidb_helper import AniDBEnrichmentHelper

async with AniDBEnrichmentHelper() as helper:
    anime_data = await helper.fetch_all_data(4563)
    search_results = await helper.search_anime_by_name("One Piece")

# Anime-Planet Helper - Class with methods
from src.enrichment.api_helpers.animeplanet_helper import AnimePlanetEnrichmentHelper

helper = AnimePlanetEnrichmentHelper()
anime_data = await helper.fetch_anime_data("one-piece")

# AnimSchedule Fetcher - Function
from src.enrichment.api_helpers.animeschedule_fetcher import fetch_animeschedule_data

data = await fetch_animeschedule_data("One Piece", output_path="schedule.json")
```

#### Crawlers Programmatic Usage

**Crawlers export top-level `fetch_*` functions** that handle normalization, caching, and I/O:

```python
# AniSearch Crawlers - Function imports
from src.enrichment.crawlers.anisearch_anime_crawler import fetch_anisearch_anime
from src.enrichment.crawlers.anisearch_character_crawler import fetch_anisearch_characters
from src.enrichment.crawlers.anisearch_episode_crawler import fetch_anisearch_episodes

# Fetch anime data (accepts ID, path, or full URL)
anime_data = await fetch_anisearch_anime("18878,dan-da-dan")
anime_data = await fetch_anisearch_anime("/18878,dan-da-dan", output_path="anime.json")

# Fetch characters
characters = await fetch_anisearch_characters(18878)

# Fetch episodes
episodes = await fetch_anisearch_episodes(18878, output_path="episodes.json")

# Anime-Planet Crawlers - Function imports
from src.enrichment.crawlers.anime_planet_anime_crawler import fetch_animeplanet_anime
from src.enrichment.crawlers.anime_planet_character_crawler import fetch_animeplanet_characters

anime_data = await fetch_animeplanet_anime("dandadan")
characters = await fetch_animeplanet_characters("dandadan", output_path="chars.json")
```

### Key Difference: Helpers vs Crawlers

- **API Helpers**: Export **classes** (e.g., `AniListEnrichmentHelper`, `JikanDetailedFetcher`)
- **Crawlers**: Export **functions** (e.g., `fetch_anisearch_anime`, `fetch_anime_planet_characters`)

### ParallelAPIFetcher Usage

**The main programmatic interface** that orchestrates all helpers:

```python
from src.enrichment.programmatic.api_fetcher import ParallelAPIFetcher

async with ParallelAPIFetcher(mal_id=21) as fetcher:
    # Fetch all data sources in parallel
    all_data = await fetcher.fetch_all_data()

    # Access individual results
    jikan_data = all_data.get("jikan")
    anilist_data = all_data.get("anilist")
    kitsu_data = all_data.get("kitsu")

    # Or fetch selectively
    selected_data = await fetcher.fetch_all_data(sources=["jikan", "anilist"])
```

### AniSearch Helper: Wrapper Around Crawlers

**The AniSearch helper is unique** - it wraps the three AniSearch crawlers:

```python
from src.enrichment.api_helpers.anisearch_helper import AniSearchEnrichmentHelper

helper = AniSearchEnrichmentHelper()

# Under the hood, calls fetch_anisearch_anime(), fetch_anisearch_characters(), etc.
all_data = await helper.fetch_all_data(anisearch_id=18878)

# Returns combined data from all three crawlers:
# {
#   "anime": {...},      # from anisearch_anime_crawler
#   "characters": [...], # from anisearch_character_crawler
#   "episodes": [...]    # from anisearch_episode_crawler
# }
```

### When to Use Each Mode

| Use Case | Mode | Example |
|----------|------|---------|
| Quick testing/debugging | **CLI** | `python -m src.enrichment.api_helpers.jikan_helper episodes 21 in.json out.json` |
| Pipeline integration | **Import** | `from src.enrichment.programmatic.api_fetcher import ParallelAPIFetcher` |
| Custom workflow | **Import** | `from src.enrichment.api_helpers.anilist_helper import AniListEnrichmentHelper` |
| Selective data source | **Import** | `from src.enrichment.crawlers.anisearch_anime_crawler import fetch_anisearch_anime` |

## Important Edge Cases & Caching Behaviors

### Cache Hit Detection and Rate Limiting Optimization

**All API helpers check `response.from_cache` attribute to optimize rate limiting:**

```python
# Jikan Helper (src/enrichment/api_helpers/jikan_helper.py)
from_cache = (
    isinstance(getattr(response, "from_cache", None), bool)
    and response.from_cache
)

if response.status == 200:
    data = await response.json()
    # Only rate limit for network requests, not cache hits
    await self._record_network_request(from_cache)
```

**Why this matters:**
- Cache hits skip rate limiting delays (e.g., 0.5s between Jikan requests)
- Enables 100-1000x speedup on repeated requests
- Prevents unnecessary waits when data is already cached

### Session Ownership and Resource Management

**Pattern A (Jikan)**: Tracks session ownership to prevent resource leaks

```python
class JikanDetailedFetcher:
    def __init__(self, anime_id: str, data_type: str, session: Optional[Any] = None):
        # Track whether we own the session
        self._owns_session = session is None
        self.session = session or _cache_manager.get_aiohttp_session("jikan")

    async def close(self) -> None:
        """Close the underlying HTTP session if we created it."""
        if self._owns_session and self.session:
            await self.session.close()
```

**Pattern B (Kitsu)**: Per-request session pattern (context manager)

```python
async def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    # Create session per request via context manager (automatic cleanup)
    async with _cache_manager.get_aiohttp_session("kitsu", timeout=...) as session:
        async with session.get(url, headers=headers, params=params) as response:
            return await response.json()
```

**Pattern C (AniList)**: Per-event-loop session management with automatic recreation

```python
async def _make_request(self, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    # Check if we need to create/recreate session for current event loop
    current_loop = asyncio.get_running_loop()
    if self.session is None or self._session_event_loop != current_loop:
        if self.session is not None:
            try:
                await self.session.close()  # Close old session
            except Exception:
                pass  # Ignore errors closing old session

        # Create new session for current event loop
        self.session = http_cache_manager.get_aiohttp_session("anilist", timeout=...,
            headers={"X-Hishel-Body-Key": "true"}  # Enable body-based caching for GraphQL
        )
        self._session_event_loop = current_loop
```

**Why this matters:**
- Prevents "Event loop is closed" errors in multi-event-loop scenarios
- Ensures proper cleanup even when event loops change
- Avoids resource leaks from unclosed sessions

### Async Context Manager Protocol

**All helpers implement `__aenter__` and `__aexit__` for context manager support:**

```python
# Recommended usage pattern
async with JikanDetailedFetcher("21", "episodes") as fetcher:
    await fetcher.fetch_detailed_data("input.json", "output.json")
# Session automatically closed on exit (if owned)

# AniList Helper
async with AniListEnrichmentHelper() as helper:
    anime_data = await helper.fetch_all_data_by_anilist_id(21)
# Session closed on exit

# Kitsu Helper (sessions created per request, context manager is no-op)
async with KitsuEnrichmentHelper() as helper:
    data = await helper.fetch_all_data(anime_id)
```

### GraphQL Body-Based Caching (AniList)

**AniList helper uses `X-Hishel-Body-Key: true` header to enable POST request caching:**

```python
self.session = http_cache_manager.get_aiohttp_session(
    "anilist",
    timeout=aiohttp.ClientTimeout(total=None),
    headers={"X-Hishel-Body-Key": "true"}  # Enable body-based caching for GraphQL
)
```

**Why this is critical:**
- GraphQL uses POST requests with different bodies for different queries
- Without body-based caching, all GraphQL requests would share the same cache key
- Hishel uses request body to generate unique cache keys per query

### Crawler Caching with Canonical Path Normalization

**Crawlers normalize anime identifiers to canonical paths for stable cache keys:**

```python
# anisearch_anime_crawler.py
def _normalize_anime_url(anime_identifier: str) -> str:
    """Convert anime identifier into canonical URL."""
    if not anime_identifier.startswith("http"):
        url = f"{BASE_ANIME_URL}{anime_identifier.lstrip('/')}"
    else:
        url = anime_identifier
    return url

@cached_result(ttl=TTL_ANISEARCH, key_prefix="anisearch_anime")
async def _fetch_anisearch_anime_data(canonical_path: str) -> Optional[Dict[str, Any]]:
    """Cache keyed on canonical path (e.g., '18878,dan-da-dan') not full URL."""
    url = f"{BASE_ANIME_URL}{canonical_path}"
    # ... fetch logic
```

**Why this matters:**
- Ensures cache reuse across different URL formats (relative, absolute, bare ID)
- All of these hit the same cache: `"18878,dan-da-dan"`, `"/18878,dan-da-dan"`, `"https://www.anisearch.com/anime/18878,dan-da-dan"`
- TTL configured centrally via `CacheConfig` in `src/cache_manager/config.py`

### Graceful Fallback on Redis Unavailability

**All caching gracefully falls back to uncached execution if Redis fails:**

```python
# HTTP caching (cache_manager/aiohttp_adapter.py)
try:
    storage = AsyncRedisStorage(redis_client, ttl=ttl)
    self._storage = storage
except Exception:
    # Fallback to uncached session if Redis unavailable
    logger.warning("Failed to set up Redis cache, using uncached session")
    self._storage = None

# Result caching (cache_manager/result_cache.py)
try:
    redis_client = get_result_cache_redis_client()
    cached_value = await redis_client.get(cache_key)
    # ... use cache
except redis.RedisError:
    # Fall back to uncached execution
    result = await func(*args, **kwargs)
```

**Why this matters:**
- Application never crashes due to Redis issues
- Degraded performance (no caching) instead of hard failure
- Useful for development without Redis or when Redis is temporarily down

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
