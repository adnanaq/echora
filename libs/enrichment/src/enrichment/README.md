# enrichment

Data enrichment library for anime records. Fetches structured data from 7 external
sources in parallel, caches results in Redis, and assembles them into a canonical
`AnimeRecord` payload for downstream vectorisation and search.

---

## Directory Layout

```
libs/enrichment/src/enrichment/
├── pipeline/           # Orchestration layer
│   ├── enrichment_pipeline.py   # EnrichmentPipeline — main entry point
│   ├── api_fetcher.py           # ApiFetcher — parallel fan-out across all sources
│   ├── id_extractor.py          # PlatformIDExtractor — URL → ids dict
│   ├── assembly.py              # Merge programmatic + AI outputs → AnimeRecord
│   └── config.py                # EnrichmentConfig (Pydantic BaseSettings)
│
├── sources/            # Per-source fetch packages (see sources/README.md)
│   ├── base/           # Shared transport, framework, configs
│   ├── mal/            # MyAnimeList — browser scraping via crawl4ai Docker
│   ├── kitsu/          # Kitsu — REST API
│   ├── anilist/        # AniList — GraphQL API
│   ├── anisearch/      # AniSearch — browser scraping via crawl4ai Docker
│   ├── anime_planet/   # Anime-Planet — browser scraping via crawl4ai Docker
│   ├── anidb/          # AniDB — XML API
│   └── animeschedule/  # AnimSchedule — REST API
│
├── utils/
│   ├── deduplication.py   # Semantic + string-based array deduplication
│   └── text_utils.py      # Text normalisation helpers
│
├── similarity/
│   └── ccip.py            # CCIP character image similarity (OpenCLIP fallback)
│
├── crawlers/           # Legacy — only anidb_character_crawler.py remains
│
└── ai_character_matcher.py  # AI-powered fuzzy character name matching (BGE-M3)
```

---

## Pipeline

### `EnrichmentPipeline`

Top-level orchestrator. Call `run(anime_entry, agent_id)` to enrich a single anime.

```python
from enrichment.pipeline.enrichment_pipeline import EnrichmentPipeline
from enrichment.pipeline.config import EnrichmentConfig

pipeline = EnrichmentPipeline(EnrichmentConfig())
result = await pipeline.run(anime_entry, agent_id="One_agent1")
```

Internally it:
1. Extracts platform IDs via `PlatformIDExtractor`
2. Fans out to all 7 sources in parallel via `ApiFetcher`
3. Writes per-source JSONL files to `temp/<agent_id>/`
4. Returns merged payload for AI assembly (Stage 4+)

**Performance**: ~5–10 s parallel vs ~30–60 s sequential.

### `ApiFetcher`

Runs all source helpers concurrently with `asyncio.gather`. Each helper is
independent — a failure in one source does not abort others.

```python
from enrichment.pipeline.api_fetcher import ApiFetcher

async with ApiFetcher() as fetcher:
    results = await fetcher.fetch_all(ids, offline_data, temp_dir)
```

### `PlatformIDExtractor`

Regex-based extraction of platform identifiers from offline-database source URLs.
Returns a dict used by every source helper's `fetch_all(ids, ...)`.

```python
from enrichment.pipeline.id_extractor import PlatformIDExtractor

ids = PlatformIDExtractor().extract(offline_data)
# {"mal_url": "https://myanimelist.net/anime/21", "kitsu_url": "...", ...}
```

### `EnrichmentConfig`

Pydantic `BaseSettings` — reads from environment or `.env`.

| Variable | Default | Purpose |
|---|---|---|
| `TEMP_DIR` | `temp` | Base directory for agent working dirs |
| `SKIP_SOURCES` | `[]` | Sources to skip (e.g. `["anidb"]`) |
| `ONLY_SOURCES` | `[]` | Run only these sources |
| `VERBOSE_LOGGING` | `false` | Extra per-source timing logs |

---

## Sources

All source packages expose a `*Helper(BaseEnrichmentHelper)` class with a single
public method:

```python
result = await helper.fetch_all(ids, offline_data, temp_dir)
# Returns {"anime": dict, "episodes": list, "characters": list} or None
```

See **[sources/README.md](src/enrichment/sources/README.md)** for full per-source
documentation including module tables, expected `ids` keys, and CLI usage.

### Quick-start CLI examples

```bash
# MAL
uv run python -m enrichment.sources.mal.mal_helper anime https://myanimelist.net/anime/21/One_Piece
uv run python -m enrichment.sources.mal.mal_helper episodes https://myanimelist.net/anime/21/One_Piece 1156
uv run python -m enrichment.sources.mal.mal_helper characters https://myanimelist.net/anime/21/One_Piece

# Kitsu
uv run python -m enrichment.sources.kitsu.kitsu_helper anime https://kitsu.app/anime/one-piece

# AniList
uv run python -m enrichment.sources.anilist.anilist_helper anime https://anilist.co/anime/21

# AniSearch
uv run python -m enrichment.sources.anisearch.anisearch_episode_crawler https://www.anisearch.com/anime/2227,one-piece

# Anime-Planet
uv run python -m enrichment.sources.anime_planet.anime_planet_helper anime https://www.anime-planet.com/anime/one-piece

# AnimSchedule
uv run python -m enrichment.sources.animeschedule.animeschedule_helper "One Piece"
```

---

## Transport Layer

### `crawl4ai_docker.py` (`sources/base/`)

All browser-based sources (MAL, AniSearch, Anime-Planet) use the shared crawl4ai
Docker REST transport instead of spawning `AsyncWebCrawler` in-process.

```python
from enrichment.sources.base.crawl4ai_docker import crawl_single_url, crawl_batch_urls

result  = await crawl_single_url(url, browser_config, crawler_config)
results = await crawl_batch_urls(urls, browser_config, crawler_config)
# Returns None (single) or list aligned to input (batch) on failure
```

Key behaviours:
- **WAF recovery**: on 403 (Cloudflare) or 405 (AWS WAF), pauses 60 s between probes,
  retries for up to 10 minutes before giving up
- **Transient retry**: up to 3 attempts for DNS failures, connection refused, page timeouts
- **Result alignment**: batch result list is always the same length as input; `None` for failures

Requires the crawl4ai Docker container to be running:

```bash
docker compose -f docker/docker-compose.dev.yml up -d crawl4ai
```

### `crawler_config.py` (`sources/base/`)

Shared browser/crawler config factories used by all Docker-based crawlers:

```python
from enrichment.sources.base.crawler_config import (
    get_docker_browser_config,
    get_docker_crawler_config,
    get_ap_rate_limiter,
)
```

---

## Caching

All API-based sources (Kitsu, AniList, AniDB, AnimSchedule) use the `http_cache`
library (Hishel + Redis, RFC 9111). Browser-based crawlers use `@cached_result`
from `http_cache.result_cache` with per-source TTLs.

| Source | TTL |
|---|---|
| MAL | 24 h |
| Kitsu | 24 h |
| AniList | 24 h |
| AniSearch | 24 h |
| Anime-Planet | 24 h |
| AniDB | 7 days |
| AnimSchedule | 24 h |

---

## Utilities

### `deduplication.py`

Deduplicates array fields using string equality or semantic cosine similarity
(via injected `TextEmbeddingModel`). Used in assembly to merge character/episode
lists from multiple sources.

### `similarity/ccip.py`

Character image similarity using the CCIP algorithm (dghs-imgutils) with OpenCLIP
as a fallback. Used in Stage 5 (AI character matching) to validate candidate pairs
before writing to the final record.

### `ai_character_matcher.py`

BGE-M3-based fuzzy character name matching. Handles multilingual names (hiragana,
katakana, romaji, romaji variants). Achieves ~99% precision / ~92% recall vs
primitive string matching.
