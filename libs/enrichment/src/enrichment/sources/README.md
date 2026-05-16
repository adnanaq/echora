# Enrichment Sources

> Part of the `enrichment` library — see [`libs/enrichment/README.md`](../../../../README.md) for the full library overview.

Per-source packages for fetching and normalising anime data. Each package
owns its models, mappers, crawlers/API clients, and a `*Helper` entry point
that implements `BaseEnrichmentHelper`.

## Directory Layout

```
sources/
├── base/                      # Shared infrastructure
│   ├── base_helper.py         # BaseEnrichmentHelper ABC + normalize_enrichment_payload
│   ├── crawl4ai_docker.py     # crawl4ai Docker REST transport (crawl_single_url / crawl_batch_urls)
│   ├── crawler_config.py      # Shared browser/crawler configs, CrawlerRateLimiter
│   ├── exceptions.py          # ServiceNotFoundError, ServiceBlockedError, …
│   ├── utils.py               # sanitize_output_path, etc.
│   └── framework/             # Template-method crawler framework
│       ├── crawler.py         # BaseCrawler[T_Source, T_Canonical]
│       ├── interfaces.py      # ITransport, IRepository
│       ├── repository.py      # FileRepository (JSONL append) + NullRepository
│       └── transport.py       # DockerTransport, RateLimitedTransport
│
├── mal/                       # MyAnimeList (browser scraping via crawl4ai Docker)
├── kitsu/                     # Kitsu (REST API via aiohttp)
├── anilist/                   # AniList (GraphQL API via aiohttp)
├── anisearch/                 # AniSearch (browser scraping via crawl4ai Docker)
├── anime_planet/              # Anime-Planet (browser scraping via crawl4ai Docker)
├── anidb/                     # AniDB (XML API via aiohttp)
└── animeschedule/             # AnimSchedule (REST API via aiohttp)
```

## Base Framework

### `BaseEnrichmentHelper`

Abstract base class every source helper implements. The pipeline calls only
`fetch_all(ids, offline_data, temp_dir)`.

```python
class BaseEnrichmentHelper(ABC):
    @abstractmethod
    async def fetch_all(
        self,
        ids: dict[str, str],
        offline_data: dict[str, Any],
        temp_dir: str | None = None,
    ) -> dict[str, Any] | None: ...
```

### `BaseCrawler[T_Source, T_Canonical]`

Template-method crawler for single-URL detail pages. Subclasses implement:

- `normalize_identifier(identifier) -> str`
- `fetch_raw_data(url) -> dict | None`
- `build_source_model(processed_raw, url) -> T_Source`
- `map_to_canonical(source_model) -> T_Canonical`

```python
crawler = AnimePlanetCharacterCrawler(DockerTransport(), NullRepository())
result = await crawler.crawl(url)
```

### `FileRepository` / `NullRepository`

Persistence abstraction used by all helpers and crawlers.

```python
repo = FileRepository("out/characters.jsonl")
repo.save(canonical_dict)   # appends one JSONL line

repo = NullRepository()     # no-op (unit tests, callers that handle writes)
repo.save(canonical_dict)
```

### `crawl4ai_docker.py`

Low-level transport for the crawl4ai Docker REST server. All browser-based
sources use this instead of spawning `AsyncWebCrawler` in-process.

```python
from enrichment.sources.base.crawl4ai_docker import crawl_single_url, crawl_batch_urls

result = await crawl_single_url(url, browser_config, crawler_config)
results = await crawl_batch_urls(urls, browser_config, crawler_config)
```

Key behaviours:
- WAF soft-block recovery: pauses up to 600 s, probes every 60 s, retries blocked URLs
- Transient error retry: up to 3 attempts for DNS failures, connection refused, page timeouts
- Result alignment: returned list is always the same length as input `urls`, `None` for failures

---

## Source Packages

### MAL (`sources/mal/`)

Browser scraping via crawl4ai Docker REST API.

| Module | Purpose |
|---|---|
| `mal_helper.py` | `MalHelper` — entry point; orchestrates anime, episodes, characters |
| `mal_anime_crawler.py` | `fetch_mal_anime(url)` |
| `mal_episode_crawler.py` | `fetch_mal_episodes(urls, output_path)` |
| `mal_episode_count_crawler.py` | `fetch_mal_episode_count(url)` — resolves "Unknown" counts |
| `mal_character_refs_crawler.py` | `fetch_mal_character_refs(url)` — list page → URL list |
| `mal_character_crawler.py` | `fetch_mal_character(url)`, `fetch_mal_characters(urls, output_path)` |
| `mal_mapper.py` | Raw scraped dicts → canonical `dict[str, Any]` |
| `mal_models.py` | Pydantic source models |
| `mal_base.py` | `normalize_mal_anime_url` |

**Expected `ids` key:** `mal_url` — full slug URL (e.g. `https://myanimelist.net/anime/21/One_Piece`)

**CLI:**
```bash
uv run python -m enrichment.sources.mal.mal_helper anime https://myanimelist.net/anime/21/One_Piece
uv run python -m enrichment.sources.mal.mal_helper episodes https://myanimelist.net/anime/21/One_Piece <count>
uv run python -m enrichment.sources.mal.mal_helper characters https://myanimelist.net/anime/21/One_Piece
```

---

### Kitsu (`sources/kitsu/`)

REST API via aiohttp with Redis HTTP cache.

| Module | Purpose |
|---|---|
| `kitsu_helper.py` | `KitsuHelper` — anime, episodes, characters |
| `kitsu_mapper.py` | Kitsu API responses → canonical dicts |
| `kitsu_models.py` | Pydantic source models |

**Expected `ids` key:** `kitsu_url` — full URL or slug URL (e.g. `https://kitsu.app/anime/one-piece`)

**CLI:**
```bash
uv run python -m enrichment.sources.kitsu.kitsu_helper anime https://kitsu.app/anime/one-piece
uv run python -m enrichment.sources.kitsu.kitsu_helper episodes https://kitsu.app/anime/one-piece
uv run python -m enrichment.sources.kitsu.kitsu_helper characters https://kitsu.app/anime/one-piece
```

---

### AniList (`sources/anilist/`)

GraphQL API via aiohttp with Redis HTTP cache.

| Module | Purpose |
|---|---|
| `anilist_helper.py` | `AniListHelper` — anime + characters |
| `anilist_mapper.py` | AniList GraphQL responses → canonical dicts |
| `anilist_anime_models.py` | Pydantic anime source models |
| `anilist_character_models.py` | Pydantic character source models |

**Expected `ids` key:** `anilist_url` — full URL (e.g. `https://anilist.co/anime/21`)

**CLI:**
```bash
uv run python -m enrichment.sources.anilist.anilist_helper anime https://anilist.co/anime/21
uv run python -m enrichment.sources.anilist.anilist_helper characters https://anilist.co/anime/21
```

---

### AniSearch (`sources/anisearch/`)

Browser scraping via crawl4ai Docker REST API using XPath extraction.

| Module | Purpose |
|---|---|
| `anisearch_helper.py` | `AniSearchHelper` — anime, episodes, characters |
| `anisearch_anime_crawler.py` | `fetch_anisearch_anime(url, output_path)` |
| `anisearch_episode_crawler.py` | `fetch_anisearch_episodes(url, output_path)` |
| `anisearch_character_refs_crawler.py` | Character list page → URL list |
| `anisearch_character_crawler.py` | `fetch_anisearch_characters(urls, output_path)` |
| `anisearch_mapper.py` | Raw XPath dicts → canonical dicts |
| `anisearch_anime_models.py` | Pydantic source models |

**Expected `ids` key:** `anisearch_url` — full URL (e.g. `https://www.anisearch.com/anime/2227,one-piece`)

Note: both `https://anisearch.com/` and `https://www.anisearch.com/` are accepted; normalized to `www` internally.

**CLI (via episode crawler):**
```bash
uv run python -m enrichment.sources.anisearch.anisearch_episode_crawler https://www.anisearch.com/anime/2227,one-piece
```

---

### Anime-Planet (`sources/anime_planet/`)

Browser scraping via crawl4ai Docker REST API. Cloudflare-protected — uses
WAF recovery logic in `crawl4ai_docker.py`.

| Module | Purpose |
|---|---|
| `anime_planet_helper.py` | `AnimePlanetHelper` — anime + characters |
| `anime_planet_anime_crawler.py` | `fetch_animeplanet_anime(url, output_path)` |
| `anime_planet_character_refs_crawler.py` | Characters list page → URL list |
| `anime_planet_character_crawler.py` | `fetch_animeplanet_characters(urls, output_path)` |
| `animeplanet_mapper.py` | Raw CSS dicts → canonical dicts |
| `anime_planet_character_models.py` | Pydantic character source models |
| `anime_planet_models.py` | Pydantic anime source models |

**Expected `ids` key:** `anime_planet_url` — full URL (e.g. `https://www.anime-planet.com/anime/one-piece`)

**CLI:**
```bash
uv run python -m enrichment.sources.anime_planet.anime_planet_helper anime https://www.anime-planet.com/anime/one-piece
uv run python -m enrichment.sources.anime_planet.anime_planet_helper characters https://www.anime-planet.com/anime/one-piece
```

---

### AniDB (`sources/anidb/`)

XML API via aiohttp with strict rate limiting (2 req/s, 1 req burst).

| Module | Purpose |
|---|---|
| `anidb_helper.py` | `AniDBHelper` — anime data via XML API |

**Expected `ids` key:** `anidb_id` — numeric AniDB ID

---

### AnimSchedule (`sources/animeschedule/`)

REST API via aiohttp. Lookup is title-search-based (no persistent anime ID).

| Module | Purpose |
|---|---|
| `animeschedule_helper.py` | `AnimescheduleHelper` — anime + broadcast schedule |
| `animeschedule_mapper.py` | API responses → canonical dicts |
| `animeschedule_models.py` | Pydantic source models |

**Expected `ids` key:** resolved via title search against `offline_data`

---

## Pipeline Integration

The `ApiFetcher` in `enrichment/pipeline/api_fetcher.py` instantiates each
helper and calls `fetch_all(ids, offline_data, temp_dir)`. The `ids` dict is
built by `IdExtractor` from the offline anime record.

```python
from enrichment.sources.mal.mal_helper import MalHelper

helper = MalHelper()
result = await helper.fetch_all(
    ids={"mal_url": "https://myanimelist.net/anime/21/One_Piece"},
    offline_data={...},
    temp_dir="/tmp/One_agent1",
)
# result = {"anime": {...}, "episodes": [...], "characters": [...]}
```

---

## Legacy

`libs/enrichment/src/enrichment/crawlers/` contains `anidb_character_crawler.py`
— a standalone script that has not been migrated to the sources/ framework.
All other crawlers that were in that directory have been migrated here.
