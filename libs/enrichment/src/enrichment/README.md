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
│   ├── id_extractor.py          # PlatformIDExtractor — source URLs → ids dict
│   ├── config.py                # EnrichmentConfig (Pydantic BaseSettings)
│   ├── metadata_merger.py       # Merges the seven provider records into one Anime
│   ├── metadata_rules.py        # Per-field rules for that merge
│   ├── link_rules.py            # Merge rules for links, images and media
│   ├── relationship_merger.py   # Merges per-source relationship data
│   ├── same_work.py             # When two provider URLs denote the same work
│   ├── same_company.py          # When two company names are the same company
│   └── same_word.py             # When two differently written words are the same
│
├── sources/            # Per-source fetch packages (see sources/README.md)
│   ├── base/           # Shared browser, page waiting, Cloudflare, framework
│   ├── mal/            # MyAnimeList — browser scraping via zendriver
│   ├── kitsu/          # Kitsu — REST API
│   ├── anilist/        # AniList — GraphQL API
│   ├── anisearch/      # AniSearch — browser scraping via zendriver
│   ├── anime_planet/   # Anime-Planet — browser scraping via zendriver
│   ├── anidb/          # AniDB — XML API (characters via zendriver)
│   └── animeschedule/  # AnimSchedule — REST API
│
├── utils/
│   ├── deduplication.py   # Semantic + string-based array deduplication
│   └── text_utils.py      # Text normalisation helpers
│
├── similarity/
│   └── ccip.py            # CCIP character image similarity (OpenCLIP fallback)
│
└── ai_character_matcher.py  # AI-powered fuzzy character name matching (BGE-M3)
```

---

## Pipeline

### `EnrichmentPipeline`

Top-level orchestrator. Enter it with `async with`, then call `enrich_anime()` for
one anime (or `enrich_batch()` for several).

```python
from enrichment.pipeline.enrichment_pipeline import EnrichmentPipeline
from enrichment.pipeline.config import EnrichmentConfig

async with EnrichmentPipeline(EnrichmentConfig()) as pipeline:
    result = await pipeline.enrich_anime(offline_data, agent_dir="One_agent1")
```

Entering the pipeline sets the process-wide browser limit and removes browsers
and profiles left behind by a killed run (`reap_orphans()`). `enrich_anime()`
then:
1. Extracts platform IDs via `PlatformIDExtractor`
2. Fans out to all 7 sources in parallel via `ApiFetcher`
3. Writes per-source JSONL files to `temp/<agent_dir>/`
4. Returns `offline_data`, `extracted_ids`, `api_data` (per source) and
   `enrichment_metadata` (total time, timing breakdown, temp directory)

`skip_services` / `only_services` limit the sources, and `fetch_characters=False` /
`fetch_episodes=False` skip those parts.

### `ApiFetcher`

Runs all source helpers concurrently with `asyncio.gather`. Each helper is
independent — a failure in one source does not abort others.

```python
from enrichment.pipeline.api_fetcher import ApiFetcher

async with ApiFetcher() as fetcher:
    results = await fetcher.fetch_all_data(ids, offline_data, temp_dir)
```

### `PlatformIDExtractor`

Regex-based extraction of platform identifiers from offline-database source URLs.
Returns the dict every source helper's `fetch_all(ids, ...)` reads.

```python
from enrichment.pipeline.id_extractor import PlatformIDExtractor

ids = PlatformIDExtractor().extract_all_ids(offline_data)
# {"mal_url": "https://myanimelist.net/anime/21/One_Piece",
#  "anidb_url": "https://anidb.net/anime/69", "kitsu_url": "https://kitsu.app/anime/12", ...}
```

### `EnrichmentConfig`

Pydantic `BaseSettings` — reads from environment or `.env`, with the
`ENRICHMENT_` prefix. Common settings:

| Variable | Default | Purpose |
|---|---|---|
| `ENRICHMENT_TEMP_DIR` | `temp` | Base directory for agent working dirs |
| `ENRICHMENT_MAX_CONCURRENT_BROWSERS` | `4` | Most Chrome browsers open at once in the process |
| `ENRICHMENT_SKIP_FAILED_APIS` | `true` | Return partial results when a source fails |
| `ENRICHMENT_VERBOSE_LOGGING` | `false` | Log the configuration and extra timings |
| `ENRICHMENT_ANIDB_MIN_REQUEST_INTERVAL` | `2.0` | Seconds between AniDB API requests (backs off up to `ENRICHMENT_ANIDB_MAX_REQUEST_INTERVAL`, 10.0) |

---

## Sources

All source packages expose a `*Helper(BaseEnrichmentHelper)` class with one
method the pipeline calls:

```python
result = await helper.fetch_all(ids, offline_data, temp_dir)
# Returns {"anime": dict, "episodes": list, "characters": list, "extras": dict} or None
```

See **[sources/README.md](sources/README.md)** for full per-source
documentation including module tables, expected `ids` keys, and CLI usage.

### Quick-start CLI examples

```bash
# MAL
uv run python -m enrichment.sources.mal.mal_helper anime https://myanimelist.net/anime/21/One_Piece mal_anime.json
uv run python -m enrichment.sources.mal.mal_helper episodes https://myanimelist.net/anime/21/One_Piece 1156 mal_episodes.json
uv run python -m enrichment.sources.mal.mal_helper characters https://myanimelist.net/anime/21/One_Piece mal_characters.json

# Kitsu
uv run python -m enrichment.sources.kitsu.kitsu_helper https://kitsu.app/anime/one-piece --output kitsu_anime.json

# AniList
uv run python -m enrichment.sources.anilist.anilist_helper --url https://anilist.co/anime/21 --output out/

# AniSearch
uv run python -m enrichment.sources.anisearch.anisearch_episode_crawler https://www.anisearch.com/anime/2227,one-piece

# Anime-Planet
uv run python -m enrichment.sources.anime_planet.anime_planet_helper anime https://www.anime-planet.com/anime/one-piece --output ap_anime.jsonl

# AniDB
uv run python -m enrichment.sources.anidb.anidb_helper anime https://anidb.net/anime/69 onepiece_anidb.json
uv run python -m enrichment.sources.anidb.anidb_helper episodes https://anidb.net/anime/69 onepiece_anidb_episodes.json
uv run python -m enrichment.sources.anidb.anidb_helper characters https://anidb.net/anime/69 onepiece_anidb_characters.jsonl
uv run python -m enrichment.sources.anidb.anidb_helper all https://anidb.net/anime/69 output_dir/

# AnimSchedule
uv run python -m enrichment.sources.animeschedule.animeschedule_helper "One Piece"
```

---

## Browser Automation

All browser-based sources (MAL, AniSearch, Anime-Planet, AniDB) use `zendriver`
(CDP-based Chrome automation). No external Docker sidecar is required. Crawlers
never call `zd.start()` or `browser.stop()` themselves; they open a browser through
`sources/base/browser.py` and wait for pages through
`sources/base/page_readiness.py`:

```python
from enrichment.sources.base.browser import browser_session
from enrichment.sources.base.page_readiness import wait_for_page

async with browser_session(headless=True) as session:
    page = await session.browser.get(url)
    await wait_for_page(page, "css-selector", url)
    html = await page.get_content()
```

`browser_session()` closes the browser in tens of milliseconds, holds a slot from
the process-wide limit `EnrichmentConfig.max_concurrent_browsers`
(`ENRICHMENT_MAX_CONCURRENT_BROWSERS`, default 4) while it is open, and marks the
browser's profile so a later run can clean up after a killed one. AniSearch and
AniDB pass `headless=False`; a batch that hits a browser crash calls
`session.restart()`, which keeps the session's settings.

`wait_for_page()` waits up to 30 s for an element only the expected page has,
then until the page's HTML has fully arrived, and reads it then. Everything the
crawlers extract comes with the HTML, so there are no fixed sleeps or scrolling;
a page whose HTML never finishes is read after 10 s with a warning.

Browsers never download what the crawlers do not read:

* Every browser fails its image, font, media and stylesheet requests. The
  crawlers parse only the HTML, and image URLs are read from attributes, so the
  extracted data is the same on every page type; scripts, XHR and fetch requests
  always load, because MAL renders parts of its pages with them and Cloudflare's
  checks are scripts.
* MAL sessions pass `allowed_site=MAL_DOMAIN`: the browser may connect only to
  `myanimelist.net` and its subdomains, so the ad and tracking servers MAL's pages
  pull in are never reached. Never set it for Anime-Planet or AniDB: their
  Cloudflare check is served from `challenges.cloudflare.com`, another website.
* Chrome's background downloads are switched off on every browser.

A crawler that needs everything loaded passes
`browser_session(..., block_unused_resources=False)`; that turns the resource and
host blocking off for its own sessions only.

Key behaviours:
- **Cloudflare**: AniDB character pages and Anime-Planet are behind Cloudflare;
  MAL and AniSearch are not. A challenge page is waited out and solved only if it
  stays (`sources/base/cloudflare_challenge.py`). AniDB's challenge clears by
  itself only in a headed browser, and the clearance an AniDB browser earns is
  stored in Redis and reused (`browser_session(clearance_site="anidb.net")`,
  `sources/base/cloudflare_clearance.py`); see `sources/README.md`.
- **lxml XPath extraction**: raw HTML parsed with lxml for fast, typed field extraction
- **Rate limiting**: per-source delays between requests to avoid IP bans

---

## Caching

All API-based sources (Kitsu, AniList, AniDB, AnimSchedule) use the `http_cache`
library (Hishel + Redis, RFC 9111). Browser-based crawlers use `@cached_result`
from `http_cache.result_cache` with per-source TTLs. A cached result's key
includes a hash of its function's source, so changing a crawler's fetch code makes
its old entries unused.

| Source | TTL |
|---|---|
| MAL | 7 days |
| Kitsu | 7 days |
| AniList | 7 days |
| AniSearch | 7 days |
| Anime-Planet | 7 days |
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
