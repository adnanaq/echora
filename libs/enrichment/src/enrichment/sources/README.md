# Enrichment Sources

> Part of the `enrichment` library — see [the enrichment README](../README.md) for the full library overview.

Per-source packages for fetching and normalising anime data. Each package
owns its models, mappers, crawlers/API clients, and a `*Helper` entry point
that implements `BaseEnrichmentHelper`.

## Directory Layout

```
sources/
├── base/                      # Shared infrastructure
│   ├── base_helper.py         # BaseEnrichmentHelper ABC + normalize_enrichment_payload
│   ├── browser.py             # browser_session(): start, limit, block, close and reap browsers
│   ├── page_readiness.py      # wait_for_page(): wait for an element, then the page's HTML
│   ├── cloudflare_challenge.py  # Wait out a Cloudflare challenge, solve it only if it stays
│   ├── cloudflare_clearance.py  # Store and reuse a site's cf_clearance cookie (Redis)
│   ├── ad_annotations.py      # Undo what Google's in-text ads insert into page content
│   ├── companies.py           # Canonical company list from per-role provider lists
│   ├── external_links.py      # Canonical ExternalLink entries from provider links
│   ├── polite_http.py         # Plain-HTTP fetching, spaced out, stopping at the first block
│   ├── exceptions.py          # ServiceNotFoundError, ServiceBlockedError, …
│   ├── utils.py               # sanitize_output_path, etc.
│   └── framework/             # Template-method crawler framework
│       ├── crawler.py         # BaseCrawler[T_Source, T_Canonical]
│       ├── interfaces.py      # IRepository
│       └── repository.py      # FileRepository (JSONL append) + NullRepository
│
├── mal/                       # MyAnimeList (browser scraping via zendriver + lxml)
├── kitsu/                     # Kitsu (REST API via aiohttp)
├── anilist/                   # AniList (GraphQL API via aiohttp)
├── anisearch/                 # AniSearch (plain HTTP + lxml)
├── anime_planet/              # Anime-Planet (browser scraping via zendriver + lxml)
├── anidb/                     # AniDB (XML API via aiohttp + zendriver for characters)
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
crawler = AnimePlanetCharacterCrawler(NullRepository())
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

---

## Source Packages

### MAL (`sources/mal/`)

Browser scraping via zendriver (CDP) + lxml XPath.

| Module | Purpose |
|---|---|
| `mal_helper.py` | `MalHelper` — entry point; orchestrates anime, episodes, characters |
| `mal_anime_crawler.py` | `fetch_mal_anime(url)` — zendriver + lxml XPath |
| `mal_episode_crawler.py` | `fetch_mal_episodes(urls, output_path)` — zendriver + lxml XPath |
| `mal_episode_count_crawler.py` | `fetch_mal_episode_count(url)` — resolves "Unknown" counts |
| `mal_character_refs_crawler.py` | `fetch_mal_character_refs(url)` — list page → URL list |
| `mal_character_crawler.py` | `fetch_mal_character(url)`, `fetch_mal_characters(urls, output_path)` |
| `mal_mapper.py` | Raw scraped dicts → canonical `dict[str, Any]` |
| `mal_models.py` | Pydantic source models |
| `mal_base.py` | `normalize_mal_anime_url` |

**Expected `ids` key:** `mal_url` — full slug URL (e.g. `https://myanimelist.net/anime/21/One_Piece`)

**CLI — anime crawler (direct):**
```bash
uv run python -m enrichment.sources.mal.mal_anime_crawler \
    https://myanimelist.net/anime/21/One_Piece

uv run python -m enrichment.sources.mal.mal_anime_crawler \
    https://myanimelist.net/anime/21/One_Piece --output one_piece.json
```

**CLI — episode crawler (direct):**
```bash
uv run python -m enrichment.sources.mal.mal_episode_crawler \
    https://myanimelist.net/anime/21/One_Piece/episode/1

uv run python -m enrichment.sources.mal.mal_episode_crawler \
    https://myanimelist.net/anime/21/One_Piece/episode/1 --output ep1.json
```

**CLI — helper (all data types):**
```bash
uv run python -m enrichment.sources.mal.mal_helper anime https://myanimelist.net/anime/21/One_Piece mal_anime.json
uv run python -m enrichment.sources.mal.mal_helper episodes https://myanimelist.net/anime/21/One_Piece <count> mal_episodes.json
uv run python -m enrichment.sources.mal.mal_helper characters https://myanimelist.net/anime/21/One_Piece mal_characters.json
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

**CLI** (anime, episodes and characters in one call):
```bash
uv run python -m enrichment.sources.kitsu.kitsu_helper https://kitsu.app/anime/one-piece --output kitsu_anime.json
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

**CLI** (by AniList URL or MAL ID; `--output` is a directory):
```bash
uv run python -m enrichment.sources.anilist.anilist_helper --url https://anilist.co/anime/21 --output out/
uv run python -m enrichment.sources.anilist.anilist_helper --mal-id 21 --output out/
```

---

### AniSearch (`sources/anisearch/`)

Plain HTTP + lxml XPath, no browser: AniSearch is not behind Cloudflare and every
page arrives complete in its HTML. `anisearch_http.py` sends Chrome's own headers,
keeps requests at least 3 s apart across the process and stops all AniSearch
traffic at the first block, because AniSearch bans clients by User-Agent and then
refuses every connection from that IP.

| Module | Purpose |
|---|---|
| `anisearch_helper.py` | `AniSearchHelper` — anime, episodes, characters |
| `anisearch_anime_crawler.py` | `fetch_anisearch_anime(url, output_path)` |
| `anisearch_episode_crawler.py` | `fetch_anisearch_episodes(url, output_path)` |
| `anisearch_character_refs_crawler.py` | Character list page → URL list |
| `anisearch_character_crawler.py` | `fetch_anisearch_characters(refs, output_path)` — refs from the character list page |
| `anisearch_http.py` | `fetch_anisearch_page(url)` — the one way crawlers fetch AniSearch |
| `anisearch_mapper.py` | Raw XPath dicts → canonical dicts |
| `anisearch_anime_models.py` | Pydantic source models |

**Expected `ids` key:** `anisearch_url` — full URL (e.g. `https://www.anisearch.com/anime/2227,one-piece`)

Note: both `https://anisearch.com/` and `https://www.anisearch.com/` are accepted; normalized to `www` internally.

**CLI (via episode crawler):**
```bash
uv run python -m enrichment.sources.anisearch.anisearch_episode_crawler https://www.anisearch.com/anime/2227,one-piece --output anisearch_episodes.jsonl
```

---

### Anime-Planet (`sources/anime_planet/`)

Browser scraping via zendriver (CDP) + lxml XPath. Behind Cloudflare, which normally only runs an invisible check. A challenge shown instead of a page is waited out and solved only if it stays (`base/cloudflare_challenge.py`); a character batch stops at a challenge that does not clear.

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
uv run python -m enrichment.sources.anime_planet.anime_planet_helper anime https://www.anime-planet.com/anime/one-piece --output ap_anime.jsonl
uv run python -m enrichment.sources.anime_planet.anime_planet_helper characters https://www.anime-planet.com/characters/monkey-d-luffy --output ap_characters.jsonl
uv run python -m enrichment.sources.anime_planet.anime_planet_helper all https://www.anime-planet.com/anime/one-piece
```

---

### AniDB (`sources/anidb/`)

XML API via aiohttp with adaptive rate limiting: at least 2 s between requests, backing off up to 10 s after errors (`ENRICHMENT_ANIDB_MIN_REQUEST_INTERVAL` / `ENRICHMENT_ANIDB_MAX_REQUEST_INTERVAL`).

Character pages are behind Cloudflare. Its "Just a moment..." page clears by itself in about 2 s in a headed browser (not headless), so it is waited out and solved only if it stays (`base/cloudflare_challenge.py`). The `cf_clearance` cookie a browser earns is stored in Redis per User-Agent and given to the next AniDB browser, which then skips the challenge (`base/cloudflare_clearance.py`, kept up to `CLOUDFLARE_CLEARANCE_MAX_TTL`). AniDB's own AntiLeech page still gets its "Please Unban Me" flow.

| Module | Purpose |
|---|---|
| `anidb_helper.py` | `AniDBHelper` — anime, episodes, and characters via XML API |
| `anidb_character_crawler.py` | `fetch_anidb_characters(char_ids)` / `fetch_anidb_character(char_id)` — character web pages via zendriver + lxml XPath |
| `anidb_xml_parser.py` | AniDB XML → source models (stateless, no I/O) |
| `anidb_mapper.py` | XML + page responses → canonical dicts |
| `anidb_models.py` | Pydantic source models |

**Expected `ids` key:** `anidb_url` — full URL (e.g. `https://anidb.net/anime/69`)

**CLI:**
```bash
uv run python -m enrichment.sources.anidb.anidb_helper anime https://anidb.net/anime/69 anidb_anime.json
uv run python -m enrichment.sources.anidb.anidb_helper episodes https://anidb.net/anime/69 anidb_episodes.json
uv run python -m enrichment.sources.anidb.anidb_helper characters https://anidb.net/anime/69 anidb_characters.jsonl
uv run python -m enrichment.sources.anidb.anidb_helper all https://anidb.net/anime/69 output_dir/
```

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
built by `PlatformIDExtractor.extract_all_ids()` from the offline anime record's
source URLs.

```python
from enrichment.sources.mal.mal_helper import MalHelper

helper = MalHelper()
result = await helper.fetch_all(
    ids={"mal_url": "https://myanimelist.net/anime/21/One_Piece"},
    offline_data={...},
    temp_dir="/tmp/One_agent1",
)
# result = {"anime": {...}, "episodes": [...], "characters": [...], "extras": {...}}
```
