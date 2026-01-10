# Crawlers

Heavy-duty web crawlers using [crawl4ai](https://github.com/unclecode/crawl4ai) for comprehensive anime data extraction.

## Purpose

These crawlers use browser automation with JavaScript rendering and Cloudflare bypass capabilities to extract structured data from anime websites. These will eventually replace the lightweight scrapers in the enrichment pipeline for more robust data extraction.

**Status**: Currently standalone scripts. Will be refactored into importable library classes for production pipeline integration.

## Crawlers vs Scrapers

**Crawlers** (`libs/enrichment/src/enrichment/crawlers/`) - **Future production code**:
- Heavy-duty browser automation using crawl4ai
- JavaScript rendering and Cloudflare bypass
- CSS extraction strategies with structured schemas
- Will be refactored into library classes
- More reliable data extraction

**Scrapers** (`libs/enrichment/src/enrichment/scrapers/`) - **Legacy code**:
- Lightweight HTTP scraping using cloudscraper + BeautifulSoup
- Library classes meant to be imported
- Currently used in enrichment pipeline
- Will be deprecated once crawlers are integrated

## Available Crawlers

### Anime-Planet Crawlers

#### `anime_planet_anime_crawler.py`
Extracts comprehensive anime information from Anime-Planet anime pages.

**CLI Usage:**
```bash
# Flexible input formats (all equivalent)
./pants run libs/enrichment/src/enrichment/crawlers/anime_planet_anime_crawler.py -- dandadan
./pants run libs/enrichment/src/enrichment/crawlers/anime_planet_anime_crawler.py -- /anime/dandadan
./pants run libs/enrichment/src/enrichment/crawlers/anime_planet_anime_crawler.py -- https://www.anime-planet.com/anime/dandadan

# Custom output file
./pants run libs/enrichment/src/enrichment/crawlers/anime_planet_anime_crawler.py -- dandadan --output /path/to/output.json
```

**Arguments:**
- `identifier` (required): Anime slug, path, or full URL
- `--output` (optional): Custom output file path (default: `animeplanet_anime.json`)

**Programmatic Usage:**
```python
from enrichment.crawlers.anime_planet_anime_crawler import fetch_animeplanet_anime

# Return data without writing file
anime_data = await fetch_animeplanet_anime(
    slug="dandadan",
    return_data=True,
    output_path=None
)

# Write to specific file
anime_data = await fetch_animeplanet_anime(
    slug="dandadan",
    return_data=True,
    output_path="/path/to/output.json"
)
```

**Parameters:**
- `slug` (required): Anime slug, path, or full URL
- `return_data` (optional): Return data dict (default: `True`)
- `output_path` (optional): File path to save JSON (default: `None`)

#### `anime_planet_character_crawler.py`
Extracts character information from Anime-Planet anime character pages.

**CLI Usage:**
```bash
# Flexible input formats (all equivalent)
./pants run libs/enrichment/src/enrichment/crawlers/anime_planet_character_crawler.py -- dandadan
./pants run libs/enrichment/src/enrichment/crawlers/anime_planet_character_crawler.py -- /anime/dandadan
./pants run libs/enrichment/src/enrichment/crawlers/anime_planet_character_crawler.py -- /anime/dandadan/characters
./pants run libs/enrichment/src/enrichment/crawlers/anime_planet_character_crawler.py -- https://www.anime-planet.com/anime/dandadan

# Custom output file
./pants run libs/enrichment/src/enrichment/crawlers/anime_planet_character_crawler.py -- dandadan --output /path/to/output.json
```

**Arguments:**
- `identifier` (required): Anime slug, path, or full URL
- `--output` (optional): Custom output file path (default: `animeplanet_characters.json`)

**Programmatic Usage:**
```python
from enrichment.crawlers.anime_planet_character_crawler import fetch_animeplanet_characters

# Return data without writing file
character_data = await fetch_animeplanet_characters(
    slug="dandadan",
    return_data=True,
    output_path=None
)

# Write to specific file
character_data = await fetch_animeplanet_characters(
    slug="dandadan",
    return_data=True,
    output_path="/path/to/output.json"
)
```

**Parameters:**
- `slug` (required): Anime slug, path, or full URL
- `return_data` (optional): Return data dict (default: `True`)
- `output_path` (optional): File path to save JSON (default: `None`)

### AniSearch Crawlers

#### `anisearch_anime_crawler.py`
Extracts anime information from AniSearch anime pages.

**CLI Usage:**
```bash
./pants run libs/enrichment/src/enrichment/crawlers/anisearch_anime_crawler.py -- dandadan
```

**Arguments:**
- `anime_id`: Anime slug, path, or full URL
- `--output` (optional): Custom output file path (default: `anisearch_anime.json`)

**Data Extracted:**
- Title, description, images
- Studios, year, status
- Genres, tags
- Related media

#### `anisearch_character_crawler.py`
Extracts character information from AniSearch character pages.

**CLI Usage:**
```bash
./pants run libs/enrichment/src/enrichment/crawlers/anisearch_character_crawler.py -- dandadan
```

**Arguments:**
- `anime_id`: Anime slug, path, or full URL
- `--output` (optional): Custom output file path (default: `anisearch_characters.json`)

**Data Extracted:**
- Name, description, images
- Character attributes
- Related anime/manga

#### `anisearch_episode_crawler.py`
Extracts episode information from AniSearch.

**CLI Usage:**
```bash
./pants run libs/enrichment/src/enrichment/crawlers/anisearch_episode_crawler.py -- 18878
```

**Arguments:**
- `anime_id`: Anime slug, path, or full URL
- `--output` (optional): Custom output file path (default: `anisearch_episodes.json`)

**Data Extracted:**
- Episode numbers, titles
- Air dates, descriptions
- Screenshots, thumbnails

### AniDB Crawlers

#### `anidb_character_crawler.py`
Extracts detailed character information from AniDB character pages using advanced anti-bot bypass techniques to handle AniDB's anti-leech protection.

**CLI Usage:**
```bash
# Fetch character by ID
./pants run libs/enrichment/src/enrichment/crawlers/anidb_character_crawler.py -- 491

# Custom output file
./pants run libs/enrichment/src/enrichment/crawlers/anidb_character_crawler.py -- 491 --output brook.json
```

**Arguments:**
- `character_id` (required): AniDB character ID (e.g., 491 for Brook)
- `--output` (optional): Custom output file path (default: `anidb_character.json`)

**Programmatic Usage:**
```python
from enrichment.crawlers.anidb_character_crawler import fetch_anidb_character

# Return data without writing file
character_data = await fetch_anidb_character(
    character_id=491,
    return_data=True,
    output_path=None
)

# Write to specific file
character_data = await fetch_anidb_character(
    character_id=491,
    return_data=True,
    output_path="brook.json"
)
```

**Parameters:**
- `character_id` (required): AniDB character ID
- `return_data` (optional): Return data dict (default: `True`)
- `output_path` (optional): File path to save JSON (default: `None`)

**Data Extracted:**
- Character names (kanji, romaji, aliases)
- Gender, age, bloodtype
- Descriptions and characteristics
- Nicknames, abilities, physical appearance

## Dependencies

These crawlers require `crawl4ai`:

```bash
# Install crawl4ai
pip install crawl4ai

# Or with uv
uv pip install crawl4ai
```

## Integration with Enrichment Pipeline

**Current State**: Standalone scripts for manual testing and exploration.

**Future State**: These crawlers will replace the scrapers in `api_fetcher.py` for production use.

**Migration Plan**:
1. Refactor crawlers into importable library classes
2. Create base crawler class (similar to `base_scraper.py`)
3. Update helpers in `api_helpers/` to use crawlers instead of scrapers
4. Deprecate `libs/enrichment/src/enrichment/scrapers/` once migration is complete

## Notes

- Crawlers output to JSON files in the project root by default.
- **Overwrite Behavior**: Each run overwrites the previous output file at the specified (or default) path.
- Includes Cloudflare bypass for sites with protection.
- Uses structured CSS extraction strategies for reliable data extraction
- Slower than scrapers due to browser automation overhead


