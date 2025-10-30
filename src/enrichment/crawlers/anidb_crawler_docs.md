### AniDB Crawlers

#### `anidb_character_crawler.py`
Extracts detailed character information from AniDB character pages. This crawler uses advanced anti-bot bypass techniques to handle AniDB's anti-leech protection.

**CLI Usage:**
```bash
# Fetch character by ID
python src/enrichment/crawlers/anidb_character_crawler.py 491

# Custom output file
python src/enrichment/crawlers/anidb_character_crawler.py 491 --output brook.json
```

**Arguments:**
- `character_id` (required): AniDB character ID (e.g., 491 for Brook).
- `--output` (optional): Custom output file path.

**Programmatic Usage:**
```python
from src.enrichment.crawlers.anidb_character_crawler import fetch_anidb_character

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
- `character_id` (required): AniDB character ID.
- `return_data` (optional): Return data dict (default: `True`).
- `output_path` (optional): File path to save JSON (default: `None`).

**Data Extracted:**
- Names (Kanji, nicknames, official names)
- Gender
- Abilities, looks, personality, role
- Supernatural abilities
