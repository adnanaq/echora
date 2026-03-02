# Vector Database Operations

## Selective Vector Updates

Script: `scripts/update_vectors.py`

Use this when you need to update specific vectors without a full reindex.

Arguments:

- `--vectors VECTOR [VECTOR ...]` (required)
- `--index N`
- `--title "TITLE"`
- `--batch-size N` (default: `100`)
- `--file PATH`

Examples:

```bash
# Update one vector for all anime
uv run python scripts/update_vectors.py --vectors title_vector

# Update multiple vectors
uv run python scripts/update_vectors.py --vectors genre_vector character_vector

# Update a specific anime by index
uv run python scripts/update_vectors.py --vectors staff_vector --index 5

# Update by title search
uv run python scripts/update_vectors.py --vectors temporal_vector --title "Bungaku"

# Custom batch size
uv run python scripts/update_vectors.py --vectors image_vector --batch-size 50
```

## Full Database Reindex

Script: `scripts/reindex_anime_database.py`

This performs full reindexing (deletes and recreates the collection).

```bash
uv run python scripts/reindex_anime_database.py
```
