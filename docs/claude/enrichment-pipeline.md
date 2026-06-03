# Enrichment Pipeline

## Main Entry Point

Script: `run_enrichment.py`

Default data source:

- `data/qdrant_storage/anime-offline-database.json`

## Script Arguments

- `--index N`: Process anime at index N (0-based)
- `--title "Title"`: Search anime by title (case-insensitive, partial match)
- `--file PATH`: Use custom database file
- `--agent "name"`: Set agent directory name (optional)
- `--skip service1 service2`: Skip specific services
- `--only service1 service2`: Fetch only specific services

Available services:

- `jikan`, `anilist`, `kitsu`, `anidb`, `anime_planet`, `anisearch`, `animeschedule`

Example usage:

```bash
# Process first anime in database
python run_enrichment.py --index 0

# Process One Piece
python run_enrichment.py --title "One Piece"

# Use custom database
python run_enrichment.py --file custom.json --index 5

# Specify agent directory
python run_enrichment.py --title "Dandadan" --agent "Dandadan_test"

# Skip specific services
python run_enrichment.py --title "Dandadan" --skip animeschedule anidb

# Only fetch specific services
python run_enrichment.py --title "Dandadan" --only anime_planet anisearch
```

Notes:

- `--skip` and `--only` are mutually exclusive
- If `--agent` is not set, agent IDs are auto-assigned via gap-filling logic

## Stage Script Directory Pattern

All stage scripts use:

```bash
python process_stage<N>.py <agent_id> [--temp-dir <base>]
```

- `agent_id`: directory name (for example `One_agent1`)
- `--temp-dir`: base directory (default: `temp`)

Directory layout:

- `temp/<agent_id>/`

## Stage 1: Metadata Extraction

Script: `process_stage1_metadata.py`

Arguments: `agent_id`, `--temp-dir` (default `temp`), `--current-anime` (legacy)

```bash
python process_stage1_metadata.py One_agent1
python process_stage1_metadata.py One_agent1 --temp-dir custom_temp
python process_stage1_metadata.py --current-anime temp/One_agent1/current_anime.json
```

## Stage 2: Episode Processing

Script: `process_stage2_episodes.py`

Arguments: `agent_id`, `--temp-dir` (default `temp`)

Required file: `episodes_detailed.json` in the agent directory

```bash
python process_stage2_episodes.py One_agent1
python process_stage2_episodes.py One_agent1 --temp-dir custom_temp
```

## Stage 3: Relationship Processing

Script: `process_stage3_relationships.py`

Arguments: `agent_id`, `--temp-dir` (default `temp`), `--current-anime` (legacy)

```bash
python process_stage3_relationships.py One_agent1
python process_stage3_relationships.py One_agent1 --temp-dir custom_temp
python process_stage3_relationships.py --current-anime temp/One_agent1/current_anime.json
```

## Stage 4: Statistics Extraction

Script: `scripts/process_stage4_statistics.py`

Arguments: `agent_id`, `--temp-dir` (default `temp`)

```bash
python scripts/process_stage4_statistics.py Dandadan_agent1
python scripts/process_stage4_statistics.py Dandadan_agent1 --temp-dir custom_temp
```

## Stage 5: AI Character Matching

Script: `process_stage5_characters.py`

Arguments: `agent_id`, `--temp-dir` (default `temp`), `--restart` (optional)

```bash
python process_stage5_characters.py One_agent1
python process_stage5_characters.py One_agent1 --restart
python process_stage5_characters.py One_agent1 --temp-dir custom_temp
```
