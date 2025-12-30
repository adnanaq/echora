# Payload Filters Reference

This document provides a comprehensive reference of all filterable payload fields in the anime vector database.

## Overview

The anime vector database supports **38+ indexed payload fields** for filtering and querying. All fields support appropriate operators based on their type (range queries for numbers, exact/partial match for keywords, etc.) and can be combined with logical operators.

## Filter Categories

1. **Basic Metadata** - Core anime information (title, type, status, etc.)
2. **Temporal Filters** - Year, season, duration
3. **Content Filters** - Genres, tags, demographics, ratings
4. **Character Filters** - Character traits, appearance
5. **Statistics Filters** - Per-platform ratings, rankings, popularity
6. **Score Filters** - Aggregate cross-platform scores

---

## 1. Basic Metadata Filters

| Field             | Type    | Description                                | Example Query                                    |
| ----------------- | ------- | ------------------------------------------ | ------------------------------------------------ |
| `id`              | keyword | Unique anime identifier                    | `{"match": {"value": "21"}}`                     |
| `title`           | keyword | Exact title matching                       | `{"match": {"value": "Cowboy Bebop"}}`           |
| `title_text`      | text    | Full-text title search                     | Uses text search operators                       |
| `type`            | keyword | Anime type (TV, MOVIE, OVA, etc.)          | `{"match": {"value": "TV"}}`                     |
| `status`          | keyword | Airing status (FINISHED, RELEASING, etc.)  | `{"match": {"value": "FINISHED"}}`               |
| `rating`          | keyword | Content rating (G, PG-13, R, etc.)         | `{"match": {"value": "PG-13"}}`                  |
| `source_material` | keyword | Original source (MANGA, LIGHT_NOVEL, etc.) | `{"match": {"value": "MANGA"}}`                  |
| `nsfw`            | bool    | Adult content flag                         | `{"match": {"value": false}}`                    |
| `sources`         | keyword | Data source platforms                      | `{"match": {"any": ["myanimelist", "anilist"]}}` |

---

## 2. Temporal Filters

| Field      | Type    | Description                                   | Example Query                  |
| ---------- | ------- | --------------------------------------------- | ------------------------------ |
| `year`     | integer | Release year                                  | `{"gte": 2020, "lte": 2024}`   |
| `season`   | keyword | Release season (WINTER, SPRING, SUMMER, FALL) | `{"match": {"value": "FALL"}}` |
| `episodes` | integer | Number of episodes                            | `{"gte": 12, "lte": 24}`       |
| `duration` | integer | Episode duration in seconds                   | `{"gte": 1200}` (20+ minutes)  |

---

## 3. Content Filters

| Field              | Type    | Description                    | Example Query                                  |
| ------------------ | ------- | ------------------------------ | ---------------------------------------------- |
| `genres`           | keyword | Anime genres                   | `{"match": {"any": ["Action", "Adventure"]}}`  |
| `tags`             | keyword | Descriptive tags               | `{"match": {"any": ["Time Travel", "Mecha"]}}` |
| `demographics`     | text    | Target demographic description | Text search                                    |
| `content_warnings` | text    | Content warning descriptions   | Text search                                    |

---

## 4. Character Filters

| Field                         | Type    | Description                       | Example Query                                     |
| ----------------------------- | ------- | --------------------------------- | ------------------------------------------------- |
| `characters.hair_color`       | keyword | Character hair colors present     | `{"match": {"any": ["pink", "blue"]}}`            |
| `characters.eye_color`        | keyword | Character eye colors present      | `{"match": {"any": ["red", "gold"]}}`             |
| `characters.character_traits` | keyword | Character personality/role traits | `{"match": {"any": ["tsundere", "protagonist"]}}` |

---

## 5. Statistics Filters (Per-Platform)

**24 per-platform statistics fields** are indexed for filtering anime by ratings, popularity, and rankings across different platforms.

### Statistics Filter Test Results

- MAL: 6/6 fields tested
- AniList: 3/3 fields tested
- AniDB: 2/2 fields tested
- Anime-Planet: 3/3 fields tested
- Kitsu: 5/5 fields tested
- AnimeSchedule: 4/4 fields tested
- **Total: 24/24 statistics filters working**

### MyAnimeList (MAL) - 6 fields

| Field                            | Type    | Description               | Example Query    |
| -------------------------------- | ------- | ------------------------- | ---------------- |
| `statistics.mal.score`           | float   | User rating (0-10 scale)  | `{"gte": 7.0}`   |
| `statistics.mal.scored_by`       | integer | Number of users who rated | `{"gte": 10000}` |
| `statistics.mal.members`         | integer | Total members tracking    | `{"gte": 50000}` |
| `statistics.mal.favorites`       | integer | Users who favorited       | `{"gte": 100}`   |
| `statistics.mal.rank`            | integer | Overall rank position     | `{"lte": 5000}`  |
| `statistics.mal.popularity_rank` | integer | Popularity rank position  | `{"lte": 10000}` |

### AniList - 3 fields

| Field                                | Type    | Description              | Example Query   |
| ------------------------------------ | ------- | ------------------------ | --------------- |
| `statistics.anilist.score`           | float   | User rating (0-10 scale) | `{"gte": 7.0}`  |
| `statistics.anilist.favorites`       | integer | Users who favorited      | `{"gte": 10}`   |
| `statistics.anilist.popularity_rank` | integer | Popularity rank position | `{"lte": 5000}` |

### AniDB - 2 fields

| Field                        | Type    | Description               | Example Query   |
| ---------------------------- | ------- | ------------------------- | --------------- |
| `statistics.anidb.score`     | float   | User rating (0-10 scale)  | `{"gte": 7.0}`  |
| `statistics.anidb.scored_by` | integer | Number of users who rated | `{"gte": 1000}` |

### Anime-Planet - 3 fields

| Field                              | Type    | Description               | Example Query   |
| ---------------------------------- | ------- | ------------------------- | --------------- |
| `statistics.animeplanet.score`     | float   | User rating (0-10 scale)  | `{"gte": 7.0}`  |
| `statistics.animeplanet.scored_by` | integer | Number of users who rated | `{"gte": 1000}` |
| `statistics.animeplanet.rank`      | integer | Overall rank position     | `{"lte": 1000}` |

### Kitsu - 5 fields

| Field                              | Type    | Description              | Example Query    |
| ---------------------------------- | ------- | ------------------------ | ---------------- |
| `statistics.kitsu.score`           | float   | User rating (0-10 scale) | `{"gte": 7.0}`   |
| `statistics.kitsu.members`         | integer | Total members tracking   | `{"gte": 1000}`  |
| `statistics.kitsu.favorites`       | integer | Users who favorited      | `{"gte": 10}`    |
| `statistics.kitsu.rank`            | integer | Overall rank position    | `{"lte": 10000}` |
| `statistics.kitsu.popularity_rank` | integer | Popularity rank position | `{"lte": 10000}` |

### AnimeSchedule - 4 fields

| Field                                | Type    | Description               | Example Query    |
| ------------------------------------ | ------- | ------------------------- | ---------------- |
| `statistics.animeschedule.score`     | float   | User rating (0-10 scale)  | `{"gte": 6.0}`   |
| `statistics.animeschedule.scored_by` | integer | Number of users who rated | `{"gte": 5}`     |
| `statistics.animeschedule.members`   | integer | Total members tracking    | `{"gte": 10}`    |
| `statistics.animeschedule.rank`      | integer | Overall rank position     | `{"lte": 10000}` |

---

## 6. Score Filters (Aggregate Cross-Platform)

Aggregated scores computed from multiple platform ratings, all normalized to 0-10 scale.

| Field                             | Type  | Description                             | Example Query  |
| --------------------------------- | ----- | --------------------------------------- | -------------- |
| `score.arithmetic_mean`           | float | Simple average of all platform scores   | `{"gte": 7.0}` |
| `score.arithmetic_geometric_mean` | float | Geometric mean (reduces outlier impact) | `{"gte": 7.0}` |
| `score.median`                    | float | Median score across platforms           | `{"gte": 7.0}` |

---

## Usage Examples

### Common Filter Patterns

#### Statistics-Based Queries

```python
# Example: Find highly-rated anime on MAL
filter = {"statistics.mal.score": {"gte": 8.0}}

# Example: Popular anime with good aggregate scores
filter = {
    "statistics.mal.members": {"gte": 100000},
    "score.arithmetic_mean": {"gte": 7.5}
}

# Example: Hidden gems (high score, low popularity)
filter = {
    "score.arithmetic_mean": {"gte": 7.5},
    "statistics.mal.members": {"lte": 50000}
}

# Example: Critically acclaimed across platforms
filter = {
    "statistics.mal.score": {"gte": 8.0},
    "statistics.anilist.score": {"gte": 8.0},
    "statistics.kitsu.score": {"gte": 8.0}
}
```

#### Content-Based Queries

```python
# Example: Recent action anime
filter = {
    "genres": {"match": {"any": ["Action"]}},
    "year": {"gte": 2020}
}

# Example: Finished TV series from Fall 2024
filter = {
    "type": {"match": {"value": "TV"}},
    "season": {"match": {"value": "FALL"}},
    "year": {"match": {"value": 2024}},
    "status": {"match": {"value": "FINISHED"}}
}

# Example: Short-form content (under 10 minutes per episode)
filter = {
    "duration": {"lte": 600}
}

# Example: Safe for all audiences
filter = {
    "nsfw": {"match": {"value": False}},
    "rating": {"match": {"any": ["G - All Ages", "PG - Children"]}}
}
```

#### Character-Based Queries

```python
# Example: Anime with pink-haired characters
filter = {
    "characters.hair_color": {"match": {"any": ["pink"]}}
}

# Example: Anime with tsundere protagonists
filter = {
    "characters.character_traits": {"match": {"any": ["tsundere", "protagonist"]}}
}
```

#### Multi-Category Complex Queries

```python
# Example: Popular recent action anime with good scores
filter = {
    "genres": {"match": {"any": ["Action"]}},
    "year": {"gte": 2023},
    "statistics.mal.score": {"gte": 7.5},
    "statistics.mal.members": {"gte": 50000}
}

# Example: Manga adaptations from specific season
filter = {
    "source_material": {"match": {"value": "MANGA"}},
    "season": {"match": {"value": "WINTER"}},
    "year": {"match": {"value": 2024}}
}

# Example: Long-running completed series with high ratings
filter = {
    "type": {"match": {"value": "TV"}},
    "status": {"match": {"value": "FINISHED"}},
    "episodes": {"gte": 50},
    "score.arithmetic_mean": {"gte": 7.0}
}
```

### Multi-Platform Combination Filters

Filters can be combined to query across multiple platforms:

```python
# High score on both MAL and AniList
multi_filter_dict = {
    "statistics.mal.score": {"gte": 7.0},
    "statistics.anilist.score": {"gte": 7.0}
}
```

```python
# Popular on MAL with high aggregate score
combo_filter_dict = {
    "statistics.mal.members": {"gte": 50000},
    "score.arithmetic_mean": {"gte": 7.0}
}
```

### Range Queries

All numeric fields support range operations:

```python
# Score between 6.0 and 8.0
filter_dict = {"statistics.mal.score": {"gte": 6.0, "lte": 8.0}}

# Rank better than (lower than) 5000
filter_dict = {"statistics.mal.rank": {"lte": 5000}}

# At least 10K ratings
filter_dict = {"statistics.mal.scored_by": {"gte": 10000}}
```

## Filter Query Operators

| Operator | Meaning                  | Example         |
| -------- | ------------------------ | --------------- |
| `gte`    | Greater than or equal to | `{"gte": 7.0}`  |
| `lte`    | Less than or equal to    | `{"lte": 5000}` |
| `gt`     | Greater than             | `{"gt": 6.9}`   |
| `lt`     | Less than                | `{"lt": 5001}`  |

## Implementation Notes

### Settings Configuration (src/config/settings.py)

The filterable fields are defined in `qdrant_indexed_payload_fields`:

```python
qdrant_indexed_payload_fields: Dict[str, str] = Field(
    default={
        # ... other fields ...

        # MAL (MyAnimeList) statistics
        "statistics.mal.score": "float",
        "statistics.mal.scored_by": "integer",
        "statistics.mal.members": "integer",
        "statistics.mal.favorites": "integer",
        "statistics.mal.rank": "integer",
        "statistics.mal.popularity_rank": "integer",

        # AniList statistics
        "statistics.anilist.score": "float",
        "statistics.anilist.favorites": "integer",
        "statistics.anilist.popularity_rank": "integer",

        # AniDB statistics
        "statistics.anidb.score": "float",
        "statistics.anidb.scored_by": "integer",

        # Anime-Planet statistics
        "statistics.animeplanet.score": "float",
        "statistics.animeplanet.scored_by": "integer",
        "statistics.animeplanet.rank": "integer",

        # Kitsu statistics
        "statistics.kitsu.score": "float",
        "statistics.kitsu.members": "integer",
        "statistics.kitsu.favorites": "integer",
        "statistics.kitsu.rank": "integer",
        "statistics.kitsu.popularity_rank": "integer",

        # AnimeSchedule statistics
        "statistics.animeschedule.score": "float",
        "statistics.animeschedule.scored_by": "integer",
        "statistics.animeschedule.members": "integer",
        "statistics.animeschedule.rank": "integer",

        # Aggregate score field
        "score.arithmetic_mean": "float",
    }
)
```

### \_build_filter() Method (src/vector/client/qdrant_client.py:694)

The QdrantClient provides a `_build_filter()` method that converts Python dictionaries to Qdrant Filter objects:

```python
def _build_filter(self, filters: Dict[str, Any]) -> Optional[Filter]:
    """Build Qdrant filter from filter dictionary.

    Supports:
    - Range filters: {"key": {"gte": value, "lte": value}}
    - Match any: {"key": {"any": [value1, value2]}}
    - Exact match: {"key": value}
    """
```

## Testing

Comprehensive tests are available in `/tmp/test_statistics_filters_refactored.py` demonstrating:

- All 24 per-platform statistics filters
- Aggregate score filtering
- Multi-platform combination queries
- Range query variations

## Future Enhancements

Potential aggregate metrics for future implementation:

1. `statistics.total_members` - Sum of members across all platforms
2. `statistics.score_variance` - Variance in scores across platforms
3. `statistics.favorites_ratio` - Favorites per member ratio
4. `statistics.platform_count` - Number of platforms with data
5. `statistics.consensus_score` - Weighted cross-platform score

These would require additional computation and indexing.
