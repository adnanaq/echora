# Payload Filters Reference

This document is a reference for all filterable payload fields and the filter API contract.

There are two layers that use filters:

- **Backend → gRPC** — construct typed `FilterCondition` proto objects (see `vector_search.proto`)
- **Internal (qdrant_db)** — `SearchFilterCondition` from `qdrant_db.contracts` (used inside the gRPC service itself)

The JSON shapes below reflect what a backend would send to a REST gateway, which maps them to proto.

---

## Filter API

### Operators

| Operator | Meaning | Value type |
|---|---|---|
| `eq` | Exact match | scalar (str, int, float, bool) |
| `ne` | Not equal — excludes a single value | scalar |
| `in` | Match any of a set | non-empty list of scalars |
| `not_in` | Exclude all values in a set | non-empty list of scalars |
| `range` | Numeric range (gte/gt/lte/lt) | object with bound keys |

### Clauses (logical grouping)

Every condition has an optional `clause` field (default: `"must"`):

| Clause | Logic | Meaning |
|---|---|---|
| `must` (default) | AND | All must-conditions must match |
| `must_not` | NOT | Points matching any must_not-condition are excluded |
| `should` | OR | At least one should-condition must match |

### Combining operators and clauses

`clause` controls **which logical bucket** a condition goes into. The `operator` controls **how the value is matched** inside that bucket. They are independent — but some combinations double-negate and invert meaning:

| Intent | Correct | Wrong (inverts meaning) |
|---|---|---|
| Exclude MUSIC and CM | `not_in ["MUSIC","CM"]` + `must` | `not_in ["MUSIC","CM"]` + `must_not` |
| Exclude CANCELLED | `ne "CANCELLED"` + `must` OR `eq "CANCELLED"` + `must_not` | `ne "CANCELLED"` + `must_not` |

---

### Examples (JSON — backend → REST gateway)

```json
// Finished TV series from 2020 or newer
{ "filters": [
  { "field": "status", "operator": "eq",    "value": "FINISHED" },
  { "field": "type",   "operator": "eq",    "value": "TV" },
  { "field": "year",   "operator": "range", "value": { "gte": 2020 } }
]}

// Exclude short-form and cancelled content
{ "filters": [
  { "field": "type",   "operator": "not_in", "value": ["MUSIC", "CM", "PV"] },
  { "field": "status", "operator": "ne",     "value": "CANCELLED" }
]}

// TV or Movie (OR logic)
{ "filters": [
  { "field": "type", "operator": "eq", "value": "TV",    "clause": "should" },
  { "field": "type", "operator": "eq", "value": "MOVIE", "clause": "should" }
]}

// 2020+ anime, not cancelled, that are TV or OVA
{ "filters": [
  { "field": "year",   "operator": "range", "value": { "gte": 2020 } },
  { "field": "status", "operator": "eq",    "value": "CANCELLED", "clause": "must_not" },
  { "field": "type",   "operator": "eq",    "value": "TV",  "clause": "should" },
  { "field": "type",   "operator": "eq",    "value": "OVA", "clause": "should" }
]}
```

---

## Filterable Fields

### 1. Basic Metadata

| Field | Index type | Values / notes |
|---|---|---|
| `entity_type` | keyword | `anime`, `character`, `episode` |
| `anime_id` | keyword | Internal anime UUID |
| `anime_ids` | keyword | Related anime UUIDs (array) |
| `title` | keyword | Exact title match — partial/semantic title search handled by vector search itself |
| `type` | keyword | `TV`, `MOVIE`, `OVA`, `ONA`, `SPECIAL`, `MUSIC`, `CM`, `PV`, `TV_SHORT` |
| `status` | keyword | `FINISHED`, `RELEASING`, `NOT_YET_RELEASED`, `CANCELLED`, `HIATUS` |
| `rating` | keyword | `G - All Ages`, `PG - Children`, `PG-13`, `R - 17+`, `R+ - Mild Nudity`, `Rx - Hentai` |
| `source_material` | keyword | `MANGA`, `LIGHT_NOVEL`, `VISUAL_NOVEL`, `GAME`, `ORIGINAL`, `OTHER`, … |
| `nsfw` | bool | `true` / `false` |
| `sources` | keyword | Data source platforms (e.g. `myanimelist`, `anilist`) |

### 2. Temporal

| Field | Index type | Notes |
|---|---|---|
| `year` | integer | Release year — use `range` operator |
| `season` | keyword | `WINTER`, `SPRING`, `SUMMER`, `FALL` |
| `duration` | integer | Episode duration in seconds — use `range` operator |

### 3. Content

| Field | Index type | Notes |
|---|---|---|
| `genres` | keyword | Array — use `in` operator (e.g. `["Action", "Adventure"]`) |
| `tags` | keyword | Array — use `in` operator |
| `demographics` | keyword | Controlled values e.g. `Shounen`, `Seinen`, `Josei`, `Shoujo`, `Kids` — use `in` operator |
| `content_warnings` | keyword | Controlled values e.g. `Violence`, `Nudity` — use `in` operator |

### 4. Statistics (per platform)

All statistics fields support `range` and `eq` operators.

| Field | Index type |
|---|---|
| `statistics.mal.score` | float |
| `statistics.mal.scored_by` | integer |
| `statistics.anilist.score` | float |
| `statistics.anidb.score` | float |
| `statistics.anidb.scored_by` | integer |
| `statistics.animeplanet.score` | float |
| `statistics.animeplanet.scored_by` | integer |
| `statistics.kitsu.score` | float |
| `statistics.animeschedule.score` | float |
| `statistics.animeschedule.scored_by` | integer |

### 5. Aggregate Scores

| Field | Index type | Description |
|---|---|---|
| `score.arithmetic_mean` | float | Simple average across all platform scores |

---

## Common Patterns

---

### Highly rated on MAL

```json
{ "filters": [
  { "field": "statistics.mal.score", "operator": "range", "value": { "gte": 8.0 } }
]}
```

```python
from qdrant_db.contracts import SearchFilterCondition, SearchRange

filters = [
    SearchFilterCondition(field="statistics.mal.score", operator="range", value=SearchRange(gte=8.0)),
]
```

---

### Popular recent action anime, not cancelled

```json
{ "filters": [
  { "field": "genres",               "operator": "in",    "value": ["Action"] },
  { "field": "year",                 "operator": "range", "value": { "gte": 2023 } },
  { "field": "statistics.mal.score", "operator": "range", "value": { "gte": 7.5 } },
  { "field": "status",               "operator": "eq",    "value": "CANCELLED", "clause": "must_not" }
]}
```

```python
filters = [
    SearchFilterCondition(field="genres", operator="in", value=["Action"]),
    SearchFilterCondition(field="year", operator="range", value=SearchRange(gte=2023)),
    SearchFilterCondition(field="statistics.mal.score", operator="range", value=SearchRange(gte=7.5)),
    SearchFilterCondition(field="status", operator="eq", value="CANCELLED", clause="must_not"),
]
```

---

### Safe for all audiences

```json
{ "filters": [
  { "field": "nsfw",   "operator": "eq", "value": false },
  { "field": "rating", "operator": "in", "value": ["G - All Ages", "PG - Children"] }
]}
```

```python
filters = [
    SearchFilterCondition(field="nsfw", operator="eq", value=False),
    SearchFilterCondition(field="rating", operator="in", value=["G - All Ages", "PG - Children"]),
]
```

---

### Hidden gems (good score, not a blockbuster)

```json
{ "filters": [
  { "field": "score.arithmetic_mean",      "operator": "range", "value": { "gte": 7.5 } },
  { "field": "statistics.mal.scored_by",   "operator": "range", "value": { "lte": 50000 } }
]}
```

```python
filters = [
    SearchFilterCondition(field="score.arithmetic_mean", operator="range", value=SearchRange(gte=7.5)),
    SearchFilterCondition(field="statistics.mal.scored_by", operator="range", value=SearchRange(lte=50000)),
]
```

---

### Completed TV series with good scores

```json
{ "filters": [
  { "field": "type",                  "operator": "eq",    "value": "TV" },
  { "field": "status",                "operator": "eq",    "value": "FINISHED" },
  { "field": "score.arithmetic_mean", "operator": "range", "value": { "gte": 7.0 } }
]}
```

```python
filters = [
    SearchFilterCondition(field="type", operator="eq", value="TV"),
    SearchFilterCondition(field="status", operator="eq", value="FINISHED"),
    SearchFilterCondition(field="score.arithmetic_mean", operator="range", value=SearchRange(gte=7.0)),
]
```

---

### Exclude short-form and cancelled content

```json
{ "filters": [
  { "field": "type",   "operator": "not_in", "value": ["MUSIC", "CM", "PV"] },
  { "field": "status", "operator": "ne",     "value": "CANCELLED" }
]}
```

```python
filters = [
    SearchFilterCondition(field="type", operator="not_in", value=["MUSIC", "CM", "PV"]),
    SearchFilterCondition(field="status", operator="ne", value="CANCELLED"),
]
```

---

### TV or Movie (OR logic)

```json
{ "filters": [
  { "field": "type", "operator": "eq", "value": "TV",    "clause": "should" },
  { "field": "type", "operator": "eq", "value": "MOVIE", "clause": "should" }
]}
```

```python
filters = [
    SearchFilterCondition(field="type", operator="eq", value="TV", clause="should"),
    SearchFilterCondition(field="type", operator="eq", value="MOVIE", clause="should"),
]
```
