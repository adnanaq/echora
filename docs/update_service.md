---
title: Update Service
date: 2026-02-19
tags:
  - services
  - update-service
  - temporal
  - partial-updates
  - nats
status: active
related:
  - "[[temporal_infrastructure]]"
  - "[[event_driven_architecture]]"
  - "[[event_schema_specification]]"
  - "[[Architecture Index]]"
---

# Update Service

## Overview

The Update Service is a **Temporal worker** (Python) responsible for all ongoing partial updates to existing data after initial ingestion. It runs Temporal workflows that fetch delta data from external sources and publish change events to NATS.

**Tech Stack**:
- `Temporal Python SDK` — durable workflow execution, per-item dynamic scheduling, crash recovery
- `nats-py` — NATS JetStream publisher
- `libs/enrichment` — reuses existing source API helpers
- Source adapter pattern — pluggable per-source fetch strategies

> [!note] Why Temporal (not APScheduler)
> The Update Service manages potentially thousands of concurrent per-anime scheduled workflows, each firing at a calculated datetime. It also needs to survive process crashes and resume mid-execution without re-doing completed work. APScheduler provides neither guarantee. See [[temporal_infrastructure|Temporal Infrastructure]] for the full rationale.

---

## Update Scope

The Update Service handles **all partial updates** to existing entities. This is not limited to episodes — any field, any entity type, any source.

| Update type | Entity | Trigger | Fetch strategy |
|-------------|--------|---------|---------------|
| Episode air tracking | `episode` | Per-anime datetime schedule | Selective (by anime + episode number) |
| Score sync | `anime` | Daily (06:00 UTC) | Full stats fetch per source |
| Status transitions | `anime` | On episode tracking events + weekly check | Selective |
| Character data refresh | `character` | Weekly per-character schedule | Selective by character ID |
| Voice actor data | `character` | Weekly, post-character refresh | Selective |
| Staff data | `anime` | Weekly | Selective or full by anime |
| Anime metadata fields | `anime` | Weekly or source-triggered | Full or selective |
| Crawler-sourced data | `anime`, `character`, `episode` | Source-defined cadence | Selective extraction |

> [!note] Future Extensibility
> New update types are added as new Temporal workflow types with their own source adapters. No changes to other services required.

---

## Workflow Types

Each update type is a distinct Temporal workflow class. The Temporal scheduler creates and manages individual workflow instances — one per entity where applicable.

### 1. `EpisodeAirTrackingWorkflow`

**Schedule**: Per-anime, at calculated expected air datetime (rebuilt daily at midnight)

```
midnight rebuild activity:
  query PostgreSQL Service (GraphQL)
    → all ONGOING anime with broadcast in next 7 days
  for each anime:
    expected_air_datetime = broadcast.day + broadcast.time + timezone
    expected_episode_num  = MAX(episode_number) in DB + 1
    create or update Temporal Schedule for this anime

at air-time workflow execution:
  1. Fetch episode from source adapters (Jikan, AniList)
  2. Verify episode_number == expected
  3. Found     → publish anime.episode.aired to NATS
  4. Not found → Temporal timer (wait 2h), retry up to 3 times
  5. Check response for status change → publish anime.updated if changed
  6. After 3 retries: emit structured alert, mark window skipped
```

**Episode number strategy**:

| Anime type | Strategy |
|------------|----------|
| Finite (known episode count) | expected = next sequential; on final episode verify status via API |
| Long-running (e.g. One Piece) | expected = DB MAX + 1; status never auto-assumed finished |

### 2. `ScoreSyncWorkflow`

**Schedule**: Daily at 06:00 UTC (single workflow instance, iterates all in-scope anime)

**Scope**: All `ONGOING` anime + anime finished within the last 90 days

```
for each anime in scope:
  fetch statistics in parallel from source adapters: MAL, AniList, Kitsu
  compute ScoreCalculations (arithmetic_mean, median, geometric_mean)
  if score delta > SCORE_DELTA_THRESHOLD:
    publish anime.updated (score + statistics fields only) to NATS
```

### 3. `CharacterRefreshWorkflow`

**Schedule**: Weekly per character (Temporal Schedule per character)

```
fetch character data from source adapters
compare against current DB state (query PostgreSQL Service)
if changed fields:
  publish character.updated to NATS
```

### 4. `AnimeMetadataRefreshWorkflow`

**Schedule**: Weekly per anime, or triggered by n8n via Command API

```
fetch anime metadata from configured sources
compare against current DB state
if changed fields (status, dates, staff, etc.):
  determine if risky (title, type, episode_count regression)
  risky change → publish risk-flagged anime.updated → Event Adapter → n8n approval gate
  safe change  → publish anime.updated directly
```

### 5. `CrawlerWorkflow` *(future)*

**Schedule**: Source-defined cadence

```
run crawler for target source
extract selective fields
compare and publish changes
```

---

## Source Adapter Pattern

Each external source is implemented as a pluggable adapter. The workflow calls the adapter; the adapter owns fetch, parse, and normalize logic.

```
UpdateWorkflow
  └─ calls SourceAdapter.fetch(entity_id, fields?)
       └─ JikanAdapter     — Jikan REST API
       └─ AniListAdapter   — AniList GraphQL
       └─ MALAdapter       — MAL REST API
       └─ KitsuAdapter     — Kitsu REST API
       └─ CrawlerAdapter   — future: site-specific crawler
```

Adapters can be:
- **Full** — fetch all fields for an entity
- **Selective** — fetch only specific fields (lower API cost)
- **Differential** — fetch only what changed since a given timestamp (where API supports it)

---

## Events Published

| Subject | Workflow | Payload |
|---------|---------|---------|
| `anime.episode.aired` | `EpisodeAirTrackingWorkflow` | `EpisodeAiredEvent` |
| `anime.updated` | `ScoreSyncWorkflow`, `AnimeMetadataRefreshWorkflow` | `AnimeUpdatedEvent` |
| `character.updated` | `CharacterRefreshWorkflow` | `CharacterUpdatedEvent` |

See [[event_schema_specification|Event Schema Specification]] for full protobuf definitions.

> [!important] Update Service → PostgreSQL only
> All events are consumed exclusively by PostgreSQL Service. The Update Service never writes to PostgreSQL or Qdrant directly. Qdrant syncs via the outbox pattern after PostgreSQL commits.

---

## Temporal Worker Configuration

| Setting | Value |
|---------|-------|
| Task queue | `update-service` |
| Workflow types | `EpisodeAirTrackingWorkflow`, `ScoreSyncWorkflow`, `CharacterRefreshWorkflow`, `AnimeMetadataRefreshWorkflow` |
| Max concurrent workflows | Configurable (default: 100) |
| Max concurrent activities | Configurable (default: 50) |
| Temporal namespace | `echora` |

See [[temporal_infrastructure|Temporal Infrastructure]] for server setup.

---

## Downstream Consumers (SAGA Pattern)

```
Update Service
  → NATS (anime.episode.aired / anime.updated / character.updated)
    → PostgreSQL Service: Event Consumer
        DB transaction (atomic):
          ├── writes to entity tables
          └── inserts row into qdrant_outbox

    → PostgreSQL Service: Outbox Worker (background loop)
        polls qdrant_outbox WHERE processed_at IS NULL
          → publishes anime.synced / character.synced to NATS
          → marks row processed_at = now()

    → NATS (anime.synced / character.synced)
        → Qdrant Service
            ├── payload update  (score, status, episode_count)
            └── vector re-embed (synopsis, tags, demographics)
```

**Qdrant update strategy**:

| Changed field | Qdrant update | Reason |
|---------------|--------------|--------|
| `score`, `status`, `episode_count` | Payload only | Filter fields; no semantic change |
| `synopsis`, `tags`, `genres`, `demographics` | Payload + Vector | Semantic meaning changes embedding |
| `character` traits, description | Payload + Vector | Semantic content |

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `TEMPORAL_HOST` | `localhost:7233` | Temporal server address |
| `TEMPORAL_NAMESPACE` | `echora` | Temporal namespace |
| `NATS_URL` | `nats://localhost:4222` | NATS server URL |
| `POSTGRES_GRAPHQL_URL` | `http://localhost:8000/graphql` | PostgreSQL Service GraphQL endpoint |
| `EPISODE_WINDOW_DAYS` | `7` | Sliding window size in days |
| `SCORE_SYNC_HOUR` | `6` | UTC hour for daily score sync |
| `SCORE_DELTA_THRESHOLD` | `0.1` | Minimum score change to publish event |
| `SCORE_SYNC_LOOKBACK_DAYS` | `90` | Days past finish to keep syncing scores |
| `EPISODE_MAX_RETRIES` | `3` | Max retries before alerting and skipping |
| `EPISODE_RETRY_WAIT_HOURS` | `2` | Wait between episode retries |

---

## Related Documentation

- [[temporal_infrastructure|Temporal Infrastructure]] — Temporal server setup, rationale, and operational guide
- [[event_driven_architecture|Event-Driven Architecture]] — NATS consumer configuration and outbox pattern
- [[event_schema_specification|Event Schema Specification]] — Event protobuf definitions
---

**Status**: Planned | **Last Updated**: 2026-02-19
