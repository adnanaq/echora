---
title: Update Worker
date: 2026-02-17
tags:
  - services
  - update-worker
  - episodes
  - scores
  - nats
status: active
related:
  - "[[event_driven_architecture]]"
  - "[[event_schema_specification]]"
  - "[[Architecture Index]]"
---

# Update Worker

## Overview

The Update Worker (`apps/update_worker/`) is the Python service responsible for all **ongoing partial updates** to anime data after initial ingestion. It runs on schedules, fetches delta data from external APIs, and publishes events to NATS for downstream processing.

> [!important] Scope Boundary
> The ingestion pipeline handles **one-time full enrichment only**. The Update Worker handles **all recurring updates** — new episodes, score changes, status transitions.

**Tech Stack**:
- `APScheduler` — cron-style job scheduling
- `nats-py` — NATS JetStream publisher
- `libs/enrichment` — reuses existing API helpers (Jikan, AniList, MAL, Kitsu)
- GraphQL client — queries PostgreSQL Service for broadcast schedules

## Jobs

### 1. Sliding Window — Episode Air Tracking

**Trigger**: Midnight rebuild (daily cron) + per-anime scheduled jobs

**Window**: 7-day rolling window of expected episode air times, rebuilt each midnight.

```
Midnight rebuild:
  query PostgreSQL Service (GraphQL)
    → all ONGOING anime with broadcast.day ∈ next 7 days
  for each anime:
    expected_air_datetime = broadcast.day + broadcast.time + timezone
    expected_episode_num  = MAX(episode_number) in DB + 1
    schedule job at expected_air_datetime

At air-time job:
  1. Fetch episode from Jikan/AniList for anime_id
  2. Verify episode_number == expected_episode_num
  3. Found     → publish anime.episode.aired to NATS
  4. Not found → reschedule +2hrs, retry up to 3 times
  5. Check API response for status change → publish anime.updated if changed
  6. After 3 retries: emit alert, skip window for this anime
```

**Episode number logic**:

| Anime type | Strategy |
|------------|----------|
| Finite (e.g. 12 eps known) | expected = next sequential; on final episode verify status via API |
| Long-running (e.g. One Piece) | expected = DB MAX + 1; status never auto-assumed finished; re-verified weekly |

> [!note] Delay Handling
> `Anime.delay_information` from DB is respected. If delay info is present, the initial air-time job accounts for it before scheduling.

### 2. Score Sync

**Trigger**: Daily cron at 06:00 UTC

**Scope**: All `ONGOING` anime + anime that finished within the last 90 days

```
For each anime in scope:
  fetch statistics in parallel: MAL, AniList, Kitsu
  compute ScoreCalculations (arithmetic_mean, median, geometric_mean)
  if score delta > threshold:
    publish anime.updated to NATS (score + statistics fields only)
```

> [!note] Score Delta Threshold
> Prevents noisy Qdrant re-embeds from minor score fluctuations. Threshold is configurable via environment variable.

## Events Published

| Subject | Trigger | Payload |
|---------|---------|---------|
| `anime.episode.aired` | Episode verified on air day | `EpisodeAiredEvent` |
| `anime.updated` | Score or status changed | `AnimeUpdatedEvent` |

See [[event_schema_specification]] for full protobuf definitions.

> [!important] Update Worker → PostgreSQL only
> Update Worker publishes events consumed **exclusively by PostgreSQL Service**. It never talks to Qdrant directly. Qdrant syncs only after PostgreSQL commits, via the outbox pattern (see below).

## Downstream Consumers

| Event | Consumer | Action |
|-------|----------|--------|
| `anime.episode.aired` | PostgreSQL Service (`postgres-episode-consumer`) | Insert episode row, update episode count |
| `anime.updated` | PostgreSQL Service (`postgres-update-consumer`) | Update score/status fields |

Qdrant is **not** a consumer of Update Worker events.

## Qdrant Sync — SAGA Pattern

PostgreSQL is the single source of truth. Qdrant syncs from PostgreSQL, never from the Update Worker.

```
Update Worker
  → NATS (anime.episode.aired / anime.updated)
    → PostgreSQL Service: Event Consumer
        DB transaction (atomic):
          ├── writes to anime / episode tables
          └── inserts row into qdrant_outbox

    → PostgreSQL Service: Outbox Worker  ← background loop inside same service
        polls qdrant_outbox WHERE processed_at IS NULL
          → publishes anime.synced to NATS
          → marks row processed_at = now()

    → NATS (anime.synced)
        → Qdrant Service
            ├── payload update  (score, status, episode_count)
            └── vector re-embed (synopsis, tags, demographics)
```

The **outbox worker** is a background task running inside PostgreSQL Service — not a separate service. It is the bridge between the DB write and the NATS publish, ensuring they are never split across a failure boundary.

**SAGA guarantee**: The DB write and the `qdrant_outbox` row are committed **atomically**. If the write fails, no outbox row is created, so Qdrant is never notified. There is no partial state.

| Scenario | Outcome |
|----------|---------|
| PG write fails | NATS redelivers to PG; Qdrant untouched |
| PG write succeeds, outbox publish fails | Outbox worker retries on next poll |
| Qdrant update fails | NATS redelivers `anime.synced`; PG already committed |
| Max retries exceeded | DLQ alert; Qdrant may be temporarily stale (but PG is consistent) |

## Qdrant Update Strategy

Qdrant can update a **point** (vector), **payload** (metadata), or both — depending on which fields changed.

| Changed Field | Qdrant Update | Reason |
|---------------|--------------|--------|
| `score` (numeric only) | Payload only | Score as filter field; no semantic content to embed |
| `score` (if semantically embedded) | Payload + Vector | e.g. embedding "critically acclaimed" vs "average rating" |
| `status` | Payload only | Filter field; no text meaning changes |
| `episode_count` | Payload only | Numeric filter; vector unaffected |
| `synopsis` | Payload + Vector | Core text field — directly affects embedding |
| `tags`, `demographics`, `genres` | Payload + Vector | Categorical meaning changes embedding |

> [!note] Score Semantic Embedding (Future Option)
> If score is given semantic meaning in the embedding (e.g. bucketed into "acclaimed / well-rated / average / poor"), then a score change crossing a bucket boundary triggers a vector re-embed. This is a product decision, not yet finalised.

When Qdrant needs to re-embed (vector update), it fetches the relevant fields from PostgreSQL Service via GraphQL rather than relying on the event payload — ensuring it always embeds the committed state.

## External API Dependencies

Reuses `libs/enrichment` helpers — no new API integrations needed:

| Source | Data Fetched |
|--------|-------------|
| Jikan (MAL) | Episode details, air status |
| AniList | Episode details, score |
| MAL | Score, statistics |
| Kitsu | Score, statistics |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `NATS_URL` | `nats://localhost:4222` | NATS server URL |
| `POSTGRES_GRAPHQL_URL` | `http://localhost:8000/graphql` | PostgreSQL Service GraphQL endpoint |
| `EPISODE_WINDOW_DAYS` | `7` | Sliding window size in days |
| `SCORE_SYNC_HOUR` | `6` | UTC hour for daily score sync |
| `SCORE_DELTA_THRESHOLD` | `0.1` | Minimum score change to publish event |
| `SCORE_SYNC_LOOKBACK_DAYS` | `90` | Days past finish to keep syncing scores |
| `EPISODE_RETRY_DELAY_HOURS` | `2` | Delay before retrying a missed episode |
| `EPISODE_MAX_RETRIES` | `3` | Max retries before alerting and skipping |

## Error Handling

- **Episode not aired after 3 retries**: emit structured alert (Slack/PagerDuty), skip to next window
- **External API failure**: log error, skip anime for this run; APScheduler will retry on next scheduled run
- **NATS publish failure**: retry with exponential backoff; if stream unavailable, hold events in memory up to configurable limit

## Related Documentation

- [[event_driven_architecture|Event-Driven Architecture]] — NATS setup, consumer configuration, full event flow
- [[event_schema_specification|Event Schema Specification]] — EpisodeAiredEvent, AnimeUpdatedEvent protobuf definitions
- [[Architecture Index|Architecture Index]] — Services overview and data flow patterns

---

**Status**: Planned | **Last Updated**: 2026-02-17
