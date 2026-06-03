---
title: Ingestion Pipeline
date: 2026-02-19
tags:
  - services
  - ingestion
  - pipeline
  - enrichment
  - nats
status: active
related:
  - "[[n8n_orchestration]]"
  - "[[event_driven_architecture]]"
  - "[[event_schema_specification]]"
  - "[[Database Schema]]"
  - "[[Architecture Index]]"
---

# Ingestion Pipeline

## Overview

The Ingestion Pipeline is the **Python service responsible for initial, full enrichment of new anime entries** and publishing them to NATS for downstream persistence. It is a one-time operation per entry — ongoing partial updates are handled by the [[update_service|Update Service]].

The pipeline is orchestrated by [[n8n_orchestration|n8n]] for incremental and bootstrap runs. n8n triggers ingestion via the [[command_api|Command API]] and manages the stage-6 review gate before final publishing.

> [!important] Scope Boundary
> The Ingestion Pipeline handles **new entries only** (entries whose `entry_key` is not in `processed_entries`). It never re-ingests existing entries. Existing entry updates are the Update Service's responsibility.

---

## Entry Identity

Each source entry is identified by a deterministic `entry_key`:

```
entry_key = sha256(join("|", sort(unique(trim(sources[])))))
```

- Derived from `sources` array only (no title, type, season, year, status, score)
- Metadata header row (`$schema`) excluded before key generation
- Stored as lowercase hex digest
- Used for cross-run deduplication in `processed_entries` table

---

## Enrichment Stages

The pipeline runs entries through 5 sequential enrichment stages. Each stage adds data from different sources and enriches the canonical model.

| Stage | Name | What it does |
|-------|------|-------------|
| Stage 1 | Metadata Validation | Validate required fields, detect issues, compute canonical field names (`episode_count`, `season`) |
| Stage 2 | Episode Enrichment | Fetch episode list and details from Jikan, AniList, Kitsu |
| Stage 3 | Character Matching | Match and enrich characters from AniList, AniDB, AnimePlanet |
| Stage 4 | Cross-Source Merge | Merge data from multiple sources, resolve conflicts, normalise scores |
| Stage 5 | Final Assembly | Assemble canonical `AnimeRecord`, validate completeness, compute `ScoreCalculations` |

Stage outputs are persisted as artifacts (S3/MinIO refs) in `ingestion_entry_stages` for replay capability. Issues detected in stages 1–5 are attached to the stage-6 review packet — they do not create separate manual pauses.

For detailed stage logic see `docs/claude/enrichment-pipeline.md`.

---

## Trigger Modes

| Mode | Trigger | Scope |
|------|---------|-------|
| `bootstrap` | Manual via Command API | All entries in a source snapshot |
| `incremental` | n8n scheduled tag check → Command API | New entries only (diff against previous snapshot) |

Both modes run the same 5-stage pipeline. The difference is the candidate entry set fed into stage 1.

---

## Stage-6 Review Gate

After stages 1–5 complete, **every entry pauses for manual review** before `anime.enriched` is published. This is owned by n8n, not the pipeline service itself.

See [[n8n_orchestration|n8n Orchestration]] for the full stage-6 review flow, decision schema, and batch selector rules.

---

## Events Published

| Subject | Trigger | Payload |
|---------|---------|---------|
| `anime.enriched` | After stage-6 `approve_ingest` decision | `AnimeEnrichedEvent` |

See [[event_schema_specification|Event Schema Specification]] for the full `AnimeEnrichedEvent` protobuf definition.

The `anime.enriched` event contains the full canonical `AnimeData`, `CharacterData[]`, `EpisodeData[]`, and `EnrichmentMetadata`. It is consumed exclusively by PostgreSQL Service.

---

## Downstream Flow

```
Ingestion Pipeline
  → NATS (anime.enriched)
    → PostgreSQL Service: postgres-consumer
        DB transaction (atomic):
          ├── insert anime, characters, episodes, xrefs, company links
          └── insert row into qdrant_outbox

    → PostgreSQL Service: Outbox Worker
        → NATS (anime.created)
          → Qdrant Service (initial vector upsert)
```

---

## Ingestion State Tables

The pipeline reads and writes the following PostgreSQL tables (managed by PostgreSQL Service via Command API):

| Table | Purpose |
|-------|---------|
| `source_snapshots` | Immutable record of each upstream artifact version |
| `ingestion_runs` | Per-execution lifecycle + run lock |
| `ingestion_entries` | Per-entry status and current stage |
| `ingestion_entry_stages` | Per-entry per-stage attempt (enables replay from stage N) |
| `ingestion_reviews` | Stage-6 manual decisions |
| `processed_entries` | Cross-run dedup index — `entry_key` → seen? |

See [[Database Schema]] for full DDL.

---

## External API Dependencies

Reuses `libs/enrichment` source helpers:

| Source | Stages |
|--------|--------|
| Jikan (MAL) | Stage 2 (episodes), Stage 3 (characters) |
| AniList | Stage 2 (episodes), Stage 3 (characters), Stage 4 (merge) |
| Kitsu | Stage 2 (episodes), Stage 4 (merge) |
| AniDB | Stage 3 (characters) |
| AnimePlanet | Stage 3 (characters) |

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `NATS_URL` | `nats://localhost:4222` | NATS server URL |
| `ENRICHMENT_SERVICE_HOST` | `localhost:50052` | Enrichment gRPC service |
| `ARTIFACT_STORAGE_URL` | `http://localhost:9000` | MinIO / S3 endpoint for stage artifacts |
| `ARTIFACT_BUCKET` | `echora-ingestion` | Bucket for stage artifacts and review files |

---

## Related Documentation

- [[n8n_orchestration|n8n Orchestration]] — orchestrates this pipeline, owns stage-6 gate
- [[command_api|Command API]] — HTTP interface n8n uses to trigger pipeline stages
- [[event_schema_specification|Event Schema Specification]] — `AnimeEnrichedEvent` definition
- [[Database Schema]] — ingestion state tables DDL
- `docs/claude/enrichment-pipeline.md` — detailed per-stage implementation notes

---

**Status**: Planned | **Last Updated**: 2026-02-19
