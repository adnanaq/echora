---
title: Architecture Index
date: 2026-02-19
tags:
  - index
  - architecture
  - navigation
status: active
---

# Architecture Index

Complete navigation hub for Echora backend architecture documentation.

## Visual Overview

📊 **[[Architecture Overview.canvas|Architecture Overview Canvas]]** - Interactive visual diagram of all services and data flows

---

## Core Architecture Documents

### 1. Event-Driven Architecture
📄 **[[event_driven_architecture|Event-Driven Architecture]]**

- NATS JetStream setup and configuration
- Event flow diagrams
- Consumer configuration (pull consumers, DLQ, retry)
- Outbox pattern for Qdrant sync (SAGA)

**Key Topics**: Model synchronization, zero schema duplication, event consumers, outbox pattern

---

### 2. Event Schema Specification
📋 **[[event_schema_specification|Event Schema Specification]]**

- Complete protobuf event definitions
- gRPC service contracts
- Schema evolution guidelines
- Code generation instructions

**Key Topics**: AnimeEnrichedEvent, AnimeCreatedEvent, AnimeSyncedEvent, service APIs

---

### 3. Database Schema
🗄️ **[[Database Schema]]**

- Complete DDL (anime, character, episode, manga, company, all relationships)
- ER diagram
- Hybrid design rationale (normalized core + JSONB metadata)
- Unresolved edge pattern, deduplication strategy
- Ingestion state tables (`source_snapshots`, `ingestion_runs`, `ingestion_entries`, `ingestion_entry_stages`, `ingestion_reviews`, `processed_entries`)

**Key Topics**: Schema design, cross-references, outbox pattern, ingestion state

---

### 4. PostgreSQL Architecture Decision
🗄️ **[[postgres_integration_architecture_decision|PostgreSQL Integration Architecture]]**

- Architectural decision rationale
- Service design (dual APIs: gRPC + GraphQL)
- Technology stack (Rust + sqlx)
- Implementation roadmap

**Key Topics**: Dedicated PostgreSQL Service, Rust backend, dual API

---

### 5. Ingestion Pipeline
🔄 **[[ingestion_pipeline|Ingestion Pipeline]]**

- One-time full enrichment of new entries (5 stages)
- Entry identity (`entry_key` = sha256 of sources)
- Bootstrap vs incremental trigger modes
- Stage-6 review gate (owned by n8n)
- `anime.enriched` event publishing

**Key Topics**: Enrichment stages, entry_key, new entries only, NATS publisher

---

### 6. Update Service
🔄 **[[update_service|Update Service]]**

- Temporal-based partial update workflows for all entity fields
- Per-anime dynamic scheduling (episode air tracking)
- Pluggable source adapter pattern (selective, full, crawler)
- Workflow types: episode, score, character, staff, metadata
**Key Topics**: Temporal workflows, per-item scheduling, durable execution, all-field updates

---

### 7. n8n Orchestration
🎛️ **[[n8n_orchestration|n8n Orchestration]]**

- Control plane for async orchestration (not in hot query path)
- Incremental ingestion flow (tag check → stages → stage-6 gate)
- Stage-6 review gate: wait/resume, batch decisions, replay
- DLQ triage + replay orchestration
- Risky update approval workflows
- Connectivity: Cloudflare Tunnel → Command API

**Key Topics**: Orchestration, human-in-the-loop, stage-6 review, incremental ingestion

---

### 8. Command API
🔌 **[[command_api|Command API]]**

- HTTP/JSON bridge between n8n and internal gRPC services
- 10 workflow endpoints (tag check, run management, stage processing, review, replay)
- OAuth2 + JWT authentication, idempotency key enforcement
- Deterministic error codes for n8n branching

**Key Topics**: n8n integration boundary, idempotency, auth, workflow contract

---

### 9. Event Adapter
🔀 **[[event_adapter|Event Adapter]]**

- NATS (Protobuf) → n8n (HTTP webhook) bridge
- Subscribes to DLQ events, risk-flagged updates, review alerts
- Redaction policy: only non-sensitive identifiers forwarded
- HMAC-signed webhook delivery

**Key Topics**: NATS-to-n8n bridge, redaction, inbound n8n triggers

---

### 10. Temporal Infrastructure
⚙️ **[[temporal_infrastructure|Temporal Infrastructure]]**

- Durable execution engine for the Update Service
- PostgreSQL-backed (same cluster, separate database)
- Why Temporal over APScheduler and n8n scheduling
- Dev setup (Docker), production (Kubernetes)
- Workflow determinism rules and operational notes

**Key Topics**: Durable execution, crash recovery, per-item scheduling, worker setup

---

### 11. Schema Ownership & Contracts
📦 **[[echora_contracts|Schema Ownership & Contracts]]**

- `anime.py` (Pydantic) is source of truth — hand-written, ingestion pipeline owns it
- `anime.proto` is manually maintained mirror — same repo, co-change rule enforced in CI
- `grpcio-tools` → Python stubs checked in; `tonic-build` → Rust compiles via `build.rs`
- `buf` lint only; `check_anime_model_proto_contract.py` validates field parity
- Distribution via GitHub Releases (`proto-v*` tags, Python + Rust tarballs)

**Key Topics**: Model drift, schema ownership, proto-as-mirror, co-change enforcement

---

## Quick Reference

### Services Overview

| Service | Language | Status | Role | Interface |
|---------|----------|--------|------|-----------|
| **`apps/vector_service`** | Python | ✅ Built | Vector search + Qdrant admin | gRPC |
| **`apps/enrichment_service`** | Python | ✅ Built | Enrichment pipeline orchestration | gRPC |
| **PostgreSQL Service** | Rust | ⏳ Planned | System of record, dual API | gRPC + GraphQL |
| **Backend (BFF)** | Rust | ⏳ Planned | External API gateway | GraphQL |
| **Agent Service** | Python | ⏳ Partial | LLM orchestration | gRPC |
| **Ingestion Pipeline** | Python | ⏳ Planned | Initial full enrichment | NATS publisher |
| **Update Service** | Python | ⏳ Planned | All ongoing partial updates | Temporal worker + NATS publisher |
| **Command API** | Python (FastAPI) | ⏳ Planned | n8n → internal bridge | REST/HTTP |
| **Event Adapter** | Python | ⏳ Planned | NATS → n8n webhook bridge | NATS consumer + HTTP publisher |

### Infrastructure

| Component | Purpose | Notes |
|-----------|---------|-------|
| **NATS JetStream** | Event bus | `ANIME_EVENTS` stream, 7-day retention |
| **PostgreSQL** | Relational database | Application DB + Temporal DB (separate databases) |
| **Qdrant** | Vector database | `anime_database` collection |
| **Redis** | Idempotency store + Command API cache | TTL-managed idempotency keys |
| **Temporal Server** | Durable workflow engine | Backend: PostgreSQL |
| **MinIO / S3** | Artifact storage | Stage artifacts, review files, decision files |
| **n8n** | Orchestration control plane | Self-hosted (dev/stage), n8n Cloud (prod) |
| **Cloudflare Tunnel** | n8n Cloud → Command API connectivity | Zero-trust, no open ports |

---

### Event Subjects

**Ingestion Pipeline / Update Service → PostgreSQL Service:**

| Subject | Publisher | Consumer | Payload |
|---------|-----------|----------|---------|
| `anime.enriched` | Ingestion Pipeline | PostgreSQL Service | `AnimeEnrichedEvent` |
| `anime.episode.aired` | Update Service | PostgreSQL Service | `EpisodeAiredEvent` |
| `anime.updated` | Update Service | PostgreSQL Service | `AnimeUpdatedEvent` |
| `character.updated` | Update Service | PostgreSQL Service | `CharacterUpdatedEvent` |

**PostgreSQL Service → Qdrant Service (via outbox — SAGA):**

| Subject | Publisher | Consumer | Payload |
|---------|-----------|----------|---------|
| `anime.created` | PostgreSQL Service | Qdrant Service | `AnimeCreatedEvent` |
| `anime.synced` | PostgreSQL Service | Qdrant Service | `AnimeSyncedEvent` |
| `anime.deleted` | PostgreSQL Service | Qdrant Service | `AnimeDeletedEvent` |
| `character.synced` | PostgreSQL Service | Qdrant Service | `CharacterSyncedEvent` |

**Internal → Event Adapter → n8n (selected events only, redacted):**

| Subject | Consumer | n8n workflow triggered |
|---------|----------|----------------------|
| `anime.dlq.*` | Event Adapter | DLQ triage + replay |
| `anime.updated` (risk-flagged) | Event Adapter | Risky update approval |
| `ingestion.review.ready` | Event Adapter | Stage-6 review alert |

> [!important] Qdrant Golden Rule
> Qdrant only consumes events from PostgreSQL Service (via outbox). Never directly from Update Service or Ingestion Pipeline. This ensures Qdrant always mirrors what PostgreSQL has committed.

---

### Technology Stack

**Backend Services (Rust)**:
- `axum` — web framework
- `async-graphql` — GraphQL server
- `sqlx` — database client (compile-time checked SQL)
- `tonic` — gRPC framework
- `async-nats` — NATS client

**Internal Services (Python)**:
- `temporalio` — Temporal Python SDK (Update Service)
- `nats-py` — NATS client
- `fastapi` — Command API + Event Adapter
- `instructor` — LLM framework (Agent Service)
- `qdrant-client` — vector DB client
- `pydantic` — data validation

**Infrastructure**:
- NATS JetStream — event bus
- PostgreSQL — relational database (application + Temporal backend)
- Qdrant — vector database
- Redis — idempotency store
- Temporal — durable workflow execution
- MinIO / S3 — artifact storage

---

## Data Flow Patterns

### Natural Language Query (LLM enabled)
```
User → Backend (GraphQL)
     → Agent Service (gRPC)
     → Qdrant Service (gRPC) + PostgreSQL Service (GraphQL)
     → Backend → User
```

### Direct Search (LLM disabled or Agent down)
```
User → Backend (GraphQL)
     → Qdrant Service (gRPC) + PostgreSQL Service (GraphQL)
     → Backend → User
```

> [!info] LLM Fallback
> Backend calls PostgreSQL Service directly for entity hydration in all modes. Agent Service is only in the path for LLM-assisted query interpretation.

### Initial Ingestion
```
n8n (tag check + incremental orchestration)
  → Command API → Ingestion Pipeline (stages 1–5)
  → n8n (stage-6 review gate: wait → decision → approve)
  → Command API → Ingestion Pipeline (publish)
  → NATS (anime.enriched)
  → PostgreSQL Service → DB + qdrant_outbox (atomic)
  → Outbox Worker → NATS (anime.created)
  → Qdrant Service (initial vector upsert)
```

### Ongoing Partial Updates
```
Update Service (Temporal workflows — episode, score, character, staff, etc.)
  → NATS (anime.episode.aired / anime.updated / character.updated)
  → PostgreSQL Service → DB + qdrant_outbox (atomic, SAGA)
  → Outbox Worker → NATS (anime.synced / character.synced)
  → Qdrant Service (payload update and/or vector re-embed)
```

### DLQ Triage + Replay
```
NATS DLQ event
  → Event Adapter (redact → forward)
  → n8n webhook → DLQ triage workflow
  → n8n → Command API (replay-stage)
  → Ingestion Pipeline / Update Service (re-execute)
```

### Risky Update Approval
```
Update Service detects risky field change (title, type, episode_count regression)
  → publishes risk-flagged anime.updated to NATS
  → Event Adapter → n8n risky update approval workflow
  → n8n waits for approver decision
  → approved → Command API → PostgreSQL Service (apply)
  → rejected → discard
```

---

## Implementation Priority

1. ✅ **Event Architecture** — NATS, protobuf schemas
2. ✅ **Documentation** — Architecture docs, event specs, service contracts
3. ⏳ **PostgreSQL Service** — System of record; everything downstream depends on it (dual APIs, outbox, ingestion state tables)
4. ⏳ **Qdrant Service** — Complete the partial `apps/vector_service` implementation; wire up NATS outbox event consumers (`anime.created`, `anime.synced`)
5. ⏳ **Ingestion Pipeline** — First path to populate PostgreSQL; stages 1–5 + NATS publishing
6. ⏳ **Command API** — FastAPI bridge for n8n; idempotency, auth, 10 workflow endpoints; requires PostgreSQL Service + Ingestion Pipeline shape
7. ⏳ **Event Adapter** — NATS → n8n webhook bridge; required before n8n can react to internal events
8. ⏳ **n8n Orchestration** — Incremental ingestion flow + stage-6 review gate; requires Command API + Event Adapter
9. ⏳ **Update Service** — Temporal worker; requires PostgreSQL Service + NATS; episode tracking, score sync, character refresh
10. ⏳ **Backend (BFF)** — External GraphQL API; requires PostgreSQL Service + Qdrant Service both operational
11. ⏳ **Agent Service** — LLM query orchestration; final layer on top of Backend + Qdrant + PostgreSQL

---

## Schema Contracts & Model Drift

`anime.py` (Pydantic) is the canonical data shape, **owned by the ingestion pipeline** — the team that discovers what external APIs return. `anime.proto` is a manually maintained mirror in the same repo.

📄 **[[echora_contracts|Schema Ownership & Contracts]]** — full details: toolchain, CI pipeline, distribution, change type rules

**Key points**:
- Co-change rule: a PR touching `anime.py` must also update `anime.proto`
- `check_anime_model_proto_contract.py` validates field parity in CI (fails on mismatch)
- Generated stubs (`_pb2.py`) are checked in; Rust services compile protos via `tonic-build`
- Distribution: GitHub Releases with `proto-v*` tags — Python tarballs + Rust tarballs
- PostgreSQL DDL is manual, but `cargo sqlx prepare` makes a missing migration a compile error

---

## External Resources

- [NATS JetStream Documentation](https://docs.nats.io/nats-concepts/jetstream)
- [Protocol Buffers](https://protobuf.dev/)
- [Temporal Documentation](https://docs.temporal.io/)
- [async-nats (Rust)](https://docs.rs/async-nats/)
- [sqlx (Rust)](https://docs.rs/sqlx/)
- [async-graphql (Rust)](https://async-graphql.github.io/)
- [n8n Documentation](https://docs.n8n.io/)

---

**Last Updated**: 2026-02-19 | **Status**: Active
