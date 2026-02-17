---
title: Architecture Index
date: 2026-02-16
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

## Core Architecture Documents

### 1. Event-Driven Architecture
📄 **[[event_driven_architecture|Event-Driven Architecture]]**

- NATS JetStream setup and configuration
- Event flow diagrams
- Code examples (Rust + Python)
- Local testing guide
- Monitoring and production considerations

**Key Topics**: Model synchronization, zero schema duplication, event consumers

---

### 2. Event Schema Specification
📋 **[[event_schema_specification|Event Schema Specification]]**

- Complete protobuf event definitions
- gRPC service contracts
- Schema evolution guidelines
- Code generation instructions

**Key Topics**: AnimeEnrichedEvent, AnimeCreatedEvent, service APIs

---

### 3. Database Schema
🗄️ **[[Database Schema]]**

- Complete DDL (anime, character, episode, manga, all relationships)
- ER diagram
- Hybrid design rationale (normalized core + JSONB metadata)
- Unresolved edge pattern for relationships
- Deduplication strategy
- Indexing guidelines
- Incremental update policy

**Key Topics**: Schema design, cross-references, outbox pattern

---

### 4. PostgreSQL Architecture Decision
🗄️ **[[postgres_integration_architecture_decision|PostgreSQL Integration Architecture]]**

- Architectural decision rationale
- Service design (dual APIs: gRPC + GraphQL)
- Technology stack (Rust + sqlx)
- Implementation roadmap

**Key Topics**: Option B selection, Rust backend integration

---

### 5. Update Worker
🔄 **[[update_worker|Update Worker]]**

- Sliding window episode air tracking (7-day rolling)
- Score sync job (daily, multi-platform)
- NATS events published and downstream consumers
- Configuration reference and error handling

**Key Topics**: APScheduler, episode verification, score delta, ongoing partial updates

---

### 6. echora-contracts (Schema Contracts)
📦 **[[echora_contracts|echora-contracts Repo]]**

- `.proto` files as single source of truth for all data shapes
- `buf generate` → Python + Rust types (never hand-written)
- `buf breaking` CI gate — prevents breaking changes without major version bump
- Qdrant payload index config + CI validation
- Distribution via GitHub Packages (consumed as a versioned dependency)

**Key Topics**: Model drift, buf toolchain, proto-as-canonical, change type rules

---

## Quick Reference

### Services Overview

| Service | Language | Role | APIs |
|---------|----------|------|------|
| **Backend (BFF)** | Rust | External API | GraphQL |
| **Agent Service** | Python | LLM orchestration | gRPC consumer |
| **PostgreSQL Service** | Rust | Data layer | gRPC + GraphQL |
| **Qdrant Service** | Python | Vector search | gRPC consumer |
| **Ingestion Pipeline** | Python | Initial full enrichment | Event publisher |
| **Update Worker** | Python | Ongoing partial updates | Event publisher + GraphQL client |

### Event Subjects

**Update Worker / Pipeline → PostgreSQL Service:**

| Subject | Publisher | Consumer | Payload |
|---------|-----------|----------|---------|
| `anime.enriched` | Ingestion Pipeline | PostgreSQL Service | AnimeEnrichedEvent |
| `anime.episode.aired` | Update Worker | PostgreSQL Service | EpisodeAiredEvent |
| `anime.updated` | Update Worker | PostgreSQL Service | AnimeUpdatedEvent |

**PostgreSQL Service → Qdrant Service (via outbox — SAGA):**

| Subject | Publisher | Consumer | Payload |
|---------|-----------|----------|---------|
| `anime.created` | PostgreSQL Service | Qdrant Service | AnimeCreatedEvent |
| `anime.synced` | PostgreSQL Service | Qdrant Service | AnimeSyncedEvent |
| `anime.deleted` | PostgreSQL Service | Qdrant Service | AnimeDeletedEvent |

> [!important] Qdrant only consumes events from PostgreSQL Service — never directly from Update Worker or Ingestion Pipeline. This ensures Qdrant always mirrors what PostgreSQL has committed.

### Technology Stack

**Backend Services (Rust)**:
- `axum` - Web framework
- `async-graphql` - GraphQL server
- `sqlx` - Database client
- `tonic` - gRPC framework
- `async-nats` - NATS client

**Internal Services (Python)**:
- `nats-py` - NATS client
- `instructor` - LLM framework
- `qdrant-client` - Vector DB client
- `pydantic` - Data validation

**Infrastructure**:
- NATS JetStream - Event bus
- PostgreSQL - Relational database
- Qdrant - Vector database

---

## Data Flow Patterns

### Natural Language Query (LLM enabled)
```
User → Backend → Agent Service → (Qdrant Service + PostgreSQL Service) → Backend → User
```

### Direct Search (LLM disabled or LLM down)
```
User → Backend → Qdrant Service (direct gRPC) + PostgreSQL Service (GraphQL) → Backend → User
```

> [!info] LLM Fallback
> Backend talks directly to Qdrant Service via gRPC when:
> - LLM is explicitly disabled by user/operator
> - Agent Service is unreachable or down
>
> PostgreSQL Service is **always** called directly by Backend for entity hydration regardless of LLM mode.

### Data Ingestion (initial)
```
Ingestion Pipeline → NATS → PostgreSQL Service → NATS → Qdrant Service
```

### Ongoing Updates (Update Worker)
```
Update Worker → NATS (anime.episode.aired) → PostgreSQL Service
Update Worker → NATS (anime.updated) → PostgreSQL Service + Qdrant Service
Update Worker ← PostgreSQL Service (broadcast schedule via GraphQL)
```

---

## Implementation Priority

1. ✅ **Event Architecture Setup** - NATS, protobuf schemas
2. ✅ **Documentation** - Architecture docs, event specs
3. ⏳ **PostgreSQL Service** - Rust implementation with dual APIs
4. ⏳ **Ingestion Integration** - Event publishing
5. ⏳ **Qdrant Integration** - Event consumption
6. ⏳ **Update Worker** - APScheduler + sliding window + score sync

---

## Related Plans

- [[../plans/2026-02-10-postgres-qdrant-data-model-design|PostgreSQL + Qdrant Data Model Design]]
- [[postgres_integration_architecture_decision#Implementation Roadmap|Implementation Roadmap]]

---

## Schema Contracts & Model Drift

All data shapes are defined as `.proto` files, **owned by the ingestion pipeline** (the service that discovers and defines what data looks like). Python and Rust types are **generated** from these — never hand-written. This eliminates model drift across services.

📄 **[[echora_contracts|Schema Ownership & Contracts]]** — full details: toolchain, CI pipeline, distribution, change type rules, Qdrant payload index config

**Key points:**
- Ingestion pipeline CI publishes `echora-types` packages to GitHub Packages on every merge
- All other services consume `echora-types` as a normal versioned dependency
- `buf breaking` gates all breaking changes — fails PR if a field is removed or type changed
- PostgreSQL DDL is manual, but `cargo sqlx prepare` makes a missing migration a compile error

---

## External Resources

- [NATS JetStream Documentation](https://docs.nats.io/nats-concepts/jetstream)
- [Protocol Buffers](https://protobuf.dev/)
- [async-nats (Rust)](https://docs.rs/async-nats/)
- [sqlx (Rust)](https://docs.rs/sqlx/)
- [async-graphql (Rust)](https://async-graphql.github.io/)

---

**Last Updated**: 2026-02-17 | **Status**: Active
