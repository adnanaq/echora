---
title: Architecture Index
date: 2026-11-14
tags:
  - index
  - architecture
  - navigation
status: active
---

# Architecture Index

The starting point for Echora's architecture documents. Each document describes the current design, and planned parts are marked as planned.

## Visual Overview

📊 **[[Architecture Overview.canvas|Architecture Overview Canvas]]** shows every service, store and data flow on one page.

## How the System Fits Together

- **One owner per store.** The PostgreSQL Service is the only way into PostgreSQL, and the vector service is the only way into Qdrant.
- **Temporal orchestrates every write.** Workflows call each service over gRPC in order, retry failed steps and keep a history of every run. No service writes through another on its own: every write is a workflow step. Reads go directly, such as the vector service reading an entity back after an event, or the backend calling search.
- **NATS carries notices after a commit.** The PostgreSQL Service's outbox is the only publisher of catalogue events. Consumers read the current state back over gRPC.
- **The backend is the frontend's only entry point.** It routes each request to the service that owns the data.
- **Entities are built from provider pages.** Works, episodes and characters each have their own pages, refresh and events. Pages are stored as received, and each entity is mapped and merged from them, with approved corrections applied on top.
- **Every field is a column or a table.** PostgreSQL holds no JSONB. The schema is normalized to fifth normal form, and every table and column is named after the model field it stores.
- **IDs are UUIDv7, assigned once when an entity is first created,** and never derived from provider URLs.
- **Echora writes its own synopses.** Provider synopses are never shown. The Agent Service writes each work's synopsis, and until it exists works show none.

---

## Documents

### Schema Ownership & Contracts
📦 **[[echora_contracts|Schema Ownership & Contracts]]**

`anime.py` is the source of truth and `anime.proto` mirrors it. Covers how Python and Rust get the generated types, how drift is caught in CI and at build time, and how fields change before and after production.

### PostgreSQL Service
🗄️ **[[postgres_integration_architecture_decision|PostgreSQL Service]]**

Why one Rust service in this repository owns the database, what it does and does not do, its gRPC callers, its tech stack, and the Rust workspace in `crates/` (`common`, `postgres_db`, `temporal_workflows`, `postgres_service`, `update_service`).

### Database Schema
🗄️ **[[Database Schema]]**

The full DDL, normalized to fifth normal form with no JSONB: provider pages and their raw responses, the links from works, episodes and characters to their pages, every field of the canonical models, relations, corrections, AI-written text, the approval queue and the outbox. Also maps every model field to its table, and covers deduplication, refresh queries and the knowledge graph: its nodes, its one edge vocabulary, relation provenance, franchises and the graph reads.

### Update Service
🔄 **[[update_service|Update Service]]**

The Temporal workflows, written in Rust, that keep works, episodes and characters current: how a refresh runs, why only references pass through Temporal, one door per entity with version-checked saves, per-provider rate limits with priority and fairness, the weekly sweep and the 30-year rule, refresh on open, episode tracking, daily scores and the AI-written synopsis.

### Ingestion Pipeline
🔄 **[[ingestion_pipeline|Ingestion Pipeline]]**

How a seed release becomes works: clean the seed, load the works with `LoadSeedReleaseWorkflow`, then enrich each new work, and its episodes and characters, with the same refresh workflows. Lists what exists today and what is still missing.

### Event-Driven Architecture
📄 **[[event_driven_architecture|Event-Driven Architecture]]**

Why NATS JetStream, thin events, when each entity's `created`, `updated` and `deleted` events are sent, the transactional outbox and its relay, the vector service consumers, delivery deadlines and backoff, dead letters, monitoring and deployment.

### Event Schema Specification
📋 **[[event_schema_specification|Event Schema Specification]]**

The protobuf messages: catalogue and notification events, the shared merge messages, and the planned `EnrichmentService`, `AgentService` and `PostgresService` methods.

### Temporal Infrastructure
⚙️ **[[temporal_infrastructure|Temporal Infrastructure]]**

The Temporal server and its PostgreSQL databases, local development with the CLI dev server, priority and fairness, queue rate limits, Worker Versioning, the Rust worker and the rules for writing workflows.

---

## Services

| Service | Language | Status | Role | Interface |
|---------|----------|--------|------|-----------|
| `apps/vector_service` | Python | Built, consumers planned | Vector search and Qdrant admin; indexes works, episodes and characters from catalogue events | gRPC |
| `apps/enrichment_service` | Python | Built, methods planned | Fetching pages, and mapping and merging works, episodes and characters | gRPC |
| `apps/review_page` | Python | Built, local | Answering seed-cleaning review cases | HTTP on `127.0.0.1` |
| PostgreSQL Service | Rust | Planned | The catalogue database's only owner | gRPC |
| Update Service | Rust | Planned | Temporal workers for refreshes, episode tracking, sweeps, synopses and seed loads | Temporal worker |
| Backend | Rust | Planned | The frontend's single entry point | GraphQL |
| Users Service | Not decided | Planned | Accounts, MAL and AniList logins, anime lists | gRPC |
| Agent Service | Python | Planned | Every AI call: writing synopses, extracting character attributes, Jev scoring of review cases, search reasoning | gRPC |
| Notification Service | Not decided | Planned | Episode and status notifications | NATS consumer |
| Frontend | Not decided | Planned | The website | Calls the backend |

`apps/agent_service/` holds only leftover `__pycache__` folders from an earlier prototype, with no source files and nothing tracked by git. The Agent Service is built new in that folder, with its logic in a library under `libs/` like the other services, and the leftover folders are deleted first.

The review page moves to PostgreSQL once the approval queue is built. It then serves one multi-user queue for review cases, risky changes and edits users suggest, with reviewer roles and an approval history.

## Infrastructure

| Component | Purpose | Notes |
|-----------|---------|-------|
| PostgreSQL 18 | Catalogue and Temporal databases | One cluster with the `echora`, `temporal` and `temporal_visibility` databases; the users service gets its own |
| Qdrant | Vector database | `anime_database` collection, owned by the vector service |
| NATS JetStream 2.15 | Event bus | `ANIME_EVENTS`, `ANIME_DLQ`, `NOTIFICATION_EVENTS` |
| Temporal 1.31+ | Durable workflows | Refreshes, episode timers, sweeps, seed loads |
| Redis | HTTP and result cache | Databases 0-15 for services, 16-31 for tests |

## Event Subjects

All published by the PostgreSQL Service outbox. See [[event_driven_architecture#Subjects|Subjects]].

| Subject | Sent when | Consumers |
|---------|-----------|-----------|
| `anime.created`, `episode.created`, `character.created` | The entity is saved for the first time | Vector Service, later "recently added" and new-anime notifications |
| `anime.updated`, `episode.updated`, `character.updated` | An existing entity is saved with changes | Vector Service |
| `anime.deleted`, `episode.deleted`, `character.deleted` | The entity is removed, such as a work merged into another | Vector Service |
| `notification.episode.aired` | Episode tracking confirmed an episode aired | Notification Service |
| `notification.anime.status.changed` | A work changes airing status | Notification Service |

---

## Data Flows

### Search
```
Frontend → Backend
  → Vector Service (gRPC search)           direct search
  → Agent Service (gRPC)                   natural-language search, when enabled
      → Vector Service + PostgreSQL Service
  → PostgreSQL Service (gRPC reads)        full works for the results
→ Frontend
```

### Opening a page
```
Frontend → Backend
  reads the entity from its cache, else PostgreSQL Service (GetAnime, GetEpisode or GetCharacter)
  returns the stored entity at once; the view writes nothing
  if due (next_refresh_at has passed), and the visitor is not a bot:
    Backend starts its refresh workflow (USE_EXISTING, priority 1), under a cap per minute
    the next view shows the refreshed data
  the cache drops an entity when its catalogue event arrives, with a short expiry as a fallback
```

### Refresh
```
RefreshAnimeWorkflow (Temporal), and the same for episodes and characters
  refresh_page per due page: EnrichmentService.FetchPage → PostgresService.StorePage
  merge_anime: PostgresService.LoadAnime → EnrichmentService.MergeAnime → PostgresService.SaveAnime
  SaveAnime checks the version, commits the work and its outbox rows,
  and creates the episodes and characters the work newly lists
→ outbox relay → NATS anime.updated → Vector Service reads back and re-indexes
→ refreshes of the new episodes and characters; WriteSynopsisWorkflow when provider synopses changed
```

### Episode tracking
```
Daily: ListUpcomingEpisodes → one track-episode workflow per episode airing this week
At air time: refresh the work's episode pages and the episode; retry every 2 hours
After airing: refresh the episode again at 1, 3 and 7 days
```

### Seed load
```
clean_seed.py + review page → cleaned_seed.jsonl
→ LoadSeedReleaseWorkflow: create new works, relink moved pages, at most 500 refreshes pending
→ RefreshAnimeWorkflow per new or changed work, then its episodes and characters
→ NATS anime.created / episode.created / character.created
```

### Approval (planned)
```
Review case, risky change or suggested edit → approval queue in PostgreSQL
→ reviewer decides on the review page
→ approved field values become corrections, applied on every merge
→ the entity is merged and saved again
```

### Franchise link check (planned)
```
CheckFranchiseLinksWorkflow, weekly and after a seed release
→ PostgreSQL Service streams the franchise links into the Enrichment Service's check
→ links worth a look become link_check items in the approval queue
→ reviewer confirms each link, or corrects it, or marks a crossover work
→ the corrections are applied on every merge and kept in a file in the repository
```

---

## Suggested Build Order

Tracked in the Linear initiative "Catalogue platform". Cleanup issues follow the build: ECHO-116 and ECHO-123. The docs themselves are ECHO-86.

1. **PostgreSQL Service** with the page, work, episode, character and outbox tables, and the store, load and save methods. ECHO-87, ECHO-93, ECHO-97, ECHO-98, ECHO-100 to ECHO-103, ECHO-133, ECHO-104, ECHO-107, ECHO-108
2. **Enrichment methods** `FetchPage`, `MergeAnime`, `MergeEpisode` and `MergeCharacter`, with the merge taking the work's ID instead of deriving it. ~~ECHO-88~~, ~~ECHO-89~~, ~~ECHO-94~~, ~~ECHO-99~~, ~~ECHO-105~~, ~~ECHO-109~~, ECHO-110
3. **Update Service** refresh workflows, `SweepCatalogueWorkflow` and episode tracking, with queue rate limits, priority and fairness. ECHO-95, ECHO-111 to ECHO-113, ECHO-117 to ECHO-119
4. **Cleaned seed and seed load**: write `cleaned_seed.jsonl`, then `LoadSeedReleaseWorkflow`; then the character list, built from works' character lists, and its load. ECHO-92, ECHO-120, ECHO-135
5. **Vector Service consumers** for the catalogue events, with the version stored in each point. ECHO-90, ECHO-106, ECHO-114, ECHO-121, ECHO-122
6. **Backend**, once GraphQL or REST is decided. Backend project
7. **Approval queue** in PostgreSQL, and the review page moved onto it. Approval queue project
8. **Knowledge graph** reads and franchises, with the franchise link check sending its candidates to the approval queue. P-ECHO-22
9. **Agent Service** for synopses, then the **Users Service** and **Notification Service**. ECHO-91, ECHO-96, ECHO-115; Users Service and Notification Service projects
10. **People** (staff and voice actors), last. People project

## External Resources

- [NATS JetStream](https://docs.nats.io/nats-concepts/jetstream)
- [Temporal](https://docs.temporal.io/)
- [Protocol Buffers](https://protobuf.dev/)
- [sqlx](https://docs.rs/sqlx/) and [async-nats](https://docs.rs/async-nats/)

---

**Status**: Active | **Last Updated**: 2026-11-14
