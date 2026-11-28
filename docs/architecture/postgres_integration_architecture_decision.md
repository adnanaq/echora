---
title: PostgreSQL Service
date: 2026-11-14
tags:
  - architecture
  - postgresql
  - rust
  - decision
status: planned
related:
  - "[[Database Schema]]"
  - "[[event_driven_architecture]]"
  - "[[event_schema_specification]]"
  - "[[update_service]]"
---

# PostgreSQL Service

## Decision

The catalogue database is owned by one service, the **PostgreSQL Service**, written in Rust and kept in this repository. No other service connects to the database. Every read and write goes through its gRPC API, the same way the vector service is the only way into Qdrant.

- **One owner per store.** The schema, migrations and queries live in one place, so no other service mirrors a table and no two services write the same rows.
- **Same repository.** The service sits beside the Python services and the Rust update service, and builds its gRPC code from the same `protos/` folder, so a proto change and both of its sides land in one PR.
- **Rust.** `sqlx` checks every query against the real schema at build time, so a model field without a column fails the build instead of a request.

## What the Service Does

- ~~**Stores every provider page losslessly**, with its responses as received and its fetch time. Nothing mapped is stored. (ECHO-97)~~
- **Stores works, episodes and characters.**
  - ~~A work's source and partial-source pages, its merged record and its approved per-field corrections. (ECHO-98)~~
  - ~~Episodes, with their pages and merged records. (ECHO-103)~~
  - ~~Characters, with their pages and merged records. (ECHO-104)~~
- **Writes events through an outbox.** See [[event_driven_architecture|Event-Driven Architecture]].
  - ~~A change and its outbox row commit in one transaction. (ECHO-98)~~
  - A relay task in the same process publishes the rows to NATS. (ECHO-106)
- ~~**Answers reads** for the vector service, the backend and the update service's sweep. (ECHO-108)~~

## What the Service Does Not Do

- **It never calls another service.** Temporal workflows call it, and call the enrichment service between its steps. See [[update_service|Update Service]]. (~~ECHO-111~~)
- ~~**It holds no enrichment rules.** Merging pages and applying corrections happen in `EnrichmentService.MergeAnime`. The service stores the result. (ECHO-105)~~
- **It does not decide what is risky.** A change that needs review is sent to the approval queue by the workflow that found it, once "risky" is defined. The service only stores the queue. (P-ECHO-15)
- **It does not hold user accounts or lists.** Those belong to the planned users service and its own database. (P-ECHO-17)

## API

The service exposes gRPC only. The planned methods are listed in [[event_schema_specification#PostgresService (planned)|Event Schema Specification]].

| Caller | Methods | Purpose |
|---|---|---|
| ~~Update Service activities (Temporal)~~ | ~~`StorePage`~~ | ~~Store each fetched page (ECHO-97)~~ |
| ~~Update Service activities (Temporal)~~ | ~~`LoadAnime`, `SaveAnime`~~ | ~~Load, merge and save a work (ECHO-98)~~ |
| ~~Update Service activities (Temporal)~~ | ~~`ListDueAnime`~~ | ~~Find the works the sweep refreshes (ECHO-108)~~ |
| ~~Vector Service~~ | ~~`GetAnime`, `GetEpisode`, `GetCharacter`, `ListEntityVersions`~~ | ~~Read the current state after an event, and compare Qdrant with PostgreSQL (ECHO-108)~~ |
| Backend | Reads for pages the frontend shows, defined with the backend | Serve the frontend (P-ECHO-16) |
| Backend | Knowledge graph reads (`FindNeighbors`, `FindReachable`, `FindPath`, `CompareEntities`) | Franchise pages, related works, a studio's other works (P-ECHO-22) |
| Agent Service (planned) | The same knowledge graph reads | Answer questions that follow links between works (P-ECHO-22) |

The frontend talks only to the backend (GraphQL), and the backend calls this service over gRPC. (P-ECHO-16)

### Writes are safe to repeat

Temporal retries failed activities, and NATS delivers events at least once, so every write accepts being called twice:

- ~~`StorePage` ignores a fetch older than the stored `fetched_at` of the same page. (ECHO-97)~~
- ~~When `StorePage` changes a stored page (`has_changed`), it bumps `version` and sets `merge_requested_at` on any linked `source` work so stale in-flight merges fail the version check and trigger a re-merge. (ECHO-98)~~
- ~~`SaveAnime` writes nothing, and sends nothing, when the work did not change. (ECHO-98)~~
- Every other write is safe to repeat.
  - ~~Pages are upserted on their page key. (ECHO-97)~~
  - ~~Entities are written by the ID the merge assigned, and detail rows are replaced per entity. (ECHO-98)~~

### Knowledge graph

> **Linear:** P-ECHO-22

Works, characters, companies and source material, and the typed links between them, form a knowledge graph in `graph_edge`. The service answers graph questions with fixed, parameterized recursive SQL, so callers never send SQL. The graph does not depend on any one caller: the backend reads it for pages, and the Agent Service will read it the same way.

- **Fixed reads.** `neighbors` (one hop), `k_hop` (several hops), `path` (how two nodes connect) and `compare` (what two nodes share), for any node type.
- **One edge vocabulary.** The `graph_edge_type` list: work relations, `FEATURES` and `APPEARS_IN` for characters, company roles, and `ADAPTED_FROM` for source material. Nothing is derived at query time.
- **Node filters.** Work type, year, status, and genre, tag or theme names narrow the results. Genres, tags and themes are filters, not nodes.
- **Hard limits on the server.** At most 5 hops, 50 results and 50 neighbours per hop, a 2-second statement timeout set with `SET LOCAL` inside each read's own transaction, and PostgreSQL's own cycle handling, a recursive `UNION` or the `CYCLE` clause.
- **Empty is a valid answer.** No path between two nodes is a result, not an error.

Nodes, edges, provenance, franchises and the SQL for each read are in [[Database Schema#Knowledge Graph|Database Schema]].

## Tech Stack

| Concern | Crate |
|---|---|
| ~~Database access and migrations~~ | ~~`sqlx` (Postgres, compile-time checked queries, `sqlx migrate`) (ECHO-93)~~ |
| ~~gRPC server and generated code~~ | ~~`tonic` and `prost` (ECHO-97)~~ |
| ~~Async runtime~~ | ~~`tokio` (ECHO-87)~~ |
| NATS | `async-nats` (ECHO-106) |
| ~~Errors in libraries~~ | ~~Error structs with a private kind, `is_…` methods such as `is_version_conflict()`, and `From` conversions so `?` works (ECHO-87)~~ |
| ~~Errors in apps~~ | ~~`anyhow`, and `postgres_service` turns library errors into gRPC statuses in one function (ECHO-97)~~ |
| ~~Logs and traces~~ | ~~`tracing` with named events (`outbox.publish.failed`) and message templates, exported through OpenTelemetry like the other services (ECHO-87)~~ |
| ~~Allocator of the PostgreSQL Service~~ | ~~`mimalloc` (ECHO-97)~~ |
| ~~Allocator of the Update Service~~ | ~~`mimalloc` (ECHO-95)~~ |
| ~~Generated gRPC code~~ | ~~`tonic-prost-build` in each crate's `build.rs` (ECHO-97)~~ |

~~Rust code follows the Microsoft Pragmatic Rust Guidelines, through the `ms-rust` skill. (ECHO-87)~~

~~`sqlx` is chosen over Diesel because the service is mostly hand-written SQL, including recursive queries, and `sqlx` checks plain SQL without a query DSL. Crate versions are pinned in `Cargo.lock` when the crate is created. (ECHO-93)~~

## Repository Layout

~~The Rust code follows the same layering as the Python code: libraries hold the logic, and each app is a thin entry point over them. The crates sit side by side in `crates/`, apart from the Python `apps/` and `libs/`, so Cargo and Pants each own their own folders. (ECHO-87)~~

```
echora/
├── Cargo.toml                       # workspace of every Rust crate below (ECHO-87)
├── Cargo.lock
├── protos/                          # shared by every service
├── .cargo/config.toml               # target CPU for the production servers (ECHO-87)
└── crates/                          # every Rust crate, side by side
    ├── common/                      # settings from env, then .env, then defaults; tracing and OpenTelemetry setup (ECHO-87)
    ├── postgres_db/                 # all SQL: pages, works, episodes, characters, corrections, approvals (ECHO-93)
    │   ├── migrations/              # sqlx migrations, the only schema source (ECHO-93)
    │   └── src/outbox.rs            # relay task that publishes outbox rows to NATS (ECHO-106)
    ├── temporal_workflows/          # ~~workflows and activities, see Update Service (ECHO-95)~~
    ├── postgres_service/            # main.rs and gRPC handlers over postgres_db (ECHO-87)
    └── update_service/              # ~~main.rs: Temporal worker running temporal_workflows (ECHO-95)~~
```

- ~~**Generated gRPC code.** Each crate that needs gRPC types generates them from `protos/` in its own `build.rs`, the way `scripts/generate-proto.py` does for Python. The services exchange bytes, never Rust types, so no shared crate is needed. (ECHO-97)~~
- ~~**Settings.** `common` reads the same environment variable names as the Python `EnvSettings`, and each app defines its own settings struct from it. (ECHO-87)~~
- ~~**Ownership.** Only `postgres_service` depends on `postgres_db`, so no other crate reaches the database. (ECHO-87)~~
- ~~**Workspace.** The root `Cargo.toml` lists every crate and declares every dependency version, the crates' own versions and the lints once, in `[workspace.dependencies]` and `[workspace.lints]`. Each crate takes them with `workspace = true`. The edition is 2024, and the minimum Rust version is 1.92, which the Temporal Rust SDK needs. (ECHO-87)~~
- ~~**Checks.** CI runs `rustfmt`, clippy with the lint set of the Microsoft Rust guidelines (the `ms-rust` skill) plus `wildcard_enum_match_arm`, `cargo-audit` for vulnerable dependencies and `cargo-udeps` for unused ones. See [[echora_contracts#The PostgreSQL Service's mapping|Schema Ownership & Contracts]] for why. (ECHO-87)~~
- ~~**Build.** Cargo builds and tests the Rust crates. Pants has no supported Rust backend, so Pants only wraps the Cargo builds to produce the Docker images, and Python targets stay as they are. (ECHO-87)~~
  - ~~`.sqlx/` holds the prepared query data, so builds without a database still check queries. (ECHO-93)~~

## Security

Nothing is authenticated in local development. Before any shared deployment:

- **Mutual TLS between services.** Each service has a certificate, so a server knows which service is calling. This is the standard identity for service-to-service gRPC. (P-ECHO-21)
- **One service, permissions per method.** `PostgresService` stays one service. An interceptor checks the caller's identity against a table of the methods each caller may use, keyed by the full method name (`/postgres_service.v1.PostgresService/DeleteAnime`), and denies anything not in the table. A CI test compares the table with the proto, so a new method cannot ship without an entry. The backend, for example, gets reads only, and only the Update Service gets the saves, relinks and deletes. Splitting reads and writes into two services would give coarser rules for the same protection, at the cost of a second proto and client. (P-ECHO-21)
- **NATS accounts.** One account per service, and only the PostgreSQL Service may publish catalogue and notification subjects. Consumers confirm a delete by reading back, so a forged delete changes nothing. (P-ECHO-21)
- **`RunPipeline` is removed** from the Enrichment Service first. It accepts any file path from its caller. (ECHO-123)

## Backups and Restore

PostgreSQL holds the only copy of what cannot be fetched again: entity IDs, review answers, approved corrections and AI-written text.

- **PostgreSQL** gets point-in-time recovery: a base backup plus continuous archiving of its write-ahead log, so it can be restored to any moment. The tool is chosen with the deployment. (P-ECHO-21)
- **Qdrant needs no backup.** The vector service's comparison job rebuilds it from PostgreSQL (see [[event_driven_architecture#3. The vector service reacts|Event-Driven Architecture]]). (ECHO-121)
- **NATS needs no backup.** It holds only notices, and PostgreSQL is current.
- **Temporal's databases** can be lost.
  - ~~The Update Service creates the `RefreshTrigger` search attribute at startup if it is missing. (ECHO-95)~~
  - The Update Service creates its schedules at startup if they are missing. (ECHO-117, ECHO-118)
  - The next sweep picks up refreshes that were in flight. (ECHO-117)

Restore PostgreSQL first, then start the services, then run the vector comparison job. (P-ECHO-21)

After a restore to an earlier moment, Qdrant may hold versions newer than PostgreSQL. The comparison job repairs every point whose version differs, not only older ones. (ECHO-121)

Outbox IDs are UUIDv7, so rows written after a restore never reuse an ID that NATS's deduplication would drop. (ECHO-106)

## Database Layout

- ~~In development, one PostgreSQL cluster holds the catalogue database, `echora`. (ECHO-93)~~
- ~~In development, Temporal runs as the CLI dev server with its own file database, so it needs no PostgreSQL. (ECHO-95)~~
- In production, Temporal's databases get a cluster of their own, or at least their own connection budget. Temporal writes and vacuums heavily, and its spikes must not slow the catalogue's queries or trip the 2-second graph timeout. (P-ECHO-21)
- The users service gets its own database when it is built, so catalogue and user data never share an owner. (P-ECHO-17)
- Tables are described in [[Database Schema|Database Schema]].

## Related Documentation

- [[Database Schema|Database Schema]] — Tables and queries
- [[event_driven_architecture|Event-Driven Architecture]] — The outbox and NATS events
- [[event_schema_specification|Event Schema Specification]] — Events and gRPC messages
- [[update_service|Update Service]] — The workflows that write through this service
- [[temporal_infrastructure|Temporal Infrastructure]] — Temporal, which shares the cluster

---

**Status**: Planned | **Last Updated**: 2026-11-14
