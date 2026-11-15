---
title: PostgreSQL Service
date: 2026-02-16
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

- **Stores every provider page losslessly**, with its raw payload, mapped record and fetch time.
- **Stores works**: which pages are a work's sources and which are its partial sources, the merged record, episodes and characters, and the approved per-field corrections.
- **Writes events through an outbox.** A change and its outbox row commit in one transaction, and a relay task in the same process publishes them to NATS. See [[event_driven_architecture|Event-Driven Architecture]].
- **Answers reads** for the vector service, the backend and the update service's sweep.

## What the Service Does Not Do

- **It never calls another service.** Temporal workflows call it, and call the enrichment service between its steps. See [[update_service|Update Service]].
- **It holds no enrichment rules.** Merging pages and applying corrections happen in `EnrichmentService.MergeWork`. The service stores the result.
- **It does not decide what is risky.** A change that needs review is sent to the approval queue by the workflow that found it, once "risky" is defined. The service only stores the queue.
- **It does not hold user accounts or lists.** Those belong to the planned users service and its own database.

## API

The service exposes gRPC only. The planned methods are listed in [[event_schema_specification#PostgresService (planned)|Event Schema Specification]].

| Caller | Methods | Purpose |
|---|---|---|
| Update Service activities (Temporal) | `StorePage`, `LoadWork`, `SaveWork`, `ListDueWorks` | Refresh, first enrichment and page moves |
| Vector Service | `GetAnime`, `ListEpisodes`, `GetCharacter` | Read the current state after an event |
| Backend | Reads for pages the frontend shows, defined with the backend | Serve the frontend |
| Agent Service (planned) | Relationship traversal | Answer questions that follow relations between works |

The frontend talks only to the backend, and the backend calls this service over gRPC. Whether the frontend uses GraphQL or REST is still open, and does not change this service either way.

### Writes are safe to repeat

Temporal retries failed activities, and NATS delivers events at least once, so every write accepts being called twice:

- `StorePage` ignores a fetch older than the stored `fetched_at` of the same page.
- `SaveWork` writes nothing, and sends nothing, when the work did not change.
- Every insert is an upsert on a natural key, such as a page key or `(anime_id, episode_number)`.

### Relationship traversal

> **Linear:** Relationship traversal project

The agent service will follow relations between works, such as sequels, shared studios and shared characters, as part of natural-language search. The service answers these with recursive SQL queries over fixed templates, so callers never send SQL:

- **Fixed primitives.** `neighbors` (one hop), `k_hop` (several hops), `path` (how two works connect) and `compare` (what two works share).
- **Fixed edge types.** The relation types from `AnimeRelationType`, plus derived edges for a shared studio, a shared character and a shared source work.
- **Hard limits on the server.** At most 5 hops, 50 results and 50 neighbours per hop, a 2-second statement timeout, and a visited list that stops cycles. The timeout is set with `SET LOCAL` inside each query's own transaction, so it never stays on a pooled connection.
- **Empty is a valid answer.** No path between two works is a result, not an error.

The SQL for each primitive is in [[Database Schema#Graph Query Primitives|Database Schema]].

## Tech Stack

| Concern | Crate |
|---|---|
| Database access and migrations | `sqlx` (Postgres, compile-time checked queries, `sqlx migrate`) |
| gRPC server and generated code | `tonic` and `prost` |
| Async runtime | `tokio` |
| NATS | `async-nats` |
| Errors | Libraries return error structs: a private kind, `is_…` methods such as `is_version_conflict()`, and `From` conversions so `?` works. The apps use `anyhow`, and `postgres_service` turns library errors into gRPC statuses in one function. |
| Logs and traces | `tracing` with named events (`outbox.publish.failed`) and message templates, exported through OpenTelemetry like the other services |
| Allocator | `mimalloc` in both apps |
| Generated gRPC code | `tonic-prost-build` in each crate's `build.rs` |

Rust code follows the Microsoft Pragmatic Rust Guidelines, through the `ms-rust` skill. Each finished file carries its compliance comment, `// Rust guideline compliant <date>`.

`sqlx` is chosen over Diesel because the service is mostly hand-written SQL, including recursive queries, and `sqlx` checks plain SQL without a query DSL. Crate versions are pinned in `Cargo.lock` when the crate is created.

## Repository Layout

> **Linear:** ECHO-87, ECHO-93, ECHO-95

The Rust code follows the same layering as the Python code: libraries hold the logic, and each app is a thin entry point over them. The crates sit side by side in `crates/`, apart from the Python `apps/` and `libs/`, so Cargo and Pants each own their own folders.

```
echora/
├── Cargo.toml                       # workspace of every Rust crate below
├── Cargo.lock
├── protos/                          # shared by every service
├── .cargo/config.toml               # target CPU for the production servers
└── crates/                          # every Rust crate, side by side
    ├── common/                      # settings from env, then .env, then defaults; tracing and OpenTelemetry setup
    ├── postgres_db/                 # all SQL: pages, works, episodes, characters, corrections, approvals
    │   ├── migrations/              # sqlx migrations, the only schema source
    │   └── src/outbox.rs            # relay task that publishes outbox rows to NATS
    ├── temporal_workflows/          # workflows and activities, see Update Service
    ├── postgres_service/            # main.rs and gRPC handlers over postgres_db
    └── update_service/              # main.rs: Temporal worker running temporal_workflows
```

- **Generated gRPC code.** Each crate that needs gRPC types generates them from `protos/` in its own `build.rs`, the way `scripts/generate-proto.py` does for Python. The services exchange bytes, never Rust types, so no shared crate is needed.
- **Settings.** `common` reads the same environment variable names as the Python `EnvSettings`, and each app defines its own settings struct from it.
- **Ownership.** Only `postgres_service` depends on `postgres_db`, so no other crate reaches the database.
- **Workspace.** The root `Cargo.toml` lists every crate and declares every dependency version, the crates' own versions and the lints once, in `[workspace.dependencies]` and `[workspace.lints]`. Each crate takes them with `workspace = true`. The edition is 2024, and the minimum Rust version is 1.92, which the Temporal Rust SDK needs.
- **Checks.** CI runs `rustfmt`, clippy with the lint set of the Microsoft Rust guidelines (the `ms-rust` skill), `cargo-audit` for vulnerable dependencies and `cargo-udeps` for unused ones.
- **Build.** Cargo builds and tests the Rust crates. Pants has no supported Rust backend, so Pants only wraps the Cargo builds to produce the Docker images, and Python targets stay as they are. `.sqlx/` holds the prepared query data, so builds without a database still check queries.

## Database Layout

- One PostgreSQL cluster in development. The catalogue and Temporal use separate databases in it (`echora`, `temporal` and `temporal_visibility`).
- In production, Temporal's databases get a cluster of their own, or at least their own connection budget. Temporal writes and vacuums heavily, and its spikes must not slow the catalogue's queries or trip the 2-second graph timeout.
- The users service gets its own database when it is built, so catalogue and user data never share an owner.
- Tables are described in [[Database Schema|Database Schema]].

## Related Documentation

- [[Database Schema|Database Schema]] — Tables and queries
- [[event_driven_architecture|Event-Driven Architecture]] — The outbox and NATS events
- [[event_schema_specification|Event Schema Specification]] — Events and gRPC messages
- [[update_service|Update Service]] — The workflows that write through this service
- [[temporal_infrastructure|Temporal Infrastructure]] — Temporal, which shares the cluster

---

**Status**: Planned | **Last Updated**: 2026-11-14
