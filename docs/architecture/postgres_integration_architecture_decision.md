---
title: PostgreSQL Integration Architecture Decision
date: 2026-02-16
tags:
  - architecture
  - postgresql
  - rust
  - decision
status: active
related:
  - "[[event_driven_architecture]]"
  - "[[event_schema_specification]]"
---

# PostgreSQL Integration Architecture Decision

**Status:** Active
**Date:** 2026-02-16
**Context:** Deciding where PostgreSQL integration should live given Rust backend + Python agent service

> [!success] Final Decision
> **PostgreSQL Service with Dual APIs (Rust)**
>
> See [[event_driven_architecture|Event-Driven Architecture]] for complete implementation details including NATS setup and event schemas.

---

## Current Architecture

**Confirmed Decisions:**
- ✅ **Backend:** Rust (separate repo, `axum` + `async-graphql` + `sqlx`) - talks directly to PostgreSQL
- ✅ **Agent Service:** Python (separate repo, being extracted in PR) - needs PostgreSQL for graph queries
- ✅ **Qdrant Service:** Python (separate repo, being extracted in PR)
- ✅ **Ingestion Pipeline:** Separate (being extracted in PR)

**Reference Documents:**
- `docs/plans/2026-02-10-postgres-qdrant-data-model-design.md` - Schema design (reference only, not final)
- `docs/implementation_plan.md` - Backend tech stack: `axum` + `async-graphql` (NOT Juniper)

---

## The Decision

**How should Agent Service (Python) access PostgreSQL for graph traversal queries?**

Both agent service and backend need PostgreSQL access:
- **Agent service**: Graph traversal queries during orchestration (returns IDs)
- **Backend**: Entity hydration for GraphQL responses (returns full objects)

---

## PostgreSQL Service with Dual APIs

### Architecture

```
PostgreSQL Database
  ↑
  |
PostgreSQL Service (Rust)
  ├→ gRPC API (for Agent Service)
  └→ GraphQL API (for Backend)
    ↑                    ↑
    |                    |
Agent Service        Backend
(gRPC client)       (GraphQL client)
```

### Implementation

**Postgres Service repo (`postgres_service`):**
```
postgres_service/
├── src/
│   ├── models/              # Rust sqlx models (single source of truth)
│   ├── repositories/        # Graph query + hydration functions
│   ├── grpc_api/            # tonic gRPC service
│   │   └── graph_service.rs # Graph traversal endpoints
│   ├── graphql_api/         # async-graphql schema
│   │   └── schema.rs        # Entity hydration queries
│   └── main.rs
├── migrations/              # sqlx migrations (only place)
└── proto/                   # gRPC definitions
```

**Agent Service repo:**
- Includes gRPC client to postgres_service
- No SQLAlchemy models needed

**Backend repo:**
- Includes GraphQL client to postgres_service
- No sqlx models needed (just types for responses)

### Pros
- ✅ **Single source of truth**: One schema implementation (Rust sqlx)
- ✅ **No duplication**: Agent and Backend don't manage schemas
- ✅ **Consistent Rust stack**: Both postgres_service and backend use sqlx
- ✅ **Single migration path**: Only sqlx migrations to manage
- ✅ **Graph logic in one place**: No duplication of complex queries
- ✅ **Type safety**: Compile-time checks for all SQL queries

### Cons
- ❌ Additional network hop for all DB queries (latency)
- ❌ More services to manage (4 total: agent, qdrant, postgres, backend)
- ❌ Need to design/maintain two APIs (gRPC + GraphQL)
- ❌ More complex deployment

### When This Makes Sense
- Schema evolves frequently
- Want guaranteed consistency between agent and backend
- Willing to accept latency tradeoff for consistency
- All other services already being extracted anyway

---

---

## Final Recommendation: PostgreSQL Service

**Decision:** Create separate `postgres_service` (Rust) with dual APIs

### Rationale

Based on research + your requirements:

1. **Backend WILL write to anime tables** - view counts, user-driven updates
2. **Multiple services writing to same tables** - clear anti-pattern (research consensus)
3. **Latency is negligible** - Extra gRPC hop adds ~2-3ms (0.1% of total request time)
4. **Research strongly favors separation** - when multiple services write to same data
5. **Future-proof** - No schema duplication, single source of truth
6. **Saga pattern coordination** - PostgreSQL service owns transaction logic

### Architecture

```
                    NATS JetStream (events)
                            ↓
                    Ingestion Pipeline
                            ↓
                    PostgreSQL Service (Rust)
                    ├─ Owns anime data schema
                    ├─ Saga coordination
                    ├─ Event publishing
                    ├─ gRPC API (graph queries)
                    └─ GraphQL API (mutations + queries)
                        ↑                    ↑
                        |                    |
                Agent Service            Backend
                (gRPC client)         (GraphQL client)
                                            ↑
                                    Qdrant Sync Service
```

### Benefits

✅ **Single source of truth** - One Rust schema implementation
✅ **No schema duplication** - No Rust + Python sync needed
✅ **Write coordination** - All writes validated in one place
✅ **Saga pattern** - Transaction rollback centralized
✅ **Event publishing** - Consistent event stream to Qdrant
✅ **Type safety** - Compile-time SQL checks (sqlx)
✅ **Performance** - ~2-3ms overhead (negligible vs LLM 500-2000ms)

### Trade-offs

⚠️ **One more service** - 5 total (agent, backend, qdrant, postgres, ingestion)
⚠️ **Network hop** - Extra 2-3ms latency (acceptable)
⚠️ **Dual APIs** - Need to maintain gRPC + GraphQL (manageable)

---

## Two Distinct API Surfaces

PostgreSQL Service exposes two fundamentally different query categories. These must not be confused — they have different callers, different SQL patterns, and different safety requirements.

### 1. Regular Queries (CRUD + Hydration)

Standard SQL — indexed lookups, INSERT/UPDATE, entity fetching. No recursion.

| Surface | Caller | Purpose |
|---------|--------|---------|
| `PostgresWriteService` (gRPC) | Ingestion Pipeline, Update Service | Receive NATS events (`AnimeEnrichedEvent`, `EpisodeAiredEvent`, `AnimeUpdatedEvent`); write to DB atomically + queue `qdrant_outbox` |
| **GraphQL API** | Backend/BFF | Entity hydration — fetch full `Anime`, `Character`, `Episode` objects by UUID; batched DataLoader pattern for N+1 prevention |
| `search` lane (fast lookup) | Backend/BFF | Deterministic entity lookup by name for typeahead; Postgres-only, no agent involved |

### 2. Graph Queries (Relationship Traversal)

`PostgresGraphService` (gRPC) — called **exclusively by the Agent Service** during `search_ai` (natural-language) queries. This is PostgreSQL's replacement for Neo4j.

These are implemented as recursive CTEs. They are fundamentally different from regular queries: they traverse a relationship graph rather than fetching pre-indexed rows.

**Four bounded primitives:**

| Primitive | Purpose | Implementation |
|-----------|---------|----------------|
| `neighbors` | One-hop adjacency from a node | Simple JOIN, no recursion |
| `k_hop` | Multi-hop expansion with depth + fanout caps | `WITH RECURSIVE` |
| `path` | Explanation chain between two nodes | `WITH RECURSIVE`, terminates at `$end` |
| `compare` | Structured diff between two entities | Multi-query (scalars + shared chars + shared studios) |

**Edge vocabulary is an explicit allowlist** — not free-form strings from the agent:

| Edge type | Source | Table |
|-----------|--------|-------|
| `SEQUEL`, `PREQUEL`, `SIDE_STORY`, `ALTERNATIVE`, `SUMMARY`, `FULL_STORY`, `CHARACTER`, `SPIN_OFF`, `ADAPTATION`, `OTHER` | Direct relation | `anime_relation` |
| `same_studio` | Derived (join) | `anime_company` where `role = 'studio'` |
| `co_appears` | Derived (join) | `anime_character` |
| `adapted_from` | Derived (join) | `anime_original_work_relation` |

The `PostgresGraphExecutor` (Python, in Agent Service) routes each `GraphIntent` to the correct parameterized SQL template based on `query_type` and `edge_types`. The agent never emits raw SQL.

**Safety invariants enforced server-side regardless of caller input:**
- All SQL is parameterized — no string interpolation from agent output
- Hard caps: `max_hops ≤ 5`, `max_results ≤ 50`, `max_fanout_per_hop ≤ 50`
- PostgreSQL `statement_timeout` per query (2 seconds) as the hard safety backstop
- Cycle prevention built into every recursive CTE via `visited` array check
- Zero-results from traversal is a valid business outcome, not an error

> [!note] Full agent contract
> See [[AGENTIC_RAG_SPEC]] for complete `GraphIntent`, `GraphResult`, orchestrator loop, and `PostgresGraphExecutor` plug-in contract.

> [!note] CTE implementations
> See [[Database Schema#Graph Query Primitives]] for the full SQL for all four primitives.

---

---

## Rust Tech Stack (Research-Backed Recommendations)

### PostgreSQL Service Stack

Based on extensive research of Rust ecosystem (2024-2026):

#### Core Database Layer

**SQLx (Recommended)** - 16.5k+ GitHub stars, industry standard for async Rust
```toml
[dependencies]
sqlx = { version = "0.8", features = ["runtime-tokio", "tls-rustls", "postgres", "macros", "migrate", "chrono", "uuid"] }
```

**Why SQLx over Diesel:**
- ✅ Truly async from ground up (Diesel async is bolt-on)
- ✅ Compile-time checked SQL queries without DSL
- ✅ Write real SQL (better for complex graph queries)
- ✅ Lighter weight, faster compile times
- ✅ Industry consensus for microservices (2024-2026)
- ✅ Excellent for connection pooling
- ⚠️ Research shows: "SQLx is not an ORM" - perfect for our use case

**Migration Tool:**
```bash
cargo install sqlx-cli --no-default-features --features postgres
```

#### GraphQL API Layer

**async-graphql (Recommended)** - 7.2k+ on crates.io, most popular Rust GraphQL
```toml
[dependencies]
async-graphql = "7.2"
async-graphql-axum = "7.2"  # Axum integration
```

**Features:**
- Full async/await support
- Type-safe schema generation from Rust structs
- Compile-time validation
- Built-in DataLoader (N+1 query prevention)
- Subscriptions support (if needed later)

#### gRPC API Layer

**tonic (De facto standard)** - Official Rust gRPC
```toml
[dependencies]
tonic = "0.14"
prost = "0.13"  # Protocol Buffers

[build-dependencies]
tonic-build = "0.14"  # Code generation
```

**Features:**
- Built on hyper + tokio
- Async, performant
- Code generation from `.proto` files
- Excellent error handling

#### Web Framework

**axum (Recommended)** - From tokio team, modern async framework
```toml
[dependencies]
axum = "0.8"
tower = "0.5"  # Middleware
tower-http = "0.6"  # HTTP middleware (CORS, tracing, etc.)
```

**Why axum:**
- Official tokio team project
- Best async performance
- Great ergonomics with extractors
- Excellent for both GraphQL and gRPC

#### Async Runtime

**tokio (Industry Standard)**
```toml
[dependencies]
tokio = { version = "1", features = ["full"] }
```

#### Serialization & Validation

```toml
[dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"
validator = "0.18"  # Input validation
```

#### Observability

```toml
[dependencies]
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
```

#### Error Handling

```toml
[dependencies]
thiserror = "2.0"  # Error types
anyhow = "1"  # Error context
```

#### Date/Time & UUIDs

```toml
[dependencies]
chrono = { version = "0.4", features = ["serde"] }
uuid = { version = "1", features = ["v4", "serde"] }
```

### Complete `Cargo.toml` Example

```toml
[package]
name = "postgres-service"
version = "0.1.0"
edition = "2021"

[dependencies]
# Database
sqlx = { version = "0.8", features = ["runtime-tokio", "tls-rustls", "postgres", "macros", "migrate", "chrono", "uuid"] }

# GraphQL
async-graphql = "7.2"
async-graphql-axum = "7.2"

# gRPC
tonic = "0.14"
prost = "0.13"

# Web Framework
axum = "0.8"
tower = "0.5"
tower-http = { version = "0.6", features = ["cors", "trace"] }

# Async Runtime
tokio = { version = "1", features = ["full"] }

# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# Utilities
chrono = { version = "0.4", features = ["serde"] }
uuid = { version = "1", features = ["v4", "serde"] }
thiserror = "2.0"
anyhow = "1"
validator = "0.18"

# Observability
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

[build-dependencies]
tonic-build = "0.14"

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
```

### Project Structure

```
postgres_service/
├── Cargo.toml
├── build.rs                    # tonic build script
├── proto/
│   └── graph_service.proto     # gRPC definitions
├── migrations/
│   └── 20260216_create_anime.sql
├── src/
│   ├── main.rs                 # Entry point
│   ├── config.rs               # Configuration
│   ├── db/
│   │   ├── mod.rs
│   │   ├── pool.rs             # Connection pool
│   │   └── models.rs           # SQLx models
│   ├── graphql/
│   │   ├── mod.rs
│   │   ├── schema.rs           # GraphQL schema
│   │   ├── query.rs            # Query resolvers
│   │   └── mutation.rs         # Mutation resolvers
│   ├── grpc/
│   │   ├── mod.rs
│   │   └── graph_service.rs    # gRPC implementation
│   └── repositories/
│       ├── mod.rs
│       ├── anime.rs            # Anime queries
│       ├── character.rs        # Character queries
│       └── graph.rs            # Graph traversal
└── tests/
    └── integration/
```

---

## Implementation Roadmap

### Phase 1: PostgreSQL Service Foundation (Week 1-2)

1. **Repository Setup**
   ```bash
   cargo new postgres-service
   cd postgres-service
   ```

2. **Database Schema Migration**
   - Port schema from `docs/plans/2026-02-10-postgres-qdrant-data-model-design.md`
   - Create initial migrations using `sqlx migrate`
   ```bash
   sqlx migrate add create_anime_tables
   ```

3. **SQLx Models**
   - Define Rust structs matching schema
   - Use `#[derive(sqlx::FromRow)]` for query mapping

4. **Connection Pool**
   - Configure `PgPool` with proper sizing
   - Health check endpoint

### Phase 2: gRPC API (Week 2-3)

1. **Protobuf Definitions**
   - Define graph query messages
   - Graph traversal operations (neighbors, k-hop, etc.)

2. **gRPC Service Implementation**
   - Implement graph traversal queries
   - Error handling and validation

3. **Integration Testing**
   - Test from Python agent service

### Phase 3: GraphQL API (Week 3-4)

1. **GraphQL Schema**
   - Define types (Anime, Character, Episode, etc.)
   - Query resolvers (by ID, search, etc.)
   - Mutation resolvers (update view counts, etc.)

2. **DataLoader Integration**
   - Prevent N+1 queries
   - Batch loading for relationships

3. **Integration Testing**
   - Test from Rust backend

### Phase 4: Saga Pattern & Events (Week 4-5)

1. **Transaction Coordination**
   - Implement saga pattern for multi-step operations
   - Rollback logic

2. **Event Publishing**
   - Integrate NATS JetStream client (`async-nats` crate)
   - Publish events on successful writes
   - Event schema definitions

3. **Qdrant Sync**
   - Event consumer service
   - Update Qdrant on anime data changes

### Phase 5: Production Readiness (Week 5-6)

1. **Observability**
   - Structured logging with tracing
   - Metrics (Prometheus)
   - Health checks

2. **Error Handling**
   - Comprehensive error types
   - Client-friendly error messages

3. **Performance Optimization**
   - Query optimization
   - Connection pool tuning
   - Caching strategy

4. **Documentation**
   - API documentation (GraphQL introspection + gRPC reflection)
   - README with setup instructions

---

## Next Steps

1. **Create `postgres_service` repository**
2. **Set up Rust project with recommended stack**
3. **Port schema from design doc to sqlx migrations**
4. **Implement gRPC API first** (for agent service)
5. **Implement GraphQL API** (for backend)
6. **Wire into existing services**
