---
title: Event Schema Specification
date: 2026-02-16
tags:
  - protobuf
  - events
  - schema
  - api
status: active
version: 1.1.0
related:
  - "[[event_driven_architecture]]"
  - "[[postgres_integration_architecture_decision]]"
---

# Event Schema Specification

## Overview

Complete Protocol Buffers (protobuf) event schemas for Echora event-driven architecture, based on the enrichment pipeline's `AnimeRecord` structure.

> [!info] Schema Version
> **Current**: 1.0.0 | **Syntax**: proto3 | **Package**: `echora.events.v1`

## Top-Level Events

### AnimeEnrichedEvent

Published by **Ingestion Pipeline** after all 5 enrichment stages (initial full enrichment only).

```protobuf
message AnimeEnrichedEvent {
  string event_id = 1;                           // UUID
  google.protobuf.Timestamp timestamp = 2;
  string anime_id = 3;                           // Anime UUID
  AnimeData anime = 4;
  repeated CharacterData characters = 5;
  repeated EpisodeData episodes = 6;
  EnrichmentMetadata enrichment_metadata = 7;
}
```

**Subject**: `anime.enriched` | **Publisher**: Ingestion Pipeline | **Consumer**: PostgreSQL Service

### AnimeCreatedEvent

Published by **PostgreSQL Service** after successful persistence.

```protobuf
message AnimeCreatedEvent {
  string event_id = 1;
  google.protobuf.Timestamp timestamp = 2;
  string anime_id = 3;
}
```

**Subject**: `anime.created` | **Publisher**: PostgreSQL Service | **Consumer**: Qdrant Service

### EpisodeAiredEvent

Published by **Update Worker** when a new episode is verified as aired.

```protobuf
message EpisodeAiredEvent {
  string event_id = 1;                           // UUID
  google.protobuf.Timestamp timestamp = 2;
  string anime_id = 3;                           // UUID
  string source = 4;                             // "jikan", "anilist"
  string aired_at = 5;                           // ISO 8601 actual air datetime
  EpisodeData episode = 6;
}
```

**Subject**: `anime.episode.aired` | **Publisher**: Update Worker | **Consumer**: PostgreSQL Service

### AnimeUpdatedEvent

Published by **Update Worker** when score or status changes. Contains only the changed fields.

```protobuf
message AnimeUpdatedEvent {
  string event_id = 1;                           // UUID
  google.protobuf.Timestamp timestamp = 2;
  string anime_id = 3;                           // UUID
  string source = 4;                             // "mal", "anilist", "kitsu"
  repeated string changed_fields = 5;            // e.g. ["score", "statistics"]
  optional string status = 6;                    // if status changed
  optional ScoreCalculations score = 7;          // if score changed
  map<string, Statistics> statistics = 8;   // per-platform stats
}
```

**Subject**: `anime.updated` | **Publisher**: Update Worker | **Consumer**: PostgreSQL Service only

> [!important] Qdrant does NOT consume this event
> Qdrant syncs only after PostgreSQL commits, via `anime.synced` (see below). This enforces the SAGA pattern — PostgreSQL is always the source of truth.

### AnimeSyncedEvent

Published by **PostgreSQL Service** (via outbox) after successfully committing any update. This is the event Qdrant consumes to stay in sync.

```protobuf
message AnimeSyncedEvent {
  string event_id = 1;                           // UUID
  google.protobuf.Timestamp timestamp = 2;
  string anime_id = 3;                           // UUID
  repeated string changed_fields = 4;            // e.g. ["score", "episode_count", "status"]
  SyncUpdateType update_type = 5;
}

enum SyncUpdateType {
  PAYLOAD_ONLY = 0;         // score, status, episode_count — no re-embed needed
  PAYLOAD_AND_VECTOR = 1;   // synopsis, tags, demographics — full re-embed required
}
```

**Subject**: `anime.synced` | **Publisher**: PostgreSQL Service (outbox worker) | **Consumer**: Qdrant Service

Qdrant decides what to update based on `changed_fields` and `update_type`:
- `PAYLOAD_ONLY` → update Qdrant point payload directly (fast, no embedding)
- `PAYLOAD_AND_VECTOR` → fetch full anime data from PostgreSQL Service (GraphQL), re-embed, update both payload and vector

## Core Data Models

See [[event_driven_architecture#Event Flow]] for complete protobuf definitions including:
- `AnimeData` (mirrors Pydantic Anime model - 24 scalar, 14 array, 10 object fields)
- `CharacterData` (mirrors Pydantic Character model)
- `EpisodeData` (mirrors Pydantic Episode model — see below)
- All supporting types (ThemeSong, Statistics, StaffData, etc.)

### EpisodeData

Mirrors `libs/common/src/common/models/anime.py::Episode`.

```protobuf
message EpisodeData {
  int32  episode_number  = 1;
  string title           = 2;
  string title_japanese  = 3;
  string title_romaji    = 4;
  string synopsis        = 5;
  int32  duration        = 6;                    // seconds
  bool   filler          = 7;
  bool   recap           = 8;
  float  score           = 9;
  string aired           = 10;                   // ISO 8601
  repeated string thumbnails          = 11;
  map<string, string> episode_pages   = 12;
  map<string, string> streaming       = 13;
}
```

### ScoreCalculations

Mirrors `libs/common/src/common/models/anime.py::ScoreCalculations`.

```protobuf
message ScoreCalculations {
  float arithmetic_geometric_mean = 1;  // arithmetic-geometric mean across platforms
  float arithmetic_mean           = 2;
  float median                    = 3;
}
```

## gRPC Services

### PostgresWriteService

```protobuf
service PostgresWriteService {
  rpc IngestAnime(AnimeEnrichedEvent) returns (IngestResponse);
  rpc UpdateAnime(UpdateAnimeRequest) returns (UpdateResponse);
  rpc DeleteAnime(DeleteAnimeRequest) returns (DeleteResponse);
}
```

### PostgresGraphService

```protobuf
service PostgresGraphService {
  rpc FindRelatedAnime(RelatedAnimeQuery) returns (RelatedAnimeResponse);
  rpc TraverseRelationshipGraph(GraphTraversalQuery) returns (GraphTraversalResponse);
}
```

## Schema Evolution

### Canonical Source of Truth

**`.proto` files are the single source of truth** for all data shapes. `libs/common/src/common/models/anime.py` and Rust structs in `postgres_service` are **generated outputs** — never hand-edited.

```
echora-contracts repo
  protos/echora/v1/anime.proto    ← only place you ever edit data shapes
          ↓  buf generate
          ├── Python (betterproto)
          │   → libs/common/models/anime.py   (generated, DO NOT EDIT)
          └── Rust (prost)
              → postgres_service/src/models/  (generated, DO NOT EDIT)
```

This eliminates the model drift problem: if the `.proto` changes, all consumers regenerate automatically. If a consumer doesn't regenerate, it won't compile.

### Toolchain

| Tool | Role |
|------|------|
| `buf` | Proto linting, breaking change detection, code generation |
| `buf generate` | Produces Python + Rust types from `.proto` |
| `buf breaking` | CI check — fails PR if breaking change introduced |
| `buf lint` | Enforces naming conventions in proto files |
| `buf push` | Publishes versioned proto schema to buf.build registry |
| `betterproto` | buf plugin — generates Pydantic-compatible Python dataclasses |
| `prost` + `tonic` | buf plugin — generates Rust structs + gRPC stubs |

### Contracts Repository

Proto files live in a **separate `echora-contracts` repo** — not owned by any single service. This keeps the contract neutral: any team can submit a PR, breaking change detection runs in one place.

```
echora-contracts/
  protos/
    echora/v1/
      anime.proto       ← Anime, Character, Episode, enums
      events.proto      ← AnimeEnrichedEvent, AnimeSyncedEvent, etc.
      services.proto    ← gRPC service definitions
  buf.yaml
  buf.gen.yaml
```

**CI on merge to main:**

```
buf push → buf.build/echora/contracts (versioned proto registry)

buf generate (Python)
  → publish `echora-types` Python package → GitHub Packages (PyPI)

buf generate (Rust)
  → publish `echora-types` Rust crate → GitHub Packages (crates.io compatible)
```

### Distribution — How Each Service Gets Types

Each service consumes the generated package as a normal dependency. No service ever clones the contracts repo or runs `buf generate` itself.

**Python services** (`echora`, `update_worker`, `qdrant_service`, `ingestion_pipeline`):
```toml
# pyproject.toml
[dependencies]
echora-types = "1.2.0"   # from GitHub Packages
```

**Rust services** (`postgres_service`, `backend`):
```toml
# Cargo.toml
[dependencies]
echora-types = { version = "1.2.0", registry = "github-packages" }
```

To update to a new schema version: `uv lock --upgrade-package echora-types` or `cargo update echora-types`. A major version bump (breaking change) will fail the build explicitly — forcing a conscious update.

### PostgreSQL DDL — The One Manual Step

PostgreSQL migration files (`sqlx migrate`) cannot be auto-generated from proto. However, the compile-time chain catches missing migrations before runtime:

```
proto field added → Rust struct regenerated → sqlx FromRow requires DB column
                                            → cargo sqlx prepare fails
                                            → CI build fails
                                            → developer knows a migration is needed
```

**Result:** Adding a field without a migration is a compile error, not a silent runtime bug.

### Qdrant Payload Index

Qdrant payload indexing (which fields are filterable/sortable) is declared in a small config file in the contracts repo:

```yaml
# qdrant_payload_index.yaml
indexed_fields:
  - name: status           type: keyword
  - name: year             type: integer
  - name: score            type: float
  - name: genres           type: keyword[]
  - name: entity_type      type: keyword
  - name: episode_count    type: integer
```

CI validates that every field listed here exists in `anime.proto`. If you remove a proto field that's still in this YAML, the contracts repo CI fails — catching Qdrant drift before any code ships.

### Change Type Rules

**Adding an optional field** ✅ safe — do this:
1. Add to `anime.proto` as `optional` with a new field number
2. Run `buf generate` → Python + Rust types update automatically
3. DB migration: `ALTER TABLE ... ADD COLUMN ... NULL`
4. Old NATS events in flight: missing field → protobuf default → null (safe)
5. Old Qdrant points: missing payload field → null (safe, no re-index needed)

**Removing a field** ⚠️ breaking — follow this sequence:
1. Mark the field number as `reserved` in the proto — **never reuse field numbers**
2. `buf breaking` will flag this as a major version bump
3. Keep DB column until all historical events are processed, then drop
4. Old Qdrant points retain the stale payload field until next full re-index
5. Bump major version in the published package

**Renaming or changing a field type** ❌ always breaking — use add + deprecate:
1. Add new field (new name/type, new field number) alongside old one
2. Dual-write both in producing services during transition
3. Once all consumers read the new field, remove the old one (follow removal rules)

### Versioning

**Versioning**: Major.Minor.Patch (semantic versioning) on the published package

| Bump | When |
|------|------|
| **Patch** | Docs/comment changes, no schema change |
| **Minor** | Add optional fields (backward compatible) |
| **Major** | Remove field, change type, rename — any breaking change |

`buf breaking --against buf.build/echora/contracts:latest` runs on every PR to the contracts repo and automatically enforces this.

## Related Documentation

- [[event_driven_architecture|Event-Driven Architecture]] - NATS setup, consumer config, event flow
- [[postgres_integration_architecture_decision|PostgreSQL Architecture]] - Service design and rationale
- [[Architecture Index|Architecture Index]] - Services overview
- [buf.build documentation](https://buf.build/docs)
- [betterproto (Python codegen)](https://github.com/danielgtaylor/python-betterproto)
- [prost (Rust codegen)](https://docs.rs/prost/)
- [Protocol Buffers field number rules](https://protobuf.dev/programming-guides/proto3/#reserved)

---

**Version**: 1.2.0 | **Updated**: 2026-02-17
