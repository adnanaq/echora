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

Published by **Update Service** when a new episode is verified as aired.

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

**Subject**: `anime.episode.aired` | **Publisher**: Update Service | **Consumer**: PostgreSQL Service

### AnimeUpdatedEvent

Published by **Update Service** when score or status changes. Contains only the changed fields.

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

**Subject**: `anime.updated` | **Publisher**: Update Service | **Consumer**: PostgreSQL Service only

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
  repeated string images              = 11;
  repeated string     sources          = 12;
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

> [!note] Full Details in [[echora_contracts|Schema Ownership & Contracts]]
> See [[echora_contracts]] for the complete schema ownership model, toolchain, CI pipeline, distribution approach, and change type rules. The key rules are summarised below.

### Canonical Source of Truth

`libs/common/src/common/models/anime.py` (hand-written Pydantic) is the source of truth.
`protos/shared_proto/v1/anime.proto` is a manually maintained mirror — a PR touching
`anime.py` must also update `anime.proto`. Generated stubs (`_pb2.py`, `_pb2.pyi`,
`_pb2_grpc.py`) are checked into `libs/common/src/shared_proto/v1/` and regenerated
by `scripts/generate-proto.py`.

### Change Type Rules

**Adding an optional field** ✅ safe:
1. Add field to `anime.py` and `anime.proto` in the same PR (assign a new field number)
2. Run `scripts/generate-proto.py` to regenerate stubs
3. DB migration: `ALTER TABLE ... ADD COLUMN ... NULL`
4. Old NATS events in flight: missing field → protobuf default → null (safe)
5. Old Qdrant points: missing payload field → null (safe, no re-index needed)

**Removing a field** ⚠️ breaking:
1. Remove from `anime.py` and `anime.proto` in the same PR
2. Mark the field number as `reserved` in the proto — **never reuse field numbers**
3. Update all consumers before or in the same PR

**Renaming or changing a field type** ❌ always breaking — add new, deprecate old:
1. Add new field (new name/type, new field number) alongside old in both files
2. Dual-write during transition if consumers are external
3. Remove old field once all consumers are migrated (follow removal rules)

### PostgreSQL DDL

Migration files cannot be auto-generated from proto. Drift is caught at compile time:

```
anime.py field added → proto updated → stubs regenerated → sqlx FromRow needs DB column
                                                         → cargo sqlx prepare fails
                                                         → CI build fails
```

## Related Documentation

- [[echora_contracts|Schema Ownership & Contracts]] - toolchain, CI pipeline, distribution, change type rules
- [[event_driven_architecture|Event-Driven Architecture]] - NATS setup, consumer config, event flow
- [[postgres_integration_architecture_decision|PostgreSQL Architecture]] - Service design and rationale
- [[Architecture Index|Architecture Index]] - Services overview
- [Protocol Buffers field number rules](https://protobuf.dev/programming-guides/proto3/#reserved)

---

**Version**: 1.2.0 | **Updated**: 2026-02-17
