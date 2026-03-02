---
title: Event Schema Specification
date: 2026-02-24
tags:
  - protobuf
  - events
  - schema
  - api
status: active
related:
  - "[[event_driven_architecture]]"
  - "[[postgres_integration_architecture_decision]]"
---

# Event Schema Specification

## Overview

Complete Protocol Buffers (protobuf) event schemas for Echora event-driven architecture, based on the enrichment pipeline's `AnimeRecord` structure.

> [!info] Schema
> **Syntax**: proto3 | **Package**: `echora.events.v1`

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

Qdrant receives this thin signal and calls `PostgresReadService.GetAnimeRecord` via gRPC to get the full `AnimeRecord` (anime + all characters + all episodes). It then runs `MultiVectorEmbeddingManager.process_anime_vectors` to create all three point types in one pass — anime point, character points, episode points.

### EpisodeAiredEvent

Published by **Update Service** when a new episode is verified as aired. The workflow that confirms the episode also checks the current anime status from the source — if it changed, `status` is included so PG can update both episode and status in a single write.

```protobuf
message EpisodeAiredEvent {
  string event_id = 1;                           // UUID
  google.protobuf.Timestamp timestamp = 2;
  string anime_id = 3;                           // UUID
  string aired_at = 4;                           // ISO 8601 actual air datetime
  EpisodeData episode = 5;
  optional AnimeStatus status = 6;               // set only if status changed at time of episode confirmation — e.g. ONGOING → FINISHED on last episode
}
```

**Subject**: `episode.aired` | **Publisher**: Update Service | **Consumer**: PostgreSQL Service

> [!note] Status absent = status unchanged
> If `status` is not set, PG keeps the existing anime status as-is. This avoids waiting up to 7 days for the weekly poll to detect a FINISHED transition when the last episode airs.

### AnimeUpdatedEvent

Published by **Update Service** when any anime field changes. Carries only the changed values — `changed_fields` is the authoritative list of what PG should write.

```protobuf
message AnimeUpdatedEvent {
  string event_id = 1;                           // UUID
  google.protobuf.Timestamp timestamp = 2;
  string anime_id = 3;                           // UUID
  repeated string changed_fields = 4;            // authoritative list of changed fields — e.g. ["score", "broadcast", "status", "synopsis"]
  AnimeData anime = 5;                           // sparse — only fields listed in changed_fields are populated
}
```

**Subject**: `anime.updated` | **Publisher**: Update Service | **Consumer**: PostgreSQL Service only

> [!note] Sparse payload + `changed_fields`
> `changed_fields` tells PG exactly which columns to update — the consumer generates a targeted UPDATE for only those columns. Fields absent from `changed_fields` are not touched in PG regardless of what value they carry in the payload.

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
  map<string, google.protobuf.Value> metadata_updates = 5; // new values for non-embedding fields — omitted when any embedding field changed
}
```

**Subject**: `anime.synced` | **Publisher**: PostgreSQL Service (outbox worker) | **Consumer**: Qdrant Service

Qdrant inspects `changed_fields` to decide the update path:
- `changed_fields` ∩ `EMBEDDING_FIELDS` non-empty → call `GetAnime` via gRPC, re-embed, `update_single_point_vector` + `update_payload(mode="merge")` — `metadata_updates` is omitted, `GetAnime` response provides all values
- `changed_fields` all metadata → read `metadata_updates` directly, `update_payload(mode="merge")` — no gRPC call, no re-embed

`EMBEDDING_FIELDS = {"synopsis", "title", "tags", "genres", "demographics"}` — defined in the Qdrant Service, not the event.

### EpisodeUpdatedEvent

Published by **Update Service** when metadata on an existing episode changes — title, synopsis, images, or other fields that were missing or incorrect at initial ingest.

```protobuf
message EpisodeUpdatedEvent {
  string event_id = 1;                           // UUID
  google.protobuf.Timestamp timestamp = 2;
  string episode_id = 3;                         // UUID
  repeated string changed_fields = 4;            // authoritative list of changed fields — e.g. ["title", "synopsis"]
  EpisodeData episode = 5;                       // sparse — only fields listed in changed_fields are populated
}
```

**Subject**: `episode.updated` | **Publisher**: Update Service | **Consumer**: PostgreSQL Service only

> [!important] Qdrant does NOT consume this event
> Qdrant syncs only after PostgreSQL commits, via `episode.synced`. This enforces the SAGA pattern — PostgreSQL is always the source of truth.

### EpisodeSyncedEvent

Published by **PostgreSQL Service** (via outbox) after successfully committing any episode update. This is the event Qdrant consumes to keep episode points in sync.

```protobuf
message EpisodeSyncedEvent {
  string event_id = 1;                           // UUID
  google.protobuf.Timestamp timestamp = 2;
  string episode_id = 3;                         // UUID
  repeated string changed_fields = 4;            // e.g. ["title", "synopsis"]
  map<string, google.protobuf.Value> metadata_updates = 5; // new values for non-embedding fields — omitted when any embedding field changed
}
```

**Subject**: `episode.synced` | **Publisher**: PostgreSQL Service (outbox worker) | **Consumer**: Qdrant Service

Same update path logic as `AnimeSyncedEvent` — Qdrant inspects `changed_fields` using `EPISODE_EMBEDDING_FIELDS = {"title", "synopsis"}`.

### CharacterUpdatedEvent

Published by **Update Service** when character data changes — description, images, traits, or other fields filled in after initial ingest. Characters can appear in multiple anime so there is no single `anime_id`.

```protobuf
message CharacterUpdatedEvent {
  string event_id = 1;                           // UUID
  google.protobuf.Timestamp timestamp = 2;
  string character_id = 3;                       // UUID
  repeated string changed_fields = 4;            // authoritative list of changed fields — e.g. ["description", "images", "character_traits"]
  CharacterData character = 5;                   // sparse — only fields listed in changed_fields are populated
}
```

**Subject**: `character.updated` | **Publisher**: Update Service | **Consumer**: PostgreSQL Service only

> [!important] Qdrant does NOT consume this event
> Qdrant syncs only after PostgreSQL commits, via `character.synced`. This enforces the SAGA pattern — PostgreSQL is always the source of truth.

### CharacterSyncedEvent

Published by **PostgreSQL Service** (via outbox) after successfully committing any character update. This is the event Qdrant consumes to keep character points in sync.

```protobuf
message CharacterSyncedEvent {
  string event_id = 1;                           // UUID
  google.protobuf.Timestamp timestamp = 2;
  string character_id = 3;                       // UUID
  repeated string changed_fields = 4;            // e.g. ["description", "character_traits"]
  map<string, google.protobuf.Value> metadata_updates = 5; // new values for non-embedding fields — omitted when any embedding field changed
}
```

**Subject**: `character.synced` | **Publisher**: PostgreSQL Service (outbox worker) | **Consumer**: Qdrant Service

Same update path logic as `AnimeSyncedEvent` — Qdrant inspects `changed_fields` using `CHARACTER_EMBEDDING_FIELDS = {"name", "description", "character_traits"}`.

## Notification Events

Events in the `NOTIFICATION_EVENTS` stream — separate from the data pipeline. Data is always committed to PostgreSQL before any notification fires. Consumers are a future notification service.

> [!note] `anime_title` is denormalised
> Notification events carry `anime_title` directly so the notification service can render a push notification without querying PostgreSQL.

### EpisodeUpcomingNotification

Published by **Update Service** (Temporal workflow timer) at T-24h and T-1h before the expected episode air time. The workflow ID is `episode-notification-{anime_id}-{episode_number}` — deterministic so it can be cancelled and rescheduled if the air time changes (hiatus, delay).

```protobuf
message EpisodeUpcomingNotification {
  string event_id = 1;
  google.protobuf.Timestamp timestamp = 2;
  string anime_id = 3;
  int32 episode_number = 4;                      // expected next episode
  google.protobuf.Timestamp expected_air_at = 5;
  string window = 6;                             // "T-24h" or "T-1h"
  string anime_title = 7;
}
```

**Subject**: `notification.episode.upcoming` | **Publisher**: Update Service (Temporal) | **Consumer**: Notification Service (future)

### EpisodeAiredNotification

Published by **PostgreSQL Service outbox worker** after an episode row is committed to the database.

```protobuf
message EpisodeAiredNotification {
  string event_id = 1;
  google.protobuf.Timestamp timestamp = 2;
  string anime_id = 3;
  int32 episode_number = 4;
  google.protobuf.Timestamp aired_at = 5;
  string anime_title = 6;
}
```

**Subject**: `notification.episode.aired` | **Publisher**: PostgreSQL Service (outbox worker) | **Consumer**: Notification Service (future)

### AnimeStatusChangedNotification

Published by **PostgreSQL Service outbox worker** after a status change is committed to the database.

```protobuf
message AnimeStatusChangedNotification {
  string event_id = 1;
  google.protobuf.Timestamp timestamp = 2;
  string anime_id = 3;
  AnimeStatus old_status = 4;
  AnimeStatus new_status = 5;
  string anime_title = 6;
}
```

**Subject**: `notification.anime.status.changed` | **Publisher**: PostgreSQL Service (outbox worker) | **Consumer**: Notification Service (future)

## Core Data Models

> [!info] Defined in shared proto — do not duplicate
> All domain types (`Anime`, `Character`, `Episode`, and every supporting message and enum) are already defined in `protos/shared_proto/v1/anime.proto` (package `shared_proto.v1`). When creating `protos/events/v1/events.proto`, **import** that package rather than redefining any types here.

> [!note] File location decided
> Events proto lives at `protos/events/v1/events.proto` — a **separate file** from `shared_proto/v1/anime.proto`. `anime.proto` is the domain model; `events.proto` is the event contract. They are distinct layers. Events import the domain, not the other way around.

```protobuf
// protos/events/v1/events.proto  (decided location — file not yet created)
syntax = "proto3";
package echora.events.v1;

import "google/protobuf/timestamp.proto";
import "google/protobuf/struct.proto";    // provides google.protobuf.Value for metadata_updates maps
import "shared_proto/v1/anime.proto";     // provides Anime, Character, Episode, AnimeRecord, …
```

### Type Map

| Event field type | Shared proto type | Pydantic model |
|---|---|---|
| anime payload | `shared_proto.v1.Anime` | `Anime` |
| character list | `shared_proto.v1.Character` | `Character` |
| episode list | `shared_proto.v1.Episode` | `Episode` |
| full record | `shared_proto.v1.AnimeRecord` | `AnimeRecord` |
| enrichment info | `shared_proto.v1.EnrichmentMetadata` | `EnrichmentMetadata` |

`AnimeRecord` (Anime + repeated Character + repeated Episode) is the ingest shape — used by `AnimeEnrichedEvent` (ingestion → PG) and `GetAnimeRecord` (initial Qdrant index). It is never used for update reads — updates use the targeted `GetAnime`, `GetCharacter`, `GetEpisode` RPCs to avoid pulling characters and episodes unnecessarily.

> [!note] Role normalisation
> `Character.role` (field 11) is typed as `CharacterRole` in both the proto and the Pydantic model. A `field_validator(mode="before")` on `Character` normalises all upstream strings at ingest time. The full verified vocabulary across all sources is documented in [[anime_relationship_and_format_type_mappings#Character Role Mappings]]. The `ProcessedCharacter` internal dataclass (enrichment only) keeps `role: str` intentionally — it is normalised on the way out when building the final `AnimeRecord`.

## gRPC Services

### PostgresWriteService

```protobuf
service PostgresWriteService {
  rpc IngestAnime(AnimeEnrichedEvent) returns (IngestResponse);
  rpc UpdateAnime(UpdateAnimeRequest) returns (UpdateResponse);
  rpc DeleteAnime(DeleteAnimeRequest) returns (DeleteResponse);
}
```

### PostgresReadService

Used by Qdrant Service for targeted reads. Always returns the current committed state from PostgreSQL.

```protobuf
service PostgresReadService {
  rpc GetAnimeRecord(GetAnimeRequest) returns (AnimeRecord);    // initial index only — full record with characters and episodes
  rpc GetAnime(GetAnimeRequest) returns (Anime);                // anime entity only — used for anime field updates
  rpc GetCharacter(GetCharacterRequest) returns (Character);    // character entity — used for character field updates
  rpc GetEpisode(GetEpisodeRequest) returns (Episode);          // episode entity — used for episode field updates
}

message GetAnimeRequest {
  string anime_id = 1;
}

message GetCharacterRequest {
  string character_id = 1;
}

message GetEpisodeRequest {
  string episode_id = 1;
}
```

`GetAnimeRecord` is called once per anime at initial index (`AnimeCreatedEvent`). `GetAnime`, `GetCharacter`, `GetEpisode` are called only when embedding fields change — never for metadata-only updates.

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
