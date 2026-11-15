---
title: Event Schema Specification
date: 2026-02-24
tags:
  - protobuf
  - events
  - schema
  - api
status: planned
related:
  - "[[event_driven_architecture]]"
  - "[[update_service]]"
  - "[[postgres_integration_architecture_decision]]"
  - "[[Database Schema]]"
---

# Event Schema Specification

## Overview

This page defines the messages the services exchange:

- **Events** on NATS, which the PostgreSQL Service outbox sends after a commit. Every catalogue event is a thin notice holding an ID and a version, never the changed data.
- **gRPC methods** that Temporal activities, event consumers and the backend call. The data itself always travels over gRPC.

The domain types (`Anime`, `Episode`, `Character` and their enums) are defined once in `protos/shared_proto/v1/anime.proto`. Events and services import them and never redefine them.

| File | Package | Status |
|---|---|---|
| `protos/shared_proto/v1/anime.proto` | `shared_proto.v1` | Exists |
| `protos/shared_proto/v1/merge.proto` | `shared_proto.v1` | Planned |
| `protos/enrichment_service/v1/enrichment_service.proto` | `enrichment_service.v1` | Exists, new methods planned |
| `protos/events/v1/events.proto` | `events.v1` | Planned |
| `protos/postgres_service/v1/postgres_service.proto` | `postgres_service.v1` | Planned |
| `protos/agent_service/v1/agent_service.proto` | `agent_service.v1` | Planned |

## Catalogue Events

> **Linear:** ECHO-90, ECHO-98, ECHO-103, ECHO-104

Works, episodes and characters are separate entities, each with its own refresh, so each has its own created, updated and deleted events. All of them use one message, and the subject says which entity and which change. They go to the `ANIME_EVENTS` stream. See [[event_driven_architecture#Thin Events|Thin Events]] for why they carry no data.

```protobuf
// protos/events/v1/events.proto
syntax = "proto3";

package events.v1;

import "google/protobuf/timestamp.proto";
import "shared_proto/v1/anime.proto";

// A work, an episode or a character was created, changed or removed.
message CatalogueEvent {
  string event_id = 1;                       // outbox row ID, also the Nats-Msg-Id
  google.protobuf.Timestamp saved_at = 2;
  string entity_id = 3;                      // the anime, episode or character ID
  int64 version = 4;                         // the entity's version after this change
  optional string anime_id = 5;              // the episode's work, set on episode events only
}
```

| Subject | Sent when | Consumers |
|---|---|---|
| `anime.created` | A work is saved for the first time | Vector Service, later "recently added" and new-anime notifications |
| `anime.updated` | An existing work is saved with changes | Vector Service |
| `anime.deleted` | A work is removed, such as when merged into another | Vector Service |
| `episode.created` | An episode is saved for the first time | Vector Service |
| `episode.updated` | An existing episode is saved with changes | Vector Service |
| `episode.deleted` | An episode is removed | Vector Service |
| `character.created` | A character is saved for the first time | Vector Service |
| `character.updated` | An existing character is saved with changes | Vector Service |
| `character.deleted` | A character is removed, such as an orphan or a merged duplicate | Vector Service |

- **Version.** Each save that changes an entity increases its `version` in the same transaction. A consumer stores the version it last handled and skips any event whose version is not newer.
- **Deletes.** A removed entity keeps its row with `deleted_at` set, and its `version` increases once more, so the delete event's version is newer than every update before it. A consumer handles a delete without reading the entity back.
- **Removing an episode.** When an episode is removed and `episode.deleted` is sent, for example when a work's merge no longer lists it, is not decided yet. Decided in ECHO-103.

## Notification Events

> **Linear:** ECHO-90, ECHO-98, ECHO-103, Notification Service project

Notification events go to the `NOTIFICATION_EVENTS` stream, from the same outbox, in the same transaction as the change they report. Unlike catalogue events, they carry what a notification shows, so the notification service can send a message without reading PostgreSQL. Each one is sent with a `Nats-TTL`, so a notification the service could not deliver in time expires.

```protobuf
// An episode was saved as aired for the first time.
message EpisodeAiredNotification {
  string event_id = 1;
  google.protobuf.Timestamp saved_at = 2;
  string anime_id = 3;
  string anime_title = 4;
  string episode_id = 5;
  int32 episode_number = 6;
  google.protobuf.Timestamp aired_at = 7;
}

// A work changed airing status.
message AnimeStatusChangedNotification {
  string event_id = 1;
  google.protobuf.Timestamp saved_at = 2;
  string anime_id = 3;
  string anime_title = 4;
  shared_proto.v1.AnimeStatus previous_status = 5;
  shared_proto.v1.AnimeStatus current_status = 6;
}
```

| Subject | Message | Consumers |
|---|---|---|
| `notification.episode.aired` | `EpisodeAiredNotification` | Notification Service (planned) |
| `notification.anime.status.changed` | `AnimeStatusChangedNotification` | Notification Service (planned) |

Reminders before an episode airs are not catalogue changes and are not sent through the outbox. Their sender is decided with the notification service.

## gRPC Services

Every method called from a Temporal activity is safe to run twice, because Temporal retries a failed activity. See [[update_service#How a Refresh Runs|How a Refresh Runs]] for the order of the calls.

### Shared messages (planned)

> **Linear:** ECHO-97

Several services exchange provider pages, corrections, generated text and entity references, the inputs and outputs of a merge, so these messages live in `shared_proto.v1`. No service proto imports another service's proto.

```protobuf
// protos/shared_proto/v1/merge.proto
syntax = "proto3";

package shared_proto.v1;

import "google/protobuf/any.proto";
import "google/protobuf/timestamp.proto";
import "shared_proto/v1/anime.proto";

enum PageKind {
  PAGE_KIND_UNSPECIFIED = 0;
  PAGE_KIND_ANIME = 1;
  PAGE_KIND_EPISODE = 2;
  PAGE_KIND_CHARACTER = 3;
}

// A provider page, named by provider, kind and the page's address on that provider.
message PageKey {
  string provider = 1;                       // 'mal', 'anilist', ...
  PageKind page_kind = 2;
  string address = 3;                        // e.g. '21'
}

// One response of a page, exactly as the provider sent it.
message PageResponse {
  string request_name = 1;                   // e.g. 'main', 'characters', 'episodes_page_2'
  string content_type = 2;                   // 'application/json', 'text/html', 'application/xml'
  bytes body = 3;
}

// A provider page as fetched and as PostgreSQL stores it. Nothing is mapped yet.
message StoredPage {
  PageKey page_key = 1;
  google.protobuf.Timestamp fetched_at = 2;
  bool is_dead = 3;                          // the provider no longer serves the page
  repeated PageResponse responses = 4;
}

// An approved change to one field, applied after every merge.
message FieldCorrection {
  string field_path = 1;                     // e.g. 'episode_count', 'broadcast.jp_time'
  google.protobuf.Any value = 2;             // the value as its own protobuf message
  string approved_by = 3;
  google.protobuf.Timestamp approved_at = 4;
}

// Text the Agent Service wrote for a field, such as the synopsis.
message GeneratedText {
  string field_path = 1;
  string text = 2;
  google.protobuf.Timestamp written_at = 3;
}

// An episode a work lists, with its ID once assigned.
message EpisodeReference {
  string episode_id = 1;                     // existing ID, or a new UUIDv7 from the merge
  int32 episode_number = 2;
  string title = 3;
  repeated PageKey page_keys = 4;
}

// A character a work lists, with its ID once assigned.
message CharacterReference {
  string character_id = 1;                   // existing ID, or a new UUIDv7 from the merge
  string name = 2;
  CharacterRole role = 3;
  int32 billing_order = 4;
  repeated PageKey page_keys = 5;
}
```

`PageKey` holds the same three parts as the page key string `libs/work_matching` uses, `provider:kind:id` such as `mal:anime:21`, with `id` named `address`. Manga pages (`mal:manga:13`) are not fetched as provider pages; they identify source material through `source_material_xref`.

`FieldCorrection.value` is a `google.protobuf.Any`: the type URL names the value's message and the bytes hold it. PostgreSQL stores the two as `value_message` and `value` (see [[Database Schema#Approval, Corrections, AI Text and Outbox|Database Schema]]).

### EnrichmentService (planned methods)

> **Linear:** ECHO-89, ECHO-94, ECHO-99, ECHO-105, ECHO-109, ECHO-110

The enrichment service owns everything enrichment does. It holds no data between calls. A caller passes in what a method needs and stores what it returns.

```protobuf
service EnrichmentService {
  rpc Health(HealthRequest) returns (HealthResponse);                    // exists
  rpc RunPipeline(RunPipelineRequest) returns (RunPipelineResponse);    // exists, replaced by the methods below
  // Fetch one provider page, as received. Nothing is mapped.
  rpc FetchPage(FetchPageRequest) returns (FetchPageResponse);
  // Map a work's stored pages, merge them, then apply corrections and AI text.
  rpc MergeWork(MergeWorkRequest) returns (MergeWorkResponse);
  rpc MergeEpisode(MergeEpisodeRequest) returns (MergeEpisodeResponse);
  rpc MergeCharacter(MergeCharacterRequest) returns (MergeCharacterResponse);
}

message FetchPageRequest {
  shared_proto.v1.PageKey page_key = 1;
}

message FetchPageResponse {
  shared_proto.v1.StoredPage page = 1;
}

message MergeWorkRequest {
  string anime_id = 1;                                        // the work's existing ID
  repeated shared_proto.v1.StoredPage source_pages = 2;
  repeated shared_proto.v1.PageKey partial_source_pages = 3;  // linked, not merged
  repeated shared_proto.v1.FieldCorrection corrections = 4;
  repeated shared_proto.v1.GeneratedText generated_texts = 5;
  repeated shared_proto.v1.EpisodeReference known_episodes = 6;
  repeated shared_proto.v1.CharacterReference known_characters = 7;
}

message MergeWorkResponse {
  shared_proto.v1.Anime anime = 1;
  repeated shared_proto.v1.EpisodeReference episodes = 2;     // every episode the pages list
  repeated shared_proto.v1.CharacterReference characters = 3; // every character the pages list
}

message MergeEpisodeRequest {
  string episode_id = 1;
  string anime_id = 2;
  repeated shared_proto.v1.StoredPage pages = 3;
  repeated shared_proto.v1.FieldCorrection corrections = 4;
}

message MergeEpisodeResponse {
  shared_proto.v1.Episode episode = 1;
}

message MergeCharacterRequest {
  string character_id = 1;
  repeated shared_proto.v1.StoredPage pages = 2;
  repeated shared_proto.v1.FieldCorrection corrections = 3;
}

message MergeCharacterResponse {
  shared_proto.v1.Character character = 1;
}
```

- **Mapping happens in the merge.** `MergeWork` runs each provider's mapper on the stored responses, merges the results with `merge_provider_records` from `libs/enrichment`, then applies the corrections and the generated text. Nothing mapped is stored.
- **IDs are assigned at the merge.** The work's ID is passed in. An episode or character the pages list keeps its ID when `known_episodes` or `known_characters` already has it, matched by episode number or page key. One the work has never listed gets a new UUIDv7.
- **The merge keeps no provider synopsis.** `Anime.synopsis` comes from a correction, else the latest generated text, else stays empty.
- The work's `sources` and `partial_sources` come from the page links, so the merge no longer reads the offline database entry.

### AgentService (planned methods)

> **Linear:** ECHO-91, ECHO-96

The agent service holds every AI call. It stores nothing. A caller passes in the sources and stores the result.

```protobuf
service AgentService {
  // Write a work's synopsis from its provider synopses and a web search.
  rpc WriteSynopsis(WriteSynopsisRequest) returns (WriteSynopsisResponse);
}

message WriteSynopsisRequest {
  string anime_id = 1;
  string title = 2;
  shared_proto.v1.AnimeType type = 3;
  optional int32 year = 4;
  shared_proto.v1.AnimeStatus status = 5;
  repeated ProviderSynopsis provider_synopses = 6;
}

message ProviderSynopsis {
  shared_proto.v1.PageKey page_key = 1;
  string synopsis = 2;
}

message WriteSynopsisResponse {
  string synopsis = 1;
  repeated string source_urls = 2;           // web pages used
  string model_name = 3;
  string prompt_version = 4;
}
```

A method for extracting character attributes follows the same shape and is defined when that work starts.

### PostgresService (planned)

> **Linear:** ECHO-97, ECHO-98, ECHO-103, ECHO-104, ECHO-107, ECHO-108, ECHO-115

The PostgreSQL Service is the only owner of the database. It stores and returns data and never calls another service.

```protobuf
// protos/postgres_service/v1/postgres_service.proto
service PostgresService {
  // Pages
  rpc StorePage(StorePageRequest) returns (StorePageResponse);

  // Works
  rpc LoadWork(LoadWorkRequest) returns (LoadWorkResponse);
  rpc SaveWork(SaveWorkRequest) returns (SaveWorkResponse);

  // Episodes and characters
  rpc LoadEpisode(LoadEpisodeRequest) returns (LoadEpisodeResponse);
  rpc SaveEpisode(SaveEpisodeRequest) returns (SaveEntityResponse);
  rpc LoadCharacter(LoadCharacterRequest) returns (LoadCharacterResponse);
  rpc SaveCharacter(SaveCharacterRequest) returns (SaveEntityResponse);

  // Seed load and review answers
  rpc FindEntitiesByPages(FindEntitiesByPagesRequest) returns (FindEntitiesByPagesResponse);
  rpc CreateWorks(CreateWorksRequest) returns (CreateWorksResponse);
  rpc RelinkPages(RelinkPagesRequest) returns (RelinkPagesResponse);
  rpc DeleteWork(DeleteWorkRequest) returns (SaveEntityResponse);

  // What is due
  rpc ListDueWorks(ListDueRequest) returns (ListDueResponse);
  rpc ListDueEpisodes(ListDueRequest) returns (ListDueResponse);
  rpc ListDueCharacters(ListDueRequest) returns (ListDueResponse);
  rpc ListUpcomingEpisodes(ListUpcomingEpisodesRequest) returns (ListUpcomingEpisodesResponse);
  rpc RecordOpen(RecordOpenRequest) returns (RecordOpenResponse);

  // AI-written text
  rpc LoadSynopsisSources(LoadSynopsisSourcesRequest) returns (LoadSynopsisSourcesResponse);
  rpc SaveGeneratedText(SaveGeneratedTextRequest) returns (SaveGeneratedTextResponse);

  // Reads, for event consumers and the backend
  rpc GetAnime(GetEntityRequest) returns (shared_proto.v1.Anime);
  rpc GetEpisode(GetEntityRequest) returns (shared_proto.v1.Episode);
  rpc ListEpisodes(GetEntityRequest) returns (ListEpisodesResponse);
  rpc GetCharacter(GetEntityRequest) returns (shared_proto.v1.Character);
}

message StorePageRequest {
  shared_proto.v1.StoredPage page = 1;
}

message StorePageResponse {
  bool has_changed = 1;                      // false when the responses matched, or the fetch was older
}

message LoadWorkRequest {
  string anime_id = 1;
}

message LoadWorkResponse {
  int64 version = 1;                         // passed back to SaveWork as expected_version
  repeated shared_proto.v1.StoredPage source_pages = 2;
  repeated shared_proto.v1.PageKey partial_source_pages = 3;
  repeated shared_proto.v1.FieldCorrection corrections = 4;
  repeated shared_proto.v1.GeneratedText generated_texts = 5;  // the latest per field
  repeated shared_proto.v1.EpisodeReference known_episodes = 6;
  repeated shared_proto.v1.CharacterReference known_characters = 7;
}

message SaveWorkRequest {
  string anime_id = 1;
  int64 expected_version = 2;
  shared_proto.v1.Anime anime = 3;
  repeated shared_proto.v1.EpisodeReference episodes = 4;
  repeated shared_proto.v1.CharacterReference characters = 5;
}

message SaveWorkResponse {
  bool has_changed = 1;
  int64 version = 2;
  repeated string created_episode_ids = 3;   // episodes to refresh next
  repeated string created_character_ids = 4; // characters to refresh next
}

message LoadEpisodeRequest {
  string episode_id = 1;
}

message LoadEpisodeResponse {
  int64 version = 1;
  string anime_id = 2;
  repeated shared_proto.v1.StoredPage pages = 3;
  repeated shared_proto.v1.FieldCorrection corrections = 4;
}

message SaveEpisodeRequest {
  string episode_id = 1;
  int64 expected_version = 2;
  shared_proto.v1.Episode episode = 3;
}

message LoadCharacterRequest {
  string character_id = 1;
}

message LoadCharacterResponse {
  int64 version = 1;
  repeated shared_proto.v1.StoredPage pages = 2;
  repeated shared_proto.v1.FieldCorrection corrections = 3;
}

message SaveCharacterRequest {
  string character_id = 1;
  int64 expected_version = 2;
  shared_proto.v1.Character character = 3;
}

message SaveEntityResponse {
  bool has_changed = 1;
  int64 version = 2;
}

message FindEntitiesByPagesRequest {
  repeated shared_proto.v1.PageKey page_keys = 1;
}

message FindEntitiesByPagesResponse {
  repeated PageOwner page_owners = 1;        // only the pages already linked
}

message PageOwner {
  shared_proto.v1.PageKey page_key = 1;
  string entity_id = 2;
  bool is_partial_source = 3;
}

message CreateWorksRequest {
  repeated NewWork works = 1;
}

message NewWork {
  string anime_id = 1;                       // a new UUIDv7, assigned by the seed load
  repeated shared_proto.v1.PageKey source_pages = 2;
  repeated shared_proto.v1.PageKey partial_source_pages = 3;
}

message CreateWorksResponse {
  repeated string created_anime_ids = 1;     // works already created by an earlier attempt are left out
}

message RelinkPagesRequest {
  repeated PageMove moves = 1;
}

message PageMove {
  shared_proto.v1.PageKey page_key = 1;
  string from_anime_id = 2;
  string to_anime_id = 3;
  bool is_partial_source = 4;
}

message RelinkPagesResponse {
  repeated string affected_anime_ids = 1;    // works to merge again
}

message DeleteWorkRequest {
  string anime_id = 1;
  int64 expected_version = 2;
  optional string merged_into_anime_id = 3;
}

message ListDueRequest {
  google.protobuf.Timestamp refreshed_before = 1;  // the sweep's fixed cutoff
  int32 older_after_years = 2;               // entities older than this refresh only when opened
  int32 page_size = 3;
  optional google.protobuf.Timestamp after_refreshed_at = 4;  // keyset: the last row of the previous page
  optional string after_id = 5;
}

message ListDueResponse {
  repeated DueEntity entities = 1;
}

message DueEntity {
  string entity_id = 1;
  optional google.protobuf.Timestamp refreshed_at = 2;
}

message ListUpcomingEpisodesRequest {
  int32 window_days = 1;
}

message ListUpcomingEpisodesResponse {
  repeated UpcomingEpisode episodes = 1;
}

message UpcomingEpisode {
  string anime_id = 1;
  optional string episode_id = 2;            // unset when the work has not listed it yet
  int32 episode_number = 3;
  google.protobuf.Timestamp expected_air_at = 4;
}

message RecordOpenRequest {
  shared_proto.v1.PageKind entity_kind = 1;
  string entity_id = 2;
  int32 refresh_after_days = 3;
}

message RecordOpenResponse {
  bool is_due = 1;
}

message LoadSynopsisSourcesRequest {
  string anime_id = 1;
}

message LoadSynopsisSourcesResponse {
  agent_service.v1.WriteSynopsisRequest request = 1;  // ready to send to WriteSynopsis
  string input_hash = 2;                     // hash of the provider synopses
}

message SaveGeneratedTextRequest {
  string anime_id = 1;
  string field_path = 2;                     // 'synopsis'
  agent_service.v1.WriteSynopsisResponse written = 3;
  string input_hash = 4;
}

message SaveGeneratedTextResponse {
  bool has_changed = 1;
}

message GetEntityRequest {
  string entity_id = 1;
}

message ListEpisodesResponse {
  repeated shared_proto.v1.Episode episodes = 1;
}
```

- **Saves check the version.** `SaveWork`, `SaveEpisode`, `SaveCharacter` and `DeleteWork` write only when the stored version still equals `expected_version`. Otherwise they return `ABORTED`, and Temporal retries the activity from its load. A save that changes nothing writes nothing and sends nothing, and only records that the refresh finished.
- **`SaveWork` creates what the work lists.** An episode or character the merge gave a new ID is created with its page links, and the response names it so the workflow can start its refresh. If another work's save created the same character first, the page link conflicts, the save returns `ABORTED`, and the retried merge finds the existing ID.
- **`StorePage`** stores a page and its responses only when its `fetched_at` is newer than the stored one, so a retried or late fetch never replaces newer data.
- **`CreateWorks` and `RelinkPages`** resolve every unresolved relation that points at a newly linked source page, in the same transaction.
- **`ListDueWorks`, `ListDueEpisodes` and `ListDueCharacters`** page by `(refreshed_at, id)` from a fixed cutoff, so a work that fails to refresh is not picked again in the same sweep.
- **`LoadSynopsisSources`** and **`SaveGeneratedText`** serve the `SynopsisWorkflow`. A synopsis whose input hash already matches is not written again.
- **The PostgreSQL proto imports the Agent Service's messages** for those two methods. If that dependency is unwanted, `WriteSynopsisRequest` and `WriteSynopsisResponse` move into `shared_proto.v1`.
- Approval-queue writes and the backend's reads are defined when those features are designed.

## Schema Evolution

> [!note] Full details in [[echora_contracts|Schema Ownership & Contracts]]
> The toolchain, CI checks and distribution are described there. The rules below cover what changes in a message.

`libs/common/src/common/models/anime.py` is the source of truth for the domain types. `protos/shared_proto/v1/anime.proto` mirrors it, and a change to one changes the other in the same PR. `scripts/check_anime_model_proto_contract.py` checks that they match.

Before any service runs in production:

- **Removing a field** deletes it and renumbers the message's fields from 1 without gaps. No numbers or names are reserved.
- **Adding or renaming a field** changes the model and the proto together, then regenerates the stubs.

Once services exchange these messages across versions in production:

- **Adding an optional field** is safe. An old message without it reads as the default.
- **Removing a field** reserves its number and name, which are never reused.
- **Renaming a field or changing its type** adds a new field beside the old one, and removes the old one once every consumer has moved.

Thin events make this rarely matter for events. A domain field added to `Anime` changes no event message, since events carry only IDs and versions.

Stored corrections and approval values are protobuf messages too, so the same rules keep old stored values readable.

PostgreSQL migrations are written by hand. A model field without a database column is caught at build time, when `cargo sqlx prepare` checks the queries against the schema.

## Related Documentation

- [[event_driven_architecture|Event-Driven Architecture]] — Streams, subjects, the outbox and consumers
- [[update_service|Update Service]] — The Temporal workflows that call these methods
- [[Database Schema|Database Schema]] — Where each message is stored
- [[echora_contracts|Schema Ownership & Contracts]] — Toolchain, CI and distribution
- [[postgres_integration_architecture_decision|PostgreSQL Service]] — The PostgreSQL Service

---

**Status**: Planned | **Last Updated**: 2026-11-14
