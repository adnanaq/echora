---
title: Event Schema Specification
date: 2026-11-14
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

| File                                                    | Package                 | Status                                                                                                        |
| ------------------------------------------------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------- |
| `protos/shared_proto/v1/anime.proto`                    | `shared_proto.v1`       | Exists                                                                                                        |
| `protos/shared_proto/v1/merge.proto`                    | `shared_proto.v1`       | ~~Planned (ECHO-97)~~ Exists                                                                                  |
| `protos/enrichment_service/v1/enrichment_service.proto` | `enrichment_service.v1` | Exists, new methods planned, see [[#EnrichmentService (planned methods)]]                                     |
| `protos/events/v1/events.proto`                         | `events.v1`             | ~~Planned (ECHO-90)~~ Exists (`CatalogueEvent`, `AnimeStatusChangedNotification`, `EpisodeAiredNotification`) |
| `protos/postgres_service/v1/postgres_service.proto`     | `postgres_service.v1`   | ~~Planned (ECHO-97)~~ Exists                                                                                  |
| `protos/agent_service/v1/agent_service.proto`           | `agent_service.v1`      | Planned (ECHO-91)                                                                                             |

## Catalogue Events

Works, episodes and characters are separate entities, each with its own refresh, so each has its own created, updated and deleted events. All of them use one message, and the subject says which entity and which change. They go to the `ANIME_EVENTS` stream. See [[event_driven_architecture#Thin Events|Thin Events]] for why they carry no data. (ECHO-90)

```protobuf
// protos/events/v1/events.proto
syntax = "proto3";

package events.v1;

import "google/protobuf/timestamp.proto";
import "shared_proto/v1/anime.proto";

// A work, an episode or a character was created, changed or removed.
message CatalogueEvent {  // (ECHO-90)
  string event_id = 1;                       // outbox row ID, also the Nats-Msg-Id
  google.protobuf.Timestamp saved_at = 2;
  string entity_id = 3;                      // the anime, episode or character ID
  int64 version = 4;                         // the entity's version after this change
  optional string anime_id = 5;              // the episode's work, set on episode events only
}
```

| Subject             | Sent when                                           | Consumers                                                                    |
| ------------------- | --------------------------------------------------- | ---------------------------------------------------------------------------- |
| ~~`anime.created`~~ | ~~A work is saved for the first time~~ | ~~Vector Service, later "recently added" and new-anime notifications (ECHO-98)~~ |
| ~~`anime.updated`~~ | ~~An existing work is saved with changes~~ | ~~Vector Service (ECHO-98)~~ |
| `anime.deleted`     | A work is removed, such as when merged into another | Vector Service (ECHO-107)                                                    |
| ~~`episode.created`~~ | ~~An episode is saved for the first time~~ | ~~Vector Service (ECHO-103)~~ |
| ~~`episode.updated`~~ | ~~An existing episode is saved with changes~~ | ~~Vector Service (ECHO-103)~~ |
| ~~`episode.deleted`~~ | ~~An episode is removed~~ | ~~Vector Service (ECHO-103)~~ |
| `character.created` | A character is saved for the first time             | Vector Service (ECHO-104)                                                    |
| `character.updated` | An existing character is saved with changes         | Vector Service (ECHO-104)                                                    |
| `character.deleted` | A character is removed as a merged duplicate        | Vector Service (ECHO-104)                                                    |
| `character.deleted` | A character no work links to any more is removed    | Vector Service (ECHO-116)                                                    |

- ~~**Version.** Each save that changes an entity increases its `version` in the same transaction. A consumer stores the version it last handled and skips any event whose version is not newer. (ECHO-98)~~
- **Deletes.** A removed entity keeps its row with `deleted_at` set, and its `version` increases once more, so the delete event's version is newer than every update before it.
  - `DeleteAnime` removes works this way. (ECHO-107)
  - ~~Episodes are removed this way. (ECHO-103)~~
  - Characters are removed this way. (ECHO-104)
  - A consumer reads a deleted entity back like any other, and `Get*` answers `NOT_FOUND`, which confirms the delete. (ECHO-114)
- **Removing an episode.** When an episode is removed and `episode.deleted` is sent, for example when a work's merge no longer lists it, is not decided yet. (ECHO-103) (no issue yet)

## Notification Events

Notification events carry what a notification shows, so the notification service can send a message without reading PostgreSQL. They come from the same outbox as catalogue events.

- They go to the `NOTIFICATION_EVENTS` stream, which allows a per-message expiry. (ECHO-90)
- ~~The status notification is written by the work's save, in the same transaction as the status change. (ECHO-98)~~
- ~~The aired notification is written by `MarkEpisodeAired`, once, when episode tracking confirms the episode aired. (ECHO-103)~~
- The relay sends each one with a `Nats-TTL`, so a notification the service could not deliver in time expires. (ECHO-106)

```protobuf
// Episode tracking confirmed an episode aired, through MarkEpisodeAired.
message EpisodeAiredNotification {  // (ECHO-90)
  string event_id = 1;
  google.protobuf.Timestamp saved_at = 2;
  string anime_id = 3;
  string anime_title = 4;
  string episode_id = 5;
  int32 episode_number = 6;
  google.protobuf.Timestamp aired_at = 7;
}

// A work changed airing status.
message AnimeStatusChangedNotification {  // (ECHO-90)
  string event_id = 1;
  google.protobuf.Timestamp saved_at = 2;
  string anime_id = 3;
  string anime_title = 4;
  shared_proto.v1.AnimeStatus previous_status = 5;
  shared_proto.v1.AnimeStatus current_status = 6;
}
```

| Subject                             | Message                          | Consumers                                 |
| ----------------------------------- | -------------------------------- | ----------------------------------------- |
| ~~`notification.episode.aired`~~ | ~~`EpisodeAiredNotification`~~ | ~~Notification Service (planned) (ECHO-103)~~ |
| ~~`notification.anime.status.changed`~~ | ~~`AnimeStatusChangedNotification`~~ | ~~Notification Service (planned) (ECHO-98)~~ |

The Notification Service that reads these subjects is planned. (P-ECHO-18)

Reminders before an episode airs are not catalogue changes and are not sent through the outbox. Their sender is decided with the notification service. (P-ECHO-18)

## gRPC Services

Every method called from a Temporal activity is safe to run twice, because Temporal retries a failed activity. See [[update_service#How a Refresh Runs|How a Refresh Runs]] for the order of the calls.

### Shared messages (planned)

~~Several services exchange provider pages, corrections, generated text and entity references, the inputs and outputs of a merge, so these messages live in `shared_proto.v1`. No service proto imports another service's proto. (ECHO-97)~~

```protobuf
// protos/shared_proto/v1/merge.proto
syntax = "proto3";

package shared_proto.v1;

import "google/protobuf/any.proto";
import "google/protobuf/timestamp.proto";
import "shared_proto/v1/anime.proto";

enum PageKind {  // (ECHO-97)
  PAGE_KIND_UNSPECIFIED = 0;
  PAGE_KIND_ANIME = 1;
  PAGE_KIND_EPISODE = 2;
  PAGE_KIND_CHARACTER = 3;
}

// A provider page, named by provider, kind and the page's address on that provider.
message PageKey {  // (ECHO-97)
  string provider = 1;                       // 'mal', 'anilist', ...
  PageKind page_kind = 2;
  string address = 3;                        // e.g. '21'
}

// One response of a page, exactly as the provider sent it.
message PageResponse {  // (ECHO-97)
  string request_name = 1;                   // e.g. 'main', 'characters', 'episodes_page_2'
  string content_type = 2;                   // 'application/json', 'text/html', 'application/xml'
  bytes body = 3;                            // exactly as the provider sent it; gRPC compresses it in transit
}

// A provider page as fetched. PostgreSQL compresses the bodies at rest; on the wire they are as received.
message StoredPage {  // (ECHO-97)
  PageKey page_key = 1;
  google.protobuf.Timestamp fetched_at = 2;
  bool is_dead = 3;                          // confirmed gone: by a reviewer, or automatically once trusted; StorePage refuses it
  repeated PageResponse responses = 4;
  bool is_not_found = 5;                     // this fetch got the provider's own "not found"; StorePage records it, keeping the old responses
}

// An approved change to one field, applied after every merge.
message FieldCorrection {  // (ECHO-98)
  string field_path = 1;                     // e.g. 'episode_count', 'broadcast.jp_time'
  google.protobuf.Any value = 2;             // the value as its own protobuf message
  string approved_by = 3;
  google.protobuf.Timestamp approved_at = 4;
}

// Text the Agent Service wrote for a field, such as the synopsis.
message GeneratedText {  // (ECHO-115)
  string field_path = 1;
  string text = 2;
  google.protobuf.Timestamp written_at = 3;
}

// An episode a work lists, with its ID once assigned.
message EpisodeReference {  // (ECHO-105)
  string episode_id = 1;                     // existing ID, or a new UUIDv7 from the merge
  int32 episode_number = 2;
  string title = 3;
  repeated PageKey page_keys = 4;
  repeated EpisodeListEntry list_entries = 5;  // what each provider's work page says about it
}

// One provider's facts about an episode, sliced out of the work's page by the work's merge.
// Most providers list episodes only inside the work's responses.
message EpisodeListEntry {  // (ECHO-105)
  string provider = 1;                       // 'mal', 'anilist', ...
  optional string title = 2;
  optional google.protobuf.Timestamp aired_at = 3;
  optional int32 duration_seconds = 4;
  optional bool is_filler = 5;
  optional bool is_recap = 6;
}

// A character a work lists, with its ID once assigned.
message CharacterReference {  // (ECHO-105)
  string character_id = 1;                   // existing ID, or a new UUIDv7 from the merge
  string name = 2;
  CharacterRole role = 3;
  int32 billing_order = 4;
  repeated PageKey page_keys = 5;
}

// One provider's own claim that the work relates to a target page, with the type that
// provider gave, kept as provenance beside the single type the merge picks.
message RelationClaim {  // (ECHO-102)
  string provider = 1;                       // 'mal', 'anilist', ...
  AnimeRelationType relation_type = 2;
  PageKey target_page = 3;
}
```

~~`PageKey` holds the same three parts as the page key string `libs/work_matching` uses, `provider:kind:id` such as `mal:anime:21`, with `id` named `address`. Manga pages (`mal:manga:13`) are not fetched as provider pages; they identify source material through `source_material_xref`. (ECHO-97)~~

~~`FieldCorrection.value` is a `google.protobuf.Any`: the type URL names the value's message and the bytes hold it. PostgreSQL stores the two as `value_message` and `value` (see [[Database Schema#Approval, Corrections, AI Text and Outbox|Database Schema]]). (ECHO-98)~~

### EnrichmentService (planned methods)

The enrichment service owns everything enrichment does. It holds no data between calls. A caller passes in what a method needs and stores what it returns.

```protobuf
import "google/protobuf/duration.proto";

service EnrichmentService {
  rpc Health(HealthRequest) returns (HealthResponse);                    // exists
  rpc RunPipeline(RunPipelineRequest) returns (RunPipelineResponse);    // exists, replaced by the methods below; removed before any shared deployment (ECHO-123)
  // Fetch one provider page, as received. Nothing is mapped.
  rpc FetchPage(FetchPageRequest) returns (FetchPageResponse);  // (ECHO-99)
  // Map a work's stored pages, merge them, then apply corrections and AI text.
  rpc MergeAnime(MergeAnimeRequest) returns (MergeAnimeResponse);  // (ECHO-105)
  rpc MergeEpisode(MergeEpisodeRequest) returns (MergeEpisodeResponse);  // (ECHO-109)
  rpc MergeCharacter(MergeCharacterRequest) returns (MergeCharacterResponse);  // (ECHO-110)
  // Match the cast entries of a work's pages that are not yet linked to a character.
  // Its own call, so a cast of a thousand runs under its own timeout. Messages are defined
  // with the character merge.
  rpc MatchCharacters(MatchCharactersRequest) returns (MatchCharactersResponse);  // (ECHO-110)
  // Find franchise links worth a reviewer's look. The links stream in; the candidates come back.
  // Messages are defined with the knowledge graph project.
  rpc FindFranchiseLinkCandidates(stream FindFranchiseLinkCandidatesRequest) returns (FindFranchiseLinkCandidatesResponse);  // (P-ECHO-22)
}

message FetchPageRequest {  // (ECHO-99)
  shared_proto.v1.PageKey page_key = 1;
  google.protobuf.Duration max_age = 2;      // the response cache answers only with a younger response; 0 always fetches
}

message FetchPageResponse {  // (ECHO-99)
  shared_proto.v1.StoredPage page = 1;
}

message MergeAnimeRequest {  // (ECHO-105)
  string anime_id = 1;                                        // the work's existing ID
  repeated shared_proto.v1.StoredPage source_pages = 2;
  repeated shared_proto.v1.PageKey partial_source_pages = 3;  // linked, not merged
  repeated shared_proto.v1.FieldCorrection corrections = 4;
  repeated shared_proto.v1.GeneratedText generated_texts = 5;
  repeated shared_proto.v1.EpisodeReference known_episodes = 6;
  repeated shared_proto.v1.CharacterReference known_characters = 7;
}

message MergeAnimeResponse {  // (ECHO-105)
  shared_proto.v1.Anime anime = 1;
  repeated shared_proto.v1.EpisodeReference episodes = 2;     // every episode the pages list
  repeated shared_proto.v1.CharacterReference characters = 3; // every character the pages list
  repeated shared_proto.v1.RelationClaim relation_claims = 4;  // each provider's relations, before the merge picks one type (ECHO-105)
}

message MergeEpisodeRequest {  // (ECHO-109)
  string episode_id = 1;
  string anime_id = 2;
  repeated shared_proto.v1.StoredPage pages = 3;
  repeated shared_proto.v1.FieldCorrection corrections = 4;
  repeated shared_proto.v1.EpisodeListEntry list_entries = 5;  // merged with the pages, so an episode with no page of its own still merges
}

message MergeEpisodeResponse {  // (ECHO-109)
  shared_proto.v1.Episode episode = 1;
}

message MergeCharacterRequest {  // (ECHO-110)
  string character_id = 1;
  repeated shared_proto.v1.StoredPage pages = 2;
  repeated shared_proto.v1.FieldCorrection corrections = 3;
}

message MergeCharacterResponse {  // (ECHO-110)
  shared_proto.v1.Character character = 1;
}
```

- **Mapping happens in the merge.** `MergeAnime` runs each provider's mapper on the stored responses, merges the results with `merge_provider_records` from `libs/enrichment`, then applies the corrections and the generated text. (ECHO-105)
  - Nothing mapped is stored, because the fetchers return pages as received. AniList and Kitsu move first. (ECHO-89)
  - MAL, Anime-Planet and aniSearch follow. (ECHO-94)
  - AniDB, AnimeSchedule and Syoboi follow. (ECHO-99)
- **IDs are assigned at the merge.** The work's ID is passed in. An episode or character the pages list keeps its ID when `known_episodes` or `known_characters` already has it, matched by episode number or page key. One the work has never listed gets a new UUIDv7. (ECHO-105)
- **The merge keeps no provider synopsis.** `Anime.synopsis` comes from a correction, else the latest generated text, else stays empty. (ECHO-105)
- The work's `sources` and `partial_sources` come from the page links, so the merge no longer reads the offline database entry. (ECHO-105)

### AgentService (planned methods)

The agent service holds every AI call. It stores nothing. A caller passes in the sources and stores the result. (ECHO-91)

```protobuf
service AgentService {
  // Write a work's synopsis from its provider synopses and a web search.
  rpc WriteSynopsis(shared_proto.v1.WriteSynopsisRequest) returns (shared_proto.v1.WriteSynopsisResponse);  // (ECHO-96)
}

// The two messages below live in shared_proto.v1, since the PostgreSQL Service also uses them,
// and no service proto imports another service's proto.

message WriteSynopsisRequest {  // (ECHO-96)
  string anime_id = 1;
  string title = 2;
  shared_proto.v1.AnimeType type = 3;
  optional int32 year = 4;
  shared_proto.v1.AnimeStatus status = 5;
  repeated ProviderSynopsis provider_synopses = 6;
}

message ProviderSynopsis {  // (ECHO-96)
  shared_proto.v1.PageKey page_key = 1;
  string synopsis = 2;
}

message WriteSynopsisResponse {  // (ECHO-96)
  string synopsis = 1;
  repeated string source_urls = 2;           // web pages used
  string model_name = 3;
  string prompt_version = 4;
}
```

A method for extracting character attributes follows the same shape and is defined when that work starts. (P-ECHO-19)

### PostgresService (planned)

The PostgreSQL Service is the only owner of the database. It stores and returns data and never calls another service.

```protobuf
// protos/postgres_service/v1/postgres_service.proto
service PostgresService {
  // Pages
  rpc StorePage(StorePageRequest) returns (StorePageResponse);  // (ECHO-97)

  // Works
  rpc LoadAnime(LoadAnimeRequest) returns (LoadAnimeResponse);  // (ECHO-98)
  rpc SaveAnime(SaveAnimeRequest) returns (SaveAnimeResponse);  // (ECHO-98)

  // Episodes and characters
  rpc LoadEpisode(LoadEpisodeRequest) returns (LoadEpisodeResponse);  // (ECHO-103)
  rpc SaveEpisode(SaveEpisodeRequest) returns (SaveEpisodeResponse);  // (ECHO-103)
  rpc LoadCharacter(LoadCharacterRequest) returns (LoadCharacterResponse);  // (ECHO-104)
  rpc SaveCharacter(SaveCharacterRequest) returns (SaveCharacterResponse);  // (ECHO-104)

  // Seed load and review answers
  rpc FindEntitiesByPages(FindEntitiesByPagesRequest) returns (FindEntitiesByPagesResponse);  // (ECHO-107)
  rpc CreateAnime(CreateAnimeRequest) returns (CreateAnimeResponse);  // (ECHO-107)
  rpc RelinkPages(RelinkPagesRequest) returns (RelinkPagesResponse);  // (ECHO-107)
  rpc DeleteAnime(DeleteAnimeRequest) returns (DeleteAnimeResponse);  // (ECHO-107)

  // What is due
  rpc ListEntityPages(ListEntityPagesRequest) returns (ListEntityPagesResponse);  // (ECHO-108)
  rpc RecordRefresh(RecordRefreshRequest) returns (RecordRefreshResponse);  // (ECHO-108)
  rpc ListDueAnime(ListDueRequest) returns (ListDueResponse);  // (ECHO-108)
  rpc ListDueEpisodes(ListDueRequest) returns (ListDueResponse);  // (ECHO-108)
  rpc ListDueCharacters(ListDueRequest) returns (ListDueResponse);  // (ECHO-108)
  rpc ListUpcomingEpisodes(ListUpcomingEpisodesRequest) returns (ListUpcomingEpisodesResponse);  // (ECHO-108)
  rpc ListScoreAnime(ListScoreAnimeRequest) returns (ListScoreAnimeResponse);  // (ECHO-108)
  rpc ListRequestedMerges(ListRequestedMergesRequest) returns (ListRequestedMergesResponse);  // (ECHO-108)
  rpc MarkEpisodeAired(MarkEpisodeAiredRequest) returns (MarkEpisodeAiredResponse);  // (ECHO-103)

  // AI-written text
  rpc LoadSynopsisSources(LoadSynopsisSourcesRequest) returns (LoadSynopsisSourcesResponse);  // (ECHO-115)
  rpc SaveGeneratedText(SaveGeneratedTextRequest) returns (SaveGeneratedTextResponse);  // (ECHO-115)

  // Reads, for event consumers and the backend
  rpc GetAnime(GetEntityRequest) returns (GetAnimeResponse);  // (ECHO-108)
  rpc GetEpisode(GetEntityRequest) returns (GetEpisodeResponse);  // (ECHO-108)
  rpc ListEpisodes(GetEntityRequest) returns (ListEpisodesResponse);  // (ECHO-108)
  rpc GetCharacter(GetEntityRequest) returns (GetCharacterResponse);  // (ECHO-108)
  rpc ListEntityVersions(ListEntityVersionsRequest) returns (ListEntityVersionsResponse);  // (ECHO-108)

  // Budgeted providers
  rpc ListDuePages(ListDuePagesRequest) returns (ListDuePagesResponse);  // (ECHO-108)

  // Knowledge graph reads, for the backend and the Agent Service; messages defined with that project
  rpc FindNeighbors(FindNeighborsRequest) returns (GraphNodesResponse);  // (P-ECHO-22)
  rpc FindReachable(FindReachableRequest) returns (GraphNodesResponse);  // (P-ECHO-22)
  rpc FindPath(FindPathRequest) returns (GraphPathsResponse);  // (P-ECHO-22)
  rpc CompareEntities(CompareEntitiesRequest) returns (CompareEntitiesResponse);  // (P-ECHO-22)

  // Franchise link check; messages defined with the knowledge graph project
  rpc ListFranchiseLinks(ListFranchiseLinksRequest) returns (stream ListFranchiseLinksResponse);  // (P-ECHO-22)
  rpc CreateLinkCheckApprovals(CreateLinkCheckApprovalsRequest) returns (CreateLinkCheckApprovalsResponse);  // (P-ECHO-22)

  // Corrections kept in the repository; messages defined with the approval queue
  rpc ExportCorrections(ExportCorrectionsRequest) returns (stream ExportCorrectionsResponse);  // (P-ECHO-15)
  rpc ImportCorrections(stream ImportCorrectionsRequest) returns (ImportCorrectionsResponse);  // (P-ECHO-15)
}

message StorePageRequest {  // (ECHO-97)
  shared_proto.v1.StoredPage page = 1;
}

message StorePageResponse {  // (ECHO-97)
  bool has_changed = 1;                      // false when the responses matched, or the fetch was older
}

message LoadAnimeRequest {  // (ECHO-98)
  string anime_id = 1;
}

message LoadAnimeResponse {  // (ECHO-98)
  int64 version = 1;                         // passed back to SaveAnime as expected_version
  repeated shared_proto.v1.StoredPage source_pages = 2;
  repeated shared_proto.v1.PageKey partial_source_pages = 3;
  repeated shared_proto.v1.FieldCorrection corrections = 4;
  repeated shared_proto.v1.EpisodeReference known_episodes = 5;      // undeleted episodes, so the merge keeps their IDs (ECHO-103)
  repeated shared_proto.v1.CharacterReference known_characters = 6;  // (ECHO-104)
  repeated shared_proto.v1.GeneratedText generated_texts = 7;  // the latest per field (ECHO-115)
}

message SaveAnimeRequest {  // (ECHO-98)
  string anime_id = 1;
  int64 expected_version = 2;
  shared_proto.v1.Anime anime = 3;
  int32 merge_version = 4;                   // the mapping and merge code that produced the save
  repeated shared_proto.v1.RelationClaim relation_claims = 5;  // stored in anime_relation_claim (ECHO-102)
  repeated shared_proto.v1.EpisodeReference episodes = 6;      // (ECHO-103)
  repeated shared_proto.v1.CharacterReference characters = 7;  // (ECHO-104)
}

message SaveAnimeResponse {  // (ECHO-98)
  bool has_changed = 1;
  int64 version = 2;
  repeated string created_episode_ids = 3;   // episodes to refresh next (ECHO-103)
  repeated string updated_episode_ids = 4;   // existing episodes whose list entries changed (ECHO-103)
  repeated string created_character_ids = 5; // characters to refresh next (ECHO-104)
  repeated string updated_character_ids = 6; // existing characters whose cast entry changed (ECHO-104)
}

message LoadEpisodeRequest {  // (ECHO-103)
  string episode_id = 1;
}

message LoadEpisodeResponse {  // (ECHO-103)
  int64 version = 1;
  string anime_id = 2;
  repeated shared_proto.v1.StoredPage pages = 3;
  repeated shared_proto.v1.FieldCorrection corrections = 4;
  repeated shared_proto.v1.EpisodeListEntry list_entries = 5;
}

message SaveEpisodeRequest {  // (ECHO-103)
  string episode_id = 1;
  int64 expected_version = 2;
  shared_proto.v1.Episode episode = 3;
  int32 merge_version = 4;
}

message LoadCharacterRequest {  // (ECHO-104)
  string character_id = 1;
}

message LoadCharacterResponse {  // (ECHO-104)
  int64 version = 1;
  repeated shared_proto.v1.StoredPage pages = 2;
  repeated shared_proto.v1.FieldCorrection corrections = 3;
}

message SaveCharacterRequest {  // (ECHO-104)
  string character_id = 1;
  int64 expected_version = 2;
  shared_proto.v1.Character character = 3;
  int32 merge_version = 4;
}

// SaveEpisode, SaveCharacter and DeleteAnime each have their own response, since buf's
// STANDARD lint (RPC_RESPONSE_STANDARD_NAME, RPC_REQUEST_RESPONSE_UNIQUE) refuses a shared one.
message SaveEpisodeResponse {  // (ECHO-103)
  bool has_changed = 1;
  int64 version = 2;
}

message SaveCharacterResponse {  // (ECHO-104)
  bool has_changed = 1;
  int64 version = 2;
}

message DeleteAnimeResponse {  // (ECHO-107)
  bool has_changed = 1;
  int64 version = 2;
}

message FindEntitiesByPagesRequest {  // (ECHO-107)
  repeated shared_proto.v1.PageKey page_keys = 1;
}

message FindEntitiesByPagesResponse {  // (ECHO-107)
  repeated PageOwner page_owners = 1;        // only the pages already linked
}

message PageOwner {  // (ECHO-107)
  shared_proto.v1.PageKey page_key = 1;
  string entity_id = 2;
  bool is_partial_source = 3;
}

message CreateAnimeRequest {  // (ECHO-107)
  repeated NewAnime works = 1;
}

message NewAnime {  // (ECHO-107)
  string anime_id = 1;                       // a new UUIDv7, assigned by the seed load
  repeated shared_proto.v1.PageKey source_pages = 2;
  repeated shared_proto.v1.PageKey partial_source_pages = 3;
}

message CreateAnimeResponse {  // (ECHO-107)
  repeated string created_anime_ids = 1;     // works already created by an earlier attempt are left out
}

message RelinkPagesRequest {  // (ECHO-107)
  repeated PageMove moves = 1;
}

message PageMove {  // (ECHO-107)
  shared_proto.v1.PageKey page_key = 1;
  string from_anime_id = 2;
  string to_anime_id = 3;
  bool is_partial_source = 4;
}

message RelinkPagesResponse {  // (ECHO-107)
  repeated string affected_anime_ids = 1;    // works to merge again
}

message DeleteAnimeRequest {  // (ECHO-107)
  string anime_id = 1;
  int64 expected_version = 2;
  optional string merged_into_anime_id = 3;
}

message ListDueRequest {  // (ECHO-108)
  google.protobuf.Timestamp due_before = 1;  // the sweep's fixed cutoff, compared with next_refresh_at
  int32 older_after_years = 2;               // entities older than this refresh only when opened
  int32 page_size = 3;
  optional google.protobuf.Timestamp after_next_refresh_at = 4;  // keyset: the last row of the previous page
  optional string after_id = 5;
  int32 merge_version = 6;                   // the current merge code; entities merged by older code are due whatever their age
}

message ListDueResponse {  // (ECHO-108)
  repeated DueEntity entities = 1;
}

message DueEntity {  // (ECHO-108)
  string entity_id = 1;
  optional google.protobuf.Timestamp next_refresh_at = 2;  // unset when due now
}

message ListUpcomingEpisodesRequest {  // (ECHO-108)
  int32 window_days = 1;
}

message ListUpcomingEpisodesResponse {  // (ECHO-108)
  repeated UpcomingEpisode episodes = 1;
}

message UpcomingEpisode {  // (ECHO-108)
  string anime_id = 1;
  optional string episode_id = 2;            // unset when the work has not listed it yet
  int32 episode_number = 3;
  google.protobuf.Timestamp expected_air_at = 4;
}

message ListEntityPagesRequest {  // (ECHO-108)
  shared_proto.v1.PageKind entity_kind = 1;
  string entity_id = 2;
}

message ListEntityPagesResponse {  // (ECHO-108)
  repeated EntityPage pages = 1;
}

message EntityPage {  // (ECHO-108)
  shared_proto.v1.PageKey page_key = 1;
  bool is_partial_source = 2;
  optional google.protobuf.Timestamp fetched_at = 3;       // unset when never fetched
  bool is_dead = 4;
  optional google.protobuf.Timestamp fetch_failed_at = 5;
  int32 fetch_failures = 6;
}

message RecordRefreshRequest {  // (ECHO-108)
  shared_proto.v1.PageKind entity_kind = 1;
  string entity_id = 2;
  RefreshOutcome outcome = 3;
  google.protobuf.Duration refresh_after = 4;  // REFRESH_AFTER_DAYS, from the Update Service's settings
}

enum RefreshOutcome {  // (ECHO-108)
  REFRESH_OUTCOME_UNSPECIFIED = 0;
  REFRESH_OUTCOME_SUCCEEDED = 1;             // every page fetched or not due
  REFRESH_OUTCOME_PAGES_FAILED = 2;          // merged, but some pages failed; due again when the first may retry
  REFRESH_OUTCOME_FAILED = 3;                // the merge or save failed; due again after a backoff
}

message RecordRefreshResponse {}  // (ECHO-108)

message ListScoreAnimeRequest {  // (ECHO-108)
  int32 lookback_days = 1;                   // finished works stay in for this many days after aired_to
  int32 page_size = 2;
  optional string after_id = 3;              // keyset: the last ID of the previous page
}

message ListScoreAnimeResponse {  // (ECHO-108)
  repeated string anime_ids = 1;
}

message ListRequestedMergesRequest {  // (ECHO-108)
  shared_proto.v1.PageKind entity_kind = 1;
  int32 limit = 2;
}

message ListRequestedMergesResponse {  // (ECHO-108)
  repeated string entity_ids = 1;
}

message LoadSynopsisSourcesRequest {  // (ECHO-115)
  string anime_id = 1;
}

message LoadSynopsisSourcesResponse {  // (ECHO-115)
  shared_proto.v1.WriteSynopsisRequest request = 1;  // ready to send to WriteSynopsis
  string input_hash = 2;                     // hash of the provider synopses
}

message SaveGeneratedTextRequest {  // (ECHO-115)
  string anime_id = 1;
  string field_path = 2;                     // 'synopsis'
  optional shared_proto.v1.WriteSynopsisResponse written_synopsis = 3;  // unset when nothing was written
  string input_hash = 4;
  GeneratedTextOutcome outcome = 5;          // recorded even when nothing was written
}

enum GeneratedTextOutcome {  // (ECHO-115)
  GENERATED_TEXT_OUTCOME_UNSPECIFIED = 0;
  GENERATED_TEXT_OUTCOME_WRITTEN = 1;
  GENERATED_TEXT_OUTCOME_TOO_LITTLE_TEXT = 2;
  GENERATED_TEXT_OUTCOME_REFUSED = 3;        // every attempt was too close to a source
}

message MarkEpisodeAiredRequest {  // (ECHO-103)
  string episode_id = 1;
}

message MarkEpisodeAiredResponse {  // (ECHO-103)
  bool has_notified = 1;                     // false when it was already written, or the episode has not aired
}

message SaveGeneratedTextResponse {  // (ECHO-115)
  bool has_changed = 1;
}

message GetEntityRequest {  // (ECHO-108)
  string entity_id = 1;
}

message GetAnimeResponse {  // (ECHO-108)
  shared_proto.v1.Anime anime = 1;
  int64 version = 2;                         // the version the vector service stores in the point
  google.protobuf.Timestamp next_refresh_at = 3;  // unset when due now; the backend decides an open's refresh from it
}

message GetEpisodeResponse {  // (ECHO-108)
  shared_proto.v1.Episode episode = 1;
  int64 version = 2;
  google.protobuf.Timestamp next_refresh_at = 3;  // unset when due now; the backend decides an open's refresh from it
}

message GetCharacterResponse {  // (ECHO-108)
  shared_proto.v1.Character character = 1;
  int64 version = 2;
  google.protobuf.Timestamp next_refresh_at = 3;  // unset when due now; the backend decides an open's refresh from it
}

message ListEpisodesResponse {  // (ECHO-108)
  repeated GetEpisodeResponse episodes = 1;
}

message ListEntityVersionsRequest {  // (ECHO-108)
  shared_proto.v1.PageKind entity_kind = 1;
  int32 page_size = 2;
  optional string after_id = 3;              // keyset: the last ID of the previous page
}

message ListEntityVersionsResponse {  // (ECHO-108)
  repeated EntityVersion entities = 1;       // visible entities only, in ID order
}

message EntityVersion {  // (ECHO-108)
  string entity_id = 1;
  int64 version = 2;
}

message ListDuePagesRequest {  // (ECHO-108)
  string provider = 1;                       // 'anidb'
  int32 page_size = 2;
}

message ListDuePagesResponse {  // (ECHO-108)
  repeated DuePage pages = 1;                // never-fetched first, then oldest fetch
}

message DuePage {  // (ECHO-108)
  shared_proto.v1.PageKey page_key = 1;
  string anime_id = 2;                       // the work to send merge_requested when the page changed
}
```

- **Saves check the version.** A save writes only when the stored version still equals `expected_version`, and otherwise returns `ABORTED`.
  - ~~`SaveAnime` checks it. (ECHO-98)~~
  - ~~`SaveEpisode` checks it. (ECHO-103)~~
  - `SaveCharacter` checks it. (ECHO-104)
  - `DeleteAnime` checks it. (ECHO-107)
  - On `ABORTED`, the merge activity loads, merges and saves again within the same attempt. (ECHO-111)
  - ~~A save that changes nothing writes nothing and sends nothing. (ECHO-98)~~
  - `RecordRefresh`, the refresh's last call, records that it finished. (ECHO-108)
- **`SaveAnime` creates what the work lists.**
  - ~~An episode the merge gave a new ID is created with its page links, and the response names it so the workflow can start its refresh. (ECHO-103)~~
  - A character the merge gave a new ID is created the same way. Before creating it, the save looks up the character's page keys. When one already belongs to a character, such as one another work's save created a moment earlier, it links that character and drops the new ID, so the save never fails on it. (ECHO-104)
- ~~**`StorePage`** stores a page and its responses only when its `fetched_at` is newer than the stored one, so a retried or late fetch never replaces newer data. It reports the page as changed when the hash of the responses differs, without reading the stored bodies. (ECHO-97)~~
- **`CreateAnime` and `RelinkPages`** resolve every unresolved relation that points at a newly linked source page, in the same transaction. They also increase the version of every work that gains or loses a page, so a merge that loaded before the move starts again. (ECHO-107)
- **`Get*` answers `NOT_FOUND`** for an entity that is deleted or not merged yet, so consumers confirm deletes by reading back. Otherwise they return the entity with its version, which the vector service stores in the point. (ECHO-108)
- **`ListEntityVersions`** pages through the visible entities of one kind with their versions, for the vector service's comparison job. (ECHO-108)
- ~~**`StorePage` records a missing page instead of deleting it.** A fetch with `is_not_found` sets `missing_since` and counts the answer, and keeps the stored responses. Two answers a day apart create a `dead_page` review item, or, once the provider is trusted, set `is_dead`. Seed releases never remove pages. See [[Database Schema#4. Writes and Deletes|Database Schema]]. (ECHO-97)~~
- **Addresses are checked.** An address that does not match its provider's pattern is refused with `INVALID_ARGUMENT`, before it reaches a URL.
  - `FetchPage` checks it. (ECHO-99)
  - ~~`StorePage` checks it. (ECHO-97)~~
- ~~**`StorePage` refuses `is_dead`.** A fetch reports what the provider answered, never that a page is dead. Only an approved `dead_page` review, or two "not found" answers a day apart once the provider's review is turned off, marks a page dead, so `StorePage` returns `INVALID_ARGUMENT` for a page with `is_dead` set. (ECHO-103)~~
- ~~**`LoadAnime` returns the work's known episodes.** `known_episodes` holds every undeleted episode with its page keys and list entries, so the merge keeps their IDs. (ECHO-103)~~
- ~~**Rule violations are not retried.** A unique violation returns `FAILED_PRECONDITION`, such as a new episode ID for a number an undeleted episode holds, and a foreign-key violation returns `INVALID_ARGUMENT`, such as an unknown lookup value. Both are permanent, so the activity does not retry them, where `INTERNAL` would be retried. (ECHO-103)~~
- ~~**`MarkEpisodeAired`** writes `notification.episode.aired` once, when the episode's `aired_at` has passed and `aired_notified_at` is unset. The tracker calls it at air time, so the notification does not depend on a save changing something. (ECHO-103)~~
- **`SaveGeneratedText` records every attempt**, written or not, so the same provider synopses do not start the synopsis workflow again. (ECHO-115)
- **`ListDuePages`** lists one budgeted provider's pages in due order with their work, for the budgeted fetcher. (ECHO-108)
- ~~**`SaveAnime` stores each episode's list entries** in `episode_list_entry`.~~
  - ~~An existing episode whose entries changed is made due at once and returned in `updated_episode_ids`, so the refresh starts it. (ECHO-103)~~
  - A character whose cast entry changed is made due at once and returned in `updated_character_ids`. (ECHO-104)
- **Works exist before their first merge.** See [[Database Schema#4. Writes and Deletes|Database Schema]].
  - `CreateAnime` writes the work's ID and page links with `merged_at` unset, and links pages that were never fetched. (ECHO-107)
  - The `Get*` and listing methods skip such works. (ECHO-108)
  - ~~The save that first sets `merged_at` writes `anime.created`. (ECHO-98)~~
- **`ListEntityPages` and `RecordRefresh`** give a refresh its due pages and record its outcome, including a refresh that changed nothing, one that left pages failed, and one that failed. (ECHO-108)
- **Message size.** Every method that carries page bodies (`StorePage`, `LoadAnime`, `MergeAnime`, `MergeEpisode`, `MergeCharacter` and their loads) runs with a 64 MB limit, set on every server and every client: `max_decoding_message_size` and `max_encoding_message_size` in `tonic`, whose decoding limit defaults to 4 MB on both ends, and `grpc.max_receive_message_length` and `grpc.max_send_message_length` in `grpcio`. Messages are gzip-compressed.
  - ~~The PostgreSQL Service sets the limit and compression on its server. It returns `RESOURCE_EXHAUSTED` only for a message over the limit, never for a passing condition, so it is not retried. (ECHO-97)~~
  - The Enrichment Service sets them on its server, and returns a provider's throttling as `UNAVAILABLE`. (ECHO-99)
  - The Update Service sets them on its clients. (ECHO-111)
- **`DeleteAnime`** removes the work's page links, its own relations and every `graph_edge` row from or to it, and turns relations other works assert towards it back into unresolved ones, all in one transaction. A work merged into another keeps `merged_into_anime_id`, so its old links can redirect. See [[Database Schema#4. Writes and Deletes|Database Schema]]. (ECHO-107)
- **A page view writes nothing.** The backend decides from `next_refresh_at` in the `Get*` response whether an opened entity is due, so a view costs one read, and a cached one none. (ECHO-108)
- **`ListRequestedMerges`** returns the entities whose `merge_requested_at` is set, through a partial index. The sweep starts their refreshes before paging the due queries. (ECHO-108)
- **`DecideApproval`** (planned with the approval queue) records a reviewer's decision and, when approved, applies it in the same transaction: a correction, a page move, a dead page or a character merge. It sets `merge_requested_at` on every entity it touched, and refuses a correction whose field path the model does not have. (P-ECHO-15)
- **`ListDueAnime`, `ListDueEpisodes` and `ListDueCharacters`** page by `(next_refresh_at, id)` from a fixed cutoff. `RecordRefresh` sets `next_refresh_at` past the cutoff for every refresh that finishes, so a sweep never picks the same entity twice. (ECHO-108)
- **`ListUpcomingEpisodes`** returns the listed episodes airing in the window, and also the next episode a work's broadcast announces before any provider lists it, with `episode_id` unset. (ECHO-108)
- **`ListScoreAnime`** pages the works whose scores refresh daily, ongoing or finished within `lookback_days`. A finished work without an end date is left to the weekly sweep. (ECHO-108)
- **`LoadSynopsisSources`** and **`SaveGeneratedText`** serve the `WriteSynopsisWorkflow`. A synopsis whose input hash already matches is not written again. A new synopsis increases the work's version. (ECHO-115)
- **`WriteSynopsisRequest` and `WriteSynopsisResponse` live in `shared_proto.v1`**, since both the Agent Service and the PostgreSQL Service use them, so no service proto imports another's. (ECHO-91)
- **`ListFranchiseLinks`** streams one row per pair of works joined by a franchise link, after the crossover rule, with whether a reviewer confirmed it. **`CreateLinkCheckApprovals`** writes a `link_check` approval request per candidate and skips a candidate that already has an open request. See [[Database Schema#Link check|Database Schema]]. (P-ECHO-22)
- **`ExportCorrections`** streams every correction, crossover mark and confirmed link, for a file kept in the repository. **`ImportCorrections`** loads that file into a rebuilt database, skipping rows it already holds. (P-ECHO-15)
- Approval-queue writes and the backend's reads are defined when those features are designed.

## Schema Evolution

> [!note] Full details in [[echora_contracts|Schema Ownership & Contracts]]
> The toolchain, CI checks and distribution are described there. The rules below cover what changes in a message.

`libs/common/src/common/models/anime.py` is the source of truth for the domain types. `protos/shared_proto/v1/anime.proto` mirrors it, and a change to one changes the other in the same PR. `scripts/check_anime_model_proto_contract.py` checks that they match.

~~Messages PostgreSQL stores follow the production rules below from the first migration that stores them: the messages of `anime.proto`, since a `FieldCorrection` value can be any of them, `FieldCorrection` itself, the approval queue's proposals and decisions, and `CatalogueEvent` in the outbox. Protobuf bytes name fields by number, so a renumbered field would silently read a stored value into the wrong field. (ECHO-98)~~

Before any service runs in production, every other message may change in place:

- **Removing a field** deletes it and renumbers the message's fields from 1 without gaps. No numbers or names are reserved.
- **Adding or renaming a field** changes the model and the proto together, then regenerates the stubs.

Once services exchange these messages across versions in production:

- **Adding an optional field** is safe. An old message without it reads as the default.
- **Removing a field** reserves its number and name, which are never reused.
- **Renaming a field or changing its type** adds a new field beside the old one, and removes the old one once every consumer has moved.

Thin events make this rarely matter for events. A domain field added to `Anime` changes no event message, since events carry only IDs and versions.

~~Stored corrections, approval values and outbox events are protobuf messages too, which is why they follow the production rules from the start. (ECHO-98)~~

~~PostgreSQL migrations are written by hand. A model field without a database column is caught at build time, when `cargo sqlx prepare` checks the queries against the schema. (ECHO-93)~~

## Related Documentation

- [[event_driven_architecture|Event-Driven Architecture]] — Streams, subjects, the outbox and consumers
- [[update_service|Update Service]] — The Temporal workflows that call these methods
- [[Database Schema|Database Schema]] — Where each message is stored
- [[echora_contracts|Schema Ownership & Contracts]] — Toolchain, CI and distribution
- [[postgres_integration_architecture_decision|PostgreSQL Service]] — The PostgreSQL Service

---

**Status**: Planned | **Last Updated**: 2026-11-14
