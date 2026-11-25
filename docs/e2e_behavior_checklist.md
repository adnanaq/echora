# E2E Behavior & Verification Checklist

Comprehensive behavior checklist extracted from completed Update Service & PostgreSQL Service tickets and architecture specs (`docs/architecture/update_service.md`, `docs/architecture/temporal_infrastructure.md`, `docs/architecture/Database Schema.md`). Designed so we can append future and earlier tickets and run full isolated real-Docker E2E verification without skipping any branch or edge case.

> **Rebuild note:** Before running full real-Docker E2E tests against the compose stack, rebuild the updated images (`docker compose --env-file .env -f docker/docker-compose.dev.yml build postgres-service enrichment-service update-service`).

---

## Services Matrix

| Ticket | Title | Primary Crates / Apps | Real-Stack Dependencies |
| --- | --- | --- | --- |
| `ECHO-87` / `ECHO-93` | Rust workspace, PostgreSQL schema & `sqlx` migrations | `common`, `postgres_db`, `postgres_service` | `postgres` |
| `ECHO-95` | Set up the Temporal dev server and `update_service` worker | `temporal_workflows`, `update_service`, `docker/` | `temporal`, `update-service` |
| `ECHO-97` | Store fetched provider pages (`StorePage`) | `postgres_db`, `postgres_service` | `postgres`, `postgres-service` |
| `ECHO-98` / `ECHO-100..102` / `ECHO-133` | Load, merge-save, graph projection & outbox (`LoadAnime`, `SaveAnime`) | `postgres_db`, `postgres_service` | `postgres`, `postgres-service` |
| `ECHO-103` | Episode load, save & air notification (`LoadEpisode`, `SaveEpisode`, `MarkEpisodeAired`) | `postgres_db`, `postgres_service` | `postgres`, `postgres-service` |
| `ECHO-104` | Character load & save (`LoadCharacter`, `SaveCharacter`) | `postgres_db`, `postgres_service` | `postgres`, `postgres-service` |
| `ECHO-108` | Add the scheduling and read methods | `postgres_db`, `postgres_service` | `postgres`, `postgres-service` |
| `ECHO-111` | Refresh a work with `RefreshAnimeWorkflow` | `temporal_workflows`, `update_service` | `temporal`, `update-service`, `postgres-service`, `enrichment-service` |
| `ECHO-112` | Refresh episodes and characters with their own workflows | `temporal_workflows`, `update_service` | `temporal`, `update-service`, `postgres-service`, `enrichment-service` |
| `ECHO-113` | Rate-limit each provider and order requests by priority & fairness | `temporal_workflows`, `update_service`, `enrichment_service`, `docker/temporal/`, `scripts/` | `temporal`, `update-service`, `enrichment-service` |

---

## 0. PostgreSQL Storage & Core RPCs (`ECHO-87`, `ECHO-93`, `ECHO-97`, `ECHO-98..104`, `ECHO-133`)

| ID | Ticket | Behavior / Branch | Expected Outcome | Current Coverage | Real E2E Status |
| --- | --- | --- | --- | --- | --- |
| `E93-01` | `ECHO-87`, `ECHO-93` | `sqlx` migrations, `*_visible` views & typed errors | Migrations apply cleanly on fresh Postgres; `Error::is_not_found`, `is_version_conflict`, `is_invalid_argument`, `is_unique_violation`, `is_foreign_key_violation` classify SQLSTATEs and map to gRPC statuses | Covered (`#[sqlx::test]`, `common` unit tests) | Verified against Postgres |
| `E97-01` | `ECHO-97` | `StorePage` lossless upsert & staleness guard | Stores raw responses lossless by `PageKey`; ignores fetches with older `fetched_at` (`has_changed == false`); handles gzip payloads cleanly | Covered (`#[sqlx::test]`, `test_postgres_round_trip_e2e.py`) | Pending `update_service` -> `postgres_service` E2E |
| `E97-02` | `ECHO-97` | `StorePage` `has_changed` triggers `merge_requested_at` | When stored response hash changes, bumps `version` and sets `merge_requested_at` on linked `source` works so concurrent stale saves fail optimistic lock | Covered (`#[sqlx::test]`) | Pending `update_service` -> `postgres_service` E2E |
| `E97-03` | `ECHO-97` | `StorePage` `is_not_found` & failure tracking | Empty responses record `fetch_failed_at` & increment `fetch_failures`; `is_not_found` tracks missing answers (promoting to `is_dead` / dead-page review after threshold) | Covered (`#[sqlx::test]`) | Pending live E2E |
| `E98-01` | `ECHO-98`, `ECHO-133` | `LoadAnime` & `SaveAnime` roundtrip + no-op skip | `LoadAnime` returns `version`, `source_pages`, `partial_source_pages`, `corrections`, `known_episodes`; `SaveAnime` checks `expected_version` (`ABORTED` on conflict), skips writes & outbox on no-op, and replaces detail rows + writes `anime.created`/`anime.updated` outbox row atomically | Covered (`#[sqlx::test]`, `test_postgres_round_trip_e2e.py`) | Pending `update_service` -> `postgres_service` E2E |
| `E101-01` | `ECHO-100..102` | `SaveAnime` graph projection & episode sync | Bidirectional relation edges recomputed under symmetric advisory lock; new/updated `EpisodeReference` rows created/synced in same transaction and returned in `created_episode_ids` / `updated_episode_ids` | Covered (`#[sqlx::test]`) | Pending `RefreshAnimeWorkflow` -> `RefreshEpisodeWorkflow` E2E |
| `E103-01` | `ECHO-103` | `LoadEpisode`, `SaveEpisode` & `MarkEpisodeAired` | `LoadEpisode` returns pages + `list_entries` + `corrections`; `SaveEpisode` enforces `expected_version` (`ABORTED`) and writes outbox; `MarkEpisodeAired` emits `notification.episode.aired` at most once (idempotent on repeat calls) | Covered (`#[sqlx::test]`, `test_merge_episode_integration.py`) | Pending `RefreshEpisodeWorkflow` E2E |
| `E104-01` | `ECHO-104` | `LoadCharacter` & `SaveCharacter` | `LoadCharacter` returns character pages + `corrections`; `SaveCharacter` enforces `expected_version` (`ABORTED`), writes animeography & voice actor details, and emits outbox event atomically | Covered (`#[sqlx::test]`, `test_merge_character_integration.py`) | Pending `RefreshCharacterWorkflow` E2E |

---

## 1. `ECHO-95` — Temporal Server, Task Queues & Worker Setup

| ID | Behavior / Branch | Expected Outcome | Current Coverage | Real E2E Status |
| --- | --- | --- | --- | --- |
| `E95-01` | Worker startup polls all 10 queues | One `Worker` polls `update-service`, all 8 `provider-<name>` queues (`anidb`, `anilist`, `anime_planet`, `animeschedule`, `anisearch`, `kitsu`, `mal`, `syoboi`), and `agent-service` | Unit + server test (`worker_config.rs`, `server.rs`) | Pending full compose E2E |
| `E95-02` | Namespace & `RefreshTrigger` search attribute | `echora` namespace exists with `RefreshTrigger` (`Keyword`) search attribute; workflows queryable via visibility | Covered (`sdk_spike.rs`) | Pending full compose E2E |
| `E95-03` | `USE_EXISTING` conflict policy & `signal_with_start` | Duplicate start with same workflow ID returns existing run instead of spawning a second workflow; `signal_with_start` delivers signal atomically | Covered (`sdk_spike.rs`, `grpc_integration.rs`) | Pending full compose E2E |
| `E95-04` | Graceful shutdown on `SIGINT`/`SIGTERM` & unreachable server error | `run_workers` exits cleanly on signal and returns error on invalid/unreachable URL | Covered (`server.rs`, `worker_config.rs`) | Verified |
| `E95-05` | Worker Deployment Versioning | `WorkerDeploymentOptions` respects `WORKER_DEPLOYMENT_NAME`, `WORKER_BUILD_ID`, `WORKER_VERSIONING_ENABLED` | Covered (`worker_config.rs`) | Pending rollout test |

---

## 2. `ECHO-108` — PostgreSQL Scheduling & Read RPCs

| ID | Behavior / Branch | Expected Outcome | Current Coverage | Real E2E Status |
| --- | --- | --- | --- | --- |
| `E108-01` | `ListEntityPages` for Anime / Episode / Character | Returns all linked pages with `is_partial_source`, `fetched_at`, `is_dead`, `fetch_failed_at`, `fetch_failures`; empty for unknown/deleted entities or `EntityType::Unspecified` -> `INVALID_ARGUMENT` | Covered (`#[sqlx::test]` + tonic service test) | Pending cross-container call |
| `E108-02` | `RecordRefresh` outcome scheduling | `Succeeded` sets `next_refresh_at = now + refresh_after_days`; `PagesFailed` sets `next_refresh_at` to earliest failed page retry time (`2^failures`h, max 168h); `Failed` sets error backoff | Covered (`#[sqlx::test]` + tonic service test) | Pending cross-container call |
| `E108-03` | `ListDueAnime` / `ListDueEpisodes` / `ListDueCharacters` | Keyset pagination (`after_next_refresh_at`, `after_id`); respects cutoff, `ON_DEMAND_REFRESH_AGE_YEARS` (works <=30 yrs + unknown year), and ongoing/upcoming episode/character filters; includes outdated `merge_version` | Covered (`#[sqlx::test]`) | Pending cross-container call |
| `E108-04` | `ListUpcomingEpisodes` & `ListScoreAnime` | Returns upcoming aired-window episodes and ongoing/recently-finished works needing daily score sync | Covered (`#[sqlx::test]`) | Pending cross-container call |
| `E108-05` | `ListRequestedMerges` & `ListDuePages` | Returns entities with `merge_requested_at IS NOT NULL`, and budgeted-provider due source pages in due order | Covered (`#[sqlx::test]`) | Pending cross-container call |
| `E108-06` | `GetAnime` / `GetEpisode` / `ListEpisodes` / `GetCharacter` | Returns merged undeleted record + `version` + `next_refresh_at`; soft-deleted or unmerged (`merged_at IS NULL` in `*_visible` views) returns `NOT_FOUND`, while `ListEntityPages`/`ListDue*` still return unmerged entities | Covered (`#[sqlx::test]` + tonic) | Pending cross-container call |
| `E108-07` | `ListEntityVersions` | Keyset-paged visible entity versions for Vector Service drift checks; excludes soft-deleted and unmerged rows | Covered (`#[sqlx::test]`) | Pending cross-container call |
| `E108-08` | `RecordRefresh(Failed)` exponential backoff | Sets `refresh_failed_at = now()`, increments `refresh_failures`, and moves `next_refresh_at = now() + min(2^refresh_failures, 168)h` so repeatedly failing entities back off from sweeps | Covered (`#[sqlx::test]`) | Pending cross-container call |

---

## 3. `ECHO-111` — Work Refresh (`RefreshAnimeWorkflow`)

| ID | Behavior / Branch | Expected Outcome | Current Coverage | Real E2E Status |
| --- | --- | --- | --- | --- |
| `E111-01` | Fixed workflow ID (`refresh-anime-<anime_id>`) & `USE_EXISTING` | Concurrent starts for the same `anime_id` join one active workflow | Covered (`refresh_anime.rs`) | Pending real DB+Enrichment E2E |
| `E111-02` | Due page selection (`is_page_due`) | Fetches never-fetched or stale (`> REFRESH_AFTER_DAYS`) source pages; skips `is_dead`, `is_partial_source`, and pages still inside `2^failures`h exponential backoff (capped at 168h) | Covered (`refresh_anime.rs`) | Pending real DB+Enrichment E2E |
| `E111-03` | Concurrent `refresh_page` (`FetchPage` + `StorePage`) | Runs all selected pages concurrently on each `provider-<name>` queue; returns claim-check `(page_key, has_changed, has_failed)` | Covered (mock ports in `refresh_anime.rs` + mock gRPC in `grpc_integration.rs`) | Pending real `enrichment_service` + `postgres_service` |
| `E111-04` | Partial-source & failed page tolerance | `is_partial_source` pages are never fetched/merged; failed `refresh_page` records failure via `StorePage` and finishes with `PagesFailed` without failing the workflow | Covered (mocked) | Pending real E2E |
| `E111-05` | `MERGE_FIRST_WAIT_MINUTES` early merge + late page coalesce | If slow provider exceeds timer after first page finishes, early merge #1 runs; late-arriving changes coalesce into follow-up merge #2 | Covered (`refresh_anime.rs`) | Pending real E2E |
| `E111-06` | `merge_version` upgrade & `pages_relinked` re-merge | Merges stored pages without fetching when `entity_merge_version < merge_version` or `pages_relinked == true` | Covered (`refresh_anime.rs`) | Pending real DB E2E |
| `E111-07` | `merge_anime` 10x `ABORTED` retry loop inside 1 attempt | Optimistic concurrency conflicts (`ABORTED`) reload/remerge/resave up to 10 times inside one activity attempt; >10 conflicts fail attempt retryably | Covered (`refresh_anime.rs`, `grpc_integration.rs`) | Pending real Postgres contention E2E |
| `E111-08` | Permanent merge/save failure records `Failed` | Non-retryable `merge_anime` error calls `RecordRefresh(Failed)` and fails the workflow cleanly | Covered (`refresh_anime.rs`) | Pending real E2E |
| `E111-09` | Non-retryable gRPC status mapping | `INVALID_ARGUMENT`, `NOT_FOUND`, `PERMISSION_DENIED`, `RESOURCE_EXHAUSTED`, `FAILED_PRECONDITION`, `UNIMPLEMENTED` -> `NonRetryable`; `UNAVAILABLE`/`DEADLINE_EXCEEDED`/`INTERNAL`/`UNKNOWN` -> `Retryable` | Covered (`refresh_anime.rs`, `grpc_integration.rs`) | Verified |
| `E111-10` | `refresh_page` 20s heartbeat & cancellation | Emits heartbeats while gRPC runs and returns `ActivityError::Cancelled` when cancelled | Covered (`refresh_anime.rs`) | Pending long real-crawl E2E |
| `E111-11` | `fetch_requested` & `merge_requested` signal loop + priority escalation | Escalates `current_priority` to highest received; `max_age_seconds == 0` forces fetch; `merge_requested` triggers re-merge check before workflow exits | Covered (`refresh_anime.rs`) | Pending real E2E |
| `E111-12` | Downstream top-level starts (`start_entity_workflows`) | Starts `RefreshEpisodeWorkflow` after each merge for created/updated episodes and `WriteSynopsisWorkflow` once after all pages settle if synopses changed; zero child workflows; zero character refreshes | Covered (`refresh_anime.rs`, `grpc_integration.rs`) | Pending real `RefreshEpisodeWorkflow` E2E |
| `E111-13` | `64 MiB` gRPC limits & gzip compression | Both `build_postgres_client` and `build_enrichment_client` send/accept gzip and allow 64 MiB messages | Covered (`grpc_integration.rs`) | Pending real large-payload E2E |

---

## 4. `ECHO-112` — Episode & Character Refreshes (`RefreshEpisodeWorkflow`, `RefreshCharacterWorkflow`)

| ID | Behavior / Branch | Expected Outcome | Current Coverage | Real E2E Status |
| --- | --- | --- | --- | --- |
| `E112-01` | Fixed workflow IDs (`refresh-episode-<id>`, `refresh-character-<id>`) + `USE_EXISTING` | Unique workflow per episode and character | Covered (`refresh_anime.rs`, `grpc_integration.rs`) | Pending real DB+Enrichment E2E |
| `E112-02` | Shared parameterized orchestration (`run_entity_refresh`) | Both workflows run `list_pages` -> `refresh_page` -> `merge_episode` / `merge_character` -> `record_refresh`, with `fetch_requested` / `merge_requested` signals and zero downstream child spawns | Covered (`refresh_anime.rs`) | Pending real DB+Enrichment E2E |
| `E112-03` | `GrpcCatalogueAdapter` episode & character merge roundtrip | `LoadEpisode -> MergeEpisode -> SaveEpisode` and `LoadCharacter -> MergeCharacter -> SaveCharacter` handle `Saved`, `Aborted` -> `VersionConflict`, and error mapping | Covered (mock gRPC in `grpc_integration.rs`) | Pending real `postgres_service` + `enrichment_service` |
| `E112-04` | Character discovery `signal_with_start_workflow(fetch_requested)` | `TemporalWorkflowStarter` starts `RefreshCharacterWorkflow` for `character_ids` and atomically sends `fetch_requested` via `signal_with_start` for `character_fetch_requests` | Covered (`grpc_integration.rs`) | Pending `ECHO-135` caller E2E |

---

## 5. `ECHO-113` — Provider Rate Limits, Priority, Fairness & Browser Pool

| ID | Behavior / Branch | Expected Outcome | Current Coverage | Real E2E Status |
| --- | --- | --- | --- | --- |
| `E113-01` | Temporal queue RPS config script | `scripts/configure_temporal_queues.sh` sets `--queue-rps-limit` for all 8 `provider-*` queues (`provider-anidb` = `0.00174` RPS) and `sweep=1.0` / `seed=1.0` fairness weights | `provider-anidb` verified live on `echora-dev-temporal` | Pending full 8-queue script run on staging/dev |
| `E113-02` | Temporal dynamic config (fairness & 1 partition) | `docker/temporal/dynamicconfig.yaml` and dev compose set `matching.enableFairness=true` and `1` read/write partition for all `provider-*` queues | Config present & dev flags verified | Pending multi-partition dispatch benchmark |
| `E113-03` | Priority & fairness key propagation | `build_priority` attaches `priority_key` (`1` user, `2` episode/review, `3` sweep/seed, `4` remerge), `fairness_key`, and `fairness_weight = 1.0` (`sweep`/`seed`) | Covered (`refresh_anime.rs`, doctests) | Pending queue contention E2E |
| `E113-04` | Worker fallback RPS & browser concurrency caps | `find_provider_worker_policy` sets `max_task_queue_activities_per_second` on all `provider-*` workers and `WorkerTuner` fixed slots on browser queues (`anidb=1`, `mal/anime_planet/anisearch=2`) | Covered (`refresh_anime.rs`, `worker_config.rs`) | Pending live slow-browser contention test |
| `E113-05` | `enrichment_service` browser pool startup limit | `serve()` calls `configure_browser_pool(settings.enrichment.enrichment_max_concurrent_browsers)` | Covered (`test_main_browser_pool.py`) | Pending rebuilt container test |
| `E113-06` | Single-container default & split-ready fetch/merge routing | `ENRICHMENT_FETCH_SERVICE_ADDRESS` & `ENRICHMENT_MERGE_SERVICE_ADDRESS` default to `ENRICHMENT_SERVICE_ADDRESS`, and route `FetchPage` vs. `Merge*` independently when set | Covered (`worker_config.rs`, `grpc_integration.rs`) | Pending split-container test |
| `E113-07` | Live queue preemption & weighted fairness sharing | Under queued contention on a 1-partition `provider-*` queue, priority `1` (`user`) dispatches before priority `3`, and `sweep` / `seed` (`weight 1.0`) share priority `3` capacity 50/50 | Priority/fairness unit-tested (`build_priority`) | Pending live queue contention E2E |
| `E113-08` | Python intra-page spacing & fail-fast circuit breaker (`PoliteHttpClient`) | `PoliteHttpClient` enforces `min_interval` across concurrent `asyncio` coroutines in 1 process, and trips `_blocked_reason` on 403/423/429/refused connection so subsequent calls fail fast without HTTP requests | Covered in `libs/enrichment` unit tests | Pending real E2E block simulation |
