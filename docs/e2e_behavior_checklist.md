# E2E Behavior & Verification Checklist

Comprehensive behavior checklist extracted from completed Update Service & PostgreSQL Service tickets and architecture specs (`docs/architecture/update_service.md`, `docs/architecture/temporal_infrastructure.md`, `docs/architecture/Database Schema.md`). Designed so we can append future and earlier tickets and run full isolated real-Docker E2E verification without skipping any branch or edge case.

> **Rebuild note:** Before running full real-Docker E2E tests against the compose stack, rebuild the updated images (`docker compose --env-file .env -f docker/docker-compose.dev.yml build postgres-service enrichment-service update-service`).

---

## Services Matrix

| Ticket | Title | Primary Crates / Apps | Real-Stack Dependencies |
| --- | --- | --- | --- |
| `ECHO-95` | Set up the Temporal dev server and `update_service` worker | `temporal_workflows`, `update_service`, `docker/` | `temporal`, `update-service` |
| `ECHO-108` | Add the scheduling and read methods | `postgres_db`, `postgres_service` | `postgres`, `postgres-service` |
| `ECHO-111` | Refresh a work with `RefreshAnimeWorkflow` | `temporal_workflows`, `update_service` | `temporal`, `update-service`, `postgres-service`, `enrichment-service` |
| `ECHO-112` | Refresh episodes and characters with their own workflows | `temporal_workflows`, `update_service` | `temporal`, `update-service`, `postgres-service`, `enrichment-service` |
| `ECHO-113` | Rate-limit each provider and order requests by priority & fairness | `temporal_workflows`, `update_service`, `enrichment_service`, `docker/temporal/`, `scripts/` | `temporal`, `update-service`, `enrichment-service` |

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
| `E108-06` | `GetAnime` / `GetEpisode` / `ListEpisodes` / `GetCharacter` | Returns merged undeleted record + `version` + `next_refresh_at`; soft-deleted or missing entity returns `NOT_FOUND` | Covered (`#[sqlx::test]` + tonic) | Pending cross-container call |
| `E108-07` | `ListEntityVersions` | Keyset-paged visible entity versions for Vector Service drift checks; excludes soft-deleted rows | Covered (`#[sqlx::test]`) | Pending cross-container call |

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
