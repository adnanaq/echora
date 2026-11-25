# `temporal_workflows`

Temporal workflows, activities, task queue constants, and deployment-based Worker Versioning helpers for Echora.

## Contents

- `UPDATE_SERVICE_TASK_QUEUE` (`update-service`): workflow execution, merges, and workflow starter activities.
- `AGENT_SERVICE_TASK_QUEUE` (`agent-service`): AI activities executed against `agent_service`.
- `PROVIDER_QUEUE_NAMES` and `provider_task_queue`: per-provider queues (`provider-anidb`, `provider-anilist`, `provider-anime_planet`, `provider-animeschedule`, `provider-anisearch`, `provider-kitsu`, `provider-mal`, `provider-syoboi`).
- `REFRESH_TRIGGER_SEARCH_ATTRIBUTE` (`RefreshTrigger`): Keyword search attribute tracking what started a refresh.
- `WorkerQueueSpec` and `build_worker_options`: constructs `WorkerOptions` with `WorkerDeploymentOptions` and `UpdateServicePorts` for a given queue profile.
- `RefreshAnimeWorkflow`, `RefreshEpisodeWorkflow`, and `RefreshCharacterWorkflow`: share one parameterized orchestration engine (`run_entity_refresh`) for `list_pages`, concurrent per-provider `refresh_page`, `MERGE_FIRST_WAIT_MINUTES` early merge timer, `merge_anime` / `merge_episode` / `merge_character` (with 10x in-attempt `ABORTED` retries), downstream top-level starts, `fetch_requested` / `merge_requested` signal handling, and `record_refresh`.
- `UpdateServiceActivities` and `ProviderActivities`: activity implementations backed by `CatalogueDataPort` and `WorkflowStarterPort` ports (with `refresh_page` 20s heartbeating and cancellation).
- `SpikeWorkflow` and `SpikeActivities`: ECHO-95 SDK spike proving `USE_EXISTING`, `signal_with_start_workflow`, priority/fairness keys, `RefreshTrigger` visibility queries, and Worker Versioning.

## Layout

- `src/lib.rs` exports queue constants, `WorkerQueueSpec`, `build_worker_options`, workflow/activity types, and spike primitives.
- `src/refresh_anime_types.rs` defines claim-check workflow, signal, and activity payload types plus workflow ID helpers.
- `src/refresh_anime_workflow.rs` implements `RefreshAnimeWorkflow`, the shared `run_entity_refresh` loop, and deterministic due/backoff/priority helpers.
- `src/refresh_entity_workflows.rs` defines `RefreshEpisodeWorkflow` and `RefreshCharacterWorkflow`.
- `src/activities.rs` implements `UpdateServiceActivities` and `ProviderActivities`.
- `src/ports.rs` defines `CatalogueDataPort`, `WorkflowStarterPort`, `UpdateServicePorts`, and `PortError`.
- `tests/refresh_anime.rs` runs unit and end-to-end Temporal workflow tests for `RefreshAnimeWorkflow`.
- `tests/sdk_spike.rs` runs integration tests against the Temporal dev server.

## Tests

```bash
docker compose --env-file .env -f docker/docker-compose.dev.yml up -d temporal
cargo test -p temporal_workflows
cargo clippy -p temporal_workflows --all-targets -- -D warnings
```

See [the crates README](../README.md) for the setup and the checks every crate shares.
