# `temporal_workflows`

Temporal workflows, activities, task queue constants, and deployment-based Worker Versioning helpers for Echora.

## Contents

- `UPDATE_SERVICE_TASK_QUEUE` (`update-service`): workflow execution, merges, and workflow starter activities.
- `AGENT_SERVICE_TASK_QUEUE` (`agent-service`): AI activities executed against `agent_service`.
- `PROVIDER_QUEUE_NAMES` and `provider_task_queue`: per-provider queues (`provider-anidb`, `provider-anilist`, `provider-anime_planet`, `provider-animeschedule`, `provider-anisearch`, `provider-kitsu`, `provider-mal`, `provider-syoboi`).
- `REFRESH_TRIGGER_SEARCH_ATTRIBUTE` (`RefreshTrigger`): Keyword search attribute tracking what started a refresh.
- `WorkerQueueSpec` and `build_worker_options`: constructs `WorkerOptions` with `WorkerDeploymentOptions` for a given queue profile.
- `SpikeWorkflow` and `SpikeActivities`: ECHO-95 SDK spike proving `USE_EXISTING`, `signal_with_start_workflow`, priority/fairness keys, `RefreshTrigger` visibility queries, and Worker Versioning.

## Layout

- `src/lib.rs` exports queue constants, `WorkerQueueSpec`, `build_worker_options`, and spike primitives.
- `tests/sdk_spike.rs` runs integration tests against the Temporal dev server.

## Tests

```bash
docker compose --env-file .env -f docker/docker-compose.dev.yml up -d temporal
cargo test -p temporal_workflows
cargo clippy -p temporal_workflows --all-targets -- -D warnings
```

See [the crates README](../README.md) for the setup and the checks every crate shares.
