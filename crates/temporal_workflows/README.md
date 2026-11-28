# `temporal_workflows`

Temporal workflows, activities, and the names every Echora worker and client must agree on.

## Contents

- `DEFAULT_TEMPORAL_NAMESPACE` (`echora-dev`): the development namespace, used when `TEMPORAL_NAMESPACE` is unset. Production sets `echora`.
- `UPDATE_SERVICE_TASK_QUEUE` (`update-service`): workflows, merges, and workflow starter activities.
- `AGENT_SERVICE_TASK_QUEUE` (`agent-service`): AI activities run by `agent_service`.
- `Provider`: the metadata providers, each with its own `provider-<slug>` queue (`provider-anidb`, `provider-anilist`, `provider-anime_planet`, `provider-animeschedule`, `provider-anisearch`, `provider-kitsu`, `provider-mal`, `provider-syoboi`).
- `REFRESH_TRIGGER_SEARCH_ATTRIBUTE` (`RefreshTrigger`): the keyword search attribute recording what started a refresh.
- `WorkerQueueSpec`: the queue a worker polls.
- `EntityType`, `RefreshTrigger`, `RefreshOutcome`, `RefreshPageKey`, `RefreshAnimeInput`, `RefreshAnimeOutcome`, `FetchRequestedSignal`: strongly typed inputs, outputs, signals, and enums for `RefreshAnimeWorkflow`.
- `CatalogueDataPort`, `WorkflowStarterPort`, `UpdateServicePorts`, `PortError`: dependency-injected port traits and structured error type keeping workflow and activity logic independent of gRPC clients.
- `UpdateServiceActivities`, `ProviderActivities`, `RefreshAnimeWorkflow`: the work-refresh workflow, its activity bundles (`list_pages`, `refresh_page`, `merge_anime`, `start_entity_workflows`, `record_refresh`), and deterministic scheduling helpers (`is_page_due`, `matches_fetch_request`, `select_pages_to_refresh`, `compute_page_backoff_seconds`, `build_priority`).
- `build_worker_options` and `build_update_service_worker_options`: configure one queue with Worker Versioning, graceful shutdown drain (`graceful_shutdown_period`), and workflow/activity registration. `None` leaves the worker unversioned and `Some(behavior)` turns versioning on with that default behavior, so the two Temporal settings cannot disagree.

## Features

- `spike`: the ECHO-95 SDK spike, `SpikeWorkflow` and `SpikeActivities`, and `build_spike_worker_options`, which registers them.
- `test-util`: `testing::start_local_environment` (a Temporal CLI dev server configured like `docker/docker-compose.dev.yml`, with `echora-dev`, `RefreshTrigger`, fairness on, and one partition per queue) and `UpdateServicePorts::noop`.

Building needs no `protoc`, since Temporal's protobuf files are compiled with the SDK's `vendored-protox` feature.

## Layout

- `src/lib.rs` holds the queue/attribute names, `Provider`, `WorkerQueueSpec`, `build_worker_options`, `build_update_service_worker_options`, and the `spike` and `testing` modules.
- `src/refresh_anime_types.rs` holds the typed workflow, activity, signal, and enum models.
- `src/ports.rs` defines `CatalogueDataPort`, `WorkflowStarterPort`, `UpdateServicePorts`, and `PortError`.
- `src/activities.rs` implements `UpdateServiceActivities` and `ProviderActivities`.
- `src/refresh_anime_workflow.rs` implements `RefreshAnimeWorkflow` and its deterministic scheduling/priority helpers.
- `tests/refresh_anime.rs` covers scheduling helpers, `ActivityEnvironment` unit tests, and end-to-end `RefreshAnimeWorkflow` tests against a local dev server.
- `tests/sdk_spike.rs` checks the names and worker options, and runs the ECHO-95 spike against a local dev server.

## Tests

```bash
cargo test -p temporal_workflows --all-features
cargo clippy -p temporal_workflows --all-targets --all-features -- -D warnings
```

The spike tests need the `spike` and `test-util` features and download the Temporal CLI on first use.

See [the crates README](../README.md) for the setup and the checks every crate shares.
