# `temporal_workflows`

Temporal workflows, activities, and the names every Echora worker and client must agree on.

## Contents

- `DEFAULT_TEMPORAL_NAMESPACE` (`echora-dev`): the development namespace, used when `TEMPORAL_NAMESPACE` is unset. Production sets `echora`.
- `UPDATE_SERVICE_TASK_QUEUE` (`update-service`): workflows, merges, and workflow starter activities.
- `AGENT_SERVICE_TASK_QUEUE` (`agent-service`): AI activities run by `agent_service`.
- `Provider`: the metadata providers, each with its own `provider-<slug>` queue (`provider-anidb`, `provider-anilist`, `provider-anime_planet`, `provider-animeschedule`, `provider-anisearch`, `provider-kitsu`, `provider-mal`, `provider-syoboi`).
- `REFRESH_TRIGGER_SEARCH_ATTRIBUTE` (`RefreshTrigger`): the keyword search attribute recording what started a refresh.
- `WorkerQueueSpec`: the queue a worker polls.
- `build_worker_options`: starts a worker options builder for one queue with Worker Versioning applied. `None` leaves the worker unversioned and `Some(behavior)` turns versioning on with that default behavior, so the two Temporal settings cannot disagree. The caller registers its workflows and activities and builds the options.

## Features

- `spike`: the ECHO-95 SDK spike, `SpikeWorkflow` and `SpikeActivities`, and `build_spike_worker_options`, which registers them. The spike proves the `USE_EXISTING` conflict policy, signal-with-start, a schedule whose `Skip` overlap policy skips runs while one is open, priority and fairness keys, `RefreshTrigger` visibility queries and counts, and Worker Versioning.
- `test-util`: `testing::start_local_environment`, which starts a Temporal CLI dev server configured like the one in `docker/docker-compose.dev.yml`, with the `echora-dev` namespace, the `RefreshTrigger` search attribute, fairness on, and one partition per queue. Each call starts its own server with in-memory storage.

Building needs no `protoc`, since Temporal's protobuf files are compiled with the SDK's `vendored-protox` feature.

## Layout

- `src/lib.rs` holds the names, `build_worker_options`, the `spike` module, and the `testing` module.
- `tests/sdk_spike.rs` checks the names and worker options, and runs the spike against a local dev server, including a versioned worker on the current deployment version.

## Tests

```bash
cargo test -p temporal_workflows --all-features
cargo clippy -p temporal_workflows --all-targets --all-features -- -D warnings
```

The spike tests need the `spike` and `test-util` features and download the Temporal CLI on first use.

See [the crates README](../README.md) for the setup and the checks every crate shares.
