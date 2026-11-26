# `update_service`

The Temporal worker binary for Echora's Update Service. It opens one Temporal client and runs one worker per task queue it owns, `update-service` and each `provider-<slug>` queue. The `agent-service` queue belongs to `agent_service` and is not polled here.

## Task Queues

| Queue | What runs on it |
| --- | --- |
| `update-service` | Workflows, merge activities, and `start_entity_workflows` |
| `provider-<slug>` | Provider page fetches, one queue per provider |

Until the Update Service workflows exist, every queue registers the ECHO-95 spike activities, and `update-service` also registers the spike workflow.

## Configuration

The binary reads the [`common`](../common/README.md) settings (`LOG_LEVEL`, `OTEL_*`) and the settings below.

The Temporal connection comes from the Temporal SDK's environment configuration, so every variable it documents works here, and a `temporal.toml` profile can replace them.

| Variable | Default | Description |
| --- | --- | --- |
| `TEMPORAL_ADDRESS` | `localhost:7233` | Temporal frontend, as `host:port` or a URL |
| `TEMPORAL_NAMESPACE` | `echora-dev` | Namespace the workers poll; production sets `echora` |
| `TEMPORAL_API_KEY` | unset | API key; turns TLS on |
| `TEMPORAL_TLS` | unset | `true` turns TLS on, `false` keeps it off even with an API key |
| `TEMPORAL_TLS_*` | unset | Client certificate and key, server CA, and server name |

An address without a scheme gets `https://` when TLS is on and `http://` otherwise.

Worker Versioning is set with these variables.

| Variable | Default | Description |
| --- | --- | --- |
| `WORKER_VERSIONING_ENABLED` | `false` | `true`, `false`, `1`, `0`, `yes` or `no` |
| `WORKER_VERSIONING_BEHAVIOR` | `pinned` | `pinned` or `auto_upgrade`, used only with versioning on |
| `WORKER_DEPLOYMENT_NAME` | `update-service` | Worker Deployment name |
| `WORKER_BUILD_ID` | crate version | Build ID; required with versioning on, since each deploy needs its own |

The binary refuses to start when a value is not one it accepts, when `WORKER_VERSIONING_BEHAVIOR` is set with versioning off, or when the namespace does not exist on the server.

## Shutdown

On `SIGINT` or `SIGTERM`, every worker stops polling, finishes the tasks it is running, and releases its slots on the server before the process exits with status 0. If any worker stops with an error, the others stop the same way and the process exits with an error.

## Running

```bash
# Locally, against the dev Temporal container
docker compose --env-file .env -f docker/docker-compose.dev.yml up -d temporal
cargo run -p update_service

# In Docker, with the rest of the dev stack
docker compose --env-file .env -f docker/docker-compose.dev.yml up -d --build update-service
```

## Docker Images

- `Dockerfile.dev` builds the debug image used by `docker/docker-compose.dev.yml`.
- `Dockerfile.prd` builds the release image for production (`target-cpu=x86-64-v3`) used by `docker/docker-compose.prd.yml`, which requires `WORKER_BUILD_ID`.
- Both images are multi-stage, run as a non-root `appuser`, and check that the process is alive with `pgrep -x update_service`.

## Layout

- `src/main.rs` sets `MiMalloc` as the allocator, starts telemetry, loads the settings, and waits for `SIGINT` or `SIGTERM`.
- `src/lib.rs` defines `UpdateServiceSettings`, `list_worker_queue_specs`, `build_all_worker_options`, and `run_workers`.
- `tests/worker_config.rs` covers settings, worker options, and `run_workers` against a local dev server.
- `tests/server.rs` runs the binary and covers both shutdown signals and startup failures.

## Tests

The tests start their own Temporal CLI dev server through `temporal_workflows`' `test-util` feature, so no running Temporal is needed. The SDK downloads the CLI on first use.

```bash
cargo test -p update_service
cargo clippy -p update_service --all-targets -- -D warnings
```

See [the crates README](../README.md) for the setup and the checks every crate shares.
