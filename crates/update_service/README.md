# `update_service`

The Temporal worker binary for Echora's Update Service. It opens one Temporal client and runs one worker per task queue it owns, `update-service` and a `provider-<name>` queue for each provider in `PAGE_FETCH_PROVIDERS`. The `agent-service` queue belongs to `agent_service` and is not polled here.

## Task Queues

| Queue | What runs on it |
| --- | --- |
| `update-service` | `RefreshAnimeWorkflow`, `list_pages`, `record_page_failure`, `merge_anime`, `start_entity_workflows`, and `record_refresh` |
| `provider-<name>` | `refresh_page` fetches for that provider, one queue per provider the enrichment service fetches |

## Configuration

The binary reads the [`common`](../common/README.md) settings (`ENVIRONMENT`, `LOG_LEVEL`, `OTEL_*`) and the settings below. When `OTEL_ENABLED` and `OTEL_ENABLE_METRICS` are enabled in `CommonSettings`, Temporal SDK Core exports worker, workflow, and activity metrics over OTLP gRPC to `OTEL_EXPORTER_OTLP_ENDPOINT` (graphed in `docker/observability/grafana/dashboards/update-service-overview.json`), and `GrpcCatalogueAdapter` records the `update_service.grpc.max_message_bytes` gauge.

The Temporal connection comes from the Temporal SDK's environment configuration, so every variable it documents works here, and a `temporal.toml` profile can replace them.

| Variable | Default | Description |
| --- | --- | --- |
| `TEMPORAL_ADDRESS` | `localhost:7233` | Temporal frontend, as `host:port` or a URL |
| `TEMPORAL_NAMESPACE` | `echora-dev` | Namespace the workers poll; production sets `echora` |
| `TEMPORAL_API_KEY` | unset | API key; turns TLS on |
| `TEMPORAL_TLS` | unset | `true` turns TLS on, `false` keeps it off even with an API key |
| `TEMPORAL_TLS_*` | unset | Client certificate and key, server CA, and server name |

An address without a scheme gets `https://` when TLS is on and `http://` otherwise.

Worker Versioning, shutdown drain, downstream gRPC endpoints, and refresh defaults are set with these variables.

| Variable | Default | Description |
| --- | --- | --- |
| `WORKER_VERSIONING_ENABLED` | `false` | `true`, `false`, `1`, `0`, `yes` or `no` |
| `WORKER_VERSIONING_BEHAVIOR` | `pinned` | `pinned` or `auto_upgrade`, used only with versioning on |
| `WORKER_DEPLOYMENT_NAME` | `update-service` | Worker Deployment name |
| `WORKER_BUILD_ID` | crate version | Build ID; required with versioning on, since each deploy needs its own |
| `WORKER_GRACEFUL_SHUTDOWN_SECONDS` | `20` | Grace window before in-flight activities receive cancellation on shutdown |
| `POSTGRES_SERVICE_ADDRESS` | `http://localhost:8004` | PostgreSQL Service gRPC URL (64 MiB + gzip) |
| `ENRICHMENT_SERVICE_ADDRESS` | `http://localhost:8002` | Enrichment Service gRPC URL (64 MiB + gzip) |
| `REFRESH_AFTER_DAYS` | `7` | Age in days before a work or page counts as due |
| `MERGE_VERSION` | `1` | Current mapping/merge code version |
| `MERGE_FIRST_WAIT_MINUTES` | `10` | Wait after the first stored page before merging |

The binary refuses to start when a value is not one it accepts, when `WORKER_VERSIONING_BEHAVIOR` is set with versioning off, when the namespace does not exist on the server, or when `RefreshTrigger` exists with a non-`Keyword` type.

## Shutdown

On `SIGINT` or `SIGTERM`, every worker stops polling, finishes the tasks it is running (cancelling activities after `WORKER_GRACEFUL_SHUTDOWN_SECONDS`), and releases its slots on the server before the process exits with status 0. If any worker stops with an error, the others stop the same way and the process exits with an error.

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
- `src/lib.rs` defines `UpdateServiceSettings`, `build_runtime_options`, `list_worker_queue_specs`, `build_all_worker_options`, and `run_workers`.
- `src/grpc_clients.rs` defines `GrpcCatalogueAdapter`, `TemporalWorkflowStarter`, gzip/64 MiB tonic client builders, and exhaustive gRPC status mapping.
- `build.rs` compiles `postgres_service.proto` and `enrichment_service.proto` with `BTreeMap` map fields.
- `tests/support/mod.rs` holds the shared in-process gRPC mock server and workflow-completion helpers.
- `tests/worker_config.rs` covers settings, worker options, OpenTelemetry runtime options, and `run_workers` against a local dev server.
- `tests/grpc_integration.rs` tests `GrpcCatalogueAdapter`, `TemporalWorkflowStarter`, gzip round-trips, `ServiceBlockedError` metadata, and `RefreshAnimeWorkflow` against in-process gRPC mocks.
- `tests/server.rs` runs the binary and covers both shutdown signals and startup failures.

## Tests

The tests start their own Temporal CLI dev server through `temporal_workflows`' `test-util` feature, so no running Temporal is needed. The SDK downloads the CLI on first use.

```bash
cargo test -p update_service
cargo clippy -p update_service --all-targets -- -D warnings
```

See [the crates README](../README.md) for the setup and the checks every crate shares.
