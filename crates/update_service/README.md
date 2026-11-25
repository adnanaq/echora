# `update_service`

The Temporal worker binary for Echora's Update Service. It opens one Temporal `Client` and runs one `Worker` per task queue (`update-service`, each `provider-<name>`, and `agent-service`), registering only the workflows and activities each queue owns.

## Task Queues

| Queue | What runs on it |
| --- | --- |
| `update-service` | Workflows, merge activities, and `start_entity_workflows` |
| `provider-<name>` | Provider page fetches (one queue per provider) |
| `agent-service` | AI activities |

## Configuration

The binary reads [`common`](../common/README.md) settings (`LOG_LEVEL`, `OTEL_*`) plus:

| Variable | Default | Description |
| --- | --- | --- |
| `TEMPORAL_ADDRESS` | `http://localhost:7233` | Temporal frontend gRPC URL |
| `TEMPORAL_NAMESPACE` | `echora` | Temporal namespace |
| `POSTGRES_SERVICE_ADDRESS` | `http://localhost:8004` | `postgres_service` gRPC URL |
| `ENRICHMENT_SERVICE_ADDRESS` | `http://localhost:8002` | Default `enrichment_service` gRPC URL |
| `ENRICHMENT_FETCH_SERVICE_ADDRESS` | `ENRICHMENT_SERVICE_ADDRESS` | Optional dedicated URL for `FetchPage` (single fetch process per provider) |
| `ENRICHMENT_MERGE_SERVICE_ADDRESS` | `ENRICHMENT_SERVICE_ADDRESS` | Optional dedicated URL for scalable `Merge*` RPCs |
| `WORKER_DEPLOYMENT_NAME` | `update-service` | Worker deployment name |
| `WORKER_BUILD_ID` | Crate version | Worker deployment build ID |
| `WORKER_VERSIONING_ENABLED` | `false` | Opts in to Worker Deployment Versioning |

At startup, `run_workers` connects to Temporal, `postgres_service`, and `enrichment_service` (configuring 64 MiB gRPC message limits and gzip compression via `build_postgres_client` / `build_enrichment_client`), initializes one `Worker` per queue wired with `GrpcCatalogueAdapter` and `TemporalWorkflowStarter`, and polls concurrently until `SIGINT` or `SIGTERM`.

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
- `Dockerfile.prd` builds the release image for production (`target-cpu=x86-64-v3`) used by `docker/docker-compose.prd.yml`.
- Both images are multi-stage, run as a non-root `appuser`, and check process health via `pgrep -x update_service`.

## Layout

- `src/main.rs` initializes `MiMalloc`, telemetry, settings, and OS signal handlers.
- `src/lib.rs` defines `UpdateServiceSettings`, `list_worker_queue_specs`, `build_all_worker_options`, and `run_workers`.
- `src/grpc_clients.rs` implements `GrpcCatalogueAdapter`, `TemporalWorkflowStarter`, and 64 MiB gzip gRPC client builders.
- `tests/grpc_integration.rs` verifies `GrpcCatalogueAdapter` against mock `PostgresService` / `EnrichmentService` servers and `TemporalWorkflowStarter` against Temporal.
- `tests/worker_config.rs` verifies settings loading, queue registration, and queue configuration.
- `tests/server.rs` verifies Temporal server connection, `SIGINT`/`SIGTERM` shutdown, and connection error handling.

## Tests

```bash
cargo test -p update_service
cargo clippy -p update_service --all-targets -- -D warnings
```

See [the crates README](../README.md) for the setup and the checks every crate shares.
