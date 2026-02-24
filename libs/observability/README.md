# Observability Library

This library provides the shared observability bootstrap for Echora services.

## What This Library Does

- Initializes OpenTelemetry traces, metrics, and structured logs in one call.
- Applies optional runtime instrumentation for gRPC server/client and aiohttp client.
- Provides trace context propagation helpers for async boundaries (NATS/Temporal style headers).
- Keeps observability behavior consistent across services.

Primary module path:
- `libs/observability/src/observability`

## Public API

Main entrypoint:
- `observability.setup_telemetry(...)`

Signal setup helpers:
- `observability.setup_logging(...)`
- `observability.setup_tracing(...)`
- `observability.setup_metrics(...)`

Instrumentation helpers:
- `observability.instrument_grpc_server()`
- `observability.instrument_grpc_client()`
- `observability.instrument_aiohttp_client()`

Context propagation helpers:
- `observability.inject_trace_context(...)`
- `observability.extract_trace_context(...)`
- `observability.inject_context_into_nats_headers(...)`
- `observability.extract_context_from_nats_headers(...)`
- `observability.inject_context_into_temporal_headers(...)`
- `observability.extract_context_from_temporal_headers(...)`

## Initialization Flow

1. Service startup calls `setup_telemetry(...)` once.
2. Logging is configured first (if enabled).
3. Tracing provider/exporter is configured (if enabled).
4. Metrics provider/exporter is configured (if enabled).
5. Optional instrumentors are enabled based on toggles.
6. Internal idempotency guard prevents duplicate initialization.

## Configuration Contract

`setup_telemetry(...)` parameters:

Required:
- `service_name`
- `version`
- `environment`
- `endpoint`

Optional toggles:
- `enable_logging`
- `enable_tracing`
- `enable_metrics`
- `enable_grpc_server_instrumentation`
- `enable_grpc_client_instrumentation`
- `enable_aiohttp_client_instrumentation`

Service configs feed these values from `ObservabilityConfig` (`common.config.observability_config`).

## Current Service Wiring

- `apps/vector_service/src/vector_service/main.py`
- `apps/enrichment_service/src/enrichment_service/main.py`

Both import directly from `observability`.

## Async Propagation (NATS / Temporal)

NATS/Temporal services are not in this repo yet. This library already includes reusable context helpers to wire traceparent headers when those services are added.

Integration design and acceptance criteria:
- `docs/observability_nats_temporal_integration.md`

## Local Stack and Related Scripts

Infra and runbooks:
- `docker/docker-compose.obs.yml`
- `docker/docker-compose.dev.yml`
- `docker/observability/`
- `docs/observability_local_runbook.md`

Useful scripts:
- `scripts/observability_smoke_check.py`
- `scripts/collect_observability_load_evidence.py`

## Run Locally

Start full app + observability stack:
- `docker compose -f docker/docker-compose.dev.yml -f docker/docker-compose.obs.yml up -d`

Start observability only (when services run from shell/Pants):
- `docker compose -f docker/docker-compose.obs.yml up -d`

Stop stack(s):
- `docker compose -f docker/docker-compose.dev.yml -f docker/docker-compose.obs.yml down -v`
- `docker compose -f docker/docker-compose.obs.yml down -v`

## UI Endpoints

- Grafana: `http://localhost:3000` (`admin` / `admin`)
- Prometheus: `http://localhost:9090`
- Alertmanager: `http://localhost:9093`
- Loki readiness: `http://localhost:3100/ready`
- Tempo readiness: `http://localhost:3200/ready`
- OTel collector metrics endpoint: `http://localhost:8889/metrics`

Dashboards:
- Folder: `http://localhost:3000/dashboards/f/ffe8d6w3ayk1sd/echora`
- Service overview: `http://localhost:3000/d/echora-service-overview/echora-service-overview`
- Trace journey: `http://localhost:3000/d/echora-trace-journey/echora-trace-journey`

## Production Target Shape

- Keep both fleet-level overview dashboards and per-service dashboards.
- Route service telemetry with stable `service.name` and environment labels.
- Use retention/sampling/PII policies from docs under `docs/observability_*.md`.
- Add service-specific alert ownership and SLO burn-rate alerts before rollout.

## Extension Rules

- Keep this library transport-agnostic.
- Add new instrumentation helpers as opt-in toggles.
- Avoid unbounded-cardinality labels/fields.
- Keep startup idempotent and safe for repeated calls.
- Add/adjust unit tests in `tests/libs/observability/unit/` when changing behavior.
