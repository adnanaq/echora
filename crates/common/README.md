# `common`

Shared configuration, error types, and OpenTelemetry/tracing setup for Echora Rust crates.

## Responsibilities

- **`config`:** Reads `ENVIRONMENT`, `DEBUG`, `LOG_LEVEL`, `OTEL_*`, and `POSTGRES_SERVICE_*` from process environment, `.env`, and defaults (matching Python `EnvSettings`).
- **`error`:** Library error struct with private `ErrorKind` and `is_*` predicates.
- **`telemetry`:** Installs structured JSON `tracing` logs and optional OTLP span export.

## Commands

```bash
cargo test -p common
cargo clippy -p common --all-targets -- -D warnings
```
