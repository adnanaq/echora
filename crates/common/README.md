# `common`

Shared settings, error type and telemetry setup for Echora's Rust services. Only applications call it; libraries receive what they need as arguments.

## Settings

`CommonSettings::from_env()` reads each setting from the process environment first, then from `.env` in the working folder, then from its default. A missing `.env` is fine. The names and order match the Python `EnvSettings`, so one `.env` serves both languages.

| Variable | Default | Meaning |
| --- | --- | --- |
| `ENVIRONMENT` | `development` | Deployment name, sent with every trace |
| `DEBUG` | `true` | Debug behaviour |
| `LOG_LEVEL` | `INFO` | One of `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `POSTGRES_SERVICE_HOST` | `0.0.0.0` | IP address the PostgreSQL Service listens on; IPv6 may be bracketed |
| `POSTGRES_SERVICE_PORT` | `8004` | Port the PostgreSQL Service listens on, 1 to 65535 |
| `OTEL_ENABLED` | `true` | Turns OpenTelemetry export on or off as a whole |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4317` | OTLP gRPC collector address |
| `OTEL_ENABLE_LOGGING` | `true` | Writes JSON log lines to standard output |
| `OTEL_ENABLE_TRACING` | `true` | Exports spans to the collector |
| `OTEL_ENABLE_METRICS` | `true` | Exports metrics to the collector |

Booleans accept `1`, `true`, `yes`, `on` and `0`, `false`, `no`, `off`, in any case. A value that does not parse stops startup with an error naming the variable.

`CommonSettings::from_lookup` takes the values from a function instead of the process, which tests use. `postgres_service_addr()` turns the host and port into a socket address.

## Telemetry

`init_telemetry(service_name, &settings)` installs the global `tracing` subscriber once per process, and returns a `TelemetryGuard`. Keep the guard alive for the life of the program. Dropping it flushes the spans and metrics still waiting to be sent.

- Log lines are JSON, filtered by `LOG_LEVEL`, or by `RUST_LOG` when that is set.
- Spans and metrics go to the OTLP collector when `OTEL_ENABLED` and their own switch are on.
- Every event has a name and a message template, such as `telemetry.initialized`, so logs can be searched by event.

## Errors

`common::Error` has a private kind. Callers ask `is_invalid_env()`, `is_dotenv()` or `is_telemetry_init()`. When `RUST_BACKTRACE` is set, the error also captures a backtrace and prints it after the message.

## Layout

- `src/config.rs` reads and checks the settings.
- `src/error.rs` holds the error type.
- `src/telemetry.rs` sets up logging, tracing and metrics.
- `tests/telemetry.rs` installs the global subscriber, which a process can do only once, so it runs as its own test binary.

## Commands

```bash
cargo test -p common
cargo clippy -p common --all-targets -- -D warnings
```

See [the crates README](../README.md) for the checks every crate shares.
