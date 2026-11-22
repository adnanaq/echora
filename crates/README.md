# Rust Crates

Echora's Rust code is one Cargo workspace. The workspace file is `Cargo.toml` at the repository root, and every crate sits side by side in this folder.

| Crate | Kind | What it holds |
| --- | --- | --- |
| [`common`](common/README.md) | Library | Settings from the environment, the shared error type, and logging and OpenTelemetry setup |
| [`postgres_db`](postgres_db/README.md) | Library | The PostgreSQL schema as `sqlx` migrations, and every query the catalogue runs |
| [`postgres_service`](postgres_service/README.md) | Binary | The PostgreSQL Service, a gRPC server over `postgres_db` |

`postgres_service` depends on both libraries. `postgres_db` depends on neither of the others, so it can be tested on its own.

## Workspace Rules

- The root `Cargo.toml` declares every dependency version, the crates' own version, the edition (2024), the minimum Rust version and the lints once. A crate's `Cargo.toml` names a dependency with `workspace = true` and adds only the features it needs.
- The lints are clippy's `pedantic` group, `unwrap_used` and `expect_used`, and `wildcard_enum_match_arm` as an error. `clippy.toml` allows `unwrap` and `expect` inside tests.
- A lint is silenced with `#[expect(..., reason = "...")]`, so a silence that is no longer needed fails the build. `#[allow]` is used only for generated code.
- The code follows the Microsoft Pragmatic Rust Guidelines in `rust-guidelines.txt`. Public library items have a short summary line and `# Examples`, `# Errors` and `# Panics` sections where they apply.
- The Python limits on file and function length do not apply to Rust.

## Setup

The tests that touch the database need the dev PostgreSQL and a `DATABASE_URL`. The `.env` file at the repository root holds it.

```bash
# Start PostgreSQL 18
docker compose -f docker/docker-compose.dev.yml up -d postgres

# Load DATABASE_URL and the other settings into the shell
set -a; source .env; set +a
```

`DATABASE_URL` has the form `postgres://<user>:<password>@127.0.0.1:5432/<db>`. Each `#[sqlx::test]` creates its own database on that server, with every migration applied, so tests never see each other's rows. The database is dropped when the test passes and kept for inspection when it fails.

## Checks

Run these before a commit. CI runs the same ones.

```bash
# Format
cargo fmt --all

# Lint without a database, from the prepared query data in .sqlx/
SQLX_OFFLINE=true cargo clippy --workspace --all-targets -- -D warnings

# Unit, integration and doc tests
cargo test --workspace

# Dependency audit, and dependencies a crate declares but never uses
cargo audit
cargo +nightly udeps --workspace --all-targets
```

Run a part of the suite while working:

```bash
cargo test -p postgres_db                      # one crate
cargo test -p postgres_db --test episode       # one integration test file
cargo test -p postgres_db save_episode         # tests whose name contains a word
cargo test --workspace --doc                   # doc examples only
```

## Queries and the Offline Data

`sqlx` checks every `query!` macro against the real schema at compile time. Builds without a database, such as CI's lint job and the Docker images, read the checked queries from `.sqlx/` at the repository root instead. After changing a query or a migration, regenerate that folder and commit it:

```bash
cargo sqlx prepare --workspace -- --all-targets

# Confirm .sqlx/ matches the code
cargo sqlx prepare --workspace --check -- --all-targets
```

## Coverage

Coverage uses `cargo-llvm-cov`, which needs the `llvm-tools-preview` component of rustup. It also counts the `postgres_service` binary that the server tests start. The arguments of a `tracing::event!` call show as uncovered even when the event runs, so those lines are expected in the report. Doc tests are not counted, since their coverage needs a nightly compiler.

```bash
# Run the tests and list every uncovered line
cargo llvm-cov --workspace --text --show-missing-lines

# The table per file, from the last run
cargo llvm-cov report

# The check CI runs: fail when line coverage falls under 99%
cargo llvm-cov report --fail-under-lines 99

# A browsable report in target/llvm-cov/html
cargo llvm-cov --workspace --open
```

## Protos

`postgres_service` builds its gRPC code from `protos/` in its own `build.rs`, with `tonic-prost-build`. Nothing generated is checked in. The Python stubs are generated separately:

```bash
uv run --extra dev python scripts/generate-proto.py
buf lint
```

## CI

- `.github/workflows/rust-ci.yml` checks formatting and runs clippy offline. It then runs the tests against PostgreSQL 18 under `cargo-llvm-cov`, and fails when line coverage falls under 99%. The coverage table goes in the run's summary, and `lcov.info` is kept as the `rust-coverage` artifact. A second job runs `cargo audit` and `cargo-udeps`. Changes to `crates/`, `protos/`, `.sqlx/`, `clippy.toml` or the Cargo files start it.
- `.github/workflows/proto-stubs-check.yml` runs `buf lint`, and `buf breaking` on the protos PostgreSQL stores.
