# `postgres_db`

PostgreSQL schema migrations, connection pool helpers, and database access for Echora.

## Responsibilities

- Owns `sqlx` migration files under `migrations/`, which serve as the single source of truth for the PostgreSQL 18 schema.
- Provides `connect_pool`, `run_migrations`, and a private-kind `Error` with predicate helpers (`is_not_found`, `is_version_conflict`, `is_unique_violation`, `is_foreign_key_violation`).
- Only `postgres_service` depends on `postgres_db`.

## Commands

```bash
cargo test -p postgres_db
cargo clippy -p postgres_db --all-targets -- -D warnings
cargo sqlx prepare --check --workspace -- --all-targets
```
