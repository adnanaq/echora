# `postgres_db`

PostgreSQL schema migrations, connection pool helpers, and database access for Echora.

## Responsibilities

- Owns `sqlx` migration files under `migrations/`, which serve as the single source of truth for the PostgreSQL 18 schema (lookup tables, shared name tables, `provider_page`, `provider_page_response` with `STORAGE EXTERNAL`, `approval_request`, `anime`, `anime_visible`, `anime_provider_page`, `anime_field_correction`, and `outbox`).
- Provides `connect_pool`, `run_migrations`, `validate_page_address`, `store_page`, `load_anime`, `save_anime`, and a private-kind `Error` with predicate helpers (`is_not_found`, `is_invalid_argument`, `is_version_conflict`, `is_unique_violation`, `is_foreign_key_violation`).
- Only `postgres_service` depends on `postgres_db`.

## Commands

```bash
cargo test -p postgres_db
cargo clippy -p postgres_db --all-targets -- -D warnings
cargo sqlx prepare --check --workspace -- --all-targets
```
