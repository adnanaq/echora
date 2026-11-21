# `postgres_db`

The PostgreSQL schema and every query of the Echora catalogue. Only `postgres_service` uses it. Callers pass a `sqlx::PgPool`, and get typed inputs and outcomes back, never SQL.

## Schema

The migrations in `migrations/` are the only source of the schema, and are written by hand. `run_migrations` applies the ones a database lacks, so the service runs it at every start.

| Migration | Tables |
| --- | --- |
| `..._create_lookup_and_name_tables.sql` | Lookup tables for every enum, such as `anime_type` and `provider`, and the shared names `genre`, `tag`, `demographic`, `content_warning` and `trait` |
| `..._create_provider_page_tables.sql` | `provider_page` and `provider_page_response`, the pages as received |
| `..._create_anime_and_outbox_tables.sql` | `anime`, `anime_visible`, `anime_provider_page`, `anime_field_correction`, `approval_request` and `outbox` |
| `..._create_anime_names_classification_images_tables.sql` | Titles, synonyms, genres, tags, demographics, content warnings, themes and images of a work |
| `..._create_anime_songs_streaming_stats_broadcast_hiatus_tables.sql` | Theme songs, trailers, links, streaming, statistics, broadcast and hiatus history |
| `..._create_anime_relations_source_material_companies_graph_tables.sql` | Relations and claims, source material, companies and `graph_edge` |
| `..._create_episode_tables.sql` | `episode`, `episode_visible`, its pages, list entries, titles, images, streaming, characters and corrections |

A migration that builds an index with `CREATE INDEX CONCURRENTLY` starts with `-- no-transaction` and holds only that statement. The full design, with the reasons for each table, is in `docs/architecture/Database Schema.md`.

## Operations

| Function | What it does |
| --- | --- |
| `connect_pool`, `run_migrations` | Open a pool and bring the schema up to date |
| `record_page_observation` | Store one fetch of a provider page: a success, the provider's "not found", or a failure |
| `load_anime`, `save_anime` | Read a work's pages, corrections, known episodes and version; save its merged record |
| `link_anime_source_page` | Link a page to a work as its source, and resolve relations that point at that page |
| `load_episode`, `save_episode` | The same for an episode |
| `mark_episode_aired` | Write the aired notification once, after the episode's air time has passed |
| `delete_episode` | Remove an episode, its page links and its graph edges, and send `episode.deleted` |
| `compute_responses_hash` | Hash a page's responses the way `record_page_observation` compares them |

## Rules Every Save Follows

- **Version check.** A save names the version it loaded. A different stored version fails with a version conflict, and the caller loads again.
- **Unchanged saves write nothing.** When the merged record hash and its format match the stored ones, the save only clears `merge_requested_at` and records `merge_version`. It raises no version and sends no event.
- **Outbox.** A changing save raises the version and writes its event in the same transaction: `anime.created` or `episode.created` on the first merge, `.updated` afterwards. A changed work status also writes `notification.anime.status.changed`.
- **One clock.** Every timestamp in a transaction is PostgreSQL's transaction time, so a save's rows and events agree with each other, whichever server wrote them.
- **No deadlocks.** Shared rows are locked and written in sorted key order. A pair of related works is locked by an advisory lock keyed by a hash of the pair, taken in ascending hash order.
- **Pages.** A fetch older than what is stored is ignored. A changed page raises the version of every entity built from it and requests its merge. Two "not found" answers at least a day apart open a `dead_page` review, or mark the page dead when that provider's review is off.
- **Validation.** Inputs are checked before any write, and a broken rule fails with an invalid-argument error.

## Errors

`postgres_db::Error` has a private kind. Callers ask one question per outcome:

- `is_not_found()` for a missing or deleted entity;
- `is_version_conflict()` for a stale `expected_version`, or a serialization failure;
- `is_invalid_argument()` for input that breaks a rule;
- `is_unique_violation()` and `is_foreign_key_violation()` for constraints PostgreSQL enforced;
- `is_migration()` and `is_database()` for everything else.

When `RUST_BACKTRACE` is set, the error also captures a backtrace and prints it after the message.

## Log Events

- `postgres_db.migrations.applied`
- `postgres_db.provider_page.recorded`, with the page key and whether it changed
- `postgres_db.anime.saved` and `postgres_db.episode.saved`, with the ID, version and event subject

## Features

- `test-util` adds `Error::unique_violation` and `Error::foreign_key_violation`, so other crates' tests can build those errors. It is never turned on outside tests.

## Tests

Unit tests sit beside the code they test. Integration tests are in `tests/`, one file per area, with shared helpers in `tests/support/`. Each test gets a fresh database with every migration applied.

```bash
cargo test -p postgres_db
cargo test -p postgres_db --test provider_page
cargo clippy -p postgres_db --all-targets -- -D warnings
cargo sqlx prepare --workspace -- --all-targets
```

See [the crates README](../README.md) for the setup and the checks every crate shares.
