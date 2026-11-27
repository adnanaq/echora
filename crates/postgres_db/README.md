# `postgres_db`

The PostgreSQL schema and every query of the Echora catalogue. Only `postgres_service` uses it. Callers pass a `sqlx::PgPool`, and get typed inputs and outcomes back, never SQL.

## Schema

The migrations in `migrations/` are the only source of the schema, and are written by hand. `run_migrations` applies the ones a database lacks, so the service runs it at every start.

| Migration | Tables |
| --- | --- |
| `..._create_lookup_and_name_tables.sql` | Lookup tables for every enum, such as `anime_type` and `provider`, the shared names `genre`, `tag`, `demographic`, `content_warning` and `trait`, and the shared codes `language` and `country` |
| `..._create_provider_page_tables.sql` | `provider_page` and `provider_page_response`, the pages as received, and `approval_request` |
| `..._create_anime_and_outbox_tables.sql` | `anime`, `anime_visible`, `anime_source`, `anime_partial_source`, `anime_field_correction` and `outbox` |
| `..._create_anime_names_classification_images_tables.sql` | Titles, synonyms, genres, tags, demographics, content warnings, themes and images of a work, and the shared `theme` table |
| `..._create_anime_songs_streaming_stats_broadcast_hiatus_tables.sql` | Opening and ending themes, trailers, external sources, streaming sources, statistics, broadcast and hiatus history, and the shared `trailer` and `external_source` tables |
| `..._create_anime_relations_source_material_companies_graph_tables.sql` | Related anime and their claims, related source material, companies and `graph_edge` |
| `..._create_episode_tables.sql` | `episode`, `episode_visible`, its pages, list entries, titles, images, streaming, characters and corrections |
| `..._create_character_tables.sql` | `character`, `character_visible`, its pages, the works it appears in (`character_animeography`), names, nicknames, traits, images, attributes, spoilers, mangaography and corrections |

Every table and column is named after the model field it stores, by the rule in `docs/architecture/Database Schema.md` (Naming Rule), which `scripts/check_postgres_schema.py` checks in CI. A value shared across works, such as a URL's platform or a theme's description, is stored once in a shared table, and a relation stores only its target page, so the target work is always found by a join. A character stores the works it appears in the same way, by each work's page.

A migration that builds an index with `CREATE INDEX CONCURRENTLY` starts with `-- no-transaction` and holds only that statement. The full design, with the reasons for each table, is in `docs/architecture/Database Schema.md`.

## Operations

| Function | What it does |
| --- | --- |
| `connect_pool`, `run_migrations` | Open a pool and bring the schema up to date |
| `record_page_observation` | Store one fetch of a provider page: a success, the provider's "not found", or a failure |
| `load_anime`, `save_anime` | Read a work's pages, corrections, known episodes and version; save its merged record |
| `link_anime_source_page` | Link a page to a work as its source, and tell the works whose relations, and the characters whose appearances, now reach it |
| `load_episode`, `save_episode` | The same for an episode |
| `mark_episode_aired` | Write the aired notification once, after the episode's air time has passed |
| `load_character`, `save_character` | The same for a character; its save stores the works it appears in by each work's page, with their graph edges |
| `list_character_anime` | List the stored works a character appears in, with the strongest role its pages give in each |
| `delete_episode` | Remove an episode, its page links and its graph edges, and send `episode.deleted` |
| `compute_responses_hash` | Hash a page's responses the way `record_page_observation` compares them |
| `list_entity_pages`, `record_refresh` | List an entity's linked pages with fetch metadata, and record refresh outcomes onto `next_refresh_at` |
| `list_due_anime`, `list_due_episodes`, `list_due_characters` | Keyset-paged due queries with `older_after_years` and `merge_version` filters |
| `list_upcoming_episodes`, `list_score_anime`, `list_requested_merges`, `list_due_pages` | Tracking window episodes, daily score works, pending merge requests, and budgeted provider due pages |
| `read_visible_anime`, `read_visible_episode`, `list_visible_episodes`, `read_visible_character`, `list_entity_versions` | Read merged, undeleted entities through `*_visible` views, each in one read-only snapshot, with page URLs from `provider_page.source` and the rows that name one target grouped into one entry; and page visible entity versions |

## Rules Every Save Follows

- **Version check.** A save names the version it loaded. A different stored version fails with a version conflict, and the caller loads again.
- **Unchanged saves write nothing.** When the merged record hash and its format match the stored ones, the save only clears `merge_requested_at` and records `merge_version`. It raises no version and sends no event.
- **Outbox.** A changing save raises the version and writes its event in the same transaction: `anime.created`, `episode.created` or `character.created` on the first merge, `.updated` afterwards. A changed work status also writes `notification.anime.status.changed`.
- **One clock.** Every timestamp in a transaction is PostgreSQL's transaction time, so a save's rows and events agree with each other, whichever server wrote them.
- **No deadlocks.** Every save takes its locks in one order: the work pages it names, then shared rows in sorted key order, then its own row. A save holds a work page's advisory lock shared, so saves naming the same page run side by side, while `link_anime_source_page` and `record_page_observation` hold it exclusively, so a page link and a save naming the page never miss each other's rows. A pair of related works is locked by an advisory lock keyed by a hash of the pair, taken in ascending hash order.
- **Pages.** A fetch stamped more than five minutes ahead of the database's clock is refused, and a fetch older than what is stored is ignored. A changed page raises the version of every entity built from it and requests its merge. Two "not found" answers at least a day apart open a `dead_page` review, or mark the page dead when that provider's review is off.
- **Validation.** Inputs are checked before any write, and a broken rule fails with an invalid-argument error.

## Errors

`postgres_db::Error` has a private kind. Callers ask one question per outcome:

- `is_not_found()` for a missing or deleted entity;
- `is_version_conflict()` for a stale `expected_version`, or a serialization failure;
- `is_invalid_argument()` for input that breaks a rule, or text PostgreSQL cannot store, such as a NUL byte;
- `is_unique_violation()` and `is_foreign_key_violation()` for constraints PostgreSQL enforced;
- `is_migration()` and `is_database()` for everything else.

When `RUST_BACKTRACE` is set, the error also captures a backtrace and prints it after the message.

## Log Events

- `postgres_db.migrations.applied`
- `postgres_db.provider_page.recorded`, with the page key and whether it changed
- `postgres_db.anime.saved`, `postgres_db.episode.saved` and `postgres_db.character.saved`, with the ID, version and event subject

## Features

- `test-util` adds `Error::unique_violation` and `Error::foreign_key_violation`, so other crates' tests can build those errors. It is never turned on outside tests.

## Tests

Unit tests sit beside the code they test. Integration tests are in `tests/`, one file per area, with shared helpers in `tests/support/`. Each test gets a fresh database with every migration applied.

- A test fails when PostgreSQL would cut a table, index or constraint name at its 63-byte limit, so a long generated name is named explicitly in its migration.
- Saves that share rows are tested running at the same moment, so a change to the lock order shows up as a deadlock in the tests.

```bash
cargo test -p postgres_db
cargo test -p postgres_db --test provider_page
cargo clippy -p postgres_db --all-targets -- -D warnings
cargo sqlx prepare --workspace -- --all-targets
```

See [the crates README](../README.md) for the setup and the checks every crate shares.
