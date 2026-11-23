# `postgres_service`

The PostgreSQL Service, the only program that talks to the catalogue database. Other services call it over gRPC. It turns each request's protobuf messages into `postgres_db` inputs, and the outcomes back into responses.

## Methods

The service is defined in `protos/postgres_service/v1/postgres_service.proto`.

| Method | What it does |
| --- | --- |
| `StorePage` | Stores one fetch of a provider page, and reports whether the page changed |
| `LoadAnime` | Returns a work's pages, corrections, known episodes and version, for its merge |
| `SaveAnime` | Saves a merged work, creates the episodes it newly lists, and returns their IDs |
| `LoadEpisode` | Returns an episode's pages, list entries, corrections and version |
| `SaveEpisode` | Saves a merged episode |
| `MarkEpisodeAired` | Writes the episode's aired notification once |
| `LoadCharacter` | Returns a character's pages, corrections and version |
| `SaveCharacter` | Saves a merged character, with the works its `animeography` names, each by its work page |

The server also answers `grpc.health.v1.Health`.

## Mapping

- Every proto message is taken apart with each field named and no `..`, and every enum value has its own arm with no `_`. A field or value added to the proto fails the build until the mapping handles it.
- An enum number this build does not know is refused with `INVALID_ARGUMENT`.
- `SaveAnime`, `SaveEpisode` and `SaveCharacter` hash a canonical form of the merged record. A character's `roles` and `anime_ids` are left out, since they are read through the works its pages name. Maps are encoded in key order, which is why `build.rs` asks `prost` for `BTreeMap`.
- `SaveCharacter` turns each `animeography` URL into a work page key and each `mangaography` URL into a source material identifier. The database joins them to `anime_source` and `source_material_xref`, so a work or source material stored after the character still reaches it. A URL that names no such page is left out.
- A page key's address is lowercased, as work matching builds page keys, so a slug written with capitals names the same stored page. Character page URLs are read in the singular (`/character/`) and in the plural (`/characters/`, as Kitsu and Anime-Planet write them).
- `StorePage` refuses a page marked dead. Only a review or the provider's two "not found" answers mark a page dead.

## Status Codes

`map_db_error` turns every library error into a status in one place:

| Library error | Status | Retried |
| --- | --- | --- |
| Version conflict | `ABORTED` | Yes, after a fresh load |
| Not found | `NOT_FOUND` | No |
| Invalid argument, foreign-key violation | `INVALID_ARGUMENT` | No |
| Unique violation | `FAILED_PRECONDITION` | No |
| Anything else | `INTERNAL` | Yes |

## Messages

Methods that carry page bodies accept and send messages up to 64 MB, compressed with gzip. Only a larger message gets `RESOURCE_EXHAUSTED`. The largest message seen is recorded as the `postgres_service.grpc.max_message_bytes` metric.

## Configuration

`DATABASE_URL` is required. The service also reads the shared settings described in [`common`](../common/README.md), such as `POSTGRES_SERVICE_HOST`, `POSTGRES_SERVICE_PORT`, `LOG_LEVEL` and the `OTEL_*` switches.

`DEAD_PAGE_REVIEW_<PROVIDER>`, such as `DEAD_PAGE_REVIEW_MAL`, turns off the review of missing pages for one provider when set to `0`, `false` or `no`. That provider's pages are then marked dead directly after two "not found" answers a day apart. Review is on for every provider by default.

At startup the service connects, applies any missing migrations, and starts serving. It stops cleanly on `SIGINT` or `SIGTERM`, finishing the requests in progress.

## Running

```bash
# Locally, against the dev PostgreSQL
set -a; source .env; set +a
cargo run -p postgres_service

# In Docker, with the rest of the dev stack
docker compose -f docker/docker-compose.dev.yml up -d --build postgres-service
docker exec echora-dev-postgres-service grpc_health_probe -addr=localhost:8004
```

## Docker Images

- `Dockerfile.dev` builds the image the dev compose file uses.
- `Dockerfile.prd` builds for the production servers' CPU (`x86-64-v3`).
- Both images are multi-stage, run as a non-root user, and check their health with `grpc_health_probe`. Both build offline from the prepared queries in `.sqlx/`.

Pants builds the production image:

```bash
./pants package crates/postgres_service:postgres-service-image
```

## Layout

- `src/main.rs` starts the server and waits for a stop signal.
- `src/service.rs` holds the gRPC handlers, the status mapping and the dead-page review setting.
- `src/anime_mapping.rs`, `src/anime_relations_mapping.rs`, `src/anime_schedule_and_stats_mapping.rs` and `src/episode_mapping.rs` turn proto messages into `postgres_db` inputs.
- `tests/server.rs` starts the built binary and checks how it starts and stops.

## Tests

The handler tests call each method against a fresh database. The server tests need `DATABASE_URL` too.

```bash
cargo test -p postgres_service
cargo clippy -p postgres_service --all-targets -- -D warnings
```

See [the crates README](../README.md) for the setup and the checks every crate shares.
