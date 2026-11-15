---
title: Schema Ownership & Contracts
date: 2026-02-17
tags:
  - contracts
  - protobuf
  - schema
  - model-drift
status: active
related:
  - "[[event_schema_specification]]"
  - "[[postgres_integration_architecture_decision]]"
  - "[[Architecture Index]]"
---

# Schema Ownership & Contracts

## Overview

The hand-written Pydantic models in `libs/common/src/common/models/anime.py` are the **single source of truth** for the catalogue's data shape. `protos/shared_proto/v1/anime.proto` mirrors them by hand, and CI checks that the two match. Every other copy of the shape is generated from the proto or checked against it at build time.

## Who Owns the Schema

**Enrichment owns the schema.** The enrichment code in `libs/enrichment` fetches from every provider (MAL, AniList, Kitsu, AniDB and the others), maps each page to the canonical model and merges the pages of a work. When a provider starts returning a useful field, the field is added to `anime.py` and `anime.proto` in the same PR as the mapper change that fills it. A field that turns out useless is removed the same way.

> [!important] Schema ownership follows data discovery
> The code that discovers what the data looks like defines the schema. Every other service consumes it.

| Service | Language | Schema role |
|---------|----------|-------------|
| Enrichment Service (`libs/enrichment`) | Python | Owner. Defines `anime.py` and mirrors it to the proto |
| PostgreSQL Service | Rust | Consumer. Stores the shape and checks its SQL against it at build time |
| Update Service | Rust | Consumer. Passes the messages between services |
| Vector Service | Python | Consumer |
| Backend | Not decided | Consumer |
| Agent Service (planned) | Python | Consumer |

## Proto Files

| Proto | Defines |
|-------|---------|
| `protos/shared_proto/v1/anime.proto` | `Anime`, `Character`, `Episode`, `AnimeRecord` and their enums, mirroring `anime.py` |
| `protos/shared_proto/v1/review_cases.proto` | Review cases and their answers, read by the review page |
| `protos/shared_proto/v1/error.proto` | `ErrorDetails`, shared by every service's responses |
| `protos/vector_service/v1/*.proto` | Vector search and admin services |
| `protos/enrichment_service/v1/enrichment_service.proto` | Enrichment service |
| `protos/shared_proto/v1/merge.proto` | Planned. Provider pages, corrections, generated text and entity references, see [[event_schema_specification]] |
| `protos/events/v1/events.proto` | Planned. NATS events |
| `protos/postgres_service/v1/postgres_service.proto` | Planned. PostgreSQL Service |
| `protos/agent_service/v1/agent_service.proto` | Planned. Agent Service, for every AI call |

## How Services Get the Types

### Python services

`scripts/generate-proto.py` generates the Python stubs (`_pb2.py`, `_pb2.pyi`, `_pb2_grpc.py`) and checks them in, so schema changes show in PR diffs:

| Protos | Stubs |
|--------|-------|
| `shared_proto/v1/` | `libs/common/src/shared_proto/` |
| `vector_service/v1/` | `apps/vector_service/src/vector_proto/` |
| `enrichment_service/v1/` | `apps/enrichment_service/src/enrichment_proto/` |

```python
from shared_proto.v1 import anime_pb2
```

### Rust crates in this repository

Each Rust crate that needs gRPC types generates them from `protos/` in its own `build.rs` with `tonic-prost-build`, at build time. Since tonic 0.14, code generation lives in `tonic-prost-build` and the runtime codec in `tonic-prost`, and `tonic-build` no longer has `configure()` or `compile_protos()`. Nothing is checked in or downloaded.

```rust
// crates/postgres_service/build.rs, the crate that serves the gRPC handlers
fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_prost_build::configure()
        .build_client(false)
        .compile_protos(
            &["../../protos/postgres_service/v1/postgres_service.proto"],
            &["../../protos"],
        )?;
    Ok(())
}
```

### Code outside this repository

A `proto-v*` tag runs `proto-release-assets.yml`, which publishes a GitHub Release with the raw protos, the Python stubs, a compiled descriptor set (`echora_descriptor_set.pb`) and checksums. Code in another repository, such as a frontend or a backend kept elsewhere, builds from a release instead of from `protos/`.

## How Drift Is Caught

> **Linear:** ECHO-93

```
libs/common/src/common/models/anime.py   source of truth, hand-written
        mirrored by hand to
protos/shared_proto/v1/anime.proto
        generated from by
Python stubs (checked in)  ·  Rust code (build.rs)  ·  PostgreSQL columns (checked by sqlx)
                                                            PostgreSQL lookup rows (checked in CI)
```

| Layer | Check |
|-------|-------|
| `anime.py` and `anime.proto` fields | `scripts/check_anime_model_proto_contract.py` fails on a field missing from either side |
| Proto style | `buf lint`, configured in `buf.yaml` |
| Proto and checked-in Python stubs | `proto-stubs-check.yml` regenerates the stubs and fails on any `git diff` |
| Proto and Rust code | Generated on every build, so it cannot be stale |
| Proto and PostgreSQL columns | `cargo sqlx prepare` fails when a query does not match the migrated schema |
| `anime.py` enums and PostgreSQL lookup tables | A planned check compares every enum's values with the rows its lookup migration inserts, and fails on a value missing from either side. `sqlx` cannot catch this, since a missing lookup row only fails when a save uses it. |

`proto-stubs-check.yml` runs the first three on every PR that touches the protos, the models or the scripts. The lookup check runs on every PR that touches `anime.py` or the migrations.

## Changing a Field

### Before production

No service exchanges these messages across versions yet, so changes are made in place:

1. Change the field in `anime.py` and `anime.proto` in the same PR.
2. When a field is removed, delete it and renumber the message's fields from 1 without gaps. No numbers or names are reserved.
3. Regenerate the stubs with `./pants run scripts/generate-proto.py`, and run `scripts/check_anime_model_proto_contract.py`.
4. Add a migration when the field is stored in its own PostgreSQL column or table, and insert the new value into its lookup table when an enum gains one.
5. Update [[Database Schema#Where Each Model Field Is Stored|Where Each Model Field Is Stored]] in the same PR. No check compares that map with `anime.py`, so the PR that changes the field keeps it current.

### Once in production

Once services run different versions side by side, changes must keep old messages readable:

- **Adding an optional field** is safe. An old message without it reads as the default.
- **Removing a field** reserves its number and name, which are never reused.
- **Renaming a field or changing its type** adds a new field beside the old one, and removes the old one once every consumer has moved.

## Versioning

> **Linear:** Deployment project

Proto releases are tagged by hand as `proto-v<N>`. Breaking changes are not detected automatically, since `buf breaking` is not configured. It should be turned on before the first production release, when the rules above start to apply.

## Related Documentation

- [[event_schema_specification|Event Schema Specification]] — Events and gRPC messages
- [[postgres_integration_architecture_decision|PostgreSQL Service]] — The Rust layout and how it builds
- [[Database Schema|Database Schema]] — Where each field is stored
- [[Architecture Index|Architecture Index]] — Services overview

---

**Status**: Active | **Last Updated**: 2026-11-14
