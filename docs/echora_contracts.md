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
  - "[[Architecture Index]]"
---

# Schema Ownership & Contracts

## Overview

The hand-written Pydantic model in `libs/common/src/common/models/anime.py` is the **single source of truth** for the shared data shape. `protos/shared_proto/v1/anime.proto` is a manually maintained mirror of it — CI enforces that the two stay in sync and that any PR touching one must also touch the other.

## Who Owns the Schema

**The ingestion pipeline owns the schema.**

The ingestion pipeline calls all external APIs (Jikan, AniList, Kitsu, AniDB, etc.), enriches data through 5 stages, and determines what fields are worth capturing. When a new API starts returning a useful field, the ingestion developer adds it to `anime.py`. When a field turns out to be useless, they remove it. The proto update is part of the same PR.

This means the schema change and the enrichment code change are **one atomic PR** in the same repo.

> [!important] Schema ownership follows data discovery
> The team that discovers and defines what data looks like owns the schema. All other services are downstream consumers of those decisions.

**Producer/consumer split:**

| Service | Schema role |
|---------|-------------|
| **Ingestion Pipeline** | Owner — defines `anime.py` and mirrors it to proto |
| PostgreSQL Service | Consumer |
| Backend (BFF) | Consumer |
| Update Service | Consumer |
| Qdrant Service | Consumer |
| Agent Service | Consumer |

## Repository Structure

```
echora/
  libs/common/src/
    common/models/
      anime.py                  ← source of truth (hand-written Pydantic models)
    shared_proto/v1/
      anime_pb2.py              ← generated grpcio stubs (checked in)
      anime_pb2.pyi             ← generated type stubs (checked in)
      anime_pb2_grpc.py         ← generated gRPC stubs (checked in)
  protos/
    shared_proto/v1/
      anime.proto               ← manually maintained proto mirror of anime.py
  scripts/
    generate-proto.py               ← regenerates shared_proto/ from protos/
    check_anime_model_proto_contract.py  ← validates field parity
  buf.yaml                      ← buf lint configuration
```

## Toolchain

| Tool | Purpose |
|------|---------|
| `buf` CLI | Proto lint only (naming conventions, style) |
| `grpcio-tools` | Generates Python stubs (`_pb2.py`, `_pb2.pyi`, `_pb2_grpc.py`) |
| `tonic-build` | Rust consumers compile protos at build time via `build.rs` |

## CI Pipeline

### On every pull request (paths that touch proto-related files)

```
buf lint
  └── enforces STANDARD naming/style rules on protos/

Co-change enforcement
  └── if anime.py is changed → anime.proto must also be changed in the same PR
      (fails with an explicit error if proto update is missing)

./pants run scripts/generate-proto.py
  └── regenerates libs/common/src/shared_proto/ from protos/

./pants run scripts/check_anime_model_proto_contract.py
  └── compares Pydantic model fields vs proto message fields
      fails on any mismatch (missing_in_proto, extra_in_proto)

git diff --exit-code libs/common/src/shared_proto/ ...
  └── fails if checked-in stubs are out of date with the proto
```

CI also packages proto artifacts as GitHub Actions artifacts (14-day retention) on every qualifying PR/push.

### On `proto-v*` tag push

`proto-release-assets.yml` creates a GitHub Release with four assets:

| Asset | Contents |
|-------|----------|
| `proto-python-artifacts-<tag>.tar.gz` | Raw `.proto` files + generated `_pb2.py`, `_pb2.pyi`, `_pb2_grpc.py` stubs + `metadata.json` + `checksums.txt` |
| `proto-python-artifacts-<tag>.tar.gz.sha256` | SHA-256 of the Python tarball |
| `proto-rust-artifacts-<tag>.tar.gz` | Raw `.proto` files + `echora_descriptor_set.pb` + `metadata.json` + `checksums.txt` |
| `proto-rust-artifacts-<tag>.tar.gz.sha256` | SHA-256 of the Rust tarball |

## How Services Consume Types

### Python services in this repo

Import directly from the checked-in stubs:

```python
from shared_proto.v1 import anime_pb2
```

Generated stubs live at `libs/common/src/shared_proto/` and are checked in so that
schema diffs are visible in PRs.

### External Rust services

Download the Rust tarball from the GitHub Release, extract the raw `.proto` files and
`echora_descriptor_set.pb`, then compile them with `tonic-build` in `build.rs`:

```rust
// build.rs
fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(false)
        .compile_protos(
            &["protos/shared_proto/v1/anime.proto"],
            &["protos"],
        )?;
    Ok(())
}
```

The descriptor set (`echora_descriptor_set.pb`) is available as an alternative for
tooling that needs a pre-compiled descriptor rather than source protos.

## Model Drift Strategy

### The layers that must stay in sync

```
libs/common/src/common/models/anime.py   ← source of truth (hand-written)
        ↑ manually mirrored to
protos/shared_proto/v1/anime.proto
        ↓ generated by generate-proto.py
libs/common/src/shared_proto/v1/
  anime_pb2.py / anime_pb2.pyi / anime_pb2_grpc.py
```

### How drift is caught

| Layer | How drift is caught |
|-------|-------------------|
| **anime.py → proto fields** | `check_anime_model_proto_contract.py` — fails CI if any Pydantic field is missing from the proto or vice versa |
| **anime.py change without proto update** | Co-change rule in CI — fails the PR explicitly |
| **proto → checked-in stubs** | `git diff --exit-code` after `generate-proto.py` — fails if stubs are stale |

### Adding a field

1. Add the field to `anime.py` (Pydantic model)
2. Add the corresponding field to `anime.proto` with a new field number in the same PR
3. Run `./pants run scripts/generate-proto.py` to regenerate stubs
4. CI validates all three layers pass before merge

### Removing a field

1. Remove from `anime.py` and `anime.proto` in the same PR
2. Mark the proto field number as `reserved` to prevent accidental reuse:
   ```protobuf
   message Anime {
     reserved 3;  // removed field — number permanently reserved
   }
   ```
3. Regenerate stubs; existing consumers of the field must be updated in the same PR or prior

### Renaming or changing type

Treat as a removal + addition:
1. Add the new field (new name/type, new field number) in both `anime.py` and `anime.proto`
2. Dual-write in producers during a transition if consumers are external
3. Remove the old field once all consumers are migrated (follow removal rules)

## Versioning

Proto releases are tagged manually as `proto-v<N>` (e.g. `proto-v1`, `proto-v2`). There is no
automated breaking-change detection — `buf breaking` is not configured. Field number stability
is enforced by convention: reserve removed field numbers in the proto.

The release tag triggers `proto-release-assets.yml`, which publishes the GitHub Release
with Python and Rust tarballs for that snapshot.

## Considerations

### buf breaking for automated proto change detection

`buf` is **already installed** in `.github/workflows/proto-stubs-check.yml` (for `buf lint`). Adding `buf breaking` requires one additional step in that workflow, immediately after the lint step:

```yaml
- name: Lint proto files
  run: buf lint

- name: Check for breaking proto changes
  if: github.event_name == 'pull_request'
  run: buf breaking protos/ --against ".git#branch=main"
```

No changes to `buf.yaml` or any other file needed. `buf` is version-pinned in the workflow (`BUF_VERSION="1.50.0"`), so this is zero-dependency.

**What it catches vs what the current setup already catches:**

| Check | Current approach | With buf breaking added |
|-------|-----------------|-------------------------|
| Field removed without `reserved` | Convention only | ✅ CI failure |
| Field number reused | Convention only | ✅ CI failure |
| Field type changed | `check_anime_model_proto_contract.py` catches parity mismatch | ✅ CI failure on proto level too |
| New field added | No check needed (safe) | ✅ Passes fine |
| Field renamed | Co-change rule catches missing proto update | ✅ CI failure |

**What it does NOT replace:** `buf breaking` is purely a proto-level gate — it has no knowledge of `anime.py`. The co-change enforcement step and `check_anime_model_proto_contract.py` remain necessary because they validate the Pydantic ↔ proto parity, which `buf` cannot do.

**What about `buf generate`?** `generate-proto.py` cannot be replaced by `buf generate`. The script applies custom import rewrites after generation (e.g. `from v1 import` → `from shared_proto.v1 import`) because `grpcio-tools` emits flat imports that don't match the package structure. `buf generate` has the same flat-import limitation with standard Python protoc plugins. Switching would add complexity without removing any code.

**Verdict:** `buf breaking` is the one meaningful addition — one step, zero new deps, catches the class of drift (proto-level breaking changes) that currently relies on convention. Worth enabling once the proto is past its initial churn phase.

## Related Documentation

- [[event_schema_specification|Event Schema Specification]] — event message definitions
- [[event_driven_architecture|Event-Driven Architecture]] — NATS consumers and event flow
- [[Architecture Index|Architecture Index]] — services overview

---

**Status**: Active | **Last Updated**: 2026-02-17
