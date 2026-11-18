# `postgres_service`

gRPC server binary owning the PostgreSQL catalogue database.

## Overview

This crate provides the `PostgresService` gRPC server (`mimalloc` allocator, `common` configuration/telemetry bootstrap, `grpc.health.v1.Health` on port `8004`, startup migrations via `postgres_db`, `StorePage`, `LoadAnime`, and `SaveAnime` with 64 MiB gzip-compressed messages, exhaustive proto-to-SQL mapping, and `DEAD_PAGE_REVIEW_<PROVIDER>` support), plus development and production Docker builds.

Required environment variable at startup: `DATABASE_URL`.

## Commands

```bash
cargo run -p postgres_service
cargo test -p postgres_service
./pants package crates/postgres_service:postgres-service-image
docker compose -f docker/docker-compose.dev.yml up -d --build postgres-service
docker exec echora-dev-postgres-service grpc_health_probe -addr=localhost:8004
```
