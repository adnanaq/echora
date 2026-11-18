# `postgres_service`

gRPC server binary owning the PostgreSQL catalogue database.

## Overview

This crate provides the service binary, `mimalloc` allocator, `common` configuration/telemetry bootstrap, `grpc.health.v1.Health` endpoint on port `8004`, and development/production Docker builds. Database migrations and catalogue RPC handlers are planned next.

## Commands

```bash
cargo run -p postgres_service
cargo test -p postgres_service
./pants package crates/postgres_service:postgres-service-image
docker compose -f docker/docker-compose.dev.yml up -d --build postgres-service
docker exec echora-dev-postgres-service grpc_health_probe -addr=localhost:8004
```
