---
title: Temporal Infrastructure
date: 2026-02-19
tags:
  - infrastructure
  - temporal
  - scheduling
  - durable-execution
status: active
related:
  - "[[update_service]]"
  - "[[Architecture Index]]"
---

# Temporal Infrastructure

## Overview

Temporal is the **durable execution engine** that powers the [[update_service|Update Service]]. It provides guaranteed workflow execution, automatic crash recovery, and dynamic per-item scheduling — capabilities that APScheduler, n8n, and custom retry logic cannot reliably replicate at production scale.

---

## Why Temporal

| Requirement | APScheduler | n8n scheduler | Temporal |
|-------------|-------------|--------------|----------|
| Per-anime dynamic schedules (1000s of items) | ❌ in-process only, no persistence | ❌ not designed for per-item | ✅ native Schedules API |
| Survive process crash, resume mid-execution | ❌ | ❌ manual restart required | ✅ event history replay |
| Deterministic retry with backoff | ⚠️ manual | ⚠️ node-level only | ✅ declarative per-activity |
| Horizontal scale (add workers) | ❌ | ❌ queue mode issues | ✅ add worker replicas |
| Production track record | Limited | Weak at this use case | Uber, Netflix, Stripe, DoorDash |

**Key guarantee**: if the Update Service worker crashes mid-workflow (e.g. during step 5 of 10), Temporal replays the event history and resumes at step 5. Completed steps are not re-executed.

---

## Infrastructure Components

### Temporal Server

The Temporal Server manages workflow state, scheduling, and task dispatch. It does not run user code — that is the worker's job.

**Backend**: PostgreSQL (same PostgreSQL cluster as application DB, separate database `temporal`)

**Deployment**: Docker container (dev/stage), Kubernetes (prod)

| Port | Purpose |
|------|---------|
| `7233` | gRPC frontend (workers + clients connect here) |
| `8080` | Temporal Web UI |

### Temporal Worker

The Update Service Python process that registers workflow and activity implementations and polls Temporal Server for tasks.

**Task queue**: `update-service`

**Horizontal scaling**: add more worker replicas; Temporal distributes tasks across all registered workers automatically.

---

## Temporal Concepts in This System

| Concept | Usage |
|---------|-------|
| **Workflow** | A durable, resumable execution unit. One workflow type per update category (`EpisodeAirTrackingWorkflow`, `ScoreSyncWorkflow`, etc.) |
| **Activity** | A single unit of work within a workflow (API call, DB query, NATS publish). Retried independently if it fails. |
| **Schedule** | A recurring trigger that starts a workflow on a cron or interval. One Schedule per airing anime for episode tracking. |
| **Namespace** | Logical isolation. All Echora workflows run in the `echora` namespace. |

---

## Namespace Configuration

| Setting | Value |
|---------|-------|
| Namespace | `echora` |
| Retention | 30 days (workflow history kept for debugging) |
| Archival | Disabled (v1); enable for long-term audit if needed |

---

## Dev Setup

```yaml
# docker/docker-compose.dev.yml addition
temporal:
  image: temporalio/auto-setup:1.24
  environment:
    - DB=postgresql
    - DB_PORT=5432
    - POSTGRES_USER=temporal
    - POSTGRES_PWD=temporal
    - POSTGRES_SEEDS=postgres
  ports:
    - "7233:7233"
  depends_on:
    - postgres

temporal-ui:
  image: temporalio/ui:2.26
  environment:
    - TEMPORAL_ADDRESS=temporal:7233
  ports:
    - "8080:8080"
```

---

## Operational Notes

- **Workflow determinism**: Temporal workflows must be deterministic. Do not call `datetime.now()`, `random()`, or perform I/O directly in workflow code — use activities for all side effects.
- **Activity retries**: Configure per-activity retry policies. External API calls (Jikan, AniList) should use exponential backoff with a max of 5 attempts.
- **Workflow timeouts**: Set `execution_timeout` on long-running workflows to prevent zombie executions.
- **Worker versioning**: Use Temporal's worker versioning (build IDs) when deploying breaking workflow changes to avoid in-flight workflow failures.

---

## Related Documentation

- [[update_service|Update Service]] — workflows and activities running on this infrastructure
- [[Architecture Index|Architecture Index]] — service overview

---

**Status**: Planned | **Last Updated**: 2026-02-19
