---
title: Temporal Infrastructure
date: 2026-11-15
tags:
  - infrastructure
  - temporal
  - scheduling
  - durable-execution
status: planned
related:
  - "[[update_service]]"
  - "[[Architecture Index]]"
---

# Temporal Infrastructure

## Overview

Temporal is the **durable execution engine** that runs the [[update_service|Update Service]]. It keeps every workflow's state, resumes a workflow after a crash without repeating finished steps, holds thousands of timers (one per tracked episode) and keeps a history of every run.

---

## Why Temporal

| Requirement | APScheduler | Temporal |
|-------------|-------------|----------|
| Thousands of timers and schedules, each at its own time | ❌ in-process only, no persistence | ✅ durable timers and the Schedules API |
| Survive a process crash, resume mid-execution | ❌ | ✅ event history replay |
| Retry with backoff | ⚠️ written by hand | ✅ declared per activity |
| Share a provider's request rate across workers | ❌ | ✅ task queue rate limits, enforced on the server |
| Put user requests ahead of background work | ❌ | ✅ task queue priority and fairness |
| Horizontal scale | ❌ | ✅ add worker replicas |

**Key guarantee**: if the Update Service worker crashes mid-workflow, Temporal replays the event history and resumes at the step that was running. Completed steps are not run again.

---

## Infrastructure Components

### Temporal Server

The Temporal Server keeps workflow state, timers and task queues, and dispatches tasks. It never runs workflow code; the workers do.

- **Image:** `temporalio/server`, 1.31 or later, the first release with task queue priority and fairness generally available. Pin the release current at setup time from the [release page](https://github.com/temporalio/temporal/releases), and use the `temporalio/admin-tools` image of the same version. (P-ECHO-21)
- **Databases:** PostgreSQL, with two databases of its own, `temporal` (executions and history) and `temporal_visibility` (what the UI lists and filters). No Elasticsearch is needed at this size.
  - ~~In development the CLI dev server keeps its own file database instead, see [[#Local Development]]. (ECHO-95)~~
  - In production they get a cluster of their own, or at least their own connection budget, since Temporal writes and vacuums heavily. (P-ECHO-21)
- **Deployment:** Docker containers in development and staging, Kubernetes in production with the official Helm chart, at a release that ships server 1.31 or later. (P-ECHO-21)

| Port | Purpose |
|------|---------|
| `7233` | gRPC frontend (workers and clients connect here) |
| `8080` | Temporal Web UI |

> [!warning] `temporalio/auto-setup` is deprecated
> The `auto-setup` image, which created the schema and namespace inside the server container, [no longer receives updates](https://hub.docker.com/r/temporalio/auto-setup) and has no release past 1.29. The schema is now set up by a one-shot `admin-tools` container before the server starts, as Temporal's own [`samples-server` compose files](https://github.com/temporalio/samples-server) do. (P-ECHO-21)

### Temporal Worker

The Update Service, a Rust process built with the [Temporal Rust SDK](https://temporal.io/blog/build-durable-applications-rust-temporal-rust-sdk-now-generally-available), registers the workflow and activity implementations and polls the Temporal Server for tasks. Its activities call the Enrichment Service, the Agent Service and the PostgreSQL Service over gRPC with `tonic`. (~~ECHO-111~~, ECHO-115)

- **Task queues:** see [[update_service#Temporal Worker Configuration|Update Service]].
  - `update-service` for workflows and merges, and `agent-service` for AI activities. (ECHO-95)
  - One `provider-<name>` queue per provider. (ECHO-113)
- **Scaling:** add worker replicas. Temporal spreads tasks across every worker polling a queue, and the rate limits below stay the same however many workers there are. (ECHO-95)

---

## Temporal Concepts in This System

| Concept | Usage |
|---------|-------|
| **Workflow** | A durable, resumable run: one refresh of a work, an episode or a character, a sweep, a seed load, or one tracked episode. |
| **Activity** | One step of a workflow, a gRPC call to another service. Retried on its own when it fails. |
| **Timer** | A durable wait, such as a tracked episode sleeping until its air time. |
| **Schedule** | A recurring start: the weekly sweep, the daily score refresh and the daily episode tracking, each with the `Skip` overlap policy so a run still going is never started twice. |
| **Signal** | A message to a running workflow, such as `merge_requested` sent to a work's refresh after a synopsis is written. |
| **Priority and fairness** | The order tasks leave a queue: user requests first, then episode tracking, then the sweep and the seed load sharing what is left. |
| **Namespace** | Echora workflows run in the `echora` namespace in production and in `echora-dev` in development. |

---

## Namespace Configuration

| Setting | Value |
|---------|-------|
| ~~Namespace~~ | ~~`echora` in production, `echora-dev` in development (ECHO-95)~~ |
| Retention | 7 days. A week of refreshes is hundreds of thousands of closed workflows, and a longer retention multiplies what the visibility database holds and what its count queries scan; measured at full volume before production. Set on `echora` by the deployed `admin-tools` step; the dev server keeps Temporal's default (P-ECHO-21) |
| Archival | Disabled; enable for long-term audit if needed |
| ~~Search attribute~~ | ~~`RefreshTrigger`, a keyword, created with `temporal operator search-attribute create --namespace echora --name RefreshTrigger --type Keyword`, and with `--search-attribute RefreshTrigger=Keyword` on the dev server (ECHO-95)~~ |

---

## Task Queue Settings

### Priority and fairness

> **Linear:** ECHO-113

[Task Queue Priority and Fairness](https://docs.temporal.io/develop/task-queue-priority-fairness) are generally available from server [1.31.0](https://github.com/temporalio/temporal/releases/tag/v1.31.0), for self-hosted servers too. That release turns on the new matcher by default, so priority keys are respected without extra configuration.

- **Priority** is on by default. Each workflow or activity sets a priority key from 1 (first) to 5, and a task without one gets 3. Activities inherit their workflow's priority unless they set their own.
- **Fairness** must be turned on in the dynamic config with `matching.enableFairness: true`, for the namespace or for the provider queues. Tasks then carry a fairness key and weight, and tasks of one priority are dispatched in proportion to their keys' weights.

The keys the Update Service uses are listed in [[update_service#Provider rate limits|Update Service]].

### Partitions

> **Linear:** ECHO-113

A task queue has 4 partitions by default, and tasks are spread across them at random. Priority and fairness are exact only within one partition, so with several partitions a sweep task on a lightly loaded partition can go ahead of a user's task on a busy one. The provider queues carry less than one task a second, far below the roughly 400 a second that needs more than one partition, so each runs on a single partition:

```yaml
# dynamic config, one entry per provider queue
matching.numTaskqueueReadPartitions:
  - constraints: { namespace: 'echora', taskQueueName: 'provider-anidb' }
    value: 1
matching.numTaskqueueWritePartitions:
  - constraints: { namespace: 'echora', taskQueueName: 'provider-anidb' }
    value: 1
```

Read and write partitions are set to the same value. On a running server, lowering them changes the write partitions first and the read partitions after, so no task is left on a partition nobody reads. The `update-service` queue keeps the default.

### Rate limits

> **Linear:** ECHO-113

Each provider queue's rate limit is set on the queue itself, not in each worker:

```bash
temporal task-queue config set \
  --namespace echora \
  --task-queue provider-anidb \
  --task-queue-type activity \
  --queue-rps-limit 0.00174 \
  --queue-rps-limit-reason "AniDB daily budget: 150 a day"
```

A limit set this way takes precedence over `max_task_queue_activities_per_second` in the workers' options, so workers deployed with different values cannot fight over it. A per-fairness-key limit (`fairness-key-rps-limit-default`) can also cap the seed load's share of a provider. The limits live in a script beside the deployment, so they are versioned with it.

### Worker Versioning

> **Linear:** ECHO-95

[Worker Versioning](https://temporal.io/blog/replay-2026-product-announcements) is generally available and replaces the older build-ID versioning. A running workflow stays on the worker version that started it, so a deploy never breaks a workflow in flight, and workflow code needs no patching branches for every change. Each workflow type picks its behaviour by how long it runs:

- **Pinned:** refreshes and `WriteSynopsisWorkflow`, which finish in minutes, so an old worker version drains quickly.
- **Moves to new code:** `SweepCatalogueWorkflow`, `LoadSeedReleaseWorkflow` and `FetchBudgetedPagesWorkflow` continue as new between batches and take the new version there. Episode trackers, which live up to two weeks, use auto-upgrade, with patching where their steps change.

~~A one-day spike with the Rust SDK proves each feature the workflows use before the Update Service is built: the `USE_EXISTING` conflict policy, signal-with-start, schedules, search attributes and visibility counts, priority and fairness keys, and Worker Versioning. The Rust SDK's worker options take deployment-based versioning through `WorkerDeploymentOptions`. (ECHO-95)~~

---

## Local Development

> **Linear:** ECHO-95

~~Local development uses the [Temporal CLI](https://docs.temporal.io/cli) dev server, one process with an in-memory or file database, the Web UI on port 8233, and no separate containers. (ECHO-95)~~

```bash
temporal server start-dev \
  --namespace echora-dev \
  --db-filename .temporal/dev.db \
  --search-attribute RefreshTrigger=Keyword \
  --dynamic-config-value matching.enableFairness=true \
  --dynamic-config-value matching.numTaskqueueReadPartitions=1 \
  --dynamic-config-value matching.numTaskqueueWritePartitions=1
```

~~One partition for every queue is fine on a dev server, which runs a single process. (ECHO-95)~~

~~When the services run in containers through `docker/docker-compose.dev.yml`, the same dev server runs there too, from the `temporalio/temporal` image with the same flags and a volume for its file database, so the containerized Update Service worker reaches it inside Compose. The CLI dev server stays for running the worker on the host. Both are for development only. (ECHO-95)~~

~~The Rust SDK's `testing` feature starts and stops a dev server from tests, through `temporal_workflows`' `test-util` feature. (ECHO-95)~~

## Deployed Setup

> **Linear:** P-ECHO-21

Staging and production run the server image against PostgreSQL, with the schema set up first:

```yaml
temporal-schema-setup:
  image: temporalio/admin-tools:${TEMPORAL_VERSION}
  restart: on-failure:6
  depends_on:
    postgres:
      condition: service_healthy
  environment:
    - POSTGRES_SEEDS=postgres
    - POSTGRES_USER=temporal
  # temporal-sql-tool --plugin postgres12 creates and migrates
  # the temporal and temporal_visibility databases; safe to rerun
  command: ["/scripts/setup-postgres.sh"]

temporal:
  image: temporalio/server:${TEMPORAL_VERSION}
  depends_on:
    temporal-schema-setup:
      condition: service_completed_successfully
  environment:
    - DB=postgres12
    - DB_PORT=5432
    - POSTGRES_SEEDS=postgres
    - POSTGRES_USER=temporal
    - DYNAMIC_CONFIG_FILE_PATH=config/dynamicconfig/echora.yaml
  ports:
    - "7233:7233"

temporal-ui:
  image: temporalio/ui:${TEMPORAL_UI_VERSION}
  environment:
    - TEMPORAL_ADDRESS=temporal:7233
  ports:
    - "8080:8080"
```

The password comes from the environment file, never from the compose file. The dynamic config file sets `matching.enableFairness`, and a one-shot `admin-tools` step after the server starts creates the `echora` namespace with a retention of 7 days when it is missing and applies the task queue rate limits.

---

## Operational Notes

- **Workflow determinism**: workflow code never reads the system clock, draws random numbers or does I/O. It uses the SDK's workflow time and runs every side effect as an activity. (P-ECHO-12)
- **Activity retries and timeouts**: set per activity. Starting values are in [[update_service#Activity timeouts|Update Service]]. (~~ECHO-111~~)
- **Workflow timeouts**: long-running workflows continue as new instead of growing one history.
  - The weekly sweep continues as new between pages. (ECHO-117)
  - The seed load continues as new between batches. (ECHO-120)
- **Payload size**: activities pass only IDs, keys and flags, never page bodies or merged records. See [[update_service#Only references pass through Temporal|Update Service]]. (~~ECHO-111~~)

---

## Writing Workflow Code

Use the [Temporal Developer Skill](https://temporal.io/blog/introducing-temporal-developer-skill) when writing or reviewing workflow and activity code with a coding agent. It covers determinism rules, activity patterns, retry policies, error handling, testing, worker configuration and versioning.

- **Install in Claude Code:** `/plugin marketplace add temporalio/agent-skills`, then install `temporal-developer` from the plugin manager and restart the session.
- **Rust:** the skill's Rust guidance is in Public Preview and leaves child workflows, Continue-as-New, cancellation and timers to the [Rust SDK documentation](https://docs.temporal.io/develop/rust) and [docs.rs](https://docs.rs/temporalio-sdk/latest/temporalio_sdk/). Check Rust specifics, such as `ActivityOptions`, `WorkerOptions` and `Priority`, there.

The rules these workflows follow, each explained in [[update_service|Update Service]]:

- Other workflows are started through the client in an activity, never as child workflows. (~~ECHO-111~~)
- An activity waiting on a long gRPC call heartbeats while it waits.
  - `refresh_page` heartbeats while `FetchPage` runs. (~~ECHO-111~~)
  - `write_synopsis` heartbeats while `WriteSynopsis` runs. (ECHO-115)
- A gRPC status a retry cannot fix is returned as a non-retryable failure. (~~ECHO-111~~)
- A workflow checks for a pending `merge_requested` before it ends. (~~ECHO-111~~)

---

## Related Documentation

- [[update_service|Update Service]] — workflows and activities running on this infrastructure
- [[postgres_integration_architecture_decision|PostgreSQL Service]] — the PostgreSQL cluster Temporal shares
- [[Architecture Index|Architecture Index]] — service overview

---

**Status**: Planned | **Last Updated**: 2026-11-15
