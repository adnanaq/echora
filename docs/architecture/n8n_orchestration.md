---
title: n8n Orchestration
date: 2026-02-19
tags:
  - orchestration
  - n8n
  - workflows
  - ingestion
  - review
status: active
related:
  - "[[command_api]]"
  - "[[event_adapter]]"
  - "[[update_service]]"
  - "[[event_driven_architecture]]"
  - "[[Database Schema]]"
  - "[[Architecture Index]]"
---

# n8n Orchestration

## Role in the System

n8n is the **async orchestration and control plane** — not an application service. It owns scheduling, human-in-the-loop decision gates, incident routing, and cross-service workflow coordination. It does **not** sit in the user-facing query path and does not own per-item stateful execution (that is Temporal's role).

> [!important] Boundary Rule
> n8n calls the [[command_api|Command API]] exclusively. It never talks directly to internal gRPC services, PostgreSQL, NATS, or Qdrant. All state mutations flow through the Command API.

---

## What n8n Owns

| Flow | Trigger | Human step? |
|------|---------|-------------|
| Incremental ingestion orchestration | Weekly scheduled tag check | Yes (stage-6 review gate) |
| Bootstrap ingestion trigger | Manual / API call | Yes (stage-6 review gate) |
| Stage-6 review gate (wait/resume) | After stages 1–5 complete | Yes — reviewer submits decisions |
| Risky update approval workflow | `anime.updated` risk flag event | Yes — approver approves/rejects |
| DLQ triage + replay orchestration | DLQ event from Event Adapter | Optional (auto or manual replay) |
| Daily score sync trigger | Schedule (06:00 UTC) | No |
| Postgres → Qdrant drift reconciliation | Schedule (weekly) | No |

---

## Deployment

| Environment | Hosting |
|-------------|---------|
| `dev` | Self-hosted Docker (local) |
| `stage` | Self-hosted Docker (VPS) |
| `prod` | n8n Cloud |

**Connectivity (prod)**: n8n Cloud cannot reach private services directly. A **Cloudflare Tunnel** (zero-trust, no open ports) exposes the [[command_api|Command API]] endpoint to n8n Cloud. Sensitive payload fields are redacted before leaving the private network.

---

## Ingestion Orchestration — State Machine

### Trigger types

| Type | What starts it | v1 scope |
|------|---------------|----------|
| `bootstrap` | Manual / API call | Full ingest of a source snapshot |
| `incremental` | Scheduled tag check | New entries since last snapshot only |
| `partial` | Temporal (score sync, field updates) | Handed off to Update Service via Command API |
| `replay` | DLQ event or operator action | Re-execute failed work idempotently |

### Incremental flow state machine

| State | Purpose | Transition(s) |
|-------|---------|---------------|
| `CheckTag` | Compare latest upstream tag against `source_snapshots` | unchanged → `NoopComplete`; changed → `CreateSnapshot` |
| `NoopComplete` | No new tag — end run | terminal `success` |
| `CreateSnapshot` | Persist new snapshot metadata | → `CreateRun` |
| `CreateRun` | Create ingestion run + acquire run lock | → `FetchArtifact` |
| `FetchArtifact` | Download JSONL artifact | → `BuildDiff`; error → `FailRun` |
| `BuildDiff` | Compute new `entry_key`s not in `processed_entries` | → `PrecheckAmbiguity` |
| `PrecheckAmbiguity` | Detect same-provider multi-link ambiguity | → `ProcessStages1To5` |
| `ProcessStages1To5` | Run enrichment stages 1–5 for all candidate entries | → `BuildStage6ReviewPacket` |
| `BuildStage6ReviewPacket` | Build consolidated review payload; emit review alert | → `WaitForStage6Decisions` |
| `WaitForStage6Decisions` | **Pause** until reviewer decisions submitted | decisions → `ApplyStage6Decisions`; timeout → `FinalizeRun` |
| `ApplyStage6Decisions` | Expand batch rules, resolve precedence, route per entry | → `IngestApproved`, `ReplaySpecificStage`, `MarkRejected` |
| `IngestApproved` | Publish approved entries via Command API | → `FinalizeRun` |
| `ReplaySpecificStage` | Re-run from stage N for selected entries | → `ProcessStages1To5` (from N) |
| `MarkRejected` | Terminal reject for entry | → `FinalizeRun` |
| `FinalizeRun` | Compute final run status from per-entry outcomes | terminal `success\|partial_success\|failed` |
| `FailRun` | Handle blocking run failure | terminal `failed` |

**Terminal status rules**:
- `success` — all entries reached terminal decision and all non-rejected decisions applied
- `partial_success` — some entries processed; deferred/timeout/retry-exhausted items remain
- `failed` — blocking run-level failure before meaningful processing

---

## Stage-6 Review Gate

All candidate entries pause at stage 6 for manual decision before ingestion. No per-entry UI — review is artifact-based.

### Review flow
1. n8n writes review artifact (stored in S3/MinIO): `review_required_<run_id>.jsonl`
2. n8n emits aggregate review alert (Slack/webhook) — one alert per run, not per entry
3. Reviewer prepares decision artifact: `decisions_<run_id>.jsonl`
4. Reviewer submits decisions via `POST /workflow/submit-stage6-decisions`
5. n8n validates, expands batch rules, routes each entry

### Decision options

| Decision | Meaning |
|----------|---------|
| `approve_ingest` | Ingest as-is |
| `approve_partial_ingest` | Ingest with `partial_enrichment=true` |
| `send_back_to_stage` | Re-run from specified stage, return to stage-6 gate |
| `reject_entry` | Terminal reject for this run |
| `defer` | Leave as `review_pending` |

### Decision precedence
- Explicit `entry_key` decision overrides matching batch decision
- Multiple conflicting batch rules: latest `submitted_at` wins
- Unresolved entries remain `review_pending`

### Decision artifact fields
- `entry_key` — required for entry-level decision
- `selector` — required for batch decision (e.g. `{"issues_count": 0}`)
- `decision` — one of the options above
- `target_stage` — required when `send_back_to_stage`
- `merge_resolution` — `approve_merge_lock | keep_separate` (optional)
- `reason`, `reviewed_by`

---

## Idempotency Key Strategy

All Command API calls from n8n carry an `Idempotency-Key` header. Keys are deterministic per action.

| Action | Key format |
|--------|-----------|
| Tag check | `tag-check:manami:<tag_name>` |
| Start incremental run | `run-incremental:<snapshot_sha256>` |
| Process entry | `process-entry:<run_id>:<entry_key>` |
| Stage-6 decision | `stage6-review:<run_id>:<entry_key>:<decision_id>` |
| Stage replay | `replay-stage:<run_id>:<entry_key>:<stage_name>:<decision_id>` |

**Duplicate behavior**:
- Same key + same request hash → return prior result (idempotent replay)
- Same key + different request hash → `409 idempotency_conflict`
- Retention window: 30 days

---

## Ingestion `entry_key` (v1 Locked)

```
entry_key = sha256(join("|", sort(unique(trim(sources[])))))
```

- Derived from `sources` array only — no title, type, season, year, status, score
- Metadata header row (`$schema`) excluded before key generation
- Stored as lowercase hex digest

---

## Workflow-as-Code

Workflows are exported as JSON and version-controlled in Git. Deployments via CI (GitHub Actions + n8n CLI import). Environment-specific credentials managed via n8n credential manager, not committed to Git.

---

## Connectivity Architecture

```
n8n Cloud
  │ HTTPS (OAuth2 + JWT)
  ▼
Cloudflare Tunnel
  │
  ▼
Command API (FastAPI)
  │ gRPC / internal
  ├─► PostgreSQL Service
  ├─► Update Service (Temporal)
  └─► Enrichment Service

NATS JetStream
  │ (DLQ events, risk flags)
  ▼
Event Adapter
  │ HTTPS webhook (redacted JSON)
  ▼
n8n Cloud
```

---

## Related Documentation

- [[command_api|Command API]] — n8n → internal services contract
- [[event_adapter|Event Adapter]] — NATS → n8n webhook bridge
- [[update_service|Update Service]] — Temporal-based partial update workflows
- [[event_driven_architecture|Event-Driven Architecture]] — NATS setup and consumer config
- [[Database Schema]] — ingestion state tables (`source_snapshots`, `ingestion_runs`, `ingestion_entries`, `ingestion_entry_stages`, `ingestion_reviews`, `processed_entries`)

---

**Status**: Active | **Last Updated**: 2026-02-19
