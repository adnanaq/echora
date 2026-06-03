---
title: Event Adapter
date: 2026-02-19
tags:
  - services
  - event-adapter
  - nats
  - n8n
  - bridge
status: active
related:
  - "[[n8n_orchestration]]"
  - "[[event_driven_architecture]]"
  - "[[command_api]]"
  - "[[Architecture Index]]"
---

# Event Adapter

## Overview

The Event Adapter is a small **Python service** that bridges NATS JetStream (binary Protobuf) to n8n (HTTP/JSON webhooks). Without it, n8n has no way to react to internal domain events — it can only be triggered by its own schedules or HTTP calls.

It is intentionally thin: subscribe, deserialise, redact, forward. No business logic lives here.

---

## Why It Exists

NATS and n8n speak different protocols:
- **NATS**: binary Protobuf messages over NATS JetStream
- **n8n**: HTTP webhook endpoints expecting JSON payloads

The Event Adapter decouples these two transports. Neither NATS nor n8n needs to know about the other's protocol.

> [!important] Redaction Policy
> Only minimal, non-sensitive identifiers leave the private network toward n8n Cloud. No raw payloads, no PII, no full entity data. The Command API is the channel for n8n to fetch further details if needed.

---

## Subscribed NATS Subjects

| Subject | n8n Workflow triggered | Payload forwarded |
|---------|----------------------|-------------------|
| `anime.dlq.*` | DLQ triage + replay orchestration | `{ event_id, subject, error_type, entity_id, source_service, attempt_count }` |
| `anime.updated` (risk-flagged only) | Risky update approval workflow | `{ event_id, anime_id, changed_fields[], risk_reason }` |
| `ingestion.review.ready` | Stage-6 review alert | `{ run_id, entry_count, review_ref }` |

> [!note] Selective forwarding
> Not every NATS event is forwarded. Only events that require n8n orchestration (DLQ events, risk-flagged updates, review notifications) pass through the adapter. High-volume operational events (`anime.synced`, `anime.created`, etc.) are never forwarded.

---

## Redaction Rules

Before forwarding to n8n Cloud:

| Field | Action |
|-------|--------|
| `anime_id`, `run_id`, `entry_key`, `event_id` | Pass through (non-sensitive identifiers) |
| `changed_fields[]` | Pass through (field names only, no values) |
| `error_type`, `source_service` | Pass through |
| Any field value (synopsis, title, score, etc.) | **Redact** — replaced with `[REDACTED]` |
| PII (voice actor names, staff names) | **Redact** |

---

## Forward Mechanism

1. Pull consumer on NATS JetStream (durable, explicit ACK)
2. Deserialise Protobuf message
3. Apply redaction rules
4. `POST` redacted JSON payload to configured n8n webhook URL
5. On HTTP 2xx: ACK message
6. On HTTP 4xx (non-retryable): ACK + log error (avoid infinite loop)
7. On HTTP 5xx or timeout: NAK (NATS redelivers after `ack_wait`)

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `NATS_URL` | `nats://localhost:4222` | NATS server URL |
| `N8N_WEBHOOK_BASE_URL` | — | Base URL of n8n webhook endpoint |
| `N8N_DLQ_WEBHOOK_PATH` | `/webhook/dlq-triage` | Path for DLQ events |
| `N8N_RISK_UPDATE_WEBHOOK_PATH` | `/webhook/risky-update` | Path for risk-flagged update events |
| `N8N_REVIEW_READY_WEBHOOK_PATH` | `/webhook/review-ready` | Path for stage-6 review alerts |
| `N8N_WEBHOOK_SECRET` | — | HMAC secret for webhook signature verification |
| `ACK_WAIT_SECONDS` | `30` | NATS ack wait before redelivery |
| `MAX_DELIVER` | `5` | Max redelivery attempts before routing to DLQ |

---

## Security

- n8n webhooks are HMAC-signed: adapter adds `X-Signature-256` header to every forward
- n8n verifies signature before processing (prevents spoofed callbacks)
- Cloudflare Tunnel provides TLS and zero-trust access control on the n8n side
- The adapter only sends to pre-configured webhook URLs — no dynamic routing from message content

---

## Related Documentation

- [[n8n_orchestration|n8n Orchestration]] — workflows triggered by this adapter
- [[event_driven_architecture|Event-Driven Architecture]] — NATS subjects, DLQ configuration
- [[command_api|Command API]] — inbound direction (n8n → platform)

---

**Status**: Planned | **Last Updated**: 2026-02-19
