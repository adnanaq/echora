# Kubernetes Deployment Plan — Echora Services + Observability

## Status: Planning
## Last Updated: 2026-02-28

---

## Overview

Deploy Echora's microservices and full observability stack to Kubernetes with a
**dev-prod parity** strategy: the development environment inside k3d mirrors the
production topology so that moving to a cloud cluster requires only changing
Helm values files and secrets — no manifest rewrites.

**Core strategy:**

- **Dev:** All services + all observability backends run inside a local k3d
  cluster. Tilt provides hot-reload developer experience.
- **Prod:** Same Helm charts, different values. Observability backends replaced
  by Grafana Cloud free tier. Application services run on a cloud k8s cluster.

### Repository Strategy

Echora is a multi-repo, multi-language system. Kubernetes/Helm configuration
lives in a **dedicated deploy repo** (`echora-deploy`), separate from all
service repos. Each service repo builds and pushes container images; the deploy
repo defines how they run in the cluster.

| Repo | Language | What it builds | Image pushed to GHCR |
|---|---|---|---|
| `echora-obs` | Python | vector-service, enrichment-service, ingestion-pipeline | `vector-service:sha-*`, `enrichment-service:sha-*` |
| `echora-backend` | Rust | Backend BFF (GraphQL) | `backend:sha-*` |
| `echora-pg` | Rust | PostgreSQL Service (gRPC + GraphQL) | `postgres-service:sha-*` |
| `echora-agent` | Python | Agent Service (LLM orchestration) | `agent-service:sha-*` |
| `echora-worker` | Python | Update Worker (APScheduler) | `update-worker:sha-*` |
| `echora-contracts` | Protobuf | Shared types (buf generate) | N/A (published as package) |
| **`echora-deploy`** | **YAML/Helm** | **All Helm charts, values, Tiltfile, ArgoCD config** | **N/A** |

**Why a separate deploy repo:**

1. **Cross-repo visibility** — one place shows what's running in the cluster
   across all services, all languages, all repos.
2. **Deploy cadence differs from code** — scaling replicas, changing resource
   limits, or rotating secrets doesn't require touching service code.
3. **ArgoCD watches one repo** — GitOps requires a single source of truth for
   cluster desired state.
4. **Independent CI triggers** — service repo CI builds images; deploy repo CI
   validates Helm charts and triggers deployments. No circular dependencies.

---

## What We Have Today (Local Docker)

```
docker-compose.dev.yml       — vector-service + enrichment-service + qdrant + redis
docker-compose.obs.yml       — otel-collector + prometheus + loki + tempo + grafana + alertmanager
docker-compose.obs.scale.yml — overlay: agent + gateway collector topology (tail sampling)
docker-compose.prd.yml       — production-like stack with resource limits, pinned versions
```

Both services have production-ready Dockerfiles (`Dockerfile.prd`):

- Multi-stage build (deps → runtime) using Python 3.12-slim + uv
- Non-root user (`appuser`, uid 1000)
- `grpc_health_probe` v0.4.37 binary with SHA256 verification
- Ports: vector-service `8001`, enrichment-service `8002`
- OTel SDK embedded in `libs/observability/` — **no code changes needed for k8s**
- Model cache env vars: `HF_HOME`, `TORCH_HOME`, `MODEL_CACHE_DIR` all point to
  `/app/model_cache`

**Model cache bind-mount:** In `docker-compose.dev.yml` the model cache volume
is bind-mounted to `${HOME}/.local/share/echora/model-cache` (on the home
partition) rather than using a Docker named volume. This is intentional — the
~2.2 GB of ML models would otherwise land on the root filesystem via Docker's
default storage driver, and on systems where `/` has limited space this fills
the root partition. The same concern applies to k3d: its default `local-path`
provisioner stores PVCs under `/var/lib/rancher/k3s/storage/` (root fs). The
k8s deployment must either use a `hostPath` volume pointing to a home-partition
directory, or configure k3d with a volume mount from the host home directory
into the k3s node.

---

## Tooling Decisions

| Concern | Tool | Rationale |
|---|---|---|
| Local k8s cluster | **k3d** | ~20s startup, wraps k3s in Docker, multi-node, mirrors prod |
| Manifest management | **Helm** | Charts for own services + third-party. Values files per env |
| Dev workflow | **Tilt** | Web dashboard, auto-rebuild on code changes, native `helm()` support |
| OTel Collector mgmt | **OTel Operator** (Helm) | CRD-based collector lifecycle, DaemonSet/Deployment modes |
| GitOps (future) | **ArgoCD** | Web UI, Helm-native, simple for small teams |
| Secrets (dev) | **SOPS + age** | Encrypted in git, decrypted at apply time |
| Secrets (prod) | **External Secrets Operator** | Syncs from AWS SM / GCP SM / Vault |
| Container registry | **GHCR** | GitHub-native, free for public repos, OCI-compliant |
| Image tags | **semver + SHA** | Immutable SHA for prod, floating `dev-latest` for dev |
| Service mesh | **None initially** | Add Linkerd later if gRPC per-request LB is needed |

### Why Helm (Not Kustomize)

Helm was chosen as the single manifest management tool because:

1. **Unified approach** — same tool for own services and third-party charts
   (OTel Operator, Prometheus, Loki, Tempo, Grafana, Qdrant)
2. **Values files per environment** — `values-dev.yaml`, `values-prod.yaml`
   drive all differences (replicas, resources, endpoints, secrets)
3. **Tilt integration** — `k8s_yaml(helm(...))` in Tiltfile renders charts
   with dev values and watches for changes
4. **ArgoCD integration** — ArgoCD natively understands Helm charts, making
   future GitOps adoption straightforward
5. **Release management** — `helm upgrade --install` provides atomic deploys
   with rollback via `helm rollback`

---

## Full Service & Infrastructure Inventory

Source of truth: `echora/docs/Architecture Index.md` (main repo, latest as of
2026-02-19).

### Application Services

| Service | Language | Status | Role | Interface | K8s Kind |
|---------|----------|--------|------|-----------|----------|
| **vector-service** | Python | ✅ Built | Vector search + Qdrant admin | gRPC :8001 | Deployment |
| **enrichment-service** | Python | ✅ Built | Enrichment pipeline orchestration | gRPC :8002 | Deployment |
| **postgres-service** | Rust | ⏳ Planned | System of record, dual API | gRPC + GraphQL | Deployment |
| **backend-bff** | Rust | ⏳ Planned | External API gateway | GraphQL :3000 | Deployment |
| **agent-service** | Python | ⏳ Partial | LLM query orchestration | gRPC :8003 | Deployment |
| **ingestion-pipeline** | Python | ⏳ Planned | Initial full enrichment | NATS publisher | Job / Deployment |
| **update-service** | Python | ⏳ Planned | Ongoing partial updates | Temporal worker + NATS | Deployment |
| **command-api** | Python (FastAPI) | ⏳ Planned | n8n → internal bridge | REST/HTTP | Deployment |
| **event-adapter** | Python | ⏳ Planned | NATS → n8n webhook bridge | NATS consumer + HTTP | Deployment |

### Infrastructure

| Component | Purpose | K8s Kind | Notes |
|-----------|---------|----------|-------|
| **NATS JetStream** | Event bus | StatefulSet (3) | `ANIME_EVENTS` (7d), `ANIME_DLQ` (30d), `NOTIFICATION_EVENTS` (3d) |
| **PostgreSQL** | App DB + Temporal backend | StatefulSet (1) | Two databases: `echora`, `temporal` |
| **Qdrant** | Vector database | StatefulSet (1) | `anime_database` collection, multi-vector |
| **Redis** | Idempotency + HTTP cache | StatefulSet (1) | TTL-managed keys |
| **Temporal Server** | Durable workflow engine | Deployment (1) | PostgreSQL-backed, port :7233 |
| **Temporal UI** | Workflow dashboard | Deployment (1) | Port :8080 |
| **MinIO** | Artifact storage (dev) | StatefulSet (1) | Stage artifacts, review files |
| **n8n** | Orchestration control plane | Deployment (1) | Self-hosted dev, n8n Cloud prod |
| **Cloudflare Tunnel** | n8n Cloud → Command API | Deployment (1) | Prod only, zero-trust |

### Observability

| Component | Purpose | K8s Kind | Notes |
|-----------|---------|----------|-------|
| **OTel Collector** | Telemetry pipeline | Deployment (dev) / DaemonSet+Deployment (prod) | All services emit OTLP |
| **Prometheus** | Metrics storage | StatefulSet (dev) / Grafana Cloud (prod) | SLO alerting |
| **Loki** | Log storage | StatefulSet (dev) / Grafana Cloud (prod) | Structured logs |
| **Tempo** | Trace storage | StatefulSet (dev) / Grafana Cloud (prod) | Distributed traces |
| **Grafana** | Dashboards | Deployment (dev) / Grafana Cloud (prod) | Pre-provisioned dashboards |
| **Alertmanager** | Alert routing | Deployment (dev) / Grafana Cloud (prod) | SLO burn-rate alerts |

---

## Target Architecture

### Development (k3d + local backends)

```
┌──────────────────────────────── k3d cluster ────────────────────────────────┐
│                                                                              │
│  namespace: echora-dev                                                      │
│  ┌────────────────────── Application Services ──────────────────────┐       │
│  │                                                                   │       │
│  │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐    │       │
│  │  │ backend-bff      │ │ agent-service    │ │ command-api      │   │       │
│  │  │ Rust/axum        │ │ Python/LLM       │ │ Python/FastAPI   │   │       │
│  │  │ :3000 GraphQL    │ │ :8003 gRPC       │ │ :8080 REST       │   │       │
│  │  └────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘  │       │
│  │           │                    │                     │            │       │
│  │  ┌────────▼─────────┐ ┌───────▼──────────┐ ┌───────▼──────────┐ │       │
│  │  │ postgres-service  │ │ vector-service    │ │ event-adapter    │ │       │
│  │  │ Rust/sqlx         │ │ Python/gRPC       │ │ Python/NATS→n8n  │ │       │
│  │  │ :8004 gRPC+GQL    │ │ :8001 gRPC        │ │ (no ext port)    │ │       │
│  │  └──────────────────┘ └──────────────────┘ └──────────────────┘ │       │
│  │                                                                   │       │
│  │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐    │       │
│  │  │ enrichment-svc   │ │ ingestion-pipe   │ │ update-service   │   │       │
│  │  │ Python/gRPC      │ │ Python/batch     │ │ Python/Temporal  │   │       │
│  │  │ :8002 gRPC       │ │ (Job/Deploy)     │ │ (worker)         │   │       │
│  │  └─────────────────┘ └─────────────────┘ └─────────────────┘    │       │
│  └───────────────────────────────────────────────────────────────────┘       │
│                                                                              │
│  namespace: infra                                                           │
│  ┌────────────────────── Data & Messaging ──────────────────────────┐       │
│  │                                                                   │       │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐  │       │
│  │  │ PostgreSQL    │ │ Qdrant       │ │ NATS JetStream            │ │       │
│  │  │ :5432         │ │ :6333/:6334  │ │ :4222 (client)            │ │       │
│  │  │ echora DB     │ │ anime_db     │ │ ANIME_EVENTS (7d)         │ │       │
│  │  │ temporal DB   │ │ collection   │ │ ANIME_DLQ (30d)           │ │       │
│  │  └──────────────┘ └──────────────┘ │ NOTIFICATION_EVENTS (3d)  │ │       │
│  │                                     └──────────────────────────┘  │       │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐             │       │
│  │  │ Redis         │ │ Temporal     │ │ Temporal UI   │            │       │
│  │  │ :6379         │ │ :7233        │ │ :8080         │            │       │
│  │  └──────────────┘ └──────────────┘ └──────────────┘             │       │
│  │                                                                   │       │
│  │  ┌──────────────┐ ┌──────────────┐                               │       │
│  │  │ MinIO (S3)    │ │ n8n           │                              │       │
│  │  │ :9000/:9001   │ │ :5678         │                              │       │
│  │  └──────────────┘ └──────────────┘                               │       │
│  └───────────────────────────────────────────────────────────────────┘       │
│                                                                              │
│  namespace: monitoring                                                      │
│  ┌────────────────────── Observability ─────────────────────────────┐       │
│  │                                                                   │       │
│  │  ALL services emit OTLP → otel-collector:4317                    │       │
│  │                                                                   │       │
│  │  ┌──────────────────────────────────────┐                        │       │
│  │  │ otel-collector (Deployment, 1)       │                        │       │
│  │  │ :4317 (gRPC) :4318 (HTTP)            │                        │       │
│  │  └──────────┬───────────────────────────┘                        │       │
│  │       ┌─────┼──────────┐                                         │       │
│  │       ▼     ▼          ▼                                         │       │
│  │  ┌────────┐ ┌────────┐ ┌───────────┐ ┌──────────────┐           │       │
│  │  │ Tempo  │ │ Loki   │ │Prometheus │ │ Alertmanager │           │       │
│  │  │ :3200  │ │ :3100  │ │ :9090     │ │ :9093        │           │       │
│  │  └───┬────┘ └───┬────┘ └─────┬─────┘ └──────────────┘           │       │
│  │      └──────────┼────────────┘                                   │       │
│  │                 ▼                                                 │       │
│  │          ┌────────────┐                                          │       │
│  │          │ Grafana     │  ← pre-provisioned dashboards           │       │
│  │          │ :3000       │                                         │       │
│  │          └────────────┘                                          │       │
│  └───────────────────────────────────────────────────────────────────┘       │
│                                                                              │
│  Tilt port-forwards: grafana:3000, vector-service:8001, backend:3000,       │
│  qdrant:6333, prometheus:9090, temporal-ui:8080, n8n:5678, minio:9001      │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Production (cloud k8s + managed services)

```
┌──────────────────────────── cloud k8s cluster ──────────────────────────────┐
│                                                                              │
│  namespace: echora-prod                                                     │
│  ┌────────────────────── Application Services ──────────────────────┐       │
│  │                                                                   │       │
│  │  backend-bff (2-3, HPA)    agent-service (1-2)                   │       │
│  │  postgres-service (2)      vector-service (2-3, HPA)             │       │
│  │  enrichment-svc (1)        ingestion-pipeline (Job)              │       │
│  │  update-service (2+)       command-api (2)                       │       │
│  │  event-adapter (1)                                                │       │
│  └───────────────────────────────────────────────────────────────────┘       │
│                                                                              │
│  namespace: infra                                                           │
│  ┌────────────────────── Data & Messaging ──────────────────────────┐       │
│  │  PostgreSQL (managed or HA StatefulSet)                           │       │
│  │  NATS JetStream (StatefulSet, 3 replicas, clustered)             │       │
│  │  Redis (managed or StatefulSet)                                   │       │
│  │  Temporal Server (HA deployment)                                  │       │
│  └───────────────────────────────────────────────────────────────────┘       │
│                                                                              │
│  namespace: monitoring                                                      │
│  ┌────────────────────── Observability ─────────────────────────────┐       │
│  │  otel-agent (DaemonSet) → otel-gateway (Deployment, 2)          │       │
│  │  + k8sattributes        + tail sampling + log redaction          │       │
│  └──────────────────────────┬────────────────────────────────────────┘       │
│                              │                                               │
└──────────────────────────────┼───────────────────────────────────────────────┘
                               │
         ┌─────────────────────┼────────────────────┐
         ▼                     ▼                    ▼
   Qdrant Cloud          Grafana Cloud         n8n Cloud
   free tier (1 GB)      free tier             via Cloudflare Tunnel
   vector DB             metrics/logs/traces   orchestration
                         dashboards/alerting
```

---

## Namespace Strategy

```
echora-dev        — development (local k3d, feature branches)
echora-stg        — staging (pre-release, optional)
echora-prod       — production (cloud k8s)
infra             — stateful data services (PostgreSQL, NATS, Qdrant, Redis,
                    Temporal, MinIO) — shared within same environment
monitoring        — OTel Collector + obs backends (shared)
```

Services talk to the collector in `monitoring` via:
`otel-collector.monitoring.svc.cluster.local:4317`

Services talk to NATS in `infra` via:
`nats.infra.svc.cluster.local:4222`

To switch environments: same Helm charts, different values file.

---

## Deployment Flow: Code Push to Running Cluster

### How it works end-to-end

```
 SERVICE REPOS (each builds + pushes its own image)
 ─────────────────────────────────────────────────
 echora-obs (Python)          → GHCR: vector-service:sha-*, enrichment-svc:sha-*
 echora-backend (Rust)        → GHCR: backend-bff:sha-*
 echora-pg (Rust)             → GHCR: postgres-service:sha-*
 echora-agent (Python)        → GHCR: agent-service:sha-*
 echora-worker (Python)       → GHCR: update-service:sha-*
     │
     │  CI builds image, pushes to GHCR with sha + semver tags
     │
     ▼
 DEPLOY REPO (echora-deploy)
 ─────────────────────────────────────────────────
 values/dev.yaml              ← image tags updated (manual or CI bot PR)
 values/prod.yaml
 charts/*/                    ← Helm chart templates (stable, rarely change)
     │
     │  Dev: Tilt watches deploy repo, auto-renders Helm, rebuilds
     │  Prod: ArgoCD detects git change, syncs to cluster
     │
     ▼
 CLUSTER
 ─────────────────────────────────────────────────
 kubectl / helm / argocd applies the desired state
 Services pull images from GHCR
 OTel Collector receives telemetry from all services
```

### Data flow through the running system

```
INITIAL INGESTION
  n8n (tag check + orchestration)
    → Command API → Ingestion Pipeline (stages 1-5)
    → n8n (stage-6 review gate)
    → Command API → Ingestion Pipeline (publish)
    → NATS (anime.enriched)
    → PostgreSQL Service → DB + qdrant_outbox (atomic)
    → Outbox Worker → NATS (anime.created)
    → Qdrant Service (GetAnimeRecord gRPC → embed → upsert)

ONGOING UPDATES
  Update Service (Temporal workflows)
    → NATS (episode.aired / anime.updated / character.updated)
    → PostgreSQL Service → DB + qdrant_outbox (atomic, SAGA)
    → Outbox Worker → NATS (anime.synced / character.synced)
    → Qdrant Service (payload update and/or vector re-embed)

USER QUERY (LLM enabled)
  User → Backend BFF (GraphQL)
    → Agent Service (gRPC) → Vector Service + PostgreSQL Service
    → Backend → User

USER QUERY (LLM disabled / fallback)
  User → Backend BFF (GraphQL)
    → Vector Service (direct gRPC) + PostgreSQL Service (GraphQL)
    → Backend → User

OBSERVABILITY (always)
  Every service → OTLP gRPC → OTel Collector
    → Tempo (traces) + Loki (logs) + Prometheus (metrics) → Grafana
```

### Dev vs Prod differences

| Aspect | Dev (k3d) | Prod (cloud k8s) |
|---|---|---|
| Image source | Built locally by Tilt | Pulled from GHCR (sha-tagged) |
| Qdrant | StatefulSet in `infra` ns | Qdrant Cloud (API key in Secret) |
| PostgreSQL | StatefulSet in `infra` ns | Managed (RDS/Cloud SQL) or HA StatefulSet |
| NATS | StatefulSet (1 replica) | StatefulSet (3 replicas, clustered) |
| Temporal | Single auto-setup container | HA deployment or Temporal Cloud |
| MinIO | StatefulSet in `infra` ns | S3 / GCS |
| n8n | Self-hosted in `infra` ns | n8n Cloud (via Cloudflare Tunnel) |
| OTel Collector | Single Deployment | DaemonSet (agent) + Deployment (gateway) |
| Obs backends | Local Prometheus/Loki/Tempo/Grafana | Grafana Cloud free tier |
| Secrets | SOPS-encrypted in deploy repo | External Secrets Operator → AWS SM |
| Replicas | 1 per service | 2-3 per service + HPA |
| Resources | Minimal (fits in 8-16 GB k3d) | Full (per service requirements) |
| Deployment | `tilt up` (auto-rebuild) | ArgoCD (GitOps, auto-sync) |

---

## echora-deploy Repo Structure

This is a **separate repository** (`echora-deploy`) — not inside any service
repo. It is the single source of truth for what runs in the cluster.

```
echora-deploy/
│
├── charts/                          # Own Helm charts for Echora services
│   ├── vector-service/
│   │   ├── Chart.yaml
│   │   ├── values.yaml              # Defaults (shared across envs)
│   │   └── templates/
│   │       ├── deployment.yaml
│   │       ├── service.yaml
│   │       ├── serviceaccount.yaml
│   │       ├── pvc.yaml             # Model cache (conditional)
│   │       └── _helpers.tpl
│   │
│   ├── enrichment-service/
│   │   ├── Chart.yaml
│   │   ├── values.yaml
│   │   └── templates/...
│   │
│   ├── backend-bff/                 # Rust GraphQL backend
│   │   ├── Chart.yaml
│   │   ├── values.yaml
│   │   └── templates/...
│   │
│   ├── postgres-service/            # Rust gRPC + GraphQL data layer
│   │   ├── Chart.yaml
│   │   ├── values.yaml
│   │   └── templates/...
│   │
│   ├── agent-service/               # Python LLM orchestration
│   │   ├── Chart.yaml
│   │   ├── values.yaml
│   │   └── templates/...
│   │
│   ├── update-worker/               # Python scheduled jobs
│   │   ├── Chart.yaml
│   │   ├── values.yaml
│   │   └── templates/...
│   │
│   └── ingestion-pipeline/          # Python batch ingestion
│       ├── Chart.yaml
│       ├── values.yaml
│       └── templates/...
│
├── infra/                           # Third-party Helm value overrides
│   ├── nats/
│   │   └── values.yaml              # NATS JetStream config
│   ├── postgresql/
│   │   └── values.yaml              # Bitnami PostgreSQL chart values
│   ├── qdrant/
│   │   └── values.yaml              # Qdrant chart values
│   ├── temporal/
│   │   └── values.yaml              # Temporal server values
│   ├── redis/
│   │   └── values.yaml              # Redis values
│   ├── otel-collector/
│   │   └── values.yaml              # OTel Collector values
│   ├── prometheus/
│   │   └── values.yaml              # kube-prometheus-stack values
│   ├── loki/
│   │   └── values.yaml              # Loki values
│   ├── tempo/
│   │   └── values.yaml              # Tempo values
│   └── grafana/
│       ├── values.yaml
│       └── dashboards/              # Provisioned from echora-obs
│           ├── vector-service-overview.json
│           ├── enrichment-service-overview.json
│           ├── service-overview.json
│           └── trace-journey.json
│
├── values/                          # Environment-specific overrides
│   ├── dev.yaml                     # k3d, local backends, 1 replica
│   ├── staging.yaml                 # Cloud cluster, limited replicas
│   └── prod.yaml                    # Cloud cluster + Grafana Cloud
│
├── secrets/                         # SOPS-encrypted, safe to commit
│   ├── .sops.yaml                   # SOPS config (age public key)
│   ├── dev.enc.yaml                 # Encrypted dev secrets
│   └── prod.enc.yaml               # Encrypted prod secrets
│
├── argocd/                          # GitOps Application CRDs (future)
│   ├── dev-apps.yaml
│   └── prod-apps.yaml
│
├── namespaces.yaml                  # echora-dev, infra, monitoring
├── Tiltfile                         # Dev workflow (references all charts)
├── helmfile.yaml                    # Declarative multi-chart installs
└── README.md
```

### Third-Party Helm Charts (installed via helmfile or standalone)

| Component | Chart | Repo | Namespace |
|---|---|---|---|
| NATS JetStream | `nats/nats` | `nats.io` | `infra` |
| PostgreSQL | `bitnami/postgresql` | `bitnami` | `infra` |
| Qdrant | `qdrant/qdrant` | `qdrant.github.io` | `infra` |
| Temporal | `temporalio/temporal` | `temporal.io` | `infra` |
| Redis | `bitnami/redis` | `bitnami` | `infra` |
| OTel Collector | `opentelemetry-collector` | `open-telemetry` | `monitoring` |
| OTel Operator (prod) | `opentelemetry-operator` | `open-telemetry` | `monitoring` |
| Prometheus stack | `kube-prometheus-stack` | `prometheus-community` | `monitoring` |
| Loki | `loki` | `grafana` | `monitoring` |
| Tempo | `tempo` | `grafana` | `monitoring` |

---

## Tilt Integration

Tilt provides the developer experience for k8s — auto-rebuild, log tailing,
port forwarding, all in a web dashboard at `http://localhost:10350`.

### Example Tiltfile

```python
# echora-deploy/Tiltfile

# --- Configuration ---
load('ext://helm_resource', 'helm_resource', 'helm_repo')

# --- Namespaces ---
k8s_yaml('namespaces.yaml')

# --- Application Services ---
docker_build(
    'echora/vector-service',
    context='..',
    dockerfile='../apps/vector_service/Dockerfile.prd',
    live_update=[
        sync('../apps/vector_service/src', '/app/apps/vector_service/src'),
        sync('../libs', '/app/libs'),
    ],
)

docker_build(
    'echora/enrichment-service',
    context='..',
    dockerfile='../apps/enrichment_service/Dockerfile.prd',
    live_update=[
        sync('../apps/enrichment_service/src', '/app/apps/enrichment_service/src'),
        sync('../libs', '/app/libs'),
    ],
)

# Render Helm charts with dev values
k8s_yaml(helm('charts/vector-service', values=['values/dev.yaml']))
k8s_yaml(helm('charts/enrichment-service', values=['values/dev.yaml']))
k8s_yaml(helm('charts/qdrant', values=['values/dev.yaml']))
k8s_yaml(helm('charts/redis', values=['values/dev.yaml']))

# --- Observability Stack (dev only) ---
helm_repo('prometheus-community', 'https://prometheus-community.github.io/helm-charts')
helm_repo('grafana', 'https://grafana.github.io/helm-charts')
helm_repo('open-telemetry', 'https://open-telemetry.github.io/opentelemetry-helm-charts')

helm_resource(
    'otel-collector',
    'open-telemetry/opentelemetry-collector',
    namespace='monitoring',
    flags=['--values=values/dev-otel-collector.yaml'],
)

helm_resource(
    'prometheus',
    'prometheus-community/kube-prometheus-stack',
    namespace='monitoring',
    flags=['--values=values/dev-prometheus.yaml'],
)

helm_resource('loki', 'grafana/loki', namespace='monitoring',
              flags=['--values=values/dev-loki.yaml'])

helm_resource('tempo', 'grafana/tempo', namespace='monitoring',
              flags=['--values=values/dev-tempo.yaml'])

# --- Port Forwards ---
k8s_resource('vector-service', port_forwards=['8001:8001'])
k8s_resource('enrichment-service', port_forwards=['8002:8002'])
k8s_resource('qdrant', port_forwards=['6333:6333', '6334:6334'])
k8s_resource('grafana', port_forwards=['3000:3000'])
k8s_resource('prometheus', port_forwards=['9090:9090'])

# --- Resource Grouping ---
k8s_resource('vector-service', labels=['app'])
k8s_resource('enrichment-service', labels=['app'])
k8s_resource('qdrant', labels=['infra'])
k8s_resource('redis', labels=['infra'])
k8s_resource('otel-collector', labels=['observability'])
k8s_resource('prometheus', labels=['observability'])
k8s_resource('loki', labels=['observability'])
k8s_resource('tempo', labels=['observability'])
k8s_resource('grafana', labels=['observability'])
```

### Tilt Developer Workflow

```bash
# 1. Start k3d cluster (one-time)
# --volume mounts host home dir into k3s node to avoid filling root partition
# with ~2.2 GB ML model cache (see "Model Cache Strategy" section)
k3d cluster create echora-dev --agents 1 \
  --volume "${HOME}/.local/share/echora/k3s-storage:/k3s-storage@agent:0" \
  --port "8001:8001@loadbalancer" \
  --port "3000:3000@loadbalancer"

# 2. Start Tilt (watches for changes, auto-rebuilds)
cd k8s && tilt up

# 3. Open dashboard
# http://localhost:10350

# 4. Edit code — Tilt auto-syncs files into running containers
# Edit libs/observability/metrics.py → Tilt rebuilds → pod restarts

# 5. Tear down
tilt down
k3d cluster delete echora-dev
```

---

## OTel Collector Topology

### Development: Single Deployment

In k3d, a single OTel Collector Deployment receives all telemetry and exports
to local Prometheus, Loki, and Tempo. This mirrors the current
`docker-compose.obs.yml` topology.

The existing `docker/observability/otel-collector-config.yaml` is reused
as-is (receivers, processors, exporters pointing to local backends).

### Production: Agent + Gateway (Two-Tier)

Managed via the **OpenTelemetry Operator** using CRDs:

**Agent (DaemonSet)** — one per node, lightweight:

```yaml
apiVersion: opentelemetry.io/v1beta1
kind: OpenTelemetryCollector
metadata:
  name: otel-agent
  namespace: monitoring
spec:
  mode: daemonset
  config:
    receivers:
      otlp:
        protocols:
          grpc: { endpoint: 0.0.0.0:4317 }
          http: { endpoint: 0.0.0.0:4318 }
    processors:
      memory_limiter: { limit_mib: 128, spike_limit_mib: 32 }
      batch: { send_batch_size: 512 }
      k8sattributes:
        extract:
          metadata:
            - k8s.namespace.name
            - k8s.pod.name
            - k8s.deployment.name
            - k8s.node.name
    exporters:
      otlp:
        endpoint: otel-gateway-collector.monitoring:4317
        tls: { insecure: true }
    service:
      pipelines:
        traces:
          receivers: [otlp]
          processors: [memory_limiter, k8sattributes, batch]
          exporters: [otlp]
        metrics:
          receivers: [otlp]
          processors: [memory_limiter, k8sattributes, batch]
          exporters: [otlp]
        logs:
          receivers: [otlp]
          processors: [memory_limiter, k8sattributes, batch]
          exporters: [otlp]
```

**Gateway (Deployment)** — centralized, 2 replicas:

```yaml
apiVersion: opentelemetry.io/v1beta1
kind: OpenTelemetryCollector
metadata:
  name: otel-gateway
  namespace: monitoring
spec:
  mode: deployment
  replicas: 2
  config:
    receivers:
      otlp:
        protocols:
          grpc: { endpoint: 0.0.0.0:4317 }
    processors:
      memory_limiter: { limit_mib: 512, spike_limit_mib: 128 }
      batch: { send_batch_size: 2048, send_batch_max_size: 4096, timeout: 2s }
      tail_sampling:
        decision_wait: 10s
        policies:
          - name: keep-errors
            type: status_code
            status_code: { status_codes: [ERROR] }
          - name: keep-slow
            type: latency
            latency: { threshold_ms: 2000 }
          - name: keep-enrichment
            type: string_attribute
            string_attribute:
              key: service.name
              values: [enrichment-service]
          - name: sample-rest
            type: probabilistic
            probabilistic: { sampling_percentage: 10 }
      transform/log-redaction:
        log_statements:
          - context: log
            statements:
              - replace_pattern(body, "(?i)(api[_-]?key|password|token|secret)\\s*[:=]\\s*\\S+", "$$1=***REDACTED***")
    exporters:
      otlphttp/grafana_cloud:
        endpoint: "${env:GRAFANA_CLOUD_OTLP_ENDPOINT}"
        headers:
          authorization: "${env:GRAFANA_CLOUD_OTLP_TOKEN}"
    service:
      pipelines:
        traces:
          receivers: [otlp]
          processors: [memory_limiter, tail_sampling, batch]
          exporters: [otlphttp/grafana_cloud]
        metrics:
          receivers: [otlp]
          processors: [memory_limiter, batch]
          exporters: [otlphttp/grafana_cloud]
        logs:
          receivers: [otlp]
          processors: [memory_limiter, transform/log-redaction, batch]
          exporters: [otlphttp/grafana_cloud]
```

**Note:** The existing `otel-collector-agent-config.yaml` and
`otel-collector-gateway-config.yaml` in `docker/observability/` already
implement this topology for Docker Compose. The k8s CRDs above are the
equivalent using the OTel Operator.

---

## gRPC on Kubernetes

### Health Checks — Native gRPC Probes (K8s 1.24+)

Kubernetes 1.24+ supports native gRPC health probes without needing the
`grpc_health_probe` binary. Both services already implement the standard
gRPC health checking protocol.

```yaml
# vector-service — needs long startup for model download (~2.2 GB)
startupProbe:
  grpc:
    port: 8001
  failureThreshold: 60      # 60 × 5s = 5 minutes max startup
  periodSeconds: 5
readinessProbe:
  grpc:
    port: 8001
  periodSeconds: 10
livenessProbe:
  grpc:
    port: 8001
  periodSeconds: 15
  failureThreshold: 3

# enrichment-service — fast startup
startupProbe:
  grpc:
    port: 8002
  failureThreshold: 12      # 12 × 5s = 60s max startup
  periodSeconds: 5
readinessProbe:
  grpc:
    port: 8002
  periodSeconds: 10
livenessProbe:
  grpc:
    port: 8002
  periodSeconds: 15
  failureThreshold: 3
```

**Why startupProbe matters:** While the startup probe is running, liveness and
readiness probes are disabled. This prevents k8s from killing the vector-service
pod during the 2-5 minute model download on first boot.

### Load Balancing

gRPC uses HTTP/2 with long-lived connections. A standard `ClusterIP` Service
does connection-level balancing (all requests on one connection hit the same
pod). Solutions by complexity:

1. **Headless Service** (`clusterIP: None`) — client-side DNS round-robin.
   Sufficient for dev and low-traffic staging.
2. **Linkerd** — transparent L7 per-request load balancing. No code changes.
   Add when scaling vector-service to multiple replicas in production.
3. **Istio** — heavier, but provides full traffic management. Overkill for
   this project's current scale.

**Decision:** Start with headless services. Add Linkerd in production phase
if load balancing becomes an issue with multiple replicas.

---

## Key Helm Values: vector-service

```yaml
# charts/vector-service/values.yaml (defaults)
replicaCount: 1

image:
  repository: echora/vector-service
  tag: "dev-latest"
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 8001

env:
  QDRANT_URL: "http://qdrant:6333"
  QDRANT_COLLECTION_NAME: "anime_database"
  TEXT_EMBEDDING_MODEL: "BAAI/bge-m3"
  IMAGE_EMBEDDING_MODEL: "ViT-L-14/laion2b_s32b_b82k"
  OTEL_ENABLED: "true"
  OTEL_EXPORTER_OTLP_ENDPOINT: "http://otel-collector.monitoring.svc.cluster.local:4317"
  CACHE_ENABLED: "true"
  REDIS_URL: "redis://redis:6379/0"

resources:
  requests:
    memory: "2Gi"
    cpu: "500m"
  limits:
    memory: "4Gi"
    cpu: "2000m"

modelCache:
  enabled: true
  size: "5Gi"
  storageClass: ""   # uses default (local-path in k3d)

# Probes use native gRPC (K8s 1.24+)
probes:
  startup:
    grpc:
      port: 8001
    failureThreshold: 60
    periodSeconds: 5
  readiness:
    grpc:
      port: 8001
    periodSeconds: 10
  liveness:
    grpc:
      port: 8001
    periodSeconds: 15
    failureThreshold: 3
```

### Dev Values Override (`values/dev.yaml`)

```yaml
# Lighter resources for k3d
vector-service:
  resources:
    requests:
      memory: "1Gi"
      cpu: "250m"
    limits:
      memory: "4Gi"     # still need headroom for models
      cpu: "2000m"

enrichment-service:
  resources:
    requests:
      memory: "256Mi"
      cpu: "100m"
    limits:
      memory: "1Gi"
      cpu: "500m"

# Dev-specific env
global:
  environment: "development"
  logLevel: "DEBUG"
```

### Prod Values Override (`values/prod.yaml`)

```yaml
vector-service:
  replicaCount: 2
  image:
    repository: ghcr.io/your-org/vector-service
    tag: ""  # set by CI/CD via --set image.tag=sha-abc123
    pullPolicy: Always
  resources:
    requests:
      memory: "4Gi"
      cpu: "1000m"
    limits:
      memory: "8Gi"
      cpu: "2000m"
  env:
    QDRANT_URL: ""           # from Secret
    QDRANT_API_KEY: ""       # from Secret
    OTEL_EXPORTER_OTLP_ENDPOINT: "http://otel-agent-collector.monitoring:4317"
    CACHE_ENABLED: "false"   # no Redis in prod initially

enrichment-service:
  replicaCount: 1
  image:
    repository: ghcr.io/your-org/enrichment-service
    tag: ""  # set by CI/CD

global:
  environment: "production"
  logLevel: "INFO"
```

---

## Secrets Management

### Development: SOPS + age

SOPS encrypts secret values in-place within YAML files. Encrypted files are
safe to commit to git. Only the `age` private key (stored locally, never
committed) can decrypt.

```bash
# One-time: generate age keypair
age-keygen -o ~/.config/sops/age/keys.txt

# Create .sops.yaml in secrets/
cat > secrets/.sops.yaml <<EOF
creation_rules:
  - path_regex: \.enc\.yaml$
    age: "age1..."  # your public key
EOF

# Encrypt secrets
sops --encrypt secrets/dev.yaml > secrets/dev.enc.yaml

# Decrypt and apply (one command)
sops -d secrets/dev.enc.yaml | kubectl apply -f -
```

### Production: External Secrets Operator

```yaml
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: cloud-secrets
  namespace: echora-prod
spec:
  provider:
    aws:
      service: SecretsManager
      region: us-east-1

---
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: qdrant-credentials
  namespace: echora-prod
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: cloud-secrets
    kind: SecretStore
  target:
    name: qdrant-credentials
  data:
    - secretKey: url
      remoteRef:
        key: echora/prod/qdrant
        property: url
    - secretKey: api-key
      remoteRef:
        key: echora/prod/qdrant
        property: api_key
```

---

## Container Registry & Image Tagging

### Registry: GHCR

```
ghcr.io/your-org/vector-service:sha-a1b2c3d
ghcr.io/your-org/enrichment-service:v1.2.3
```

### Tagging Strategy

| Tag Format | Mutable? | Use For | In K8s Manifests? |
|---|---|---|---|
| `sha-<7chars>` | No | Production deployments | **Yes (prod)** |
| `v1.2.3` | No | Release tracking | Yes (prod) |
| `v1.2` | Yes (floating) | Latest minor tracking | No |
| `latest` | Yes | **Never in K8s** | **Never** |
| `main-<sha>` | No | Staging auto-deploy | Yes (staging) |
| `dev-latest` | Yes | Local k3d development | Yes (dev only) |

### CI/CD Image Build (GitHub Actions)

```yaml
# .github/workflows/build-images.yml
on:
  push:
    branches: [main]
    paths: ['apps/**', 'libs/**', 'pyproject.toml', 'uv.lock']

jobs:
  build:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        service:
          - { name: vector-service, dockerfile: apps/vector_service/Dockerfile.prd }
          - { name: enrichment-service, dockerfile: apps/enrichment_service/Dockerfile.prd }
    steps:
      - uses: actions/checkout@v4

      - name: Docker meta
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ghcr.io/${{ github.repository_owner }}/${{ matrix.service.name }}
          tags: |
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha,prefix=sha-
            type=ref,event=branch
            type=raw,value=dev-latest,enable=${{ github.ref == 'refs/heads/main' }}

      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - uses: docker/build-push-action@v6
        with:
          context: .
          file: ${{ matrix.service.dockerfile }}
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

---

## Resource Recommendations

### Application Services

| Service | CPU Request | CPU Limit | Memory Request | Memory Limit | Notes |
|---|---|---|---|---|---|
| vector-service | 500m | 2000m | 2Gi | 4Gi | BGE-M3 + ViT-L-14 models (~2.2 GB) |
| enrichment-service | 250m | 1000m | 512Mi | 1Gi | API calls + browser automation |
| postgres-service | 250m | 1000m | 256Mi | 512Mi | Rust, sqlx, dual API |
| backend-bff | 250m | 1000m | 256Mi | 512Mi | Rust, axum, GraphQL |
| agent-service | 500m | 2000m | 1Gi | 2Gi | LLM client calls, context assembly |
| ingestion-pipeline | 250m | 1000m | 512Mi | 1Gi | Batch enrichment, API calls |
| update-service | 250m | 500m | 256Mi | 512Mi | Temporal worker, API fetches |
| command-api | 100m | 500m | 128Mi | 256Mi | FastAPI, thin translation layer |
| event-adapter | 100m | 250m | 128Mi | 256Mi | NATS → HTTP, minimal processing |

### Infrastructure (dev)

| Service | CPU Request | CPU Limit | Memory Request | Memory Limit | Notes |
|---|---|---|---|---|---|
| PostgreSQL | 250m | 1000m | 512Mi | 1Gi | echora DB + temporal DB |
| Qdrant | 250m | 1000m | 512Mi | 2Gi | Vector DB, HNSW index |
| NATS JetStream | 100m | 500m | 256Mi | 512Mi | 3 streams, single node dev |
| Redis | 100m | 250m | 128Mi | 256Mi | Idempotency + HTTP cache |
| Temporal Server | 250m | 500m | 256Mi | 512Mi | Workflow state, PG backend |
| Temporal UI | 50m | 250m | 64Mi | 128Mi | Dashboard |
| MinIO | 100m | 500m | 256Mi | 512Mi | Artifact storage |
| n8n | 100m | 500m | 256Mi | 512Mi | Self-hosted orchestration |

### Observability (dev)

| Service | CPU Request | CPU Limit | Memory Request | Memory Limit | Notes |
|---|---|---|---|---|---|
| otel-collector | 100m | 500m | 128Mi | 512Mi | Single topology |
| Prometheus | 250m | 500m | 512Mi | 1Gi | Metrics + alert rules |
| Loki | 100m | 500m | 256Mi | 512Mi | Log storage |
| Tempo | 100m | 500m | 256Mi | 512Mi | Trace storage |
| Grafana | 100m | 250m | 128Mi | 256Mi | Dashboards |
| Alertmanager | 50m | 100m | 64Mi | 128Mi | Alert routing |

### Dev k3d cluster sizing

**Total estimated memory:** ~13-16 GB (all services + infra + obs).

**Recommendation:** k3d with 2 agents, Docker allocated **16 GB RAM** minimum.
On constrained machines, start with only the services you're actively developing
plus their direct dependencies — Tilt makes it easy to disable resources in
the dashboard.

**Phased startup strategy** (for machines with limited RAM):
1. **Core only** (~5 GB): postgres-service + vector-service + qdrant + redis + NATS + otel-collector
2. **+ Pipeline** (~8 GB): Add enrichment-service + ingestion-pipeline + temporal + minio
3. **+ Full stack** (~16 GB): Add all remaining services + full obs stack

---

## Open Questions / Decisions Needed

### 1. Model Cache Strategy (Most Important)

The vector service downloads ~2.2 GB of ML models (BGE-M3 + ViT-L-14) on first
boot. In k8s this needs a decision:

| Option | Pros | Cons | Recommended for |
|---|---|---|---|
| **PVC (ReadWriteOnce)** | Simple, reused on restart | Locks to 1 node, 5-min cold start on new nodes | Dev / single-replica staging |
| **Pre-bake models into image** | Zero cold start, deterministic | ~3 GB image, longer CI build | Production |
| **Init container download** | Flexible | Re-downloads on every pod restart | Avoid |

**Current recommendation:** PVC for dev (simple, k3d only has 1-2 nodes).
Pre-bake for production when CI/CD is mature.

**Disk space consideration:** The dev Docker Compose bind-mounts the model cache
to `${HOME}/.local/share/echora/model-cache` because root `/` has limited space
and Docker's default storage fills root. In k3d, the same problem exists —
`local-path` provisioner writes to `/var/lib/rancher/k3s/storage/` on the k3s
node container, which maps to Docker's root storage. Two solutions:

1. **k3d volume mount** (recommended): Mount a host directory into the k3s node
   at cluster creation time, then configure the PVC's `storageClass` or
   `hostPath` to use it:
   ```bash
   k3d cluster create echora-dev --agents 1 \
     --volume "${HOME}/.local/share/echora/k3s-storage:/k3s-storage@agent:0" \
     --port "8001:8001@loadbalancer" \
     --port "3000:3000@loadbalancer"
   ```
   Then use a `hostPath` volume in the vector-service Helm chart pointing to
   `/k3s-storage/model-cache`.

2. **Move Docker root** (`data-root` in daemon.json): Relocate all of Docker's
   storage to a partition with more space. Fixes the problem globally but
   requires Docker daemon restart.

### 2. Separate Qdrant Clusters Per Environment?

Qdrant Cloud free tier = 1 cluster. Options:

- **Single cluster, separate collections** per env (dev: `anime_database_dev`,
  prod: `anime_database`) — simplest
- **Separate free-tier accounts** per env — more isolation
- **Local Qdrant in k3d for dev** (chosen), cloud only for prod — best isolation

**Decision:** Local Qdrant for dev (runs in k3d). Qdrant Cloud only for prod.
No collection naming conflicts.

### 3. gRPC External Exposure

For now, services are internal (service-to-service). ClusterIP is sufficient.
When external access is needed:

- **Headless Service** for internal gRPC (default)
- **LoadBalancer** for external access to vector-service
- **Linkerd** for per-request load balancing at scale

### 4. Future Services

The Helm chart structure is designed to accommodate additional services.
When a new service is added:

1. Create `charts/<service-name>/` in echora-deploy with Chart.yaml + templates
2. Add service block to `values/dev.yaml` and `values/prod.yaml`
3. Add `docker_build()` + `k8s_yaml(helm(...))` to Tiltfile
4. Add to CI/CD build matrix

No existing charts or infrastructure need modification.

---

## K8s Readiness Audit

Audit of all currently built services for Kubernetes migration readiness.
Covers configuration, inter-service communication, health checks, and
required changes.

### Vector Service (`apps/vector_service/`)

**Overall: Ready** — minimal changes needed.

| Area | Status | Detail |
|------|--------|--------|
| Configuration | Ready | All settings via Pydantic `BaseSettings` → env vars. `libs/common/config/settings.py` |
| Qdrant connection | Ready | `QDRANT_URL` + `QDRANT_API_KEY` already supported in `QdrantConfig` |
| gRPC health | Ready | Standard `HealthServicer` with `SERVING`/`NOT_SERVING` in `main.py` |
| Graceful shutdown | Ready | SIGTERM handler → sets `NOT_SERVING` → `server.stop(grace=5)` → closes Qdrant client |
| OTel endpoint | Ready | `OTEL_EXPORTER_OTLP_ENDPOINT` env var, passed through to `setup_telemetry()` |
| Redis/Cache | Ready | `RedisConfig` in `libs/common/config/`, `EmbeddingCache` (L3) in `libs/vector_processing/cache.py`. DI via `runtime.py` — processors receive optional cache. Fail-open: Redis down = works without cache. |
| Service name | OK | Hardcoded `"echora-vector-service"` in `main.py` — acceptable, matches k8s service name |
| Model cache | Ready | `EMBEDDING_CACHE_DIR` env var controls model download path |

**Helm env vars mapping** (from docker-compose.dev.yml):

```yaml
env:
  ENVIRONMENT: "development"
  QDRANT_URL: "http://qdrant.infra.svc.cluster.local:6333"
  OTEL_EXPORTER_OTLP_ENDPOINT: "http://otel-collector.monitoring.svc.cluster.local:4317"
  REDIS_URL: "redis://redis.infra.svc.cluster.local:6379/0"
```

### Enrichment Service (`apps/enrichment_service/`)

**Overall: Ready** — needs PVC for data files.

| Area | Status | Detail |
|------|--------|--------|
| Configuration | Ready | Env-var driven, has `CacheConfig` in `libs/http_cache/` with full Redis settings |
| gRPC health | Ready | Standard `HealthServicer` in `main.py` |
| Graceful shutdown | Ready | SIGTERM handler, 5s grace period |
| HTTP cache (Hishel) | Ready | `CacheConfig` in `libs/http_cache/config.py`: `CACHE_ENABLED`, `REDIS_URL`, `FORCE_CACHE`, per-service TTLs |
| Inter-service calls | N/A | Only calls external APIs (Jikan, AniList, Kitsu, etc.) — no internal gRPC yet |
| NATS integration | N/A | Not integrated yet — will be needed when ingestion pipeline publishes events |
| File system | Needs PVC | Reads `/app/data` (anime-offline-database.json), writes `/app/assets/seed_data` |
| Service name | OK | Hardcoded `"echora-enrichment-service"` in `main.py` |

**Helm env vars mapping:**

```yaml
env:
  ENVIRONMENT: "development"
  OTEL_EXPORTER_OTLP_ENDPOINT: "http://otel-collector.monitoring.svc.cluster.local:4317"
  CACHE_ENABLED: "true"
  REDIS_URL: "redis://redis.infra.svc.cluster.local:6379/0"
  FORCE_CACHE: "true"
volumes:
  - name: data
    persistentVolumeClaim:
      claimName: enrichment-data  # anime-offline-database.json + seed data
```

### Observability Library (`libs/observability/`)

**Overall: Ready** — 3 minor improvements recommended.

| Area | Status | Detail |
|------|--------|--------|
| `setup_telemetry()` | Ready | Accepts endpoint, service_name, environment as params — all caller-controlled |
| Context propagation | Ready | NATS + Temporal inject/extract using W3C TraceContext (`context.py`) |
| Metrics registry | Ready | 13 instruments registered in `metrics.py` |
| gRPC interceptors | Ready | Server interceptor auto-instruments all gRPC calls |
| OTLP exporter TLS | Needs fix | `insecure=True` hardcoded in `interceptors.py` — fine for dev, breaks prod with TLS |
| Sample rates | Minor | `log_sample_rate` and `trace_sample_ratio` not exposed as env vars — currently passed as function args from each service's `main.py` |
| `setup_logging()` default | Minor | Has `localhost:4317` as fallback default — works because callers always override, but not ideal |

### Docker Compose

**No changes needed.** Docker Compose remains the quick local dev path (`docker compose up`
without k8s). The Helm charts will set the same env vars that docker-compose.dev.yml
currently sets. The two systems coexist — Docker Compose for quick local dev, k8s/Tilt
for k8s-mimicking dev.

---

## Code Changes Required

Changes ordered by priority. Items 1-3 are needed before Phase 1 k8s deployment.
Item 4+ are nice-to-haves that improve operational control.

### 1. OTLP exporter TLS toggle (required for prod)

**File:** `libs/observability/src/observability/interceptors.py`

`insecure=True` is hardcoded for the OTLP gRPC exporter. In dev (within-cluster
traffic) this is correct, but prod with Grafana Cloud requires TLS.

```python
# Current (hardcoded)
OtlpGrpcSpanExporter(endpoint=endpoint, insecure=True)

# Fix: respect OTEL_EXPORTER_OTLP_INSECURE env var (OTel SDK convention)
import os
insecure = os.getenv("OTEL_EXPORTER_OTLP_INSECURE", "true").lower() == "true"
OtlpGrpcSpanExporter(endpoint=endpoint, insecure=insecure)
```

One-line change per exporter call (span, metric, log exporters).

### 2. Qdrant API key passthrough (required for prod)

**File:** `libs/qdrant_db/src/qdrant_db/client.py`

Already supported — `QdrantConfig` has `qdrant_api_key: str | None = None`.
The client init in `apps/vector_service/src/vector_service/runtime.py` already
passes `api_key` conditionally. **No code change needed** — just set the env var
in prod Helm values.

### 3. Enrichment service PVC for data files

The enrichment service reads `anime-offline-database.json` from `/app/data` and
writes to `/app/assets/seed_data`. In k8s, this needs a PersistentVolumeClaim.

For dev with Tilt, a `hostPath` volume pointing to the local data directory is
simplest. For prod, a proper PVC or init container that downloads the data file.

### 4. Sample rate env vars (nice-to-have)

Expose `TRACE_SAMPLE_RATIO` and `LOG_SAMPLE_RATE` as env vars in the observability
config so Helm values can control them per environment (100% in dev, 10% in prod).
Currently these are hardcoded in each service's `main.py` call to `setup_telemetry()`.

---

## Ongoing Discussions

Active design decisions that are not yet resolved. These will be finalized before
or during implementation.

### 1. Caching Strategy for Vector Search

**Status:** Research complete, decision needed on implementation scope

#### The Problem

The query flow is: Frontend → Backend (Rust) → gRPC → Vector Service (Python/Qdrant).
With LLM enabled: Frontend → Backend → Agent Service (LLM) → Vector Service → Qdrant.
With LLM disabled: Frontend → Backend → Vector Service → Qdrant.

Natural language queries have a "many-to-one" problem — "anime like Cowboy Bebop",
"shows similar to Cowboy Bebop", and "Cowboy Bebop recommendations" are semantically
identical but textually different. Traditional exact-match caching misses these.

#### Why NOT Hishel

Hishel (used by enrichment service) is **not suitable** for vector search caching:

| Factor | Hishel | Redis (direct) |
|--------|--------|----------------|
| Protocol | HTTP-only (RFC 9111) | Protocol-agnostic |
| gRPC support | None — gRPC has no `Cache-Control`/`ETag` semantics | Full support via key-value |
| Cache key | URL + HTTP headers | Arbitrary (embedding hash, query hash, etc.) |
| Vector-aware | No | Yes (RediSearch module adds vector similarity) |

Hishel stays in the enrichment service for what it does well — caching HTTP responses
from external anime APIs (AniList, Kitsu, AniDB). The vector service needs Redis directly.

#### Two Caching Layers

Research identifies two distinct caching opportunities with different latency profiles:

**Layer 1 — Embedding Cache (text → vector)**

Cache the expensive model inference step. BGE-M3 takes ~200ms per embedding; a Redis
lookup takes <1ms. Embeddings are **deterministic** — same text + same model = same
vector — so the cache is 100% safe with zero false positive risk.

```
Key:   embed:{model_name}:{sha256(text)}
Value: float[] embedding (JSON-serialized or msgpack)
TTL:   24 hours (embeddings never change for same model version)
```

Performance: RedisVL `EmbeddingsCache` benchmarks show 6.86x speedup (0.0455s → 0.0066s).
This is the highest-impact, lowest-risk caching layer.

Ref: https://redis.io/docs/latest/develop/ai/redisvl/user_guide/embeddings_cache/

**Layer 2 — Search Result Cache (query → results)**

Cache the Qdrant search results themselves. Two approaches:

| Approach | Cache hit rate | Latency on hit | False positives | Complexity |
|----------|---------------|----------------|-----------------|------------|
| **Exact-match** (hash query+filters) | Low (only identical queries) | <1ms | Zero | Trivial |
| **Semantic cache** (embed query, similarity search cached queries) | Higher (catches rephrases) | 5-200ms (needs embedding + similarity search) | 1-5% | Moderate |

Exact-match key design:
```
Key:   search:{sha256(embedding_bytes + filters_json + str(limit))}
Value: JSON-serialized search results
TTL:   15min–4h (anime data is relatively stable)
```

Semantic cache uses a dedicated Qdrant collection (or Redis with RediSearch) to store
query-response pairs. On new query, embed it, search the cache collection for nearest
neighbor. If similarity > threshold (recommended: 0.90), return cached result.

Ref: https://qdrant.tech/articles/semantic-cache-ai-data-retrieval/

#### Recommended Architecture

Multi-layer caching, each layer progressively more expensive to check but catching
more queries:

```
Frontend query arrives at Rust Backend
  │
  ├─ L1: In-process LRU (moka crate)         ← <1μs, hot queries, per-instance
  │   Key: hash(query_text + filters)
  │   Capacity-bounded, no network round-trip
  │
  ├─ L2: Redis exact-match                    ← <1ms, shared across Rust instances
  │   Key: search:{hash(embedding + filters + limit)}
  │   TTL: 1-4 hours
  │
  │  MISS → call Vector Service via gRPC
  │
  ▼
Python Vector Service
  │
  ├─ L3: Embedding cache (Redis)              ← <1ms vs ~200ms model inference
  │   Key: embed:{model}:{hash(text)}
  │   TTL: 24h (deterministic, zero risk)
  │
  ├─ Qdrant vector search                     ← 5-50ms (Qdrant has internal HNSW
  │   (no external cache needed here —           warm cache via mmap)
  │    Qdrant's internal mmap/HNSW cache
  │    is already optimized)
  │
  └─ Return results → Rust backend caches in L1 + L2
```

**Key design decisions:**
- **L1 + L2 live in the Rust backend**, not the Python vector service. Caching at the
  backend avoids the gRPC call entirely — a larger latency saving than caching within
  the Python service after the gRPC call is already made.
- **L3 (embedding cache) lives in the Python vector service** because that's where
  BGE-M3 model inference happens.
- **Do NOT add a cache between Python and Qdrant.** Qdrant has its own internal warm
  cache via memory-mapped HNSW indexes. Another layer adds complexity without
  proportional benefit when Qdrant search is already 5-50ms.
- **Semantic caching (Phase 3, optional)** would replace or augment L2. Only worth
  implementing if query analytics show significant paraphrase patterns.

#### Cache Invalidation

Anime data changes via the Update Service → NATS → PostgreSQL → outbox → Qdrant path.
Cache invalidation options:

- **TTL-based (recommended start):** 1-4 hour TTL. Anime metadata changes infrequently
  enough that slightly stale results are acceptable. Score updates happen daily at
  06:00 UTC; episode air events are real-time but don't change search relevance.
- **Event-driven (future):** Subscribe to `anime.synced` events in the Rust backend.
  On receipt, invalidate L1 + L2 entries for that `anime_id`. Requires key design that
  allows per-anime invalidation (e.g., secondary index of anime_id → cache keys).

#### Implementation Phases

**Phase 1 — DONE:**
1. ~~Add `RedisConfig` to `libs/common/config/`~~ — Done. Standalone `RedisConfig(BaseModel)`
   with `redis_url`, `redis_max_connections`, `redis_socket_connect_timeout`, `redis_socket_timeout`.
   Integrated into `Settings` with env var routing and overlap assertions.
2. ~~Add embedding cache (L3) in the Python vector service~~ — Done. Custom async Redis
   wrapper (`EmbeddingCache` in `libs/vector_processing/cache.py`) with DI into
   `TextProcessor` and `VisionProcessor`. Key: `emb:{model}:{sha256}`, 7-day TTL,
   MGET/MSET for batch ops, fail-open on Redis errors. Unit tests cover hit/miss/batch/fail-open.

**Phase 2 (when Rust backend is built):**
3. Add in-process LRU cache (L1) in the Rust backend using `moka` crate
   (https://github.com/moka-rs/moka — 1.9k stars, async support, capacity-bounded).
4. Add Redis exact-match result cache (L2) in the Rust backend.
5. Add cache observability: Prometheus counters for `cache_hits_total` /
   `cache_misses_total` per layer, latency histograms by cache layer.

**Phase 3 (if analytics justify it):**
6. Add semantic caching using Qdrant two-collection pattern or RedisVL `SemanticCache`.
   Only worthwhile if query analytics show >20% of queries are rephrases of previous
   queries.

#### Current State in Codebase

- `REDIS_URL=redis://redis:6379/0` set in docker-compose for vector service — consumed by `RedisConfig`
- `libs/common/config/redis_config.py` — standalone `RedisConfig(BaseModel)`, decoupled from Hishel
- `libs/vector_processing/cache.py` — `EmbeddingCache` class (async Redis, fail-open, MGET/MSET)
- `apps/vector_service/runtime.py` — creates `EmbeddingCache` from `settings.redis.redis_url` if set,
  injects into `TextProcessor` and `VisionProcessor` via constructor DI
- `apps/vector_service/main.py` — closes embedding cache on shutdown
- `libs/http_cache/config.py` still owns Hishel-specific Redis config via `CacheConfig(BaseSettings)` —
  separate concern from embedding cache

#### Tools & Libraries

| Layer | Python | Rust |
|-------|--------|------|
| Embedding cache (L3) | Custom async Redis wrapper (`EmbeddingCache`) — **implemented** | N/A |
| In-process LRU (L1) | N/A | `moka` crate (async, concurrent, capacity-bounded) |
| Redis exact-match (L2) | N/A | `fred` or `redis-rs` crate |
| Semantic cache (L2 upgrade) | `redisvl` (`SemanticCache`) or Qdrant 2nd collection | `qdrant-client` crate |
| Monitoring | OTel metrics (existing) | OTel metrics (existing) |

#### References

- Redis EmbeddingsCache: https://redis.io/docs/latest/develop/ai/redisvl/user_guide/embeddings_cache/
- Qdrant semantic cache: https://qdrant.tech/articles/semantic-cache-ai-data-retrieval/
- Semantic cache with Qdrant + Rust: https://www.shuttle.dev/blog/2024/05/30/semantic-caching-qdrant-rust
- moka Rust cache: https://github.com/moka-rs/moka
- Redis LangCache (managed semantic cache, preview): https://redis.io/blog/spring-release-2025/
- Semantic cache implementation walkthrough: https://techcodex.io/blog/2025/06/semantic-cache-basics/

**Phase 1 complete.** RedisConfig + embedding cache (L3) implemented and tested.
Phase 2 depends on Rust backend availability.

---

## Prerequisites Checklist

### For Development (k3d)

- [ ] Install k3d: `brew install k3d` or `curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash`
- [ ] Install Tilt: `curl -fsSL https://raw.githubusercontent.com/tilt-dev/tilt/master/scripts/install.sh | bash`
- [ ] Install Helm: `brew install helm`
- [ ] Install SOPS + age: `brew install sops age`
- [ ] Docker running with >= 8 GB RAM allocated

### For Production (future)

- [ ] **Grafana Cloud** — grafana.com/auth/sign-up
  - Note OTLP gateway URL + Instance ID + API token
  - e.g. `https://otlp-gateway-prod-us-east-0.grafana.net/otlp`
- [ ] **Qdrant Cloud** — cloud.qdrant.io
  - Create free cluster (GCP us-east4 or AWS eu-west-1)
  - Note cluster URL + API key
- [ ] Cloud k8s cluster (Oracle Always Free or Civo)
- [ ] GHCR configured on GitHub repo

### Cloud k8s Options

| Provider | Monthly Cost | Notes |
|---|---|---|
| Oracle Cloud Always Free | **$0** | 4 ARM vCPUs + 24 GB RAM |
| Civo | ~$5–10 | k3s-based, fastest setup, $250 free credit |
| DigitalOcean LKE | ~$12 | Simple UI, $200 free credit |
| Linode/Akamai LKE | ~$10 | $100 free credit |

---

## Implementation Order

Implementation follows the service dependency order from the Architecture Index.
Each phase is a self-contained PR or set of PRs with a clear validation step.

### Phase 1 — Deploy Repo Bootstrap + Built Services

**Goal:** Get the two already-built services (vector-service, enrichment-service)
running in k3d with full observability — proving the entire toolchain works.

```
[ ] Create echora-deploy repo (GitHub)
[ ] Install prerequisites (k3d, Tilt, Helm, SOPS, age)
[ ] Create k3d cluster with home-partition volume mount
    k3d cluster create echora-dev --agents 1 \
      --volume "${HOME}/.local/share/echora/k3s-storage:/k3s-storage@agent:0"
[ ] Write namespaces.yaml (echora-dev, infra, monitoring)
[ ] Write Helm charts: vector-service, enrichment-service
[ ] Write Helm values for infra: qdrant, redis (dev StatefulSets)
[ ] Install OTel Collector via Helm (single deployment topology)
[ ] Install Prometheus, Loki, Tempo, Grafana, Alertmanager via Helm
[ ] Port Grafana datasources + dashboards to Helm values
    (copy JSON dashboards from echora-obs/docker/observability/grafana/)
[ ] Port Prometheus alert rules to Helm values
    (copy from echora-obs/docker/observability/prometheus-alert-rules.yaml)
[ ] Write values/dev.yaml with local backend configuration
[ ] Write Tiltfile: docker_build + helm + port-forwards + resource groups
[ ] Add QDRANT_API_KEY support to qdrant client (echora-obs repo, 1 line)
[ ] Set up SOPS + age for dev secrets
[ ] Validate: tilt up → services healthy → Grafana shows metrics/logs/traces
```

### Phase 2 — NATS + PostgreSQL + Temporal Infrastructure

**Goal:** Add the event bus, relational DB, and workflow engine — the foundation
that all remaining services depend on.

```
[ ] Add NATS JetStream Helm chart (infra/nats/values.yaml)
    Configure: ANIME_EVENTS (7d), ANIME_DLQ (30d), NOTIFICATION_EVENTS (3d)
[ ] Add PostgreSQL Helm chart (infra/postgresql/values.yaml)
    Configure: two databases (echora, temporal)
[ ] Add Temporal Server + UI Helm charts (infra/temporal/values.yaml)
    Configure: PG backend, echora namespace, 30-day retention
[ ] Add MinIO Helm chart (infra/minio/values.yaml)
    Configure: echora-ingestion bucket
[ ] Update Tiltfile: add infra resources + port-forwards
[ ] Validate: NATS streams created, PG accepts connections, Temporal UI at :8080
```

### Phase 3 — Remaining Application Services (as they are built)

**Goal:** Add Helm charts for each service as it reaches "built" status. Each
chart follows the same pattern — only the values differ.

```
[ ] postgres-service chart (Rust, gRPC + GraphQL, depends on PG + NATS)
[ ] ingestion-pipeline chart (Python, depends on NATS + enrichment-service + MinIO)
[ ] command-api chart (Python/FastAPI, depends on PG + NATS + ingestion-pipeline)
[ ] event-adapter chart (Python, depends on NATS, forwards to n8n)
[ ] n8n chart (self-hosted dev, depends on command-api webhook URL)
[ ] update-service chart (Python/Temporal worker, depends on Temporal + NATS)
[ ] agent-service chart (Python/LLM, depends on vector-service + postgres-service)
[ ] backend-bff chart (Rust/GraphQL, depends on postgres-service + vector-service)
[ ] Update Tiltfile for each new chart
[ ] Update values/dev.yaml for each new service
[ ] Validate per-service: tilt up → service healthy → events flow → traces visible
```

### Phase 4 — CI/CD + Image Registry

**Goal:** Automated image builds from all service repos, deployed to k3d or
cloud cluster via the deploy repo.

```
[ ] Set up GHCR on all service repo GitHub projects
[ ] Write .github/workflows/build-images.yml per service repo (matrix build)
    Use docker/metadata-action for semver + SHA tagging
[ ] Write values/prod.yaml (Grafana Cloud, Qdrant Cloud, GHCR image refs)
[ ] Write values/staging.yaml
[ ] Optional: CI bot that opens PR on echora-deploy when new image is pushed
[ ] Validate: git push → image built → image pulls in k3d
```

### Phase 5 — Production Deployment

**Goal:** Same charts, cloud cluster, managed backends.

```
[ ] Create Grafana Cloud + Qdrant Cloud accounts
[ ] Provision cloud k8s cluster (Oracle Always Free or Civo)
[ ] Install External Secrets Operator, configure with cloud secret store
[ ] Write OTel Operator CRDs (agent DaemonSet + gateway Deployment)
[ ] Configure Cloudflare Tunnel for n8n Cloud → command-api
[ ] Deploy with prod values: helm upgrade --install -f values/prod.yaml
[ ] Validate: all services healthy, Grafana Cloud shows live signals
[ ] Validate: ingestion pipeline → NATS → PG → outbox → Qdrant flow works
```

### Phase 6 — Production Hardening

**Goal:** Resilience, scaling, and operational maturity.

```
[ ] Pre-bake ML models into vector-service image (CI pipeline)
[ ] Add HorizontalPodAutoscaler for vector-service + backend-bff
[ ] NATS clustering (3 replicas, JetStream replication factor 3)
[ ] Add Linkerd for gRPC per-request load balancing
[ ] Add tail-based trace sampling to OTel gateway collector
[ ] Set up ArgoCD for GitOps deployment from echora-deploy repo
[ ] Add echora-stg namespace + staging values
[ ] Add NetworkPolicies (restrict cross-namespace traffic)
[ ] PodDisruptionBudgets for critical services
[ ] Temporal worker versioning (build IDs for safe workflow deploys)
```

---

## Cost Estimate

### Development (Local)

| Resource | Cost |
|---|---|
| k3d (local Docker) | $0 |
| Tilt (open source) | $0 |
| Helm (open source) | $0 |
| SOPS + age (open source) | $0 |
| **Total** | **$0** |

### Production (Free Tier)

| Resource | Cost |
|---|---|
| Qdrant Cloud (free tier, 1 GB) | $0 |
| Grafana Cloud (free tier) | $0 |
| GHCR (public repo) | $0 |
| GitHub Actions (2000 min/month) | $0 |
| Oracle Cloud Always Free k8s | $0 |
| **Total** | **$0** |

### Production (Beyond Free Tiers)

- Grafana Cloud: $8/1000 active series, $0.50/GB logs, $0.50/GB traces
- k8s compute: ~$10–30/month for 2-node cluster
- Object storage (self-hosted Loki/Tempo): ~$0.023/GB/month

---

## Related Docs

### This repo (echora-obs)

- `docs/observability_local_runbook.md` — local dev observability stack
- `docs/observability_deep_dive.md` — OTel instrumentation details
- `docs/observability_slos.md` — SLO definitions and burn-rate calculations
- `docs/observability_trace_journey.md` — trace visualization
- `docs/observability_nats_temporal_integration.md` — async context propagation
- `docs/observability_retention_tiering.md` — storage retention policies
- `docs/security_hardening_strategy.md` — security considerations
- `libs/observability/README.md` — observability library reference

### Main repo (echora) — canonical architecture docs

- `docs/Architecture Index.md` — master navigation hub for all architecture docs
- `docs/event_driven_architecture.md` — NATS JetStream, event flows, outbox/SAGA pattern
- `docs/event_schema_specification.md` — all protobuf event definitions + gRPC services
- `docs/Database Schema.md` — PostgreSQL DDL, ER diagram, ingestion state tables
- `docs/postgres_integration_architecture_decision.md` — PostgreSQL Service dual API design
- `docs/ingestion_pipeline.md` — 5-stage enrichment, entry_key, trigger modes
- `docs/update_service.md` — Temporal workflows, source adapter pattern, SAGA
- `docs/temporal_infrastructure.md` — Temporal server setup, dev/prod deployment
- `docs/n8n_orchestration.md` — orchestration control plane, stage-6 review gate
- `docs/command_api.md` — n8n → internal gRPC bridge, idempotency, auth
- `docs/event_adapter.md` — NATS → n8n webhook bridge, redaction policy
- `docs/echora_contracts.md` — schema ownership, proto toolchain, CI pipeline
