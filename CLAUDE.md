# CLAUDE.md

This file is the root guidance for Claude Code when working in this repository.

## Repository Purpose

`echora` is a microservice for semantic search over anime content using vector embeddings and Qdrant.
It supports text, image, and multimodal search, and includes production-oriented health checks, monitoring, and CORS support.

## Core Repo Map

- `apps/service/`: FastAPI API layer
- `libs/`: Shared business/domain libraries
- `tests/`: Test suite
- `docker/`: Local dev stack definitions
- `scripts/`: Operational and data-processing scripts

## Default Workflow

Use Pants as the primary interface for test/lint/check in this monorepo.

```bash
# Install deps
uv sync --extra dev

# Run vector service locally for development
./pants run apps/vector_service:vector_service

# Run enrichment service locally for development
./pants run apps/enrichment_service:enrichment_service

# Test
./pants test ::

# Format/lint/type-check
./pants fmt lint check ::
```

## High-Signal Commands

```bash
# Full local stack
docker compose -f docker/docker-compose.dev.yml up -d

# Service health
curl http://localhost:8002/health

# Qdrant health
curl http://localhost:6333/health
```

## Task-Specific Docs (Progressive Disclosure)

Open these only when relevant to the current task:

- Development setup, test variants, and quality commands:
  `docs/claude/development-workflows.md`
- Service architecture and component details:
  `docs/claude/architecture.md`
- Enrichment pipeline and stage-by-stage scripts:
  `docs/claude/enrichment-pipeline.md`
- Qdrant/vector maintenance operations:
  `docs/claude/vector-db-operations.md`
- Class/member ordering conventions:
  `docs/claude/code-conventions.md`

## Environment (Critical)

- `QDRANT_URL` (default: `http://localhost:6333`)
- `QDRANT_COLLECTION_NAME` (default: `anime_database`)
- `TEXT_EMBEDDING_MODEL` (default: `BAAI/bge-m3`)
- `IMAGE_EMBEDDING_MODEL` (default: `ViT-L-14/laion2b_s32b_b82k`)
- `VECTOR_SERVICE_PORT` (default: `8002`)
