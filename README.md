# Echora

A anime data/search platform built as gRPC services on top of Qdrant and enrichment pipelines. The repo is a Pants monorepo with modular libraries for vector processing, database operations, and enrichment workflows.

## Features

- **Unified Multi-Vector Architecture**: High-performance semantic search using text and image vectors
- **Advanced Text Search**: BGE-M3 embeddings (1024-dim) for cross-platform semantic matching
- **Visual Search**: OpenCLIP ViT-L/14 embeddings (768-dim) for multi-source image similarity
- **Multi-Vector Fusion**: Native Qdrant RRF/DBSF fusion for optimal hybrid results
- **Modular Monorepo**: Clean separation of concerns with Pants build system

## Monorepo Structure

```text
echora/
├── apps/                              # Applications
│   ├── vector_service/                # gRPC search/admin service
│   │   └── src/vector_service/
│   │       └── routes/                # Search and admin gRPC handlers
│   ├── enrichment_service/            # gRPC enrichment orchestration service
│   │   └── src/enrichment_service/
│   └── agent_service/                 # gRPC agent service
│       └── src/agent_service/
│           ├── routes/                # Agent gRPC handlers
│           └── utils/
├── libs/                              # Shared libraries
│   ├── common/                        # Common models and configuration
│   │   └── src/common/
│   │       ├── config/                # Settings and configuration
│   │       ├── grpc/                  # Shared gRPC utilities
│   │       ├── models/                # Shared data models (AnimeRecord, etc.)
│   │       └── utils/                 # ID generation, datetime helpers
│   ├── enrichment/                    # Anime data enrichment pipeline
│   │   └── src/enrichment/
│   │       ├── pipeline/              # Multi-stage enrichment pipeline
│   │       ├── similarity/            # Character similarity (CCIP)
│   │       ├── sources/               # External source integrations
│   │       │   ├── base/              # Base helper + crawler framework
│   │       │   ├── anidb/
│   │       │   ├── anilist/
│   │       │   ├── anime_planet/
│   │       │   ├── animeschedule/
│   │       │   ├── anisearch/
│   │       │   ├── kitsu/
│   │       │   └── mal/               # MyAnimeList (anime, characters, episodes)
│   │       └── utils/
│   ├── http_cache/                    # HTTP response caching (Redis-backed)
│   │   └── src/http_cache/            # Cache manager, aiohttp adapter, result cache
│   ├── observability/                 # OpenTelemetry bootstrap (logs, traces, metrics)
│   │   └── src/observability/         # Telemetry setup, gRPC interceptor, metric registry
│   ├── qdrant_db/                     # Qdrant vector database client
│   │   └── src/qdrant_db/
│   │       ├── collection/            # Collection lifecycle + schema builder
│   │       └── utils/                 # Retry, dedup utilities
│   ├── vector_db_interface/           # Provider-agnostic vector DB interface
│   │   └── src/vector_db_interface/
│   │       └── interfaces/            # ABCs: search, document, collection, monitor
│   └── vector_processing/             # Vector embedding generation
│       └── src/vector_processing/
│           ├── embedding_models/      # Model backends
│           │   ├── text/              # Text model implementations (BGE-M3, etc.)
│           │   └── vision/            # Vision model implementations (OpenCLIP)
│           ├── processors/            # Embedding manager, reranker processor
│           ├── reranking/             # Cross-encoder reranking
│           └── utils/                 # Image downloading, caching
├── scripts/                           # Utility scripts (reindexing, validation, etc.)
├── tests/                             # Test suite (mirrors source structure)
│   ├── conftest.py                    # Root fixtures (settings, clients)
│   ├── apps/                          # App-level tests
│   │   ├── vector_service/            # unit/, integration/
│   │   └── enrichment_service/        # unit/
│   ├── libs/                          # Per-library test suites
│   │   ├── common/                    # unit/, integration/
│   │   ├── enrichment/                # unit/ (sources, pipeline, utils), integration/
│   │   ├── http_cache/                # unit/, integration/
│   │   ├── observability/             # unit/
│   │   ├── qdrant_db/                 # unit/ (collection), integration/
│   │   ├── vector_db_interface/       # unit/, integration/
│   │   └── vector_processing/         # unit/ (embedding_models, processors, reranking)
│   └── scripts/                       # Script tests
└── data/                              # Data storage (Qdrant, anime databases)
```

## Quick Start

Steps 0–2 are required for both paths. Then pick **Path A (Docker)** or
**Path B (local)** — you do not need both.

### 0. Prerequisites

- **Python 3.13** — pinned in `.python-version`; Pants, ty and ruff all target 3.13
- **Docker** and Docker Compose
- **UV** package manager

```bash
# Install UV (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"   # add to ~/.bashrc or ~/.zshrc
uv --version
```

### 1. Configure environment

```bash
cp .env.example .env
```

**Required.** `ENVIRONMENT` has no default — the services raise
`ValueError` on startup without it. `.env.example` ships with
`ENVIRONMENT=development` already set, so copying it is enough.

The Docker path sets its own environment inline and works without `.env`;
the local path does not.

### 2. Get the anime database

`data/` is gitignored, so the seed dataset is **not** in the clone. The
enrichment service reads it from:

```text
assets/seed_data/anime-offline-database.json
```

Download it from the [manami-project/anime-offline-database](https://github.com/manami-project/anime-offline-database)
project (this repo was last built against the `2026-02` tag — the tag is
recorded in the file's own `$schema` field) and place it at that path.

Skip this only if you will not run the enrichment pipeline. Without it the
enrichment service still starts and reports healthy — `RunPipeline` is what
fails.

---

### Path A — Docker (recommended)

```bash
docker compose -f docker/docker-compose.dev.yml up -d
```

| Service | Address |
| --- | --- |
| vector_service (gRPC) | `localhost:8001` |
| enrichment_service (gRPC) | `localhost:8002` |
| Qdrant dashboard | <http://localhost:6333/dashboard> |
| Redis | `localhost:6379` |
| RedisInsight | <http://localhost:5540> (add host `redis`, port `6379`) |

> **First run takes ~15 minutes.** It builds two images and then downloads the
> embedding models (BGE-M3 + OpenCLIP, several GB) into the `echora_model-cache`
> volume. `vector-service` stays `starting` until that finishes — this is why
> its healthcheck has a 15-minute `start_period`. Subsequent starts are fast;
> the models are only re-downloaded if you run `down -v`.

### Path B — Local (Pants)

```bash
# 1. Create the venv (UV reads .python-version)
uv venv
uv sync
.venv/bin/python --version        # Python 3.13.x

# 2. Start only the backing services
docker compose -f docker/docker-compose.dev.yml up -d qdrant redis

# 3. Run a service
./pants run apps/vector_service:vector_service        # gRPC on :8001
./pants run apps/enrichment_service:enrichment_service # gRPC on :8002
```

Pants resolves its own interpreter from `PATH`, independently of the venv. If
`./pants` cannot find a 3.13 interpreter:

```bash
uv python install 3.13   # creates ~/.local/bin/python3.13
```

---

### 3. Verify

Both services are **gRPC only** — there are no HTTP endpoints, so `curl` will
not work. Use the standard gRPC health protocol:

```bash
# From inside the containers (grpc_health_probe is installed in the image)
docker exec echora-vector-service     grpc_health_probe -addr=localhost:8001
docker exec echora-enrichment-service grpc_health_probe -addr=localhost:8002

# Qdrant is plain HTTP
curl http://localhost:6333/healthz
```

```bash
docker compose -f docker/docker-compose.dev.yml ps       # health status
docker compose -f docker/docker-compose.dev.yml logs -f vector-service
```

### 4. Optional — observability

Telemetry is **off by default** in dev. The OTel collector lives in a separate
stack on its own network, so leaving it enabled without that stack running just
produces `StatusCode.UNAVAILABLE` export errors.

```bash
# once per machine
docker network create echora_observability-network

# start the observability stack
docker compose -f docker/docker-compose.obs.yml up -d

# start the dev stack wired into it
docker compose -f docker/docker-compose.dev.yml \
               -f docker/docker-compose.obs-link.yml up -d
```

Grafana: <http://localhost:3000>. See the header of
`docker/docker-compose.obs-link.yml` for details.

### Shutting down

```bash
docker compose -f docker/docker-compose.dev.yml down     # keeps all data
docker compose -f docker/docker-compose.dev.yml down -v  # also wipes Qdrant
                                                         # data and the model
                                                         # cache (forces a
                                                         # multi-GB re-download)
```

## Development Workflow

This project supports both UV and Pants for development:

- **UV**: Faster (~200ms) for code quality checks (formatting, linting) during iteration
- **Pants**: Hermetic builds for tests, scripts, validation, and CI/CD (handles monorepo dependencies)

**Recommendation**: Use UV for quick formatting/linting iteration, Pants for running tests/scripts and pre-commit validation.

### Testing

**Note**: Use Pants for running tests as it handles monorepo dependencies automatically. UV requires PYTHONPATH setup for imports from `libs/`.

```bash
# Run all tests
./pants test ::

# Run tests for a specific library
./pants test libs/qdrant_db::

# Run a specific test file
./pants test tests/libs/qdrant_db/unit/test_qdrant_client.py

# Run integration tests only
./pants test :: -- -m integration

# Run with coverage
./pants test --coverage ::

# Run tests matching keyword
./pants test :: -- -k test_client

# Verbose output with short traceback
./pants test :: -- -v --tb=short
```

### Code Quality

```bash
# Format code
uv run ruff format .
./pants fmt ::

# Lint code
uv run ruff check --fix .
./pants lint ::

# Type check (ty works standalone)
uv run ty check scripts/ libs/ apps/

# Format, lint, and check (Pants-only, recommended before commits)
./pants fmt lint check ::
```

### Running Scripts

**Note**: Use Pants for running scripts as it handles monorepo dependencies automatically. UV requires PYTHONPATH setup for imports from `libs/`.

```bash
# Reindex anime database
./pants run scripts/reindex_anime_database.py

# Update vectors
./pants run scripts/update_vectors.py -- --vectors title_vector

# Validate enrichment database
./pants run scripts/validate_enrichment_database.py

# View script help
./pants run scripts/update_vectors.py -- --help
```

### Pants-Only Commands

```bash
# List all targets
./pants list ::

# List targets in a specific directory
./pants list libs/qdrant_db::

# Show dependencies
./pants dependencies scripts/reindex_anime_database.py

# Show dependents
./pants dependents libs/common::

# Count lines of code
./pants count-loc ::
```

## Libraries

### `libs/common`

Shared models and configuration used across all libraries and the main application.

- **Models**: `Anime`, `Character`, `Episode`, and `AnimeRecord` Pydantic models
- **Config**: Settings management with pydantic-settings
- **Utils**: ID generation and datetime utilities

### `libs/qdrant_db`

Qdrant vector database client with strict typed contracts and batch operations.

- Async operations with typed request/response contracts (`SearchRequest`, `BatchOperationResult`)
- Automatic retry with exponential backoff for transient failures
- Multi-vector and hybrid (RRF/DBSF) search support
- Collection lifecycle management with race-safe initialization

### `libs/vector_processing`

Vector embedding generation and processing.

- **Text Models**: FlagEmbedding (BGE-M3), HuggingFace Transformers, Sentence Transformers
- **Vision Models**: OpenCLIP
- **Processors**: Multi-vector embedding manager, cross-encoder reranker
- **Field Mapping**: Anime-specific field extraction and preprocessing

## Configuration

### Environment Detection

**REQUIRED**: The service requires `ENVIRONMENT` to be explicitly set for production safety. No default value is provided to prevent accidental deployment with development settings.

```bash
# Development - debug enabled, verbose logging (respects user overrides)
ENVIRONMENT=development

# Staging - debug enabled, moderate logging, WAL enabled (respects user overrides)
ENVIRONMENT=staging

# Production - ENFORCED safety settings (ignores user overrides)
ENVIRONMENT=production
```

**Environment-Specific Behavior:**

| Setting             | Development       | Staging          | Production               |
| ------------------- | ----------------- | ---------------- | ------------------------ |
| `debug`             | `True` (default)  | `True` (default) | **`False` (enforced)**   |
| `log_level`         | `DEBUG` (default) | `INFO` (default) | **`WARNING` (enforced)** |
| `qdrant_enable_wal` | user choice       | `True` (default) | **`True` (enforced)**    |
| `model_warm_up`     | user choice       | user choice      | **`True` (enforced)**    |

**Defaults**: "default" means the value is applied only if you don't explicitly set it in your `.env` file or environment variables.

**Production Safety**: Production mode **always enforces** critical settings to prevent accidental debug mode or verbose logging in production. User-provided values are ignored for security.

**Development/Staging**: These environments respect your custom configuration. Set `DEBUG=false` or `LOG_LEVEL=ERROR` in `.env` to override the defaults.

**Docker Deployment:**

```dockerfile
ENV ENVIRONMENT=production
```

**Kubernetes Deployment:**

```yaml
env:
  - name: ENVIRONMENT
    value: "production"
```

### Application Settings

Create a `.env` file or set environment variables:

```env
# Environment (REQUIRED - must be explicitly set)
ENVIRONMENT=development

# Service
VECTOR_SERVICE_HOST=0.0.0.0
VECTOR_SERVICE_PORT=8001
ENRICHMENT_SERVICE_HOST=0.0.0.0
ENRICHMENT_SERVICE_PORT=8002

# Database
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=anime_database

# Embedding Models
# flagembedding is required for sparse/hybrid search (see Vector Architecture)
TEXT_EMBEDDING_PROVIDER=flagembedding
TEXT_EMBEDDING_MODEL=BAAI/bge-m3
IMAGE_EMBEDDING_PROVIDER=openclip
IMAGE_EMBEDDING_MODEL=ViT-L-14/laion2b_s32b_b82k

# Model Cache
MODEL_CACHE_DIR=./cache
```

## Dependency Management

### Using UV (Recommended)

```bash
# Install all dependencies
uv sync

# Update dependencies
uv lock --upgrade

# Run scripts
uv run python script.py
```

## Services And gRPC Contracts

- Services in this repo:
  - `apps/vector_service` (gRPC on `:8001`)
  - `apps/enrichment_service` (gRPC on `:8002`)
- Active vector gRPC methods:
  - `VectorAdminService`: `Health`, `GetStats`
  - `VectorSearchService`: `Search`
- Active enrichment gRPC methods:
  - `EnrichmentService`: `Health`, `RunPipeline`
- Proto sources:
  - `protos/vector_service/v1/`
  - `protos/enrichment_service/v1/`
- After any `.proto` change, regenerate checked-in stubs:

```bash
./pants run scripts/generate-proto.py
```

## Architecture

### Vector Architecture

The service uses a unified multi-vector architecture optimized for million-query scale:

**Text Vectors**:

- `text_vector`: 1024-dimensional BGE-M3 embeddings covering titles, synopses, and metadata across all entity types (Anime, Characters, Episodes).

**Image Vectors**:

- `image_vector`: 768-dimensional OpenCLIP ViT-L/14 embeddings for visual similarity of covers and character art. Stored as a Qdrant **multivector** (MAX_SIM) so one point can hold several images; HNSW is disabled on it because MAX_SIM is asymmetric.

**Sparse Vectors**:

- `text_sparse_vector`: lexical/keyword vector with the IDF modifier, used for sparse and hybrid text search.

When more than one of these signals is active in a single query, results are fused server-side via Qdrant's Query API using RRF (default) or DBSF.

### Technology Stack

- **Build System**: Pants 2.29.1
- **Language**: Python 3.13
- **RPC Framework**: gRPC (`grpc.aio`)
- **Vector Database**: Qdrant with HNSW indexing
- **HTTP Cache**: Redis (RFC 9111-compliant via Hishel, used by enrichment pipeline)
- **Text Embeddings**: BGE-M3 (1024-dim, multilingual)
- **Image Embeddings**: OpenCLIP ViT-L/14 (768-dim)
- **Package Manager**: UV
- **Testing**: pytest, pytest-asyncio
- **Type Checking**: ty
- **Formatting**: ruff format, ruff check

## Contributing

1. Install dependencies: `uv sync`
2. Make changes in appropriate library or application code
3. Add tests: `libs/*/tests/` or `tests/`
4. Run tests: `./pants test ::`
5. Format code: `./pants fmt ::`
6. Submit PR

## Related Documentation

- [Pants Documentation](https://www.pantsbuild.org/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [gRPC Documentation](https://grpc.io/docs/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Ty Documentation](https://docs.astral.sh/ty/)
