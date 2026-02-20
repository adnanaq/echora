# Echora

A production-ready anime data/search platform built as gRPC services on top of Qdrant and enrichment pipelines. The repo is a Pants monorepo with modular libraries for vector processing, database operations, and enrichment workflows.

## Features

- **Unified Multi-Vector Architecture**: High-performance semantic search using text and image vectors
- **Advanced Text Search**: BGE-M3 embeddings (1024-dim) for cross-platform semantic matching
- **Visual Search**: OpenCLIP ViT-L/14 embeddings (768-dim) for multi-source image similarity
- **Multi-Vector Fusion**: Native Qdrant RRF/DBSF fusion for optimal hybrid results
- **Modular Monorepo**: Clean separation of concerns with Pants build system

## Monorepo Structure

```text
echora/
├── apps/                          # Applications
│   ├── vector_service/            # gRPC search/admin service
│   │   └── src/vector_service/
│   └── enrichment_service/        # gRPC enrichment orchestration service
│       └── src/enrichment_service/
├── libs/                          # Shared libraries
│   ├── common/                    # Common models and configuration
│   │   └── src/common/
│   │       ├── config/            # Settings and configuration
│   │       └── models/            # Shared data models (AnimeRecord, etc.)
│   ├── enrichment/                # Anime data enrichment pipeline
│   │   └── src/enrichment/
│   │       ├── api_helpers/       # External API integrations (AniList, Kitsu, etc.)
│   │       ├── crawlers/          # Web crawlers (Crawl4ai-based)
│   │       ├── programmatic/      # Multi-stage enrichment pipeline
│   │       └── similarity/        # Character similarity (CCIP)
│   ├── http_cache/                # HTTP response caching (Redis-backed)
│   │   └── src/http_cache/        # Cache manager, aiohttp adapter
│   ├── qdrant_db/                 # Qdrant database client
│   │   ├── src/qdrant_db/         # Client implementation with retry logic
│   │   └── tests/                 # Unit and integration tests
│   ├── vector_db_interface/       # Database-agnostic vector interface
│   │   └── src/vector_db_interface/
│   └── vector_processing/         # Vector embedding processing
│       ├── src/vector_processing/
│       │   ├── embedding_models/  # Model implementations (FastEmbed, OpenCLIP)
│       │   ├── processors/        # Text/Vision processors, field mapping
│       │   └── utils/             # Image downloading, caching
│       └── tests/                 # Unit tests
├── scripts/                       # Utility scripts (reindexing, validation, etc.)
├── tests/                         # Test suite (mirrors source structure)
│   ├── conftest.py                # Root fixtures (settings, clients)
│   ├── integration/               # Cross-library integration tests
│   ├── apps/vector_service/       # Service tests (unit/, integration/)
│   ├── libs/                      # Per-library test suites
│   │   ├── common/                # Common library tests
│   │   ├── enrichment/            # Enrichment tests (unit/, integration/)
│   │   ├── http_cache/            # HTTP cache tests (unit/, integration/)
│   │   ├── qdrant_db/             # Qdrant DB tests (unit/, integration/)
│   │   ├── vector_db_interface/   # Vector DB interface tests
│   │   └── vector_processing/     # Vector processing tests (unit/)
│   ├── scripts/                   # Script tests
│   └── utils/                     # Test utilities
└── data/                          # Data storage (Qdrant, anime databases)
```

## Quick Start

### Prerequisites

- Python 3.12
- Docker and Docker Compose
- UV package manager (recommended) or pip

### Using Docker (Recommended)

```bash
# Start all services
docker compose -f docker/docker-compose.dev.yml up -d

# Services:
# - vector_service gRPC: localhost:8002
# - enrichment_service gRPC: localhost:8010
```

### Local Development

#### 1. Install UV Package Manager

```bash
# Install UV (one-time setup)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add UV to your PATH (add to ~/.bashrc or ~/.zshrc)
export PATH="$HOME/.local/bin:$PATH"

# Verify installation
uv --version
```

#### 2. Install Python 3.12 and Create Virtual Environment

```bash
# UV will automatically download Python 3.12 and create venv
uv venv

# Verify Python version
.venv/bin/python --version  # Should show Python 3.12.x
```

#### 3. Install Dependencies

```bash
# Install all project dependencies (including dev tools)
uv sync

# This creates/updates:
# - .venv/ (virtual environment with Python 3.12)
# - uv.lock (dependency lock file)
```

#### 4. Start Qdrant Database

```bash
docker compose -f docker/docker-compose.dev.yml up -d qdrant

# Access Qdrant UI dashboard
# http://localhost:6333/dashboard
```

#### 5. Run the Service

**Using Pants (Recommended)** — handles monorepo dependencies automatically:

```bash
# Run vector service (gRPC on :8002)
./pants run apps/vector_service/:vector_service

# Run enrichment service (gRPC on :8010)
./pants run apps/enrichment_service/:enrichment_service

# Run tests
./pants test ::

# Run scripts
./pants run scripts/reindex_anime_database.py
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
./pants test libs/qdrant_db/tests/unit/test_qdrant_client.py

# Run integration tests only
./pants test tests/integration::

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

Qdrant vector database client with retry logic and batch operations.

- Async operations with connection pooling
- Automatic retry with exponential backoff
- Multi-vector search support
- Comprehensive test suite (55 tests)

### `libs/vector_processing`

Vector embedding generation and processing.

- **Text Models**: FastEmbed, HuggingFace Transformers, Sentence Transformers
- **Vision Models**: OpenCLIP
- **Processors**: Text and vision processing with caching
- **Field Mapping**: Anime-specific field extraction and preprocessing

## Testing

```bash
# Run all tests (unit + integration)
./pants test ::

# Skip integration tests (fast, no DB/models required)
./pants test :: -- -m "not integration"

# Run ONLY integration tests (requires Qdrant + ML models)
docker compose -f docker/docker-compose.dev.yml up -d qdrant
./pants test :: -- -m integration

# Run tests for specific library
./pants test libs/qdrant_db/tests/unit::

# Run tests in specific directory
./pants test tests/integration::

# Run with coverage
./pants test --coverage ::
```

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
VECTOR_SERVICE_PORT=8002

# Database
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=anime_database

# Embedding Models
TEXT_EMBEDDING_PROVIDER=sentence-transformers
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

# Add a new dependency
uv add package-name

# Add a dev dependency
uv add --dev package-name

# Update dependencies
uv lock --upgrade

# Run scripts
uv run python script.py

# List installed packages
uv pip list

# Manage Python versions
uv python list                 # List installed Python versions
uv python install 3.12         # Install Python 3.12
```

### Using Pip

```bash
# Install in editable mode
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"
```

## Services And gRPC Contracts

- Services in this repo:
  - `apps/vector_service` (gRPC on `:8002`)
  - `apps/enrichment_service` (gRPC on `:8010`)
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

- `image_vector`: 768-dimensional OpenCLIP ViT-L/14 embeddings for visual similarity of covers and character art.

### Technology Stack

- **Build System**: Pants 2.29.1
- **Language**: Python 3.12
- **RPC Framework**: gRPC (`grpc.aio`)
- **Vector Database**: Qdrant with HNSW indexing
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
