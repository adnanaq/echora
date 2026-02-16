# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains backend search services and shared libraries for anime discovery:

- `apps/agent_service`: internal gRPC agent service (`SearchAI`) for natural-language orchestration.
- `apps/service`: FastAPI vector/admin service for vector operations and management.
- Shared libraries for Qdrant access, embeddings, enrichment, and agent orchestration.

Data roles:
- PostgreSQL is the canonical source of truth for identities and relationships.
- Qdrant is the semantic retrieval index for text/image vector search.

## Development Commands

### Local Development Setup

```bash
# Install dependencies
uv sync

# Install with dev dependencies (includes pytest, ty, etc.)
uv sync --extra dev

# Start Qdrant database only
docker compose -f docker/docker-compose.dev.yml up -d qdrant

# Primary entrypoint (internal agent service)
./pants run apps/agent_service:agent_service

# Secondary vector/admin service
./pants run apps/service:service
```

### Docker Development (Recommended)

```bash
# Start full stack (service + database)
docker compose -f docker/docker-compose.dev.yml up -d

# View logs
docker compose -f docker/docker-compose.dev.yml logs -f vector-service

# Stop services
docker compose -f docker/docker-compose.dev.yml down
```

### Testing

**Note**: Use Pants for running tests as it handles monorepo dependencies automatically.

```bash
# Run all tests
./pants test ::

# Run tests for a specific library
./pants test libs/qdrant_db::

# Run a specific test file
./pants test tests/libs/qdrant_db/integration/test_qdrant_client_integration.py

# Run integration tests only (requires Qdrant + ML models)
./pants test :: -- -m integration

# Skip integration tests (fast, no DB/models required)
./pants test :: -- -m "not integration"

# Run tests matching keyword
./pants test :: -- -k test_client

# Verbose output with short traceback
./pants test :: -- -v --tb=short
```

### Code Quality

**Code Quality Tools**:

- Type checking: ty
- Formatting: ruff format (replaces black)
- Import sorting: ruff (replaces isort)
- Linting: ruff check (replaces autoflake and flake8)

```bash
# Format code
uv run ruff format .
./pants fmt ::

# Lint code
uv run ruff check --fix .
./pants lint ::

# Type check (ty works standalone)
uv run ty check scripts/ libs/ apps/

# Format, lint, and check (recommended before commits)
./pants fmt lint check ::
```

### Type Safety Protocol

**MANDATORY**: All code must pass ty type checking before commits.

```bash
# Check all source files
uv run ty check scripts/ libs/ apps/

# Check specific library
uv run ty check libs/http_cache/
```

### Service Health Checks

```bash
# Check vector service health
curl http://localhost:8002/health

# Check Qdrant health
curl http://localhost:6333/health

# Get database statistics
curl http://localhost:8002/api/v1/admin/stats

# Agent service runs on gRPC (default :50051)
# Use grpcurl or a gRPC client to call: agent.v1.AgentSearchService/SearchAI
```

## Architecture Overview

### Core Architecture Pattern

The repository follows a multi-service architecture with clear separation of concerns:

**External BFF (separate repo)** → **Internal services (`apps/agent_service`, `apps/service`)** → **Data Layer (PostgreSQL + Qdrant)**

### Key Architectural Components

#### 1. Agent Service (`apps/agent_service/src/agent_service/server.py`)

- Internal gRPC `SearchAI` endpoint for natural-language queries
- Planner/executor/sufficiency orchestration loop via `libs/agent_core`
- Executes retrieval against Qdrant and (when available) PostgreSQL graph executor

#### 2. FastAPI Application (`apps/service/src/service/main.py`)

- Async application with lifespan management
- Global Qdrant client initialization with health checks
- CORS middleware and structured logging
- Graceful startup/shutdown with dependency validation

#### 3. Configuration System (`libs/common/src/common/config/settings.py`)

- Pydantic-based settings with environment variable support
- Comprehensive validation for all configuration parameters
- Support for multiple embedding providers and models
- Performance tuning parameters (quantization, HNSW, batch sizes)

#### 4. Multi-Vector Processing (`libs/vector_processing/src/vector_processing/`)

- **QdrantClient**: Advanced vector database operations with quantization support
- **TextProcessor**: BGE-M3 embeddings for semantic text search (1024-dim)
- **VisionProcessor**: OpenCLIP ViT-L/14 embeddings for image search (768-dim)
- **Fine-tuning modules**: Character recognition, art style classification, genre enhancement

#### 5. Vector Service Operational Endpoints (`apps/service/src/service/routes/`)

- **Admin Router**: Database management, statistics, and reindexing

#### 6. Agent Service RPC Endpoint (`apps/agent_service/src/agent_service/server.py`)

- **gRPC Service**: `agent.v1.AgentSearchService`
- **Method**: `SearchAI`
- **Role**: Internal natural-language orchestration endpoint for BFF calls

#### 7. Data Enrichment Pipeline (`libs/enrichment/src/enrichment/`)

- **API Helpers**: Integration with 6+ external anime APIs (AniList, Kitsu, AniDB, etc.)
- **Crawlers**: Heavy-duty browser automation using crawl4ai for robust data extraction
- **Scrapers**: Web scraping with Cloudflare bypass capabilities
- **Multi-stage AI Pipeline**: Modular prompt system for data enhancement
- **Auto-Agent Assignment**: Automatic agent ID assignment for concurrent processing with gap-filling logic

### Enrichment Pipeline Usage

**Script**: `run_enrichment.py` - Main entry point for programmatic enrichment

**Database**: Reads from `data/qdrant_storage/anime-offline-database.json` (39,244+ anime entries)

**Arguments**:

- `--index N`: Process anime at index N (0-based)
- `--title "Title"`: Search for anime by title (case-insensitive, partial match)
- `--file PATH`: Use custom database file (optional)
- `--agent "name"`: Specify agent directory name (optional, auto-generated if not provided)
- `--skip service1 service2`: Skip specific services (e.g., `--skip jikan anidb`)
- `--only service1 service2`: Only fetch specific services (e.g., `--only anime_planet`)

**Available Services**: `jikan`, `anilist`, `kitsu`, `anidb`, `anime_planet`, `anisearch`, `animeschedule`

**Example Usage**:

```bash
# Process first anime in database
python run_enrichment.py --index 0

# Process One Piece
python run_enrichment.py --title "One Piece"

# Use custom database
python run_enrichment.py --file custom.json --index 5

# Specify agent directory
python run_enrichment.py --title "Dandadan" --agent "Dandadan_test"

# Skip specific services
python run_enrichment.py --title "Dandadan" --skip animeschedule anidb

# Only fetch from specific services
python run_enrichment.py --title "Dandadan" --only anime_planet anisearch
```

**Notes**:

- `--skip` and `--only` are mutually exclusive
- **Auto-Agent Assignment**: Pipeline automatically assigns agent IDs using gap-filling logic if `--agent` not specified

### Stage Script Directory Detection

All stage scripts follow a consistent pattern for multi-agent concurrent processing. Each stage accepts an `agent_id` positional argument that specifies the directory name within the temp directory.

**Common Pattern**: `python process_stage<N>.py <agent_id> [--temp-dir <base>]`

- `agent_id`: Directory name (e.g., `One_agent1`, `Dandadan_agent1`)
- `--temp-dir`: Base directory path (default: `temp`) - optional

**Multi-agent Directory Structure**: `temp/<agent_id>/` (e.g., `temp/One_agent1/`, `temp/Dandadan_agent1>/`)

**Note**: When using `run_enrichment.py`, agent IDs are assigned automatically. Manual specification only needed for independent stage script execution.

#### Stage 1: Metadata Extraction (`process_stage1_metadata.py`)

**Arguments**: `agent_id` (positional), `--temp-dir` (default: `temp`), `--current-anime` (legacy support)

**Example Usage**:

```bash
# Recommended: Use agent_id
python process_stage1_metadata.py One_agent1

# Custom temp directory
python process_stage1_metadata.py One_agent1 --temp-dir custom_temp

# Legacy: Use file path
python process_stage1_metadata.py --current-anime temp/One_agent1/current_anime.json
```

#### Stage 2: Episode Processing (`process_stage2_episodes.py`)

**Arguments**: `agent_id` (positional), `--temp-dir` (default: `temp`)

**Required File**: `episodes_detailed.json` (must exist in agent directory)

**Example Usage**:

```bash
# Recommended: Use agent_id
python process_stage2_episodes.py One_agent1

# Custom temp directory
python process_stage2_episodes.py One_agent1 --temp-dir custom_temp
```

#### Stage 3: Relationship Processing (`process_stage3_relationships.py`)

**Arguments**: `agent_id` (positional), `--temp-dir` (default: `temp`), `--current-anime` (legacy support)

**Example Usage**:

```bash
# Recommended: Use agent_id
python process_stage3_relationships.py One_agent1

# Custom temp directory
python process_stage3_relationships.py One_agent1 --temp-dir custom_temp

# Legacy: Use file path
python process_stage3_relationships.py --current-anime temp/One_agent1/current_anime.json
```

#### Stage 4: Statistics Extraction (`process_stage4_statistics.py`)

**Arguments**: `agent_id` (positional), `--temp-dir` (default: `temp`)

**Example Usage**:

```bash
# Recommended: Use agent_id
python scripts/process_stage4_statistics.py Dandadan_agent1

# Custom temp directory
python scripts/process_stage4_statistics.py Dandadan_agent1 --temp-dir custom_temp
```

#### Stage 5: AI Character Matching (`process_stage5_characters.py`)

**Arguments**: `agent_id` (positional), `--temp-dir` (default: `temp`), `--restart` (optional flag)

**Example Usage**:

```bash
# Process with resume support (recommended)
python process_stage5_characters.py One_agent1

# Force restart from scratch
python process_stage5_characters.py One_agent1 --restart

# Custom temp directory
python process_stage5_characters.py One_agent1 --temp-dir custom_temp
```

### Vector Database Management

#### Selective Vector Updates (`scripts/update_vectors.py`)

Update specific vectors without full reindexing.

**Arguments**:

- `--vectors VECTOR [VECTOR ...]`: Vector names to update (required)
- `--index N`: Update specific anime by index (0-based)
- `--title "TITLE"`: Update anime matching title (partial match)
- `--batch-size N`: Anime per batch (default: 100)
- `--file PATH`: Custom data file path

**Example Usage**:

```bash
# Update single vector for all anime
uv run python scripts/update_vectors.py --vectors title_vector

# Update multiple vectors
uv run python scripts/update_vectors.py --vectors genre_vector character_vector

# Update specific anime by index
uv run python scripts/update_vectors.py --vectors staff_vector --index 5

# Update by title search
uv run python scripts/update_vectors.py --vectors temporal_vector --title "Bungaku"

# Custom batch size
uv run python scripts/update_vectors.py --vectors image_vector --batch-size 50
```

#### Full Database Reindexing (`scripts/reindex_anime_database.py`)

Complete reindexing with all vectors. Deletes and recreates collection.

```bash
uv run python scripts/reindex_anime_database.py
```

### Multi-Vector Collection Design

The service uses a single Qdrant collection with named vectors:

- `text_vector`: 1024-dimensional BGE-M3 embeddings for semantic search
- `image_vector`: 768-dimensional OpenCLIP ViT-L/14 embeddings for cover art, posters, banners

This design enables efficient multimodal search while maintaining data locality and reducing storage overhead.

### Configuration-Driven Model Selection

The service supports multiple embedding providers through configuration:

- **Text Models**: BGE-M3, BGE-small/base/large-v1.5, custom HuggingFace models
- **Vision Models**: OpenCLIP ViT-L/14, OpenCLIP ViT-B/32 (primary: ViT-L/14)
- **Provider Flexibility**: Easy switching between embedding providers per modality

### Performance Optimization Features

- **Vector Quantization**: Binary/Scalar/Product quantization for 40x speedup potential
- **HNSW Tuning**: Optimized parameters for anime-specific search patterns
- **Payload Indexing**: Fast filtering on genre, year, type, status fields
- **Hybrid Search**: Single-request API for combined text+image queries
- **GPU Acceleration**: Support for GPU-accelerated model inference

## Environment Variables

### Critical Configuration

- `QDRANT_URL`: Vector database URL (default: http://localhost:6333)
- `QDRANT_COLLECTION_NAME`: Collection name (default: anime_database)
- `TEXT_EMBEDDING_MODEL`: Text model (default: BAAI/bge-m3)
- `IMAGE_EMBEDDING_MODEL`: Image model (default: ViT-L-14/laion2b_s32b_b82k)

### Performance Tuning

- `QDRANT_ENABLE_QUANTIZATION`: Enable vector quantization (default: false)
- `QDRANT_QUANTIZATION_TYPE`: Quantization type (scalar, binary, product)
- `MODEL_WARM_UP`: Pre-load models during startup (default: false)
- `MAX_BATCH_SIZE`: Maximum batch size for operations (default: 500)

### Service Configuration

- `VECTOR_SERVICE_PORT`: Service port (default: 8002)
- `DEBUG`: Enable debug mode (default: true)
- `LOG_LEVEL`: Logging level (default: INFO)
