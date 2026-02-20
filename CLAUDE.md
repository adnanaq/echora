# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a specialized microservice for semantic search over anime content using vector embeddings and Qdrant database. The service provides text, image, and multimodal search capabilities with production-ready features including health checks, monitoring, and CORS support.

## Development Commands

### Local Development Setup

```bash
# Install dependencies
uv sync

# Install with dev dependencies (includes pytest, ty, etc.)
uv sync --extra dev

# Start Qdrant database only
docker compose -f docker/docker-compose.dev.yml up -d qdrant

# Run service locally for development
./pants run apps/vector_service:vector_service
```

### Docker Development (Recommended)

```bash
# Start full stack (service + database + redis + redisinsight)
docker compose -f docker/docker-compose.dev.yml up -d

# View logs
docker compose -f docker/docker-compose.dev.yml logs -f vector-service

# Stop services
docker compose -f docker/docker-compose.dev.yml down
```

### RedisInsight (Redis GUI)

RedisInsight provides a graphical interface for debugging and monitoring the HTTP cache.

```bash
# Start Redis with RedisInsight
docker compose -f docker/docker-compose.dev.yml up -d redis redisinsight

# Access RedisInsight GUI
# Open browser: http://localhost:5540

# First-time setup in RedisInsight:
# 1. Add Database → Host: redis, Port: 6379
# 2. Test connection → Connect
```

**Use cases:**
- View cached HTTP responses (keys, TTLs, content)
- Debug cache hit/miss patterns
- Monitor cache memory usage
- Manually inspect/delete cache entries
- Analyze cache key distribution

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
# Check service health
curl http://localhost:8002/health

# Check Qdrant health
curl http://localhost:6333/health

# Get database statistics
curl http://localhost:8002/api/v1/admin/stats
```

## Architecture Overview

### Core Architecture Pattern

The service follows a layered microservice architecture with clear separation of concerns:

**API Layer** (`apps/service/`) → **Business Logic** (libs) → **Database Layer** (Qdrant)

### Directory Structure

```text
echora/
├── apps/
│   └── service/                    # FastAPI microservice
│       └── src/service/
│           ├── main.py            # Application entrypoint
│           ├── dependencies.py    # Dependency injection
│           └── routes/            # API endpoints
│               └── admin.py       # Admin operations
│
├── libs/                          # Shared libraries (business logic)
│   ├── common/                    # Shared configuration & models
│   │   ├── config/                # Settings (Qdrant, Embedding, Service)
│   │   └── models/                # AnimeRecord data model
│   ├── vector_processing/         # Embedding generation
│   │   ├── processors/            # Text/Image processors
│   │   └── embedding_models/      # BGE-M3, OpenCLIP
│   ├── qdrant_db/                 # Vector database client
│   ├── http_cache/                # HTTP caching (Hishel + Redis)
│   ├── enrichment/                # Data enrichment pipeline
│   └── vector_db_interface/       # Abstract DB interface
│
├── tests/                         # Test suite (mirrors libs/)
├── data/                          # Data files & Qdrant storage
├── docker/                        # Docker Compose configurations
└── scripts/                       # Utility scripts
```

### Key Components

- **FastAPI Service** (`apps/service/`) - API endpoints with dependency injection
- **Configuration** (`libs/common/config/`) - Pydantic BaseSettings (Qdrant, Embedding, Service configs)
- **Vector Processing** (`libs/vector_processing/`) - BGE-M3 text (1024-dim) + OpenCLIP image (768-dim) embeddings
- **Qdrant Client** (`libs/qdrant_db/`) - Vector DB operations with quantization & multi-vector support
- **HTTP Cache** (`libs/http_cache/`) - RFC 9111 compliant caching with Redis backend
- **Enrichment Pipeline** (`libs/enrichment/`)

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

## Code Conventions

### Class Member Ordering

Order class members top to bottom as follows:

```
 1. ClassVar / class-level constants     (TYPE_MAPPING = {...}, etc.)
 2. __slots__
 3. __init__                             (always first method)
 4. Representation dunders              (__repr__, __str__, __eq__, __hash__, __len__)
 5. @property                           (getter → setter → deleter as a unit, one property at a time)
 6. @classmethod
 7. @staticmethod                        (rare; prefer classmethod or plain function)
 8. Public instance methods             (async before sync when both exist for same operation)
 9. Protected methods  (_name)          (async before sync when both exist)
10. Private methods   (__name)          (very rare in this codebase)
11. Context-manager protocol last       (__enter__/__exit__, __aenter__/__aexit__)
```

**Key rules:**

- **Visibility grouping over decorator grouping.** All public methods go in section 8, all protected in section 9 — regardless of whether they're async. Do not scatter protected helpers between public methods.

- **Logical cohesion within a section.** Within sections 8 and 9, group related operations together (e.g. `encode_text` followed by `encode_texts_batch`, not separated by unrelated methods). No strict alphabetical ordering inside a section.

- **Async before sync for the same operation.** When an async and sync variant exist in the same section, put async first:
  ```python
  async def encode_text(self, text: str) -> list[float]: ...
  def encode_text_sync(self, text: str) -> list[float]: ...  # sync variant
  ```

- **`@property` pairs stay together.** Getter, setter, and deleter for the same property must be adjacent.

**Special cases:**

- **Pydantic `BaseModel`**: Field declarations come first (no `__init__`). Validator methods after fields. Regular methods after validators.
- **`@dataclass`**: Fields are the class body; if methods are added follow sections 3–11.
- **gRPC servicers**: Thin adapter classes — method order should mirror the proto service definition order, no reordering needed.

> **Enforcement**: ruff has no class-member-order rule; enforce via PR review.
