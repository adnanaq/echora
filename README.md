# Echora

A production-ready semantic search microservice for anime content using an 11-vector architecture with Qdrant vector database. Built as a Pants monorepo with modular libraries for vector processing, database operations, and enrichment pipelines.

## 🎯 Features

- **11-Vector Semantic Architecture**: 9 text vectors + 2 image vectors for comprehensive anime understanding
- **Advanced Text Search**: BGE-M3 embeddings (1024-dim) across multiple semantic domains
- **Visual Search**: OpenCLIP ViT-L/14 embeddings (768-dim) for cover art and character similarity
- **Character-Focused Search**: Specialized character vector + character image vector search
- **Multi-Vector Fusion**: Native Qdrant RRF/DBSF fusion for optimal search results
- **Modular Monorepo**: Clean separation of concerns with Pants build system
- **Production Ready**: Health checks, monitoring, comprehensive test suite

## 📊 Project Stats

- **Lines of Code**: ~102k LOC
- **Files**: 180 Python files
- **Test Coverage**: 50%+ with unit and integration tests
- **Libraries**: 3 core libraries (`common`, `qdrant_db`, `vector_processing`)

## 🏗️ Monorepo Structure

```text
echora/
├── apps/                          # Applications
│   └── service/                   # Vector search service
│       └── src/service/           # FastAPI application (main.py, api/, etc.)
├── libs/                          # Shared libraries
│   ├── common/                    # Common models and configuration
│   │   └── src/common/
│   │       ├── config/            # Settings and configuration
│   │       └── models/            # Shared data models (AnimeEntry, etc.)
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
├── tests/                         # Cross-library integration tests
│   ├── integration/               # Integration tests
│   └── libs/                      # Per-library test suites
└── data/                          # Data storage (Qdrant, anime databases)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12
- Docker and Docker Compose
- UV package manager (recommended) or pip

### Using Docker (Recommended)

```bash
# Start all services
docker compose up -d

# Service available at http://localhost:8002
curl http://localhost:8002/health
```

### Local Development

#### 1. Install Dependencies

```bash
# Using UV (recommended - fast and reliable)
uv sync

# Or using pip
pip install -e .
```

#### 2. Start Qdrant Database

```bash
docker compose up -d qdrant

# Access Qdrant UI dashboard
# http://localhost:6333/dashboard
```

#### 3. Run the Service

```bash
# Using UV
uv run python -m src.main

# Or directly
python -m src.main
```

## 🛠️ Development with Pants

This project uses [Pants](https://www.pantsbuild.org/) for build orchestration, dependency management, and testing in the monorepo.

### Common Pants Commands

```bash
# List all targets
./pants list ::

# List targets in a specific directory
./pants list libs/qdrant_db::

# Run all tests
./pants test ::

# Run tests for a specific library
./pants test libs/qdrant_db::

# Run a specific test file
./pants test libs/qdrant_db/tests/unit/test_qdrant_client.py

# Run integration tests only
./pants test tests/integration::

# Format code
./pants fmt ::

# Lint code
./pants lint ::

# Type check with mypy
./pants check ::

# Count lines of code
./pants count-loc ::
```

### Running Scripts

```bash
# Reindex anime database
./pants run scripts/reindex_anime_database.py

# Update vectors
./pants run scripts/update_vectors.py

# Validate enrichment database
./pants run scripts/validate_enrichment_database.py
```

## 📚 Libraries

### `libs/common`
Shared models and configuration used across all libraries and the main application.
- **Models**: `AnimeEntry` and related data structures
- **Config**: Settings management with pydantic-settings

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

## 🧪 Testing

```bash
# Run all tests (unit + integration)
./pants test ::

# Skip integration tests (fast, no DB/models required)
./pants test :: -- -m "not integration"

# Run ONLY integration tests (requires Qdrant + ML models)
docker compose up -d qdrant
./pants test :: -- -m integration

# Run tests for specific library
./pants test libs/qdrant_db/tests/unit::

# Run tests in specific directory
./pants test tests/integration::

# Run with coverage
./pants test --coverage ::
```

### Test Organization

- **Unit Tests**: Mock external dependencies, fast execution (~seconds)
  - `libs/*/tests/unit/` - Library unit tests
  - `tests/unit/` - Application unit tests
  - No external dependencies required
- **Integration Tests**: Real database and models, slower execution (~minutes)
  - `libs/*/tests/integration/` - Library integration tests
  - `tests/integration/` - Application integration tests
  - Marked with `@pytest.mark.integration` or `pytestmark = pytest.mark.integration`
  - Requires: Qdrant DB, ML models (BGE-M3, OpenCLIP), real embeddings

## 🔧 Configuration

Create a `.env` file or set environment variables:

```env
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

## 📦 Dependency Management

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
```

### Using Pip

```bash
# Install in editable mode
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"
```

## 🌐 API Endpoints

- `GET /health` - Service health status
- `GET /api/v1/admin/stats` - Database statistics
- `POST /api/v1/search` - Multi-vector semantic search (coming soon)

## 🏛️ Architecture

### Vector Architecture

**Text Vectors (9)**:
1. `title_vector` - Title, synopsis, background
2. `character_vector` - Character descriptions and roles
3. `genre_vector` - Genres and themes
4. `staff_vector` - Staff and production information
5. `temporal_vector` - Airing dates and season
6. `streaming_vector` - Streaming platforms
7. `related_vector` - Related anime and adaptations
8. `franchise_vector` - Franchise information
9. `episode_vector` - Episode summaries (hierarchical averaging)

**Image Vectors (2)**:
1. `image_vector` - Cover art similarity
2. `character_image_vector` - Character visual similarity

### Technology Stack

- **Build System**: Pants 2.29.1
- **Language**: Python 3.12
- **Web Framework**: FastAPI + Uvicorn
- **Vector Database**: Qdrant with HNSW indexing
- **Text Embeddings**: BGE-M3 (1024-dim, multilingual)
- **Image Embeddings**: OpenCLIP ViT-L/14 (768-dim)
- **Package Manager**: UV
- **Testing**: pytest, pytest-asyncio
- **Type Checking**: mypy
- **Formatting**: black, isort

## 🤝 Contributing

1. Install dependencies: `uv sync`
2. Make changes in appropriate library or application code
3. Add tests: `libs/*/tests/` or `tests/`
4. Run tests: `./pants test ::`
5. Format code: `./pants fmt ::`
6. Submit PR

## 📝 License

[Add your license here]

## 🔗 Related Documentation

- [Pants Documentation](https://www.pantsbuild.org/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
