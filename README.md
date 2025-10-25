# Anime Vector Service

A specialized microservice for semantic search over anime content using a 14-vector architecture with Qdrant vector database.

## Features

- **14-Vector Semantic Architecture**: 12 text vectors + 2 image vectors for comprehensive anime understanding
- **Advanced Text Search**: BGE-M3 embeddings across multiple semantic domains (title, character, genre, technical, etc.)
- **Visual Search**: OpenCLIP ViT-L/14 embeddings for cover art and character image similarity
- **Character-Focused Search**: Specialized character vector + character image vector search
- **Multi-Vector Fusion**: Native Qdrant RRF/DBSF fusion for optimal search results
- **Production Ready**: Health checks, monitoring, database management

## Quick Start

### Using Docker (Recommended)

```bash
# Start services
docker compose up -d

# Service available at http://localhost:8002
curl http://localhost:8002/health
```

### Local Development

```bash
# Install dependencies using UV (recommended)
uv sync --dev

# Start Qdrant database
docker compose up -d qdrant

# Run service
uv run python -m src.main
```

#### Alternative: Using pip

```bash
# Install dependencies with pip
pip install -e .

# Start Qdrant database
docker compose up -d qdrant

# Run service
python -m src.main
```

## API Architecture

This service provides semantic search capabilities through a 14-vector architecture:

- **Text Vectors (12)**: title, character, genre, technical, staff, review, temporal, streaming, related, franchise, episode, identifiers
- **Image Vectors (2)**: general image similarity, character image similarity

### Available Endpoints

- `GET /health` - Service health status
- `GET /api/v1/admin/stats` - Database statistics

## Configuration

Environment variables:

```env
# Service
VECTOR_SERVICE_HOST=0.0.0.0
VECTOR_SERVICE_PORT=8002

# Database
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=anime_database

# Embedding Models
TEXT_EMBEDDING_MODEL=BAAI/bge-m3
IMAGE_EMBEDDING_MODEL=ViT-L-14/laion2b_s32b_b82k
```

## Dependency Management

This project uses **UV** for fast and reliable dependency management:

- `pyproject.toml`: Project configuration and dependencies
- `uv.lock`: Lock file with exact dependency versions
- `.venv/`: Virtual environment managed by UV

### Key Commands

```bash
# Install dependencies
uv sync --dev

# Add a new dependency
uv add package-name

# Add a development dependency
uv add --dev package-name

# Run scripts
uv run python script.py

# Update dependencies
uv lock --upgrade
```

## Technology Stack

- **FastAPI**: REST API framework
- **Qdrant**: Vector database with HNSW indexing and multi-vector support
- **BGE-M3**: Multi-lingual text embeddings (1024-dim)
- **OpenCLIP ViT-L/14**: High-quality image embeddings (768-dim)
- **Docker**: Containerized deployment
- **UV**: Fast Python package manager

