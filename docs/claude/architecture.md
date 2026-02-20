# Architecture Overview

## Core Architecture Pattern

The service follows a layered microservice architecture:

`API Layer (apps/service/) -> Business Logic (libs/) -> Database Layer (Qdrant)`

## Directory Structure

```text
echora/
|- apps/
|  `- service/                    # FastAPI microservice
|     `- src/service/
|        |- main.py               # Application entrypoint
|        |- dependencies.py       # Dependency injection
|        `- routes/               # API endpoints
|           `- admin.py           # Admin operations
|
|- libs/                          # Shared libraries (business logic)
|  |- common/                     # Shared configuration & models
|  |  |- config/                  # Settings (Qdrant, Embedding, Service)
|  |  `- models/                  # AnimeRecord data model
|  |- vector_processing/          # Embedding generation
|  |  |- processors/              # Text/Image processors
|  |  `- embedding_models/        # BGE-M3, OpenCLIP
|  |- qdrant_db/                  # Vector database client
|  |- http_cache/                 # HTTP caching (Hishel + Redis)
|  |- enrichment/                 # Data enrichment pipeline
|  `- vector_db_interface/        # Abstract DB interface
|
|- tests/                         # Test suite (mirrors libs/)
|- data/                          # Data files & Qdrant storage
|- docker/                        # Docker Compose configurations
`- scripts/                       # Utility scripts
```

## Key Components

- FastAPI Service (`apps/service/`): API endpoints with dependency injection
- Configuration (`libs/common/config/`): Pydantic BaseSettings
- Vector Processing (`libs/vector_processing/`): text/image embedding generation
- Qdrant Client (`libs/qdrant_db/`): vector DB operations with quantization and multi-vector support
- HTTP Cache (`libs/http_cache/`): RFC 9111-compliant caching with Redis backend
- Enrichment Pipeline (`libs/enrichment/`): API helpers, crawlers/scrapers, and multi-stage enhancement

## Multi-Vector Collection Design

The service uses a single Qdrant collection with named vectors:

- `text_vector`: 1024-dimensional BGE-M3 embeddings for semantic search
- `image_vector`: 768-dimensional OpenCLIP ViT-L/14 embeddings for artwork assets

This supports multimodal search while keeping data locality in one collection.

## Configuration-Driven Model Selection

Supported configuration model families include:

- Text: BGE-M3, BGE-small/base/large-v1.5, and custom HuggingFace models
- Vision: OpenCLIP ViT-L/14 and ViT-B/32

## Performance Optimization Features

- Vector quantization: binary/scalar/product modes
- HNSW tuning for search performance
- Payload indexing for metadata filters
- Hybrid search for combined text+image queries
- GPU acceleration support for model inference
