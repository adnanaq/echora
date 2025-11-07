# Anime Vector Service

A specialized gRPC microservice for AI-powered semantic search over anime content using a 14-vector architecture with Qdrant vector database.

## Features

- **gRPC API**: High-performance binary protocol with async support
- **AI Query Parsing**: Natural language query understanding with LLM-powered intent detection
- **14-Vector Semantic Architecture**: 12 text vectors + 2 image vectors for comprehensive anime understanding
- **Advanced Text Search**: BGE-M3 embeddings across multiple semantic domains (title, character, genre, technical, etc.)
- **Visual Search**: OpenCLIP ViT-L/14 embeddings for cover art and character image similarity
- **Character-Focused Search**: Specialized character vector + character image vector search
- **Multi-Vector Fusion**: Native Qdrant RRF/DBSF fusion for optimal search results
- **Multimodal Search**: Combined text + image search capabilities
- **Production Ready**: Health checks, monitoring, database management

## Quick Start

### Using Docker (Recommended)

```bash
# Start services
docker compose -f docker/docker-compose.dev.yml up -d

# gRPC service available at localhost:50051
# Test health check
grpcurl -plaintext localhost:50051 grpc.health.v1.Health/Check
```

### Local Development

```bash
# Install dependencies using UV (recommended)
uv sync --dev

# Start Qdrant database
docker compose -f docker/docker-compose.dev.yml up -d qdrant

# Run gRPC service
uv run python -m src.main
```

#### Alternative: Using pip

```bash
# Install dependencies with pip
pip install -e .

# Start Qdrant database
docker compose -f docker/docker-compose.dev.yml up -d qdrant

# Run gRPC service
python -m src.main
```

## API Architecture

This service provides AI-powered semantic search through a gRPC API with three services:

### gRPC Services

#### 1. Health Service (Standard gRPC Health Check)
- **`Check(service)`** - Check service health status
- **`Watch(service)`** - Stream health status updates

#### 2. AdminService (Database Management)
- **`GetStats()`** - Get vector database statistics
  - Returns: collection info, document count, vector configuration, optimizer status

#### 3. AgentService (AI-Powered Search)
- **`Search(query, image_data?)`** - Natural language search with AI query parsing
  - Accepts: text query + optional base64 image
  - AI automatically detects intent and selects appropriate search type:
    - Text search (semantic anime title/genre/theme search)
    - Image search (visual similarity using cover art)
    - Multimodal search (combined text + image)
    - Character search (character-focused queries)
  - Returns: ranked anime IDs + reasoning explanation

### Vector Architecture

14-vector semantic architecture for comprehensive search:

- **Text Vectors (12)**: title, character, genre, technical, staff, review, temporal, streaming, related, franchise, episode, identifiers
- **Image Vectors (2)**: general image similarity, character image similarity

## Testing the gRPC Service

### Using grpcurl

```bash
# Check health
grpcurl -plaintext localhost:50051 grpc.health.v1.Health/Check

# Get database stats
grpcurl -plaintext localhost:50051 admin.AdminService/GetStats

# Search for anime (requires JSON input)
grpcurl -plaintext -d '{"query": "highly rated action anime"}' \
  localhost:50051 agent.AgentService/Search
```

### Using Python Client

```python
import grpc
from protos import agent_pb2, agent_pb2_grpc, admin_pb2, admin_pb2_grpc

# Connect to server
channel = grpc.insecure_channel('localhost:50051')

# Use AdminService
admin_stub = admin_pb2_grpc.AdminServiceStub(channel)
stats = admin_stub.GetStats(admin_pb2.GetStatsRequest())
print(f"Collection: {stats.collection_name}, Documents: {stats.total_documents}")

# Use AgentService for AI search
agent_stub = agent_pb2_grpc.AgentServiceStub(channel)
response = agent_stub.Search(agent_pb2.SearchRequest(
    query="anime about pirates and adventure"
))
print(f"Found {len(response.anime_ids)} results")
print(f"Reasoning: {response.reasoning}")
```

## Configuration

Environment variables:

```env
# gRPC Service
GRPC_SERVER_PORT=50051

# Database
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=anime_database

# Embedding Models
TEXT_EMBEDDING_MODEL=BAAI/bge-m3
IMAGE_EMBEDDING_MODEL=ViT-L-14/laion2b_s32b_b82k

# AI Query Parser (Ollama)
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=qwen2.5:7b

# Logging
LOG_LEVEL=INFO
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

- **gRPC**: High-performance RPC framework with Protocol Buffers
- **Qdrant**: Vector database with HNSW indexing and multi-vector support
- **Atomic Agents**: AI agent framework for query parsing and tool orchestration
- **Ollama**: Local LLM inference for natural language query understanding
- **BGE-M3**: Multi-lingual text embeddings (1024-dim)
- **OpenCLIP ViT-L/14**: High-quality image embeddings (768-dim)
- **Docker**: Containerized deployment
- **UV**: Fast Python package manager

## Protocol Buffers

The service uses Protocol Buffers for API definitions. Proto files are located in `protos/`:

- `agent.proto` - AgentService for AI-powered search
- `admin.proto` - AdminService for database management
- `health.proto` - Standard gRPC health checking

To regenerate Python code from proto files:

```bash
python -m grpc_tools.protoc -I. \
  --python_out=. \
  --grpc_python_out=. \
  --pyi_out=. \
  protos/*.proto
```

