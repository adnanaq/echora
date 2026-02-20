# Development Workflows

## Local Development Setup

```bash
# Install dependencies
uv sync

# Install with dev dependencies (includes pytest, ty, etc.)
uv sync --extra dev

# Start Qdrant database only
docker compose -f docker/docker-compose.dev.yml up -d qdrant

# Run service locally for development
./pants run apps/service:service
```

## Docker Development (Recommended)

```bash
# Start full stack (service + database + redis + redisinsight)
docker compose -f docker/docker-compose.dev.yml up -d

# View logs
docker compose -f docker/docker-compose.dev.yml logs -f vector-service

# Stop services
docker compose -f docker/docker-compose.dev.yml down
```

## RedisInsight (Redis GUI)

```bash
# Start Redis with RedisInsight
docker compose -f docker/docker-compose.dev.yml up -d redis redisinsight

# Open browser: http://localhost:5540
```

First-time setup in RedisInsight:

1. Add Database -> Host: `redis`, Port: `6379`
2. Test connection -> Connect

Use cases:

- View cached HTTP responses (keys, TTLs, content)
- Debug cache hit/miss patterns
- Monitor cache memory usage
- Manually inspect/delete cache entries
- Analyze cache key distribution

## Testing

Use Pants for tests to handle monorepo dependencies consistently.

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

## Code Quality

Code quality tools:

- Type checking: `ty`
- Formatting: `ruff format`
- Import sorting/linting: `ruff`

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

## Type Safety Protocol

All code should pass ty type checking before commits.

```bash
# Check all source files
uv run ty check scripts/ libs/ apps/

# Check specific library
uv run ty check libs/http_cache/
```

## Service Health Checks

```bash
# Check service health
curl http://localhost:8002/health

# Check Qdrant health
curl http://localhost:6333/health

# Get database statistics
curl http://localhost:8002/api/v1/admin/stats
```

## Environment Variables

### Critical Configuration

- `QDRANT_URL`: Vector database URL (default: `http://localhost:6333`)
- `QDRANT_COLLECTION_NAME`: Collection name (default: `anime_database`)
- `TEXT_EMBEDDING_MODEL`: Text model (default: `BAAI/bge-m3`)
- `IMAGE_EMBEDDING_MODEL`: Image model (default: `ViT-L-14/laion2b_s32b_b82k`)

### Performance Tuning

- `QDRANT_ENABLE_QUANTIZATION`: Enable vector quantization (default: `false`)
- `QDRANT_QUANTIZATION_TYPE`: Quantization type (`scalar`, `binary`, `product`)
- `MODEL_WARM_UP`: Pre-load models during startup (default: `false`)
- `MAX_BATCH_SIZE`: Maximum batch size for operations (default: `500`)

### Service Configuration

- `VECTOR_SERVICE_PORT`: Service port (default: `8002`)
- `DEBUG`: Enable debug mode (default: `true`)
- `LOG_LEVEL`: Logging level (default: `INFO`)
