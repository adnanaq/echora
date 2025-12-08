# Embedding Model Initialization Strategy

## Current State (After ECHO-31 Refactor)

**Status**: Embedding models are NOT initialized during FastAPI app startup.

**Rationale**:
- Current API only provides admin endpoints (health, stats, collection info)
- No search endpoints exist yet that would require embedding generation
- Vector write operations are handled by CLI scripts and Lambda functions
- Faster startup time without loading large ML models

**Model Initialization in Scripts**:
- `scripts/reindex_anime_database.py` - Initializes TextProcessor and VisionProcessor
- `scripts/update_vectors.py` - Initializes TextProcessor and VisionProcessor
- `tests/conftest.py` - Initializes processors for test suite

Each script creates its own instances as needed for standalone execution.

## Future State: Agent Service Architecture

### When to Initialize Models

**Trigger**: When implementing search endpoints (Phase 5 in implementation_plan.md)

**Agent Service Core Responsibilities**:
1. Receive natural language queries from BFF Service
2. Use LLM to parse queries into structured search requests
3. **Generate embeddings for search queries** (requires models loaded)
4. Execute multi-vector search against Qdrant
5. Return ranked anime IDs to BFF

**Key Insight**: The Agent Service's primary job is generating embeddings for EVERY search request. Models must be loaded at startup.

### Recommended Initialization Pattern

**Option: Eager Loading at Startup** (RECOMMENDED for Agent Service)

**Implementation in `src/main.py` lifespan**:

```python
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize services on startup and cleanup on shutdown."""
    logger.info("Initializing Qdrant client and embedding models...")

    async_qdrant_client = None
    try:
        # Initialize AsyncQdrantClient
        if settings.qdrant_api_key:
            async_qdrant_client = AsyncQdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key
            )
        else:
            async_qdrant_client = AsyncQdrantClient(url=settings.qdrant_url)

        # Initialize embedding processors
        text_processor = TextProcessor(settings)
        vision_processor = VisionProcessor(settings)
        embedding_manager = MultiVectorEmbeddingManager(
            text_processor=text_processor,
            vision_processor=vision_processor,
            settings=settings,
        )

        # Initialize QdrantClient
        qdrant_client_instance = await QdrantClient.create(
            settings=settings,
            async_qdrant_client=async_qdrant_client,
            url=settings.qdrant_url,
            collection_name=settings.qdrant_collection_name,
        )

        # Store all clients on app state
        app.state.qdrant_client = qdrant_client_instance
        app.state.async_qdrant_client = async_qdrant_client
        app.state.embedding_manager = embedding_manager
        app.state.text_processor = text_processor
        app.state.vision_processor = vision_processor

        # Health check
        healthy = await app.state.qdrant_client.health_check()
        if not healthy:
            logger.error("Qdrant health check failed!")
            raise RuntimeError("Vector database is not available")

        logger.info("Agent service initialized successfully with embedding models")
        yield

    finally:
        logger.info("Shutting down agent service...")
        if async_qdrant_client:
            try:
                await async_qdrant_client.close()
                logger.info("AsyncQdrantClient closed successfully")
            except Exception as e:
                logger.error(f"Error closing AsyncQdrantClient: {e}")
```

**Dependency Injection**:

```python
# src/dependencies.py additions

async def get_embedding_manager(request: Request) -> MultiVectorEmbeddingManager:
    """Dependency that provides embedding manager instance."""
    if not hasattr(request.app.state, "embedding_manager"):
        raise RuntimeError("Embedding manager not available.")
    return request.app.state.embedding_manager

async def get_text_processor(request: Request) -> TextProcessor:
    """Dependency that provides text processor instance."""
    if not hasattr(request.app.state, "text_processor"):
        raise RuntimeError("Text processor not available.")
    return request.app.state.text_processor

async def get_vision_processor(request: Request) -> VisionProcessor:
    """Dependency that provides vision processor instance."""
    if not hasattr(request.app.state, "vision_processor"):
        raise RuntimeError("Vision processor not available.")
    return request.app.state.vision_processor
```

**Search Endpoint Example**:

```python
# src/api/search.py (future implementation)

from fastapi import APIRouter, Depends
from ..dependencies import get_qdrant_client, get_embedding_manager

router = APIRouter()

@router.post("/search")
async def search_anime(
    query: str,
    qdrant_client: QdrantClient = Depends(get_qdrant_client),
    embedding_manager: MultiVectorEmbeddingManager = Depends(get_embedding_manager),
):
    """Natural language search for anime."""
    # Generate query embedding
    query_vector = await embedding_manager.generate_text_embedding(query)

    # Execute search
    results = await qdrant_client.search(
        vector_name="text",
        query_vector=query_vector,
        limit=10
    )

    return results
```

### Why Eager Loading for Agent Service

**Pros**:
1. **Models ready immediately** - No latency on first request
2. **Predictable performance** - Every request has consistent response time
3. **Simpler code** - No lazy loading complexity
4. **Agent Service always needs models** - Core responsibility is embedding generation

**Cons**:
1. **Longer startup time** - Loading BGE-M3 and OpenCLIP takes ~10-30 seconds
2. **Higher memory footprint** - Models stay in RAM even during idle periods

**For Agent Service, pros outweigh cons** because:
- Service exists specifically to generate embeddings for search
- Production deployments expect models to be ready
- Health checks can verify model loading succeeded
- Memory cost is acceptable for dedicated search service

## Monorepo Migration Path

### Phase 1: Current Pattern (Standalone Repo)
- Eager loading in `src/main.py` lifespan
- Models stored on `app.state`
- Dependency injection via `Depends()`

### Phase 2: Monorepo with EmbeddingModelFactory

**Structure**:
```
libs/
  vector_processing/
    embedding_models/
      factory.py          # EmbeddingModelFactory
      text_processor.py   # Receives pre-initialized models
      vision_processor.py # Receives pre-initialized models

apps/
  agent/
    src/
      main.py             # Uses EmbeddingModelFactory
```

**Factory Pattern**:

```python
# libs/vector_processing/embedding_models/factory.py

class EmbeddingModelFactory:
    """Singleton factory for managing embedding models across services."""

    _instance = None
    _text_model = None
    _vision_model = None

    @classmethod
    async def get_instance(cls, settings: Settings):
        if cls._instance is None:
            cls._instance = cls()
            await cls._instance._initialize_models(settings)
        return cls._instance

    async def _initialize_models(self, settings: Settings):
        """Load models once, reuse across all processors."""
        # Load BGE-M3 model
        # Load OpenCLIP model
        # Store on class attributes

    def create_text_processor(self, settings: Settings) -> TextProcessor:
        """Create TextProcessor with pre-loaded model."""
        return TextProcessor(settings, model=self._text_model)

    def create_vision_processor(self, settings: Settings) -> VisionProcessor:
        """Create VisionProcessor with pre-loaded model."""
        return VisionProcessor(settings, model=self._vision_model)
```

**Agent Service Usage**:

```python
# apps/agent/src/main.py

from libs.vector_processing.embedding_models.factory import EmbeddingModelFactory

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize factory (loads models once)
    factory = await EmbeddingModelFactory.get_instance(settings)

    # Create processors with pre-loaded models
    text_processor = factory.create_text_processor(settings)
    vision_processor = factory.create_vision_processor(settings)

    # Rest of initialization...
```

**Benefits**:
- Single source of truth for model initialization
- Models loaded once, shared across multiple processor instances
- Easier testing with mock factory
- Cleaner separation of concerns

### Phase 3: Full Monorepo Maturity

**QdrantClient becomes pure DB client**:
- No embedding logic
- Only database operations
- Receives vectors from external processors

**All embedding logic in `libs/vector_processing/`**:
- Shared by Agent Service, CLI scripts, Lambda functions
- Consistent embedding generation across all services
- Single codebase for model updates

## Testing Strategy

**Current**: Each test file initializes its own processors (session-scoped fixtures)

**Future with Search Endpoints**: Mock embedding manager in tests to avoid loading real models

```python
# tests/conftest.py

@pytest.fixture
def mock_embedding_manager():
    """Mock embedding manager for fast tests."""
    manager = MagicMock(spec=MultiVectorEmbeddingManager)
    manager.generate_text_embedding.return_value = [0.1] * 1024
    manager.generate_image_embedding.return_value = [0.1] * 768
    return manager
```

## Performance Considerations

**Startup Time**:
- Without models: ~1-2 seconds
- With models (BGE-M3 + OpenCLIP): ~10-30 seconds
- Acceptable for production deployments with proper health checks

**Memory Usage**:
- BGE-M3: ~2GB RAM
- OpenCLIP ViT-L/14: ~1.5GB RAM
- Total: ~4GB for models + application overhead
- Requires appropriate container resource limits

**Request Latency**:
- Text embedding generation: ~50-200ms
- Image embedding generation: ~100-300ms
- Acceptable for interactive search use cases

## Migration Checklist

When implementing search endpoints:

- [ ] Add embedding model initialization to lifespan
- [ ] Store processors on app.state
- [ ] Create dependency injection functions
- [ ] Update health check to verify model loading
- [ ] Add model warmup step (optional, for consistent first-request latency)
- [ ] Update Docker resource limits for model memory requirements
- [ ] Add startup timeout configuration for slower model loading
- [ ] Document model versions and update procedures
- [ ] Create monitoring for model inference latency
- [ ] Test graceful degradation if model loading fails

## References

- Implementation Plan: `docs/implementation_plan.md` (Phase 5: Agent Service)
- Monorepo Plan: `MONOREPO_MIGRATION_PLAN.md`
- Current Processors: `src/vector/processors/`
- Current Scripts: `scripts/reindex_anime_database.py`, `scripts/update_vectors.py`
