# Technical Documentation

## Development Environment and Stack

### Technology Choices

#### Core Framework

- **FastAPI 0.115+**: Chosen for high-performance async capabilities, automatic OpenAPI documentation, and excellent type safety with Pydantic
- **Python 3.12+**: Latest Python for performance improvements, improved type hints, and modern language features
- **Uvicorn**: ASGI server for production-grade async request handling

#### Vector Database Architecture

- **Qdrant 1.14+**: Selected for its superior performance with HNSW indexing, multi-vector support, and production-ready features
- **HNSW Algorithm**: Hierarchical Navigable Small World for fast approximate nearest neighbor search
- **Advanced Quantization**: Binary, scalar, and product quantization for 40x speedup potential
- **11-Vector Primary Collection**: Main anime collection with named vectors (9×1024-dim text + 2×768-dim visual)
- **Dual-Collection Architecture**:
  - **Primary Collection**: 38K+ anime entries with comprehensive 13-vector semantic coverage
  - **Episode Collection**: Granular episode-level search with BGE-M3 chunking (Phase 3.5)
  - **Slug-Based Linking**: Cross-collection relationships using title-based IDs
- **Hybrid Dense+Sparse**: Support for both semantic embeddings and explicit feature matching (Phase 4)

#### AI/ML Stack

- **BGE-M3 (BAAI/bge-m3)**: State-of-the-art multilingual embedding model with 1024 dimensions, supporting 8192 token context
- **OpenCLIP ViT-L/14**: Vision transformer for 768-dimensional image embeddings with commercial-friendly licensing
- **Multi-Provider Support**: FastEmbed, HuggingFace, Sentence Transformers for dynamic model selection
- **PyTorch 2.0+**: Backend ML framework with optimized inference and LoRA fine-tuning support

### Development Setup

#### Prerequisites

**System Requirements:**

- Python 3.12+ for modern language features
- Docker and Docker Compose for containerization
- Git for version control
- 8GB+ RAM required for model loading
- GPU optional but recommended for image processing

#### Local Development

**Setup Process:**

1. Clone repository from version control
2. Install Python dependencies via requirements.txt
3. Start Qdrant database using Docker Compose
4. Run service using Python module execution
5. Access API documentation at localhost:8002/docs

#### Docker Development

**Container-based Development:**

- Full stack deployment using docker compose
- Service health verification via health endpoint
- Isolated development environment

### Configuration Management

#### Environment Variables

The service uses Pydantic Settings for type-safe configuration:

**Vector Service Configuration:**

- VECTOR_SERVICE_HOST: Service host address (default: 0.0.0.0)
- VECTOR_SERVICE_PORT: Service port (default: 8002)
- DEBUG: Enable debug mode (default: true)

**Qdrant Database Configuration:**

- QDRANT_URL: Database server URL (default: http://localhost:6333)
- QDRANT_COLLECTION_NAME: Collection name (default: anime_database)

**Embedding Models Configuration:**

- TEXT_EMBEDDING_MODEL: Text model (default: BAAI/bge-m3)
- IMAGE_EMBEDDING_MODEL: Image model (default: jinaai/jina-clip-v2)

**Performance Tuning:**

- QDRANT_ENABLE_QUANTIZATION: Enable quantization (default: false)
- MODEL_WARM_UP: Pre-load models (default: false)

#### Configuration Validation

- **Field Validation**: Pydantic validators ensure valid distance metrics, embedding providers, and log levels
- **Type Safety**: All configuration fields are strictly typed
- **Environment Override**: Settings can be overridden via environment variables or `.env` file

### Key Technical Decisions

#### Multi-Vector Architecture

- **Decision**: Store text, picture, and thumbnail vectors separately in same collection
- **Rationale**: Enables targeted search types while maintaining data locality
- **Implementation**: Named vectors in Qdrant with different dimensions

#### Embedding Model Selection

- **BGE-M3**: Chosen for multilingual support, large context window, and state-of-the-art performance
- **JinaCLIP v2**: Selected for superior vision-language understanding compared to OpenAI CLIP
- **Model Caching**: HuggingFace cache directory for faster subsequent loads

#### Async Architecture

- **FastAPI Async**: All endpoints are async for non-blocking I/O
- **Qdrant Async Client**: Ensures database operations don't block request handling
- **Lifespan Management**: Proper async initialization and cleanup

#### Error Handling Strategy

- **HTTP Exceptions**: Proper status codes with detailed error messages
- **Validation Errors**: Pydantic automatically handles request validation
- **Database Errors**: Graceful degradation when Qdrant is unavailable
- **Logging**: Structured logging with configurable levels

### Design Patterns in Use

#### Dependency Injection

- **Settings**: Cached settings instance using `@lru_cache`
- **Qdrant Client**: Global instance initialized during lifespan
- **Router Dependencies**: Future support for authentication/authorization

#### Factory Pattern

- **Client Creation**: QdrantClient factory with configuration-based initialization
- **Embedding Processors**: Factory methods for different embedding providers

#### Repository Pattern

- **Vector Operations**: Abstracted through QdrantClient interface
- **Data Models**: Pydantic models for request/response validation
- **Configuration**: Settings class encapsulates all configuration logic

#### Observer Pattern (Future)

- **Health Monitoring**: Health check observers for different components
- **Metrics Collection**: Performance metric observers

### Performance Optimization

#### Vector Database Optimization

- **HNSW Parameters**: Configurable `ef_construct` and `M` parameters for index tuning
- **Quantization**: Optional scalar/binary quantization for memory efficiency
- **Payload Indexing**: Indexed fields for fast metadata filtering
- **Memory Mapping**: Configurable threshold for disk vs memory storage

#### Model Performance

- **Model Warming**: Optional pre-loading during service startup
- **Cache Management**: HuggingFace model cache with configurable directory
- **Batch Processing**: Efficient batch embedding generation

#### API Performance

- **Async Processing**: Non-blocking request handling
- **Connection Pooling**: Efficient Qdrant client connection management
- **Response Compression**: Automatic FastAPI compression
- **CORS Optimization**: Configurable CORS settings

### Technical Constraints

#### Memory Constraints

- **Model Loading**: BGE-M3 + JinaCLIP v2 require ~4GB RAM combined
- **Vector Storage**: 384-dim + 2�512-dim vectors per anime = ~5KB per document
- **Index Memory**: HNSW index requires additional memory proportional to dataset size

#### Performance Constraints

- **Embedding Generation**: Text: ~50ms, Image: ~200ms per item
- **Vector Search**: ~10ms for 100K vectors with HNSW
- **Concurrent Limits**: ~100 simultaneous requests before degradation

#### Storage Constraints

- **Vector Size**: 100K anime � 5KB vectors = ~500MB vector storage
- **Payload Size**: Metadata adds ~2KB per anime document
- **Index Overhead**: HNSW index adds ~30% storage overhead

### Development Tools

#### Code Quality

- **Black**: Code formatting (configured in pyproject.toml)
- **isort**: Import sorting
- **autoflake**: Unused import removal
- **mypy**: Static type checking (future)

#### Testing Framework

- **pytest**: Unit and integration testing
- **pytest-asyncio**: Async test support
- **httpx**: HTTP client for API testing
- **pytest-mock**: Mocking for isolated tests

#### API Documentation

- **FastAPI OpenAPI**: Automatic API documentation
- **Swagger UI**: Interactive API explorer at `/docs`
- **ReDoc**: Alternative documentation at `/redoc`

#### Monitoring and Observability

- **Structured Logging**: JSON-formatted logs with timestamps
- **Health Endpoints**: `/health` for service and database status
- **Error Tracking**: Exception logging with context
- **Performance Metrics**: Response time logging (future Prometheus integration)

### Deployment Considerations

#### Container Optimization

- **Multi-stage Build**: Separate build and runtime stages
- **Layer Caching**: Optimized Dockerfile layer ordering
- **Security**: Non-root user, minimal base image
- **Size Optimization**: .dockerignore for build context reduction

#### Production Settings

- **Debug Mode**: Disabled in production
- **Logging**: INFO level with structured format
- **CORS**: Restricted origins for security
- **Health Checks**: Docker health check configuration

#### Scaling Considerations

- **Stateless Design**: No local state, suitable for horizontal scaling
- **Database Sharing**: Multiple instances can share same Qdrant cluster
- **Load Balancing**: Standard HTTP load balancing compatible
- **Resource Requirements**: 2 CPU cores, 4GB RAM per instance recommended

### Security Considerations

#### API Security

- **Input Validation**: Pydantic models prevent injection attacks
- **CORS Configuration**: Configurable origin restrictions
- **Error Information**: Careful error message exposure
- **Request Limits**: Configurable batch and search limits

#### Data Security

- **No Sensitive Data**: Only public anime metadata stored
- **TLS Termination**: HTTPS recommended for production
- **Access Logging**: Request logging for audit trails

#### Infrastructure Security

- **Container Security**: Non-root user, minimal privileges
- **Network Security**: Internal Qdrant communication
- **Secret Management**: Environment variable configuration
- **Update Strategy**: Regular security updates for dependencies

### Million-Query Vector Database Optimization Analysis

#### **Comprehensive Architecture Assessment (Phase 2.5)**

**Current State Analysis:**

- Repository contains 65+ AnimeEntry schema fields with comprehensive anime metadata
- Existing 3-vector architecture: text (384-dim BGE-M3) + picture + thumbnail (512-dim JinaCLIP v2)
- Proven scale: 38,894+ anime entries in MCP server implementation
- Current performance: 80-350ms query latency, 50+ RPS throughput

**Optimization Strategy for Million-Query Scale:**

#### **12-Vector Semantic Architecture**

**Technical Decision:** Single comprehensive collection with 12 named vectors

- **10 Text Vectors (1024-dim BGE-M3 each):** title_vector, character_vector, genre_vector, staff_vector, review_vector, temporal_vector, streaming_vector, related_vector, franchise_vector, episode_vector
- **1 Visual Vector (512-dim JinaCLIP v2):** image_vector (unified picture/thumbnail/images)
- **Rationale:** Data locality optimization, atomic consistency, reduced complexity

#### **Performance Optimization Configuration**

**Vector Quantization Strategy:**

- **High-Priority Vectors:** Scalar quantization (int8) for semantic-rich vectors (title, character, genre, review, image)
- **Medium-Priority Vectors:** Scalar quantization with disk storage for moderate-usage vectors
- **Low-Priority Vectors:** Binary quantization (32x compression) for utility vectors (franchise, episode, sources, identifiers)
- **Memory Reduction Target:** 75% reduction (15GB → 4GB for 30K anime, 500GB → 125GB for 1M anime)

**HNSW Parameter Optimization:**

```python
# Anime-specific HNSW optimization
high_priority_hnsw = {
    "ef_construct": 256,  # Higher for better anime similarity detection
    "m": 64,             # More connections for semantic richness
    "ef": 128            # Search-time optimization
}

medium_priority_hnsw = {
    "ef_construct": 200,
    "m": 48,
    "ef": 64
}

low_priority_hnsw = {
    "ef_construct": 128,
    "m": 32,
    "ef": 32
}
```

**Payload Optimization Strategy:**

- **Index Almost Everything (~60+ fields):** All structured data fields for filtering, sorting, and frontend functionality
- **Payload-Only (No Index):** Only URLs, technical metadata (enrichment_metadata, enhanced_metadata), and possibly large embedded text that's fully vectorized
- **Computed Fields:** popularity_score, content_richness_score, search_boost_factor, character_count

#### **Scalability Projections**

**Performance Targets Validated:**

- **Query Latency:** 100-500ms for complex multi-vector searches (85% improvement from current)
- **Memory Usage:** ~32GB peak for 1M anime with optimization (vs ~200GB unoptimized)
- **Throughput:** 300-600 RPS sustained mixed workload (12x improvement)
- **Concurrent Users:** 100K+ concurrent support (100x improvement)
- **Storage:** 175GB total optimized vs 500GB unoptimized (65% reduction)

#### **Technical Implementation Patterns**

**Rollback-Safe Implementation Strategy:**

- **Configuration-First:** All optimizations start with settings.py changes
- **Parallel Methods:** New 11-vector methods alongside existing 3-vector methods
- **Graceful Fallbacks:** All systems degrade to current functionality on failure
- **Feature Flags:** Production toggles without code deployment
- **Atomic Sub-Phases:** 2-4 hour implementation windows with independent testing

**Memory Management Patterns:**

- **Priority-Based Storage:** High-priority vectors in memory, medium on disk-cached, low on disk-only
- **Connection Pooling:** 50 concurrent connections with health monitoring
- **Memory Mapping:** 50MB threshold for large collection optimization
- **Garbage Collection:** Optimized for large vector operations

#### **Frontend Integration Technical Specifications**

**Customer-Facing Payload Design:**
Based on comprehensive AnimeEntry schema analysis and 11-vector architecture:

- **Search Results (Fast Loading):** Essential display fields for listing pages
- **Detail View (Complete):** All 65+ fields for comprehensive anime pages
- **Filtering (Performance):** ~60+ indexed fields for real-time filtering on all structured data
- **Computed Performance Fields:** Ranking scores, popularity metrics, content richness indicators
- **Vector Coverage:** All semantic content embedded in 14 specialized vectors for similarity search

**API Performance Optimization:**

- **Response Compression:** Automatic FastAPI gzip compression
- **Field Selection:** Dynamic payload field selection based on request type
- **Batch Operations:** Optimized for 1000-item batch processing
- **Streaming Responses:** Large result set streaming support

#### **Production Deployment Technical Requirements**

**Infrastructure Specifications:**

- **Minimum System:** 64GB RAM, 16 CPU cores for 1M anime scale
- **Database Configuration:** Qdrant cluster with 3 replicas, sharding by vector priority
- **Caching Architecture:** Redis cluster with L1 (in-memory) + L2 (Redis) + L3 (disk) tiers
- **Network:** 10Gbps for inter-service communication under load

**Monitoring and Observability:**

- **Vector-Specific Metrics:** Per-vector performance, quantization effectiveness, memory allocation
- **Search Analytics:** Query patterns, latency distribution, cache hit rates
- **Resource Monitoring:** Memory usage per vector type, CPU utilization patterns
- **SLA Targets:** 99.9% uptime, <200ms 95th percentile latency, <0.1% error rate

### Future Technical Enhancements

#### Phase 2.5 Vector Optimization (Current Focus)

- **14-Vector Collection Implementation**: Complete semantic search coverage
- **Quantization Deployment**: 75% memory reduction with maintained accuracy
- **Performance Validation**: Million-query scalability testing
- **Frontend Integration**: Customer-facing payload optimization

#### Phase 3 Production Scale Optimization

- **Redis Caching**: Multi-tier query result caching layer
- **Prometheus Metrics**: Comprehensive vector database monitoring
- **Authentication**: JWT-based API authentication with rate limiting
- **Load Testing**: Million-query performance validation

#### Phase 3.5 Episode Collection Architecture (In Progress)

- **Dual-Collection Design**: Separate episode collection for granular search
- **BGE-M3 Episode Chunking**: Hierarchical averaging with equal weighting
- **Cross-Collection Linking**: Slug-based ID strategy for anime-episode relationships
- **Smart Search Integration**: Episode-specific search capabilities
- **Testing and Validation**: Performance impact assessment

#### Phase 4 Enterprise Data Enrichment

- **API Pipeline Optimization**: Concurrent processing for 1,000-10,000 anime/day
- **AI Enhancement**: 6-stage pipeline with confidence scoring and quality validation
- **Horizontal Scaling**: Multi-agent coordination for distributed processing
- **Advanced Analytics**: Processing optimization and predictive scaling

#### Phase 4.1.2 Sparse Vector Integration (Updated for Dual Collections)

- **Metadata Sparse Vectors**: Categorical features (genre, studio, year) with static weights
- **Behavioral Sparse Vectors**: Future API usage patterns and user preferences
- **Unified Search Enhancement**: RRF fusion across dense and sparse vectors
- **Simplified Implementation**: Leveraging dual-collection architecture for cleaner integration

#### Phase 5 Advanced AI Features

- **Model Fine-tuning**: LoRA adaptation for anime-specific improvements
- **Global Distribution**: CDN integration and multi-region deployment
- **Advanced Search**: Context-aware search and intelligent query understanding
- **Enterprise Analytics**: Business intelligence integration and predictive analytics

## ML Validation Framework (Phase 3)

### Embedding Quality Validation

#### Model Drift Detection

**Implementation Strategy:**

```python
# Historical metrics with rolling windows
class EmbeddingQualityMonitor:
    alert_bands = {
        "genre_clustering": {"excellent": 0.75, "warning": 0.65, "critical": 0.60},
        "studio_similarity": {"excellent": 0.70, "warning": 0.60, "critical": 0.55},
        "temporal_consistency": {"excellent": 0.80, "warning": 0.70, "critical": 0.65}
    }
```

**Key Metrics:**

- **Distribution Shift Detection**: Wasserstein distance across BGE-M3 1024 dimensions
- **Semantic Coherence**: Genre clustering purity, studio visual consistency
- **Trend Analysis**: 7-day and 30-day rolling windows with statistical significance testing

#### Cross-Modal Validation

**Contrastive Testing Protocol:**

```python
# Same anime text+image should be more similar than random pairs
positive_similarities = cosine_similarity(text_embedding, same_anime_image)
negative_similarities = cosine_similarity(text_embedding, random_anime_image)

# Statistical validation with Mann-Whitney U test
assert mannwhitneyu(positive_similarities, negative_similarities).pvalue < 0.001
```

### Search Quality Validation

#### Gold Standard Dataset

**Expert-Curated Test Cases (500 queries):**

- **Genre Categories**: Shounen, shoujo, seinen, josei with character archetypes
- **Studio Styles**: Visual consistency validation across production houses
- **Temporal Queries**: Era-specific anime and sequel/franchise relationships
- **Edge Cases**: Ambiguous queries and boundary conditions

#### Hard Negative Sampling

**Critical Validation Tests:**

```python
hard_negatives = {
    "genre_confusion": {
        "query": "Romantic comedy anime like Toradora",
        "negatives": ["Attack on Titan", "Monster", "Ghost in the Shell"],
        "threshold": "<0.3"  # Should be very dissimilar
    }
}
```

#### Automated Metrics Pipeline

- **Precision@K, Recall@K, NDCG**: Traditional relevance metrics
- **Mean Reciprocal Rank (MRR)**: Critical for "find specific anime" queries
- **Semantic Consistency**: Within-result coherence measurement

### A/B Testing Framework

#### User Simulation Models

**Advanced Click Modeling:**

```python
class CascadeClickModel:
    """Users scan top-to-bottom, click first satisfying result"""

class DependentClickModel:
    """Models examination vs attractiveness separately"""
    examination_probs = [0.95, 0.85, 0.70, 0.50, 0.30, 0.15]  # Position-based
```

#### Search Algorithm Comparison

- **Statistical Significance Testing**: Proper experimental design
- **Engagement Metrics**: CTR, satisfaction scores, dwell time simulation
- **Performance Analysis**: Response time vs quality trade-offs

## Sparse Vector Integration (Phase 4)

### Information-Theoretic Feature Weighting

#### Feature Discriminative Power Analysis

**Optimized Weight Configuration:**

```python
optimized_weights = {
    "genre": 1.0,           # Strong semantic signal
    "demographics": 0.9,     # High user preference correlation
    "studio": 0.7,          # Multiple studios dilute signal
    "source_material": 0.6,  # Moderate preference indicator
    "year_bucket": 0.4,     # Avoid temporal overfitting
    "award_winner": 1.2,    # Quality signal boost
}
```

#### Adaptive Weight Learning

**From User Interaction Data:**

```python
# Learn from click logs using L1-regularized logistic regression
def learn_weights_from_clicks(click_data):
    features = extract_sparse_features(clicked_results)
    labels = apply_position_bias_correction(clicks)
    lr = LogisticRegression(penalty='l1', alpha=0.01)
    return dict(zip(feature_names, abs(lr.coef_[0])))
```

### Behavioral Sparse Vectors

#### Implicit Feedback Processing

**Sophisticated Signal Weighting:**

```python
feedback_weights = {
    # Positive signals
    "completed": 1.0,           "rewatched": 1.5,
    "high_rating": 1.2,         "added_to_favorites": 1.3,

    # Negative signals
    "dropped_early": -0.8,      "skipped_episodes": -0.3,
    "low_rating": -0.6,

    # Collaborative features
    "similar_user_liked": 0.8,  "trending_in_community": 0.6
}
```

#### Collaborative Filtering Integration

**"Users who liked this also liked" features:**

```python
def add_collaborative_features(anime_id, user_interactions):
    similar_users = find_similar_users(user_interactions)
    cf_features = {}
    for user in similar_users[:10]:
        user_prefs = get_user_preferences(user)
        for category, strength in user_prefs.items():
            cf_features[f"cf_{category}"] += strength * 0.1
    return cf_features
```

### Advanced Fusion Algorithms

#### Learnable Fusion Implementation

**Context-Aware Weight Adaptation:**

```python
context_weights = {
    "exploratory": {"dense": 0.8, "sparse_meta": 0.2, "sparse_behavior": 0.0},
    "targeted": {"dense": 0.4, "sparse_meta": 0.4, "sparse_behavior": 0.2},
    "personalized": {"dense": 0.3, "sparse_meta": 0.2, "sparse_behavior": 0.5}
}
```

#### Neural Fusion Networks

**For Large-Scale Systems:**

```python
class NeuralFusionNetwork(nn.Module):
    def __init__(self):
        self.dense_encoder = nn.Linear(1024, 256)    # BGE-M3 encoding
        self.sparse_encoder = nn.Linear(100, 256)    # Sparse features
        self.attention = nn.MultiheadAttention(256, 8)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1)
        )
```

### Recommendation Quality Evaluation

#### Diversity and Personalization Metrics

**Multi-Dimensional Diversity Calculation:**

```python
def calculate_anime_diversity(anime_i, anime_j):
    return (
        0.4 * genre_jaccard_distance(anime_i, anime_j) +
        0.2 * studio_diversity_binary(anime_i, anime_j) +
        0.2 * temporal_diversity_normalized(anime_i, anime_j) +
        0.2 * demographic_diversity_binary(anime_i, anime_j)
    )
```

#### Anti-Pattern Detection

**"Shounen + MAPPA" Collapse Prevention:**

```python
def detect_recommendation_collapse(recommendations):
    genre_entropy = calculate_genre_distribution_entropy(recommendations)
    studio_concentration = calculate_studio_concentration_ratio(recommendations)

    assert genre_entropy > 2.0  # Sufficient genre diversity
    assert studio_concentration < 0.4  # No single studio dominance
```

#### Personalization Coverage Analysis

**Inter-Group Diversity Measurement:**

```python
def personalization_coverage(recommendations, user_profiles):
    profile_groups = cluster_user_profiles(user_profiles)
    inter_group_diversity = 0

    for group_a, group_b in combinations(profile_groups, 2):
        recs_a = [recommendations[user] for user in group_a]
        recs_b = [recommendations[user] for user in group_b]
        diversity = calculate_recommendation_set_diversity(recs_a, recs_b)
        inter_group_diversity += diversity

    return inter_group_diversity / len(combinations(profile_groups, 2))
```

### Implementation Architecture

#### Extending Existing 14-Vector Collection (CHOSEN APPROACH)

**Rationale**: Preserve 38K+ anime entries, maintain data locality, unified search capabilities.

**Collection Extension Strategy:**

```python
# src/vector/qdrant_client.py - Modify existing collection creation
def _create_multi_vector_config(self) -> Dict:
    """Extend existing 11-vector config with sparse vectors"""
    vectors_config = {
        # EXISTING: 11 dense vectors (preserve compatibility)
        "title_vector": VectorParams(size=1024, distance=Distance.COSINE),
        "character_vector": VectorParams(size=1024, distance=Distance.COSINE),
        "genre_vector": VectorParams(size=1024, distance=Distance.COSINE),
        "staff_vector": VectorParams(size=1024, distance=Distance.COSINE),
        "review_vector": VectorParams(size=1024, distance=Distance.COSINE),
        "temporal_vector": VectorParams(size=1024, distance=Distance.COSINE),
        "streaming_vector": VectorParams(size=1024, distance=Distance.COSINE),
        "related_vector": VectorParams(size=1024, distance=Distance.COSINE),
        "franchise_vector": VectorParams(size=1024, distance=Distance.COSINE),
        "episode_vector": VectorParams(size=1024, distance=Distance.COSINE),
        "image_vector": VectorParams(size=768, distance=Distance.COSINE),
        "character_image_vector": VectorParams(size=768, distance=Distance.COSINE),

        # NEW: Sparse vectors (backwards compatible addition)
        "metadata_sparse": {},     # Categorical features (genre, studio, year, etc.)
        "behavioral_sparse": {}    # Future: API usage patterns, user preferences
    }
    return vectors_config

# Feature flag for gradual rollout
ENABLE_SPARSE_VECTORS = self.settings.enable_sparse_vectors  # Default: False
```

**Zero-Downtime Migration:**

```python
async def migrate_collection_with_sparse_vectors(self):
    """Add sparse vectors to existing collection without data loss"""

    # 1. Check if collection needs sparse vector support
    existing_config = await self.client.get_collection(self.collection_name)
    has_sparse = "metadata_sparse" in existing_config.config.params.vectors

    if not has_sparse and self.settings.enable_sparse_vectors:
        logger.info("Adding sparse vector support to existing collection...")

        # 2. Update collection schema (Qdrant supports adding vectors)
        await self.client.update_collection(
            collection_name=self.collection_name,
            vectors_config={
                **existing_config.config.params.vectors,  # Preserve existing
                "metadata_sparse": {},  # Add sparse support
                "behavioral_sparse": {}
            }
        )

        # 3. Backfill sparse vectors for existing data (optional, gradual)
        if self.settings.backfill_sparse_vectors:
            await self._backfill_sparse_vectors_gradually()
```

**Static Weight Implementation:**

```python
# src/vector/sparse_processor.py - New module for sparse vector generation
class AnimeStaticSparseProcessor:
    """Generate sparse vectors with anime-domain static weights"""

    def __init__(self):
        # Anime domain expertise encoded as static weights
        self.static_weights = {
            # Core preference signals (high weight)
            "genre_action": 1.0,         "genre_romance": 1.0,
            "genre_comedy": 1.0,         "genre_drama": 1.0,
            "demographic_shounen": 0.9,   "demographic_shoujo": 0.9,
            "demographic_seinen": 0.9,    "demographic_josei": 0.9,

            # Production signals (medium weight)
            "studio_mappa": 0.7,          "studio_madhouse": 0.7,
            "studio_bones": 0.7,          "studio_ghibli": 0.8,  # Higher for quality
            "source_manga": 0.6,          "source_novel": 0.6,

            # Temporal signals (low weight to avoid overfitting)
            "year_2020s": 0.4,            "year_2010s": 0.4,
            "year_2000s": 0.3,            "year_1990s": 0.3,

            # Quality signals (boost)
            "award_winner": 1.2,          "highly_rated": 1.1,
            "popular": 0.8,               "trending": 0.6,

            # Format signals
            "episodes_1_cour": 0.5,       "episodes_2_cour": 0.5,
            "episodes_long_running": 0.3, # Lower preference for very long series
        }

    def generate_metadata_sparse_vector(self, anime: AnimeEntry) -> SparseVector:
        """Convert anime metadata to weighted sparse vector"""
        indices = []
        values = []

        # Genre features
        for genre in anime.genres:
            feature_name = f"genre_{genre.lower().replace(' ', '_')}"
            if feature_name in self.static_weights:
                indices.append(self._get_feature_index(feature_name))
                values.append(self.static_weights[feature_name])

        # Studio features
        for studio in anime.studios:
            feature_name = f"studio_{studio.lower().replace(' ', '_')}"
            if feature_name in self.static_weights:
                indices.append(self._get_feature_index(feature_name))
                values.append(self.static_weights[feature_name])

        # Year bucket features
        year_bucket = f"year_{anime.year // 10 * 10}s"  # 2020s, 2010s, etc.
        if year_bucket in self.static_weights:
            indices.append(self._get_feature_index(year_bucket))
            values.append(self.static_weights[year_bucket])

        # Quality features
        if anime.awards and len(anime.awards) > 0:
            indices.append(self._get_feature_index("award_winner"))
            values.append(self.static_weights["award_winner"])

        if anime.score and anime.score > 8.0:
            indices.append(self._get_feature_index("highly_rated"))
            values.append(self.static_weights["highly_rated"])

        return SparseVector(indices=indices, values=values)
```

**Unified Search Integration:**

```python
# src/vector/qdrant_client.py - Extend existing search methods
async def search_with_sparse_fusion(
    self,
    query: str,
    limit: int = 20,
    enable_sparse: bool = None,
    fusion_weights: Dict[str, float] = None
) -> List[Dict]:
    """Enhanced search with optional sparse vector fusion"""

    # Default fusion weights (static for now)
    if fusion_weights is None:
        fusion_weights = {
            "dense_semantic": 0.6,    # BGE-M3 semantic understanding
            "sparse_metadata": 0.4,   # Explicit feature matching
        }

    # Check if sparse vectors are enabled
    enable_sparse = enable_sparse or self.settings.enable_sparse_vectors

    if enable_sparse:
        # Hybrid dense + sparse search using Qdrant fusion
        results = await self.client.query_points(
            collection_name=self.collection_name,
            query=QueryRequest([
                # Dense semantic search (existing functionality)
                NearestQuery(
                    nearest=self.text_processor.encode_text(query),
                    using="title_vector",  # Primary semantic vector
                ),

                # Sparse metadata search (NEW)
                NearestQuery(
                    nearest=self._generate_query_sparse_vector(query),
                    using="metadata_sparse",
                ),
            ]),
            limit=limit,
            fusion=Fusion.RRF  # Reciprocal Rank Fusion (proven algorithm)
        )
    else:
        # Fallback to existing dense-only search
        results = await self.search(query, limit)  # Existing method

    return results

def _generate_query_sparse_vector(self, query: str) -> SparseVector:
    """Convert search query to sparse vector for explicit matching"""
    # Simple query analysis for sparse features
    query_lower = query.lower()

    indices = []
    values = []

    # Genre detection in query
    genre_keywords = {
        "action": "genre_action", "romance": "genre_romance",
        "comedy": "genre_comedy", "drama": "genre_drama"
    }
    for keyword, feature in genre_keywords.items():
        if keyword in query_lower:
            indices.append(self._get_feature_index(feature))
            values.append(1.0)

    # Studio detection in query
    studio_keywords = {
        "mappa": "studio_mappa", "madhouse": "studio_madhouse",
        "ghibli": "studio_ghibli", "bones": "studio_bones"
    }
    for keyword, feature in studio_keywords.items():
        if keyword in query_lower:
            indices.append(self._get_feature_index(feature))
            values.append(1.0)

    return SparseVector(indices=indices, values=values)
```

#### Real-Time Adaptation

**Online Learning Framework:**

```python
def online_weight_adaptation(recent_feedback_window=7):
    recent_data = get_recent_interactions(recent_feedback_window)
    updated_weights = incremental_weight_update(recent_data)

    # Gradual transition to avoid sudden changes
    fusion_weights = 0.9 * current_weights + 0.1 * updated_weights
    return fusion_weights
```

### Performance Targets

#### Validation Framework Benchmarks

- **Embedding Quality**: >0.75 genre clustering, >0.70 studio consistency
- **Search Quality**: >0.80 Precision@5, >0.75 NDCG, <0.001 statistical significance
- **Recommendation Quality**: >0.70 diversity, >0.30 personalization coverage

#### Sparse Vector Performance Benchmarks

**Current Baseline (Dense-Only 14-Vector Architecture):**

```python
# Performance metrics from existing 38K+ anime dataset
baseline_performance = {
    "text_search_latency": "80ms",          # BGE-M3 title_vector search
    "image_search_latency": "250ms",        # OpenCLIP image_vector search
    "multimodal_search_latency": "350ms",   # Combined text + image
    "memory_usage_per_anime": "~5KB",       # 11 vectors + payload
    "total_memory_38k_anime": "~190MB",     # Current proven scale
    "concurrent_requests": "50+ RPS",       # Tested throughput
    "precision_at_5": "0.82",              # Estimated from query patterns
    "semantic_coherence": "0.78",          # Genre clustering baseline
}
```

**Target Performance with Sparse Vector Integration:**

```python
# Projected performance after Phase 4 implementation
sparse_enhanced_performance = {
    # Latency targets (backwards compatible)
    "dense_only_search": "80ms",           # Unchanged - fallback mode
    "sparse_only_search": "60ms",          # Faster - no embedding generation
    "hybrid_dense_sparse": "120ms",        # +50% for fusion benefits
    "learnable_fusion": "140ms",           # +75% for adaptive learning

    # Memory efficiency targets
    "sparse_vector_overhead": "<15%",      # <30MB additional for 38K anime
    "total_memory_with_sparse": "~220MB",  # Acceptable growth
    "backfill_memory_impact": "<5%",       # During gradual deployment

    # Accuracy improvement targets
    "precision_at_5_improvement": ">10%",  # 0.82 → 0.90+ target
    "genre_preference_accuracy": ">15%",   # Better genre matching
    "studio_style_matching": ">20%",       # Explicit studio preferences
    "demographic_targeting": ">25%",       # Shounen/Shoujo/Seinen/Josei

    # Scalability targets
    "1M_anime_memory_projection": "~5.5GB", # Linear scaling validation
    "1M_anime_query_latency": "<200ms",     # Sub-linear scaling target
    "concurrent_requests_maintained": "50+ RPS", # No degradation
}
```

**Integration Performance Validation:**

```python
# Phase 5 integration benchmarks with existing codebase
integration_benchmarks = {
    # Backwards compatibility validation
    "existing_api_endpoints": "100% compatible",    # No breaking changes
    "current_search_performance": "No degradation", # With sparse disabled
    "38k_anime_migration_time": "<30 minutes",      # Zero-downtime target
    "rollback_time": "<5 minutes",                  # Emergency rollback

    # Feature flag performance impact
    "sparse_disabled_overhead": "<1%",              # Feature flag cost
    "sparse_enabled_cold_start": "<10s",            # First search with sparse
    "backfill_performance_impact": "<5%",           # During gradual rollout

    # Monitoring integration
    "validation_monitoring_overhead": "<2%",        # Quality tracking cost
    "api_logging_overhead": "<3%",                  # Pattern collection cost
    "health_check_response_time": "<50ms",          # Enhanced health checks
}
```

**Performance Testing Framework:**

```python
# Comprehensive benchmarking for validation
class PerformanceBenchmarkSuite:
    def __init__(self):
        self.test_datasets = {
            "small": "1K anime",      # Quick validation
            "medium": "10K anime",    # Integration testing
            "large": "38K anime",     # Current production scale
            "xlarge": "100K anime",   # Future scale validation
        }

    async def benchmark_search_latency(self, dataset_size, query_type):
        """Measure search performance across different configurations"""

        # Test configurations
        configs = [
            {"mode": "dense_only", "vectors": ["title_vector"]},
            {"mode": "dense_multi", "vectors": ["title_vector", "character_vector", "genre_vector"]},
            {"mode": "sparse_only", "vectors": ["metadata_sparse"]},
            {"mode": "hybrid_fusion", "vectors": ["title_vector", "metadata_sparse"]},
        ]

        results = {}
        for config in configs:
            latencies = []
            for _ in range(100):  # 100 queries per config
                start_time = time.time()
                await self.execute_search(query_type, config)
                latency = (time.time() - start_time) * 1000  # Convert to ms
                latencies.append(latency)

            results[config["mode"]] = {
                "mean_latency": statistics.mean(latencies),
                "p95_latency": statistics.quantiles(latencies, n=20)[18],  # 95th percentile
                "p99_latency": statistics.quantiles(latencies, n=100)[98], # 99th percentile
            }

        return results

    async def benchmark_accuracy_improvement(self):
        """Measure recommendation quality improvement with sparse vectors"""

        # Test query sets
        test_queries = [
            {"query": "shounen action anime", "expected_demographic": "shounen"},
            {"query": "Studio Ghibli style films", "expected_studio_similarity": "ghibli"},
            {"query": "romance anime from 2020s", "expected_genre_year": ["romance", "2020s"]},
        ]

        accuracy_results = {}
        for query_data in test_queries:
            # Dense-only results
            dense_results = await self.search_dense_only(query_data["query"])

            # Hybrid dense+sparse results
            hybrid_results = await self.search_hybrid(query_data["query"])

            # Calculate accuracy improvement
            dense_accuracy = self.calculate_accuracy(dense_results, query_data)
            hybrid_accuracy = self.calculate_accuracy(hybrid_results, query_data)

            improvement = hybrid_accuracy - dense_accuracy
            accuracy_results[query_data["query"]] = {
                "dense_accuracy": dense_accuracy,
                "hybrid_accuracy": hybrid_accuracy,
                "improvement_percentage": improvement / dense_accuracy * 100
            }

        return accuracy_results

    def calculate_memory_scaling(self, anime_count):
        """Project memory usage at different scales"""

        base_memory_per_anime = {
            "dense_vectors": 4.8,      # KB: 11 vectors (9×1024×4 + 2×768×4 bytes)
            "payload_metadata": 0.2,   # KB: JSON payload
            "sparse_vectors": 0.5,     # KB: Estimated sparse vector overhead
        }

        total_memory_mb = anime_count * sum(base_memory_per_anime.values()) / 1024

        return {
            "anime_count": anime_count,
            "memory_per_anime_kb": sum(base_memory_per_anime.values()),
            "total_memory_mb": total_memory_mb,
            "total_memory_gb": total_memory_mb / 1024,
            "projected_viable": total_memory_mb < 16000,  # <16GB target
        }
```

**Real-World Performance Validation:**

````python
# Continuous performance monitoring for production deployment
class ProductionPerformanceMonitor:
    def __init__(self):
        self.performance_targets = {
            "search_latency_p95": 200,      # ms
            "search_latency_p99": 500,      # ms
            "memory_usage_limit": 8000,     # MB
            "cpu_usage_limit": 70,          # %
            "concurrent_requests": 50,      # RPS
            "error_rate_limit": 0.1,        # %
        }

    async def validate_performance_targets(self):
        """Continuous validation against production targets"""

        current_metrics = await self.collect_system_metrics()

        violations = []
        for metric, target in self.performance_targets.items():
            if current_metrics[metric] > target:
                violations.append({
                    "metric": metric,
                    "current": current_metrics[metric],
                    "target": target,
                    "severity": "critical" if current_metrics[metric] > target * 1.5 else "warning"
                })

        if violations:
            await self.trigger_performance_alerts(violations)

        return {
            "status": "healthy" if not violations else "degraded",
            "violations": violations,
            "all_metrics": current_metrics
        }

## LangGraph Smart Query Integration (Phase 2.6) - TENTATIVE

### Architecture Overview

**Design Philosophy**: Enhance existing 11-vector search capabilities with intelligent LLM-powered query analysis without modifying the core vector architecture.

#### **Core Components**
```python
# Unified query flow architecture
User Query → LangGraph Workflow → Intent Analysis → Vector Selection → Tool Execution → Results

# Key architectural components
1. QueryState: Workflow state management
2. LLM Analyzer: Intent detection and query enhancement
3. Vector Selector: Intelligent vector and weight selection
4. Tool Router: smart_search_tool vs smart_recommendation_tool
5. Result Formatter: Unified response structure
````

### **Technical Implementation**

#### **1. LangGraph Workflow Architecture**

```python
# src/ai/langgraph_workflow.py
from langgraph import StateGraph, END
from typing import TypedDict, List, Dict, Literal

class QueryState(TypedDict):
    """Workflow state for unified query processing"""
    user_query: str
    intent: Literal["search", "recommendation"]
    enhanced_query: str
    vector_weights: List[Dict[str, Any]]  # [{"name": "genre_vector", "weight": 1.0}]
    results: List[Dict[str, Any]]
    metadata: Dict[str, Any]

def create_unified_query_workflow() -> StateGraph:
    """Single endpoint workflow with intelligent routing"""
    workflow = StateGraph(QueryState)

    # Core workflow nodes
    workflow.add_node("analyze_intent", analyze_user_intent)
    workflow.add_node("enhance_query", enhance_search_query)
    workflow.add_node("select_vectors", select_optimal_vectors)
    workflow.add_node("execute_tools", execute_vector_tools)
    workflow.add_node("format_results", format_unified_response)

    # Conditional routing based on intent
    workflow.add_conditional_edges(
        "analyze_intent",
        route_by_intent,
        {
            "search": "enhance_query",
            "recommendation": "enhance_query"
        }
    )

    # Linear flow after intent detection
    workflow.add_edge("enhance_query", "select_vectors")
    workflow.add_edge("select_vectors", "execute_tools")
    workflow.add_edge("execute_tools", "format_results")
    workflow.add_edge("format_results", END)

    workflow.set_entry_point("analyze_intent")
    return workflow.compile()
```

#### **2. LLM Intelligence System**

```python
# src/ai/query_analyzer.py
class SmartQueryAnalyzer:
    """LLM-powered query analysis and enhancement"""

    def __init__(self, llm_provider: str = "openai"):
        self.llm_client = self._init_llm_client(llm_provider)
        self.vector_context = self._load_vector_descriptions()

    async def analyze_user_intent(self, query: str) -> Dict[str, Any]:
        """Determine search vs recommendation intent with confidence scoring"""

        prompt = f"""
        Analyze this anime query and determine user intent:
        Query: "{query}"

        Intent Types:
        - "search": User wants to find specific anime matching criteria
          Examples: "ninja anime", "romance from 2020", "Studio Ghibli films"
        - "recommendation": User wants suggestions based on preferences
          Examples: "like Naruto", "similar to Attack on Titan", "if I enjoyed..."

        Return JSON:
        {{
            "intent": "search|recommendation",
            "confidence": 0.0-1.0,
            "reasoning": "brief explanation"
        }}
        """

        response = await self.llm_client.generate(prompt)
        return json.loads(response)

    async def select_optimal_vectors(self, query: str, intent: str) -> List[Dict[str, Any]]:
        """Intelligent vector selection with weights from 11-vector architecture"""

        vector_descriptions = {
            "title_vector": "Anime titles, synopsis, background - semantic content matching",
            "character_vector": "Character names, descriptions, relationships - character-focused search",
            "genre_vector": "Genres, themes, demographics - categorical classification",
            "staff_vector": "Directors, studios, voice actors - production team matching",
            "review_vector": "Awards, recognition, critical reception - quality indicators",
            "temporal_vector": "Air dates, broadcast info - time-based search",
            "streaming_vector": "Platform availability, licensing - accessibility filtering",
            "related_vector": "Franchise connections, sequels - relationship mapping",
            "franchise_vector": "Multimedia content, themes - franchise context",
            "episode_vector": "Episode details, filler status - episode-specific search",
            "image_vector": "Cover art, posters, visual style - aesthetic matching",
            "character_image_vector": "Character visual recognition - character identification"
        }

        prompt = f"""
        Select the most relevant vectors for this anime query:
        Query: "{query}"
        Intent: {intent}

        Available vectors (choose 3-5 most relevant):
        {json.dumps(vector_descriptions, indent=2)}

        Return JSON with vector selection and weights:
        {{
            "vectors": [
                {{"name": "vector_name", "weight": 0.0-1.0, "reasoning": "why selected"}},
                ...
            ],
            "strategy": "explanation of selection strategy"
        }}

        Weight Guidelines:
        - 1.0: Primary semantic match for the query
        - 0.8-0.9: Strong supporting context
        - 0.6-0.7: Moderate relevance
        - 0.4-0.5: Weak supporting signal
        """

        response = await self.llm_client.generate(prompt)
        return json.loads(response)

    async def enhance_query(self, query: str, intent: str) -> str:
        """Query optimization for vector search"""

        if intent == "search":
            prompt = f"""
            Optimize this search query for semantic vector search:
            Original: "{query}"

            Enhance for better semantic matching:
            - Add relevant keywords and synonyms
            - Include context that helps with anime domain
            - Optimize for embedding similarity
            - Keep concise and focused

            Return only the enhanced query text.
            """
        else:  # recommendation
            prompt = f"""
            Optimize this recommendation query for similarity search:
            Original: "{query}"

            Extract similarity factors:
            - Identify what the user liked about referenced anime
            - Focus on transferable qualities (themes, style, genre)
            - Remove specific names, focus on characteristics
            - Optimize for finding similar content

            Return only the enhanced query text.
            """

        enhanced = await self.llm_client.generate(prompt)
        return enhanced.strip()
```

#### **3. Enhanced Tool Methods**

```python
# src/vector/qdrant_client.py - NEW methods
async def smart_search_tool(
    self,
    enhanced_query: str,
    vector_weights: List[Dict[str, Any]],
    limit: int = 5
) -> List[Dict[str, Any]]:
    """LLM-guided search with intelligent vector selection and weighting"""

    try:
        # Generate query embedding
        query_embedding = self.embedding_manager.text_processor.encode_text(enhanced_query)
        if not query_embedding:
            return []

        # Build weighted vector queries
        prefetch_queries = []
        for weight_config in vector_weights:
            vector_name = weight_config["name"]
            weight = weight_config.get("weight", 1.0)

            if vector_name in self.settings.vector_names:
                prefetch_query = Prefetch(
                    using=vector_name,
                    query=query_embedding,
                    limit=limit * 2,  # Get more results for better fusion
                )
                prefetch_queries.append(prefetch_query)

        # Execute weighted multi-vector search using existing infrastructure
        response = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetch_queries,
            query=FusionQuery(fusion=Fusion.RRF),  # Use proven RRF fusion
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        # Format results for unified response
        return self._format_smart_search_results(response, vector_weights)

    except Exception as e:
        logger.error(f"Smart search tool failed: {e}")
        # Graceful fallback to existing search
        return await self.search_text_comprehensive(enhanced_query, limit)

async def smart_recommendation_tool(
    self,
    enhanced_query: str,
    vector_weights: List[Dict[str, Any]],
    limit: int = 5
) -> List[Dict[str, Any]]:
    """LLM-guided recommendations with similarity-focused vector selection"""

    # Same implementation as smart_search_tool
    # The intelligence is in the LLM's query enhancement and vector selection
    return await self.smart_search_tool(enhanced_query, vector_weights, limit)

def _format_smart_search_results(
    self,
    response: Any,
    vector_weights: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Format search results with smart query metadata"""

    results = []
    for point in response.points:
        payload = point.payload if point.payload else {}

        # Enhanced result with smart query context
        result = {
            "id": str(point.id),
            "anime_id": payload.get("id", str(point.id)),
            "title": payload.get("title", ""),
            "synopsis": payload.get("synopsis", ""),
            "genres": payload.get("genres", []),
            "rating": payload.get("rating", ""),
            "episodes": payload.get("episodes", 0),
            "status": payload.get("status", ""),
            "cover_image": payload.get("images", {}).get("jpg", {}).get("image_url", ""),

            # Smart query specific metadata
            "smart_score": point.score,
            "fusion_score": point.score,
            "vector_breakdown": {
                weight["name"]: weight["weight"]
                for weight in vector_weights
            }
        }
        results.append(result)

    return results
```

#### **4. Unified API Endpoint**

```python
# src/api/smart_query.py - NEW router
from langgraph import CompiledGraph

router = APIRouter()

# Global workflow instance
workflow: Optional[CompiledGraph] = None

@app.on_event("startup")
async def initialize_smart_query():
    """Initialize LangGraph workflow during app startup"""
    global workflow
    workflow = create_unified_query_workflow()

class UnifiedQueryRequest(BaseModel):
    """Single endpoint for all query types"""
    query: str = Field(..., description="Natural language query", min_length=1)
    limit: int = Field(default=5, ge=1, le=20, description="Number of results")
    enable_smart: bool = Field(default=True, description="Enable smart query processing")

class UnifiedQueryResponse(BaseModel):
    """Smart query response with metadata"""
    query: str = Field(..., description="Original query")
    intent: str = Field(..., description="Detected intent: search or recommendation")
    enhanced_query: str = Field(..., description="LLM-enhanced query")
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    metadata: Dict[str, Any] = Field(..., description="Processing metadata")

@router.post("/query", response_model=UnifiedQueryResponse)
async def unified_query_endpoint(request: UnifiedQueryRequest):
    """Single endpoint for intelligent anime queries with LLM enhancement"""

    try:
        # Feature flag check
        if not request.enable_smart or not workflow:
            # Fallback to existing search
            from ..main import qdrant_client
            results = await qdrant_client.search_text_comprehensive(
                request.query, request.limit
            )
            return UnifiedQueryResponse(
                query=request.query,
                intent="search",
                enhanced_query=request.query,
                results=results,
                metadata={"mode": "fallback", "smart_enabled": False}
            )

        # Execute LangGraph workflow
        initial_state = QueryState(
            user_query=request.query,
            intent="",
            enhanced_query="",
            vector_weights=[],
            results=[],
            metadata={}
        )

        # Run workflow with timeout
        final_state = await asyncio.wait_for(
            workflow.ainvoke(initial_state),
            timeout=30.0  # 30 second timeout
        )

        return UnifiedQueryResponse(
            query=request.query,
            intent=final_state["intent"],
            enhanced_query=final_state["enhanced_query"],
            results=final_state["results"],
            metadata={
                "mode": "smart",
                "vectors_used": len(final_state["vector_weights"]),
                "processing_time": final_state["metadata"].get("processing_time", 0),
                "llm_analysis": final_state["metadata"].get("llm_analysis", {})
            }
        )

    except asyncio.TimeoutError:
        logger.warning(f"Smart query timeout for: {request.query}")
        # Fallback to traditional search
        from ..main import qdrant_client
        results = await qdrant_client.search_text_comprehensive(
            request.query, request.limit
        )
        return UnifiedQueryResponse(
            query=request.query,
            intent="search",
            enhanced_query=request.query,
            results=results,
            metadata={"mode": "fallback", "reason": "timeout"}
        )

    except Exception as e:
        logger.error(f"Smart query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Smart query processing failed: {str(e)}"
        )
```

### **Performance Considerations**

#### **Response Time Breakdown**

```python
# Expected performance profile
smart_query_latency = {
    "llm_intent_analysis": "0.5-1.0s",     # OpenAI/Anthropic API call
    "llm_vector_selection": "0.5-1.0s",    # Vector selection reasoning
    "llm_query_enhancement": "0.3-0.5s",   # Query optimization
    "vector_search": "80-120ms",           # Existing 11-vector search
    "result_formatting": "10-20ms",        # Response preparation
    "total_target": "<3.0s",               # End-to-end target
}

# Performance optimizations
optimization_strategies = {
    "llm_caching": "Cache similar query analyses for 1 hour",
    "parallel_processing": "Run LLM analysis and embedding generation in parallel",
    "timeout_handling": "30s timeout with fallback to traditional search",
    "batch_llm_calls": "Combine intent + vector selection into single LLM call",
    "connection_pooling": "Reuse LLM client connections",
}
```

#### **Resource Requirements**

```python
# Additional resource usage
resource_overhead = {
    "memory": "+100-200MB for LLM client libraries",
    "cpu": "+10-15% during LLM processing",
    "network": "+1-3 API calls per query to LLM provider",
    "storage": "Minimal - only caching LLM responses temporarily",
}

# Cost considerations
cost_analysis = {
    "openai_gpt4": "$0.03 per 1K tokens (~$0.0001 per query)",
    "anthropic_claude": "$0.015 per 1K tokens (~$0.00005 per query)",
    "daily_cost_1000_queries": "$0.05-0.10",
    "monthly_cost_30k_queries": "$1.50-3.00",
}
```

### **Error Handling and Fallbacks**

#### **Graceful Degradation Strategy**

```python
# Multi-level fallback system
fallback_hierarchy = [
    "smart_query_full",           # LLM + weighted vector search
    "smart_query_cached",         # Cached LLM + vector search
    "traditional_comprehensive",  # Existing search_text_comprehensive()
    "traditional_simple",         # Basic search() method
    "error_response",            # Graceful error with empty results
]

# Fallback triggers
fallback_conditions = {
    "llm_timeout": "30 seconds",
    "llm_error": "API error or invalid response",
    "vector_error": "Qdrant connection issues",
    "parsing_error": "JSON parsing failures",
    "validation_error": "Response validation failures",
}
```

### **Monitoring and Observability**

#### **Smart Query Metrics**

```python
# Key performance indicators
smart_query_metrics = {
    # Latency metrics
    "llm_analysis_latency": "Time for intent detection + vector selection",
    "total_query_latency": "End-to-end response time",
    "fallback_rate": "Percentage of queries falling back to traditional search",

    # Quality metrics
    "intent_detection_accuracy": "Manual validation of intent classification",
    "vector_selection_relevance": "Relevance of selected vectors to query",
    "result_quality_improvement": "A/B test comparing smart vs traditional",

    # System metrics
    "llm_api_error_rate": "LLM provider API failures",
    "cache_hit_rate": "LLM response caching effectiveness",
    "resource_utilization": "Memory and CPU overhead",
}

# Dashboard integration
monitoring_integration = {
    "prometheus_metrics": "Custom metrics for smart query performance",
    "grafana_dashboards": "Real-time smart query monitoring",
    "alerting_rules": "Thresholds for fallback rate, latency, errors",
    "health_checks": "LLM connectivity in service health endpoint",
}
```

### **Testing Strategy**

#### **Comprehensive Test Suite**

```python
# Test categories for smart query functionality
test_coverage = {
    "unit_tests": {
        "llm_intent_detection": "Test intent classification accuracy",
        "vector_selection_logic": "Validate vector relevance scoring",
        "query_enhancement": "Test query optimization quality",
        "tool_method_execution": "Verify weighted search functionality",
    },

    "integration_tests": {
        "workflow_execution": "End-to-end LangGraph workflow",
        "api_endpoint": "Unified query endpoint functionality",
        "fallback_scenarios": "Error handling and graceful degradation",
        "performance_benchmarks": "Latency and resource usage validation",
    },

    "quality_tests": {
        "a_b_comparison": "Smart vs traditional search quality",
        "domain_query_accuracy": "Anime-specific query handling",
        "edge_case_handling": "Ambiguous and complex queries",
        "multilingual_support": "Non-English query processing",
    }
}
```

### **Deployment Strategy**

#### **Feature Flag Rollout**

```python
# Gradual deployment approach
rollout_strategy = {
    "phase_1": "Internal testing with feature flag disabled",
    "phase_2": "5% traffic to smart query endpoint",
    "phase_3": "25% traffic with A/B testing",
    "phase_4": "75% traffic with performance monitoring",
    "phase_5": "100% traffic with traditional search as fallback",
}

# Configuration management
feature_flags = {
    "ENABLE_SMART_QUERY": "Master switch for smart query functionality",
    "SMART_QUERY_PERCENTAGE": "Percentage of traffic to route to smart endpoint",
    "LLM_PROVIDER": "OpenAI or Anthropic selection",
    "LLM_TIMEOUT": "Timeout before fallback (default: 30s)",
    "CACHE_LLM_RESPONSES": "Enable LLM response caching",
}
```

### **Future Enhancements**

#### **Advanced Features (Post-MVP)**

```python
# Planned enhancements after initial implementation
future_features = {
    "conversational_memory": "Multi-turn conversation context",
    "personalized_vectors": "User-specific vector weight learning",
    "real_time_adaptation": "Dynamic vector selection based on feedback",
    "custom_llm_fine_tuning": "Anime-domain specific LLM optimization",
    "hybrid_reasoning": "Combine symbolic reasoning with vector search",
    "advanced_fusion": "Neural fusion networks for vector combination",
}
```

```

```
