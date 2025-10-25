# Validation Framework Documentation

This directory contains a comprehensive ML validation framework for the anime vector search service. The framework provides quality assurance, performance monitoring, and statistical testing capabilities for the 14-vector architecture.

## Overview

The validation framework is designed to ensure semantic correctness, embedding quality, and search performance of the anime vector search system. It implements industry-standard ML evaluation metrics, statistical testing, and user behavior simulation.

## Module Architecture

```
src/validation/
├── __init__.py                    # Framework exports and imports
├── vector_field_mapping.py       # 14-vector architecture definitions
├── vector_system_validator.py    # Complete system validation suite
├── dataset_analyzer.py           # Dynamic query generation from dataset
├── search_quality.py            # Search quality metrics and gold standards
├── embedding_quality.py         # Model drift detection and embedding analysis
└── ab_testing.py                # Statistical A/B testing framework
```

---

## 📊 vector_field_mapping.py

**Purpose**: Defines the complete 14-vector architecture field mappings and utility functions for vector management.

### Key Components

#### `VECTOR_FIELD_MAPPINGS`
Comprehensive mapping of all 14 vectors to their respective data fields:

**Text Vectors (BGE-M3, 1024-dim each):**
- `title_vector`: Title information and basic descriptions
- `character_vector`: Character names, descriptions, relationships
- `genre_vector`: Comprehensive classification and content categorization
- `staff_vector`: Directors, composers, studios, voice actors
- `review_vector`: Recognition and achievements
- `temporal_vector`: Semantic temporal data and scheduling
- `streaming_vector`: Platform availability and licensing
- `related_vector`: Franchise connections with URLs
- `franchise_vector`: Multimedia content and franchise materials
- `episode_vector`: Detailed episode information

**Visual Vectors (OpenCLIP ViT-L/14, 768-dim each):**
- `image_vector`: General anime visual content (covers, posters, banners)
- `character_image_vector`: Character visual content from character images

#### Utility Functions
- `get_vector_fields(vector_name)`: Get fields indexed in a specific vector
- `get_vector_description(vector_name)`: Get description of vector contents
- `get_text_vectors()` / `get_image_vectors()`: Get vector lists by type
- `is_vector_populated(vector_name)`: Check if vector contains meaningful data
- `get_searchable_vectors()`: Get vectors suitable for search validation

### Usage Example
```python
from src.validation.vector_field_mapping import get_vector_fields, get_searchable_vectors

# Get fields for character vector
character_fields = get_vector_fields("character_vector")
# Returns: ["characters"]

# Get all searchable vectors
searchable = get_searchable_vectors()
# Returns list excluding known empty vectors like review_vector
```

---

## 🔍 vector_system_validator.py

**Purpose**: Comprehensive validation suite for the 14-vector anime search system, testing individual vectors and multi-vector fusion effectiveness.

### Key Components

#### `VectorSystemValidator`
Main validation class that orchestrates complete system testing.

**Key Methods:**
- `validate_all_vectors()`: Complete 14-vector system validation
- `validate_semantic_relevance()`: Semantic correctness validation
- `_test_individual_vector()`: Test single vector with domain queries
- `_test_multi_vector_search()`: Test fusion methods effectiveness
- `_generate_recommendations()`: Generate actionable improvement suggestions

#### Validation Process
1. **Dynamic Query Generation**: Uses `DatasetAnalyzer` to generate queries from actual data
2. **Individual Vector Testing**: Tests each of 14 vectors with vector-specific queries
3. **Multi-Vector Fusion Testing**: Tests `search_complete()`, `search_text_comprehensive()`, etc.
4. **Semantic Relevance Validation**: Ensures results are actually relevant to users
5. **Performance Metrics**: Response time, success rate, recommendation generation

#### Validation Metrics
- **Success Rate**: Percentage of queries passing validation (target: >80%)
- **Response Time**: Average search response time (target: <500ms)
- **Semantic Relevance**: Jaccard similarity between query and results
- **Fusion Effectiveness**: RRF fusion improvement over single vectors

### Usage Example
```python
from src.validation.vector_system_validator import VectorSystemValidator

# Initialize with Qdrant client
validator = VectorSystemValidator(qdrant_client)

# Run complete validation
results = await validator.validate_all_vectors()

# Check results
overall_success = results["overall_success_rate"]
recommendations = results["recommendations"]
```

### Validation Targets
- Individual vector success rate: >70%
- Overall system success rate: >80%
- Response time: <500ms per query
- Multi-vector fusion improvement: measurable vs single vectors

---

## 📈 dataset_analyzer.py

**Purpose**: Analyzes actual dataset content to generate realistic validation queries instead of using hardcoded assumptions.

### Key Components

#### `DatasetAnalyzer`
Intelligent dataset analysis for dynamic validation.

**Key Methods:**
- `analyze_dataset()`: Complete dataset content analysis
- `generate_dynamic_queries()`: Create validation queries from real data
- `_sample_dataset_points()`: Sample dataset for analysis
- `_analyze_vector_populations()`: Determine which vectors contain meaningful data
- `_analyze_genres()` / `_analyze_content_types()`: Content distribution analysis

#### Analysis Capabilities
1. **Content Characteristics**: Synopsis availability, studio distribution, content themes
2. **Vector Population Analysis**: Which vectors are actually populated with meaningful data
3. **Genre/Type Distribution**: Most common genres, demographics, content types
4. **Dynamic Query Generation**: Creates vector-specific test queries from actual content

#### Query Generation Strategy
- **Title Queries**: Uses meaningful words from actual titles, skipping common articles
- **Genre Queries**: Tests most frequent genres with sufficient data
- **Technical Queries**: Tests content types and formats from dataset
- **Temporal Queries**: Tests status and temporal data patterns
- **Vector-Specific**: Generates appropriate queries for each vector type

### Usage Example
```python
from src.validation.dataset_analyzer import DatasetAnalyzer

# Initialize with Qdrant client
analyzer = DatasetAnalyzer(qdrant_client)

# Analyze dataset
profile = await analyzer.analyze_dataset()
print(f"Total points: {profile['total_points']}")

# Generate dynamic queries
queries = await analyzer.generate_dynamic_queries()
title_queries = queries.get("title_vector", [])
```

### Analysis Output
- Dataset size and characteristics
- Vector population percentages
- Genre/type/status distributions
- Sample titles for reference
- Vector-specific validation queries

---

## ⭐ search_quality.py

**Purpose**: Comprehensive search quality validation with gold standard datasets, automated metrics, and hard negative sampling.

### Key Components

#### `GoldStandardDataset`
Expert-curated test datasets for search validation.

**Test Categories:**
- **Genre-based**: shounen action, shoujo romance, seinen psychological
- **Studio-based**: Studio Ghibli, MAPPA studio
- **Character archetypes**: ninja characters, magical girls
- **Temporal**: 90s classics, modern 2020s anime
- **Complex multi-faceted**: dark psychological, family adventure

**Hard Negative Tests:**
- Genre confusion detection (romance vs action)
- Demographic confusion (cute girls vs dark psychological)
- Temporal confusion (classic vs modern)

#### `SearchQualityValidator`
Automated search quality validation with ML metrics.

**Key Methods:**
- `validate_search_function()`: Complete search function validation
- `validate_hard_negatives()`: Hard negative confusion detection
- `generate_quality_report()`: Comprehensive quality assessment
- `_calculate_query_metrics()`: Per-query metric calculation

#### Evaluation Metrics
- **Precision@K**: Relevant results in top K positions
- **Recall@K**: Coverage of relevant results in top K
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **MRR**: Mean Reciprocal Rank for specific anime queries
- **Success Rate**: Percentage of queries meeting quality thresholds

### Usage Example
```python
from src.validation.search_quality import SearchQualityValidator

# Initialize validator
validator = SearchQualityValidator()

# Validate search function
async def my_search(query: str, limit: int):
    return await qdrant_client.search_complete(query, limit)

results = await validator.validate_search_function(my_search)

# Check metrics
precision = results["metrics"]["average_precision_at_5"]
success_rate = results["metrics"]["success_rate"]
```

### Quality Thresholds
- Precision@5: >60% (target >80% for excellent)
- Success rate: >70% (target >80% for excellent)
- Hard negative rejection: <30% similarity threshold

---

## 🧠 embedding_quality.py

**Purpose**: Embedding quality monitoring with model drift detection, semantic coherence metrics, and cross-modal validation.

### Key Components

#### `EmbeddingQualityMonitor`
Comprehensive embedding quality assessment.

**Key Methods:**
- `compute_embedding_quality_metrics()`: Complete quality assessment
- `detect_distribution_shift()`: Wasserstein distance drift detection
- `validate_cross_modal_consistency()`: Text-image embedding validation
- `get_trend_analysis()`: Historical trend analysis
- `generate_quality_report()`: Quality assessment with recommendations

#### Quality Metrics
1. **Genre Clustering Purity**: How well embeddings cluster by genre (silhouette score)
2. **Studio Visual Consistency**: Intra-studio vs inter-studio similarity for images
3. **Temporal Consistency**: Franchise/sequel relationship preservation
4. **Embedding Space Quality**: Dimensionality, variance, participation ratio

#### Drift Detection
- **Distribution Shift**: Wasserstein distance between current and reference embeddings
- **Dimension-wise Analysis**: Per-dimension drift detection
- **Alert Thresholds**: >10% dimensions showing drift triggers alert
- **Trend Analysis**: 7-day and 30-day rolling windows

#### Cross-Modal Validation
- **Contrastive Testing**: Same anime text vs image similarity
- **Statistical Significance**: Mann-Whitney U test (p < 0.001)
- **Effect Size**: Cohen's d for practical significance

### Usage Example
```python
from src.validation.embedding_quality import EmbeddingQualityMonitor

# Initialize monitor
monitor = EmbeddingQualityMonitor(history_days=30)

# Compute quality metrics
metrics = monitor.compute_embedding_quality_metrics(
    embeddings=text_embeddings,
    metadata=anime_metadata,
    vector_type="text"
)

# Check drift
drift_results = monitor.detect_distribution_shift(
    current_embeddings, reference_embeddings
)
```

### Alert Bands
- **Excellent**: Genre clustering >75%, Studio similarity >70%
- **Warning**: Genre clustering 65-75%, Studio similarity 60-70%
- **Critical**: Genre clustering <60%, Studio similarity <55%

---

## 🧪 ab_testing.py

**Purpose**: Statistical A/B testing framework for search algorithm comparison with user behavior simulation.

### Key Components

#### `CascadeClickModel`
User behavior simulation using cascade click model.

**Behavior Model:**
- Users scan results top-to-bottom
- Position bias decreases with rank
- Click first satisfying result and stop
- Satisfaction threshold configurable

#### `DependentClickModel`
Advanced click model separating examination and attractiveness.

**Features:**
- Position-dependent examination probabilities
- Relevance-dependent attractiveness weights
- Multiple clicks possible per session
- More realistic user behavior modeling

#### `ABTestingFramework`
Complete statistical testing framework.

**Key Methods:**
- `compare_search_algorithms()`: Full A/B test comparison
- `calculate_statistical_power()`: Statistical power analysis
- `_perform_statistical_tests()`: t-tests and effect size calculation
- `_generate_recommendation()`: Evidence-based recommendations

#### Statistical Tests
- **t-tests**: Statistical significance testing
- **Effect Size**: Cohen's d for practical significance
- **Power Analysis**: Sample size adequacy assessment
- **Confidence Intervals**: Uncertainty quantification

### Usage Example
```python
from src.validation.ab_testing import ABTestingFramework

# Initialize framework
ab_framework = ABTestingFramework(significance_level=0.05)

# Compare two search algorithms
async def algorithm_a(query: str, limit: int):
    return await client.search_complete(query, limit)

async def algorithm_b(query: str, limit: int):
    return await client.search_text_comprehensive(query, limit)

def relevance_evaluator(query: str, results: List[Dict]):
    # Your relevance scoring logic
    return [0.8, 0.6, 0.4, 0.2, 0.1]  # Example scores

results = await ab_framework.compare_search_algorithms(
    algorithm_a=algorithm_a,
    algorithm_b=algorithm_b,
    test_queries=["shounen action", "studio ghibli"],
    relevance_evaluator=relevance_evaluator,
    num_simulations=1000
)

print(results["recommendation"])
```

### Simulation Metrics
- **Cascade Model**: Satisfaction rate, average click position
- **Dependent Model**: Click rate, average clicks per session
- **Statistical**: t-statistics, p-values, effect sizes
- **Business**: User engagement, satisfaction rates

---

## 🚀 Usage Patterns

### Complete System Validation
```python
from src.validation import VectorSystemValidator, DatasetAnalyzer

# Initialize components
analyzer = DatasetAnalyzer(qdrant_client)
validator = VectorSystemValidator(qdrant_client)

# Run complete validation
results = await validator.validate_all_vectors()

# Check success rate
if results["overall_success_rate"] > 0.8:
    print("✅ System validation passed")
else:
    print("❌ System needs improvement")
    for rec in results["recommendations"]:
        print(f"- {rec}")
```

### Search Quality Assessment
```python
from src.validation import SearchQualityValidator

validator = SearchQualityValidator()

# Define search function to test
async def search_function(query: str, limit: int):
    return await qdrant_client.search_complete(query, limit)

# Run validation
results = await validator.validate_search_function(search_function)

# Generate quality report
report = validator.generate_quality_report()
print(f"Quality Grade: {report['quality_grade']}")
```

### Embedding Quality Monitoring
```python
from src.validation import EmbeddingQualityMonitor

monitor = EmbeddingQualityMonitor()

# Monitor text embeddings
text_metrics = monitor.compute_embedding_quality_metrics(
    embeddings=text_embeddings,
    metadata=metadata,
    vector_type="text"
)

# Check for drift
drift = monitor.detect_distribution_shift(current_emb, reference_emb)
if drift["drift_detected"]:
    print("⚠️ Model drift detected!")
```

### A/B Testing
```python
from src.validation import ABTestingFramework

ab_test = ABTestingFramework()

# Compare search algorithms
comparison = await ab_test.compare_search_algorithms(
    algorithm_a=search_complete,
    algorithm_b=search_text_comprehensive,
    test_queries=test_queries,
    relevance_evaluator=evaluate_relevance
)

print(f"Recommendation: {comparison['recommendation']}")
```

---

## 📋 Framework Integration

### With Main Application
The validation framework integrates seamlessly with the main anime vector service:

```python
# In your main application
from src.validation import VectorSystemValidator

# Initialize during startup
validator = VectorSystemValidator(qdrant_client)

# Add validation endpoint
@app.post("/api/v1/admin/validate")
async def validate_system():
    results = await validator.validate_all_vectors()
    return results
```

### Automated Monitoring
Set up periodic validation for production monitoring:

```python
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler

async def daily_validation():
    results = await validator.validate_all_vectors()
    if results["overall_success_rate"] < 0.8:
        # Send alert
        send_alert("System validation below threshold")

scheduler = AsyncIOScheduler()
scheduler.add_job(daily_validation, 'cron', hour=2)  # Daily at 2 AM
```

---

## 🎯 Performance Targets

### Individual Vector Performance
- **Success Rate**: >70% per vector
- **Response Time**: <500ms per query
- **Semantic Relevance**: >60% Jaccard similarity

### Overall System Performance
- **Success Rate**: >80% overall
- **Multi-Vector Fusion**: Measurable improvement over single vectors
- **Hard Negative Rejection**: <30% confusion rate

### Embedding Quality
- **Genre Clustering**: >65% purity (excellent: >75%)
- **Studio Consistency**: >60% similarity (excellent: >70%)
- **Cross-Modal Consistency**: Statistically significant (p < 0.001)
- **Model Drift**: <10% dimensions showing drift

### Search Quality
- **Precision@5**: >60% (excellent: >80%)
- **NDCG@5**: >60%
- **MRR**: >70%
- **Success Rate**: >70% (excellent: >80%)

---

## 🔧 Configuration

### Environment Variables
```bash
# Validation settings
VALIDATION_SAMPLE_SIZE=100          # Dataset sample size for analysis
VALIDATION_SIGNIFICANCE_LEVEL=0.05  # Statistical significance threshold
VALIDATION_DRIFT_THRESHOLD=0.1      # Model drift detection threshold
VALIDATION_HISTORY_DAYS=30          # Metrics retention period
```

### Alert Thresholds
Configure in `embedding_quality.py`:
```python
alert_bands = {
    "genre_clustering": {"excellent": 0.75, "warning": 0.65, "critical": 0.60},
    "studio_similarity": {"excellent": 0.70, "warning": 0.60, "critical": 0.55},
    "temporal_consistency": {"excellent": 0.80, "warning": 0.70, "critical": 0.65},
}
```

---

## 📊 Reporting and Analytics

### Quality Reports
All validators generate comprehensive reports with:
- Current performance metrics
- Trend analysis over time
- Alert levels and thresholds
- Actionable recommendations
- Historical comparisons

### Export Capabilities
```python
# Export validation results
results = await validator.validate_all_vectors()
with open("validation_report.json", "w") as f:
    json.dump(results, f, indent=2)

# Generate quality report
report = validator.generate_quality_report()
```

---

## 🛠️ Extension Points

### Custom Metrics
Add new quality metrics by extending base classes:

```python
class CustomEmbeddingMonitor(EmbeddingQualityMonitor):
    def compute_custom_metric(self, embeddings, metadata):
        # Your custom metric implementation
        pass
```

### Custom Test Cases
Add domain-specific test cases:

```python
# Extend GoldStandardDataset
gold_standard.anime_domain_queries["custom_test"] = {
    "query": "your custom query",
    "expected_results": ["expected", "anime", "titles"],
    "expected_genres": ["Action"],
}
```

### Integration Hooks
Add custom validation hooks:

```python
# Custom validation function
async def custom_validator(results):
    # Your custom validation logic
    return validation_score

# Register with framework
validator.add_custom_validator(custom_validator)
```

---

This validation framework provides production-ready quality assurance for the anime vector search service, ensuring semantic correctness, performance optimization, and continuous monitoring of the 14-vector architecture.