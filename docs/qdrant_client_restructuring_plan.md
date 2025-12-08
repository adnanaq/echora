# QdrantClient Restructuring Plan for Monorepo Migration

**Document Version:** 1.0
**Date:** 2025-11-30
**Scope:** Full Optimization (Base Class + Utilities + Async Factories + Consolidation)
**Status:** Planning Phase

---

## Executive Summary

This document outlines the restructuring plan for QdrantClient and related components to align with the future monorepo migration (Phase 4 of `MONOREPO_MIGRATION_PLAN.md`). The goal is to implement these changes **now** in the current codebase structure to minimize work during the actual monorepo migration.

**Key Objectives:**
1. ✅ Implement abstract `VectorDBClient` base class for pluggability
2. ✅ Extract utilities (ID generation, validation, filter building) into separate modules
3. ✅ Add async factory methods for `TextProcessor` and `VisionProcessor`
4. ✅ Consolidate `AnimeFieldMapper` to single shared instance
5. ✅ Delete 300+ lines of commented-out code
6. ✅ Restructure directories to match future monorepo layout

**Timeline:** This is pre-monorepo work that makes Phase 4 migration trivial.

---

## Current State Analysis

### Strengths ✅
- **Clean separation**: QdrantClient is already separated from vector generation
- **No circular dependencies**: All dependencies are one-way
- **Async factory pattern**: QdrantClient.create() already implemented
- **Dependency injection**: AsyncQdrantClient injected correctly
- **Type safety**: Full type hints throughout

### Areas for Improvement ⚠️
- **No abstract base class**: QdrantClient is concrete-only
- **Embedded utilities**: ID generation, validation, filter building are private methods
- **Synchronous processor initialization**: TextProcessor/VisionProcessor initialize models in `__init__`
- **Redundant AnimeFieldMapper**: Three independent instances created
- **Commented code bloat**: 300+ lines of outdated commented methods

---

## Phase 1: Directory Structure Changes

### Current Structure
```
src/
├── vector/
│   ├── client/
│   │   └── qdrant_client.py (1742 lines)
│   └── processors/
│       ├── text_processor.py (786 lines)
│       ├── vision_processor.py (886 lines)
│       └── embedding_manager.py (536 lines)
```

### New Structure (Pre-Monorepo)
```
src/
├── vector/
│   ├── client/
│   │   ├── __init__.py
│   │   ├── base.py                      # NEW: VectorDBClient abstract base class
│   │   ├── qdrant_client.py             # MODIFIED: Implements VectorDBClient
│   │   └── utils/                       # NEW: Extracted utilities
│   │       ├── __init__.py
│   │       ├── id_generator.py          # NEW: Point ID generation
│   │       ├── vector_validator.py      # NEW: Vector validation
│   │       └── filter_builder.py        # NEW: Qdrant filter construction
│   └── processors/
│       ├── text_processor.py            # MODIFIED: Async factory
│       ├── vision_processor.py          # MODIFIED: Async factory
│       ├── embedding_manager.py         # MODIFIED: Receives shared AnimeFieldMapper
│       └── anime_field_mapper.py        # EXISTING: Will be shared singleton
└── utils/
    └── retry.py                         # EXISTING: Already extracted
```

### Future Monorepo Structure (Reference)
```
libs/
├── vector_db_interface/
│   └── src/
│       └── vector_db_interface/
│           ├── __init__.py
│           └── base.py                  # MOVES HERE: VectorDBClient base class
├── qdrant_client/
│   └── src/
│       └── qdrant_client/
│           ├── __init__.py
│           ├── client.py                # MOVES HERE: QdrantClient implementation
│           └── utils/                   # MOVES HERE: All utilities
│               ├── __init__.py
│               ├── id_generator.py
│               ├── vector_validator.py
│               └── filter_builder.py
└── vector_processing/
    └── src/
        └── vector_processing/
            ├── __init__.py
            ├── text_processor.py        # MOVES HERE
            ├── vision_processor.py      # MOVES HERE
            ├── embedding_manager.py     # MOVES HERE
            └── anime_field_mapper.py    # MOVES HERE
```

**Migration Impact:** By creating `base.py` and `utils/` now, the monorepo migration becomes a simple file move operation.

---

## Phase 2: Abstract Base Class Design

### 2.1 VectorDBClient Interface

**File:** `src/vector/client/base.py`

```python
"""Abstract base class for vector database clients.

This interface defines the contract that all vector database implementations
must satisfy, enabling pluggability between different vector databases
(Qdrant, Pinecone, Weaviate, etc.).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client.models import PointStruct, ScoredPoint, Filter


class VectorDBClient(ABC):
    """Abstract base class for vector database operations.

    All vector database clients must implement this interface to ensure
    consistent behavior across different vector database providers.
    """

    # ==================== Collection Management ====================

    @abstractmethod
    async def create_collection(
        self,
        collection_name: str,
        vector_config: Dict[str, Any],
        **kwargs: Any,
    ) -> bool:
        """Create a new collection with the specified vector configuration.

        Args:
            collection_name: Name of the collection to create
            vector_config: Vector configuration (dimensions, distance metric, etc.)
            **kwargs: Provider-specific configuration options

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def delete_collection(self, collection_name: str) -> bool:
        """Delete an existing collection.

        Args:
            collection_name: Name of the collection to delete

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists.

        Args:
            collection_name: Name of the collection to check

        Returns:
            True if collection exists, False otherwise
        """
        pass

    # ==================== Document Operations ====================

    @abstractmethod
    async def add_documents(
        self,
        documents: List[PointStruct],
        batch_size: int = 100,
    ) -> Dict[str, Any]:
        """Add documents to the collection in batches.

        Args:
            documents: List of PointStruct objects with vectors and payloads
            batch_size: Number of documents per batch

        Returns:
            Dictionary with success/failure counts and details
        """
        pass

    @abstractmethod
    async def get_by_id(
        self,
        point_id: str,
        with_vectors: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a document by its ID.

        Args:
            point_id: ID of the document to retrieve
            with_vectors: Whether to include vector data

        Returns:
            Document payload (and vectors if requested), or None if not found
        """
        pass

    # ==================== Vector Update Operations ====================

    @abstractmethod
    async def update_single_vector(
        self,
        point_id: str,
        vector_name: str,
        vector_data: List[float],
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> bool:
        """Update a single named vector for an existing point.

        Args:
            point_id: ID of the point to update
            vector_name: Name of the vector to update
            vector_data: New vector embedding
            max_retries: Maximum retry attempts on transient failures
            retry_delay: Initial delay between retries (exponential backoff)

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def update_batch_vectors(
        self,
        updates: List[Dict[str, Any]],
        dedup_policy: str = "last-wins",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> Dict[str, Any]:
        """Update multiple vectors across multiple points in a batch.

        Args:
            updates: List of update dictionaries with point_id, vector_name, vector_data
            dedup_policy: How to handle duplicates (last-wins, first-wins, fail, warn)
            max_retries: Maximum retry attempts on transient failures
            retry_delay: Initial delay between retries

        Returns:
            Dictionary with success/failed counts and per-update results
        """
        pass

    # ==================== Search Operations ====================

    @abstractmethod
    async def search_single_vector(
        self,
        vector_name: str,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[ScoredPoint]:
        """Search using a single vector.

        Args:
            vector_name: Name of the vector to search on
            query_vector: Query vector embedding
            limit: Maximum number of results
            filters: Optional filters to apply
            score_threshold: Minimum similarity score

        Returns:
            List of scored points ordered by relevance
        """
        pass

    @abstractmethod
    async def search_multi_vector(
        self,
        vector_queries: List[Dict[str, Any]],
        fusion_algorithm: str = "rrf",
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[ScoredPoint, Dict[str, float]]]:
        """Search using multiple vectors with score fusion.

        Args:
            vector_queries: List of {vector_name, query_vector, weight} dicts
            fusion_algorithm: Fusion method (rrf, dbsf)
            limit: Maximum number of results
            filters: Optional filters to apply

        Returns:
            List of (scored_point, score_breakdown) tuples
        """
        pass

    # ==================== Health & Statistics ====================

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the database connection is healthy.

        Returns:
            True if healthy, False otherwise
        """
        pass

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics.

        Returns:
            Dictionary with collection stats (count, vectors, disk usage, etc.)
        """
        pass
```

### 2.2 QdrantClient Implementation Changes

**File:** `src/vector/client/qdrant_client.py`

**Changes:**
1. Add import: `from .base import VectorDBClient`
2. Change class declaration: `class QdrantClient(VectorDBClient):`
3. Keep all existing methods (they already match the interface)
4. Update docstrings to reference interface contract

**Example:**
```python
from .base import VectorDBClient

class QdrantClient(VectorDBClient):
    """Qdrant-specific implementation of VectorDBClient.

    This class provides all vector database operations using Qdrant as the
    backend. It implements the VectorDBClient interface to ensure compatibility
    with other vector database providers.
    """

    # Existing implementation remains the same
    # All methods already match the VectorDBClient interface
```

---

## Phase 3: Utility Extraction

### 3.1 ID Generator Utility

**File:** `src/vector/client/utils/id_generator.py`

```python
"""Point ID generation utilities for vector database operations.

This module provides utilities for converting application-level IDs (like anime_id)
into vector database point IDs using consistent hashing strategies.
"""

import hashlib
from typing import Union


class PointIDGenerator:
    """Generate consistent point IDs from application identifiers.

    This class uses MD5 hashing to create deterministic, fixed-length IDs
    suitable for vector database point identification.
    """

    @staticmethod
    def generate(identifier: Union[str, int]) -> str:
        """Generate a point ID from an identifier.

        Uses MD5 hashing to create a consistent 32-character hex string
        that can be used as a vector database point ID.

        Args:
            identifier: Application identifier (e.g., anime_id)

        Returns:
            32-character hexadecimal string (MD5 hash)

        Example:
            >>> generator = PointIDGenerator()
            >>> point_id = generator.generate("anime_12345")
            >>> len(point_id)
            32
        """
        # Convert to string if integer
        id_str = str(identifier)

        # Generate MD5 hash
        hash_obj = hashlib.md5(id_str.encode())

        return hash_obj.hexdigest()

    @staticmethod
    def generate_batch(identifiers: list[Union[str, int]]) -> list[str]:
        """Generate point IDs for multiple identifiers.

        Args:
            identifiers: List of application identifiers

        Returns:
            List of point IDs in the same order as input
        """
        return [PointIDGenerator.generate(id) for id in identifiers]
```

**Usage in QdrantClient:**
```python
from .utils.id_generator import PointIDGenerator

class QdrantClient(VectorDBClient):
    def __init__(self, ...):
        self._id_generator = PointIDGenerator()

    async def update_single_vector(self, anime_id: str, ...):
        point_id = self._id_generator.generate(anime_id)
        # ... rest of implementation
```

### 3.2 Vector Validator Utility

**File:** `src/vector/client/utils/vector_validator.py`

```python
"""Vector validation utilities for quality control.

This module provides validation logic for vector updates, ensuring that
vectors have correct names, dimensions, and data types before database operations.
"""

from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class VectorValidator:
    """Validate vector updates against expected schema.

    This class encapsulates validation logic for vector names and dimensions,
    preventing invalid data from reaching the vector database.
    """

    def __init__(
        self,
        allowed_vector_names: List[str],
        vector_dimensions: dict[str, int],
    ):
        """Initialize validator with schema constraints.

        Args:
            allowed_vector_names: List of valid vector names
            vector_dimensions: Dict mapping vector names to expected dimensions
        """
        self.allowed_vector_names = set(allowed_vector_names)
        self.vector_dimensions = vector_dimensions

    def validate_vector_update(
        self,
        vector_name: str,
        vector_data: List[float],
    ) -> Tuple[bool, Optional[str]]:
        """Validate a vector update operation.

        Args:
            vector_name: Name of the vector to update
            vector_data: Vector embedding to validate

        Returns:
            Tuple of (is_valid, error_message)
            - (True, None) if valid
            - (False, error_message) if invalid
        """
        # Check vector name
        if vector_name not in self.allowed_vector_names:
            return False, (
                f"Invalid vector name '{vector_name}'. "
                f"Allowed: {sorted(self.allowed_vector_names)}"
            )

        # Check dimension
        expected_dim = self.vector_dimensions.get(vector_name)
        if expected_dim is None:
            return False, f"No dimension configured for vector '{vector_name}'"

        actual_dim = len(vector_data)
        if actual_dim != expected_dim:
            return False, (
                f"Vector '{vector_name}' dimension mismatch. "
                f"Expected {expected_dim}, got {actual_dim}"
            )

        # Check data types
        if not all(isinstance(x, (int, float)) for x in vector_data):
            return False, f"Vector '{vector_name}' contains non-numeric values"

        return True, None

    def validate_batch_updates(
        self,
        updates: List[dict],
    ) -> Tuple[List[dict], List[dict]]:
        """Validate a batch of vector updates.

        Args:
            updates: List of update dicts with vector_name and vector_data

        Returns:
            Tuple of (valid_updates, invalid_updates_with_errors)
        """
        valid_updates = []
        invalid_updates = []

        for update in updates:
            vector_name = update.get("vector_name")
            vector_data = update.get("vector_data")

            is_valid, error = self.validate_vector_update(vector_name, vector_data)

            if is_valid:
                valid_updates.append(update)
            else:
                invalid_updates.append({
                    **update,
                    "error": error,
                })

        return valid_updates, invalid_updates
```

**Usage in QdrantClient:**
```python
from .utils.vector_validator import VectorValidator

class QdrantClient(VectorDBClient):
    def __init__(self, settings: Settings, ...):
        self._validator = VectorValidator(
            allowed_vector_names=settings.vector_names,
            vector_dimensions=settings.vector_dimensions,
        )

    async def update_single_vector(self, ...):
        is_valid, error_msg = self._validator.validate_vector_update(
            vector_name, vector_data
        )
        if not is_valid:
            logger.error(f"Validation failed: {error_msg}")
            return False
        # ... rest of implementation
```

### 3.3 Filter Builder Utility

**File:** `src/vector/client/utils/filter_builder.py`

```python
"""Qdrant filter construction utilities.

This module provides utilities for converting application filter dictionaries
into Qdrant-specific Filter objects.
"""

from typing import Any, Dict, List, Optional
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    Range,
)
import logging

logger = logging.getLogger(__name__)


class FilterBuilder:
    """Build Qdrant Filter objects from application filter dictionaries.

    This class encapsulates the logic for translating high-level filter
    specifications into Qdrant's filter model.
    """

    @staticmethod
    def build_filter(filter_dict: Dict[str, Any]) -> Optional[Filter]:
        """Build a Qdrant Filter from a filter dictionary.

        Args:
            filter_dict: Dictionary with filter conditions
                Examples:
                    {"genre": ["Action", "Adventure"]}  # Match any
                    {"year": {"gte": 2020, "lte": 2023}}  # Range
                    {"status": "Currently Airing"}  # Exact match

        Returns:
            Qdrant Filter object, or None if empty filter
        """
        if not filter_dict:
            return None

        conditions: List[FieldCondition] = []

        for key, value in filter_dict.items():
            condition = FilterBuilder._build_field_condition(key, value)
            if condition:
                conditions.append(condition)

        if not conditions:
            return None

        return Filter(must=conditions)

    @staticmethod
    def _build_field_condition(key: str, value: Any) -> Optional[FieldCondition]:
        """Build a single field condition.

        Args:
            key: Field name (supports dot notation for nested fields)
            value: Condition value (can be primitive, list, or dict)

        Returns:
            FieldCondition or None if invalid
        """
        # Handle list values (match any)
        if isinstance(value, list):
            return FieldCondition(
                key=key,
                match=MatchAny(any=value),
            )

        # Handle range conditions (dict with gte, lte, gt, lt)
        if isinstance(value, dict):
            range_params = {}

            if "gte" in value:
                range_params["gte"] = value["gte"]
            if "lte" in value:
                range_params["lte"] = value["lte"]
            if "gt" in value:
                range_params["gt"] = value["gt"]
            if "lt" in value:
                range_params["lt"] = value["lt"]

            if range_params:
                return FieldCondition(
                    key=key,
                    range=Range(**range_params),
                )

            logger.warning(f"Invalid range condition for key '{key}': {value}")
            return None

        # Handle primitive values (exact match)
        return FieldCondition(
            key=key,
            match=MatchValue(value=value),
        )

    @staticmethod
    def build_multi_condition_filter(
        must_conditions: Optional[List[Dict[str, Any]]] = None,
        should_conditions: Optional[List[Dict[str, Any]]] = None,
        must_not_conditions: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Filter]:
        """Build a complex filter with AND/OR/NOT logic.

        Args:
            must_conditions: All conditions must match (AND)
            should_conditions: At least one must match (OR)
            must_not_conditions: None can match (NOT)

        Returns:
            Qdrant Filter with complex boolean logic
        """
        must_clauses = []
        should_clauses = []
        must_not_clauses = []

        if must_conditions:
            for cond_dict in must_conditions:
                filter_obj = FilterBuilder.build_filter(cond_dict)
                if filter_obj and filter_obj.must:
                    must_clauses.extend(filter_obj.must)

        if should_conditions:
            for cond_dict in should_conditions:
                filter_obj = FilterBuilder.build_filter(cond_dict)
                if filter_obj and filter_obj.must:
                    should_clauses.extend(filter_obj.must)

        if must_not_conditions:
            for cond_dict in must_not_conditions:
                filter_obj = FilterBuilder.build_filter(cond_dict)
                if filter_obj and filter_obj.must:
                    must_not_clauses.extend(filter_obj.must)

        if not (must_clauses or should_clauses or must_not_clauses):
            return None

        return Filter(
            must=must_clauses if must_clauses else None,
            should=should_clauses if should_clauses else None,
            must_not=must_not_clauses if must_not_clauses else None,
        )
```

**Usage in QdrantClient:**
```python
from .utils.filter_builder import FilterBuilder

class QdrantClient(VectorDBClient):
    def __init__(self, ...):
        self._filter_builder = FilterBuilder()

    async def search_single_vector(self, ..., filters: Optional[Dict] = None):
        qdrant_filter = self._filter_builder.build_filter(filters)
        # ... rest of implementation
```

---

## Phase 4: Async Factory Methods for Processors

### 4.1 TextProcessor Async Factory

**File:** `src/vector/processors/text_processor.py`

**Changes:**

```python
class TextProcessor:
    """Text embedding processor with async initialization."""

    def __init__(
        self,
        settings: Settings,
        field_mapper: "AnimeFieldMapper",  # NEW: Receive shared instance
    ):
        """Initialize processor WITHOUT loading models.

        Args:
            settings: Application settings
            field_mapper: Shared AnimeFieldMapper instance
        """
        self.settings = settings
        self._field_mapper = field_mapper  # Use injected instance

        # Models NOT initialized here - deferred to create()
        self.text_model: Optional[Any] = None
        self.embedding_size: Optional[int] = None

    @classmethod
    async def create(
        cls,
        settings: Settings,
        field_mapper: "AnimeFieldMapper",
    ) -> "TextProcessor":
        """Async factory method for TextProcessor.

        This method initializes the processor and loads embedding models
        asynchronously, preventing blocking during startup.

        Args:
            settings: Application settings
            field_mapper: Shared AnimeFieldMapper instance

        Returns:
            Fully initialized TextProcessor instance
        """
        processor = cls(settings, field_mapper)

        # Async model initialization
        await processor._initialize_models()

        return processor

    async def _initialize_models(self) -> None:
        """Initialize embedding models asynchronously."""
        # Model loading logic here (currently in __init__)
        # This allows for async model downloads, GPU checks, etc.

        logger.info(f"Loading text model: {self.settings.text_embedding_model}")

        # Example: Load FastEmbed model asynchronously
        self.text_model = await asyncio.to_thread(
            TextEmbedding,
            self.settings.text_embedding_model,
            cache_dir=self.settings.model_cache_dir,
        )

        self.embedding_size = self.text_model.dim

        logger.info(f"Text model loaded: {self.embedding_size}D embeddings")
```

### 4.2 VisionProcessor Async Factory

**File:** `src/vector/processors/vision_processor.py`

**Changes:**

```python
class VisionProcessor:
    """Vision embedding processor with async initialization."""

    def __init__(
        self,
        settings: Settings,
        field_mapper: "AnimeFieldMapper",  # NEW: Receive shared instance
    ):
        """Initialize processor WITHOUT loading models.

        Args:
            settings: Application settings
            field_mapper: Shared AnimeFieldMapper instance
        """
        self.settings = settings
        self._field_mapper = field_mapper  # Use injected instance

        # Models NOT initialized here
        self.vision_model: Optional[Any] = None
        self.vision_preprocess: Optional[Any] = None
        self.embedding_size: Optional[int] = None
        self.device: Optional[str] = None

    @classmethod
    async def create(
        cls,
        settings: Settings,
        field_mapper: "AnimeFieldMapper",
    ) -> "VisionProcessor":
        """Async factory method for VisionProcessor.

        This method initializes the processor and loads vision models
        asynchronously, preventing blocking during startup.

        Args:
            settings: Application settings
            field_mapper: Shared AnimeFieldMapper instance

        Returns:
            Fully initialized VisionProcessor instance
        """
        processor = cls(settings, field_mapper)

        # Async model initialization
        await processor._initialize_models()

        return processor

    async def _initialize_models(self) -> None:
        """Initialize vision models asynchronously."""
        logger.info(f"Loading vision model: {self.settings.image_embedding_model}")

        # Example: Load OpenCLIP model asynchronously
        model, _, preprocess = await asyncio.to_thread(
            open_clip.create_model_and_transforms,
            self.settings.image_embedding_model,
            pretrained=self.settings.image_embedding_pretrained,
        )

        self.vision_model = model
        self.vision_preprocess = preprocess
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vision_model.to(self.device)

        # Get embedding dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224).to(self.device)
            dummy_output = self.vision_model.encode_image(dummy_input)
            self.embedding_size = dummy_output.shape[1]

        logger.info(
            f"Vision model loaded: {self.embedding_size}D embeddings on {self.device}"
        )
```

### 4.3 Update Main Initialization

**File:** `src/main.py`

**Changes:**

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan with async model initialization."""

    # Create shared AnimeFieldMapper instance (singleton)
    field_mapper = AnimeFieldMapper()

    # Initialize processors with async factories
    text_processor = await TextProcessor.create(
        settings=settings,
        field_mapper=field_mapper,  # Shared instance
    )

    vision_processor = await VisionProcessor.create(
        settings=settings,
        field_mapper=field_mapper,  # Shared instance
    )

    # Create embedding manager with shared components
    embedding_manager = MultiVectorEmbeddingManager(
        text_processor=text_processor,
        vision_processor=vision_processor,
        settings=settings,
        field_mapper=field_mapper,  # Same shared instance
    )

    # Create QdrantClient (already async)
    qdrant_client = await QdrantClient.create(...)

    # Store in app state
    app.state.text_processor = text_processor
    app.state.vision_processor = vision_processor
    app.state.embedding_manager = embedding_manager
    app.state.qdrant_client = qdrant_client
    app.state.field_mapper = field_mapper

    yield

    # Cleanup (if needed)
```

---

## Phase 5: AnimeFieldMapper Consolidation

### 5.1 Remove Lazy Loading Pattern

**Current Pattern (Multiple instances):**
```python
# In TextProcessor
self._field_mapper: Optional["AnimeFieldMapper"] = None  # Lazy init

@property
def field_mapper(self) -> "AnimeFieldMapper":
    if self._field_mapper is None:
        from .anime_field_mapper import AnimeFieldMapper
        self._field_mapper = AnimeFieldMapper()
    return self._field_mapper
```

**New Pattern (Injected singleton):**
```python
# In TextProcessor.__init__
def __init__(self, settings: Settings, field_mapper: "AnimeFieldMapper"):
    self.settings = settings
    self._field_mapper = field_mapper  # Use injected instance directly
```

### 5.2 Update MultiVectorEmbeddingManager

**File:** `src/vector/processors/embedding_manager.py`

**Changes:**

```python
class MultiVectorEmbeddingManager:
    """Orchestrates multi-vector generation using shared components."""

    def __init__(
        self,
        text_processor: TextProcessor,
        vision_processor: VisionProcessor,
        settings: Settings,
        field_mapper: AnimeFieldMapper,  # NEW: Receive shared instance
    ):
        """Initialize manager with dependency injection.

        Args:
            text_processor: Text embedding processor (already initialized)
            vision_processor: Vision embedding processor (already initialized)
            settings: Application settings
            field_mapper: Shared AnimeFieldMapper instance
        """
        self.text_processor = text_processor
        self.vision_processor = vision_processor
        self.settings = settings
        self.field_mapper = field_mapper  # Use shared instance

        # Remove: self.field_mapper = AnimeFieldMapper()  # DELETE THIS LINE
```

### 5.3 Initialization Order

**Correct dependency injection flow:**

```python
# 1. Create singleton AnimeFieldMapper (no dependencies)
field_mapper = AnimeFieldMapper()

# 2. Create processors with shared field_mapper
text_processor = await TextProcessor.create(settings, field_mapper)
vision_processor = await VisionProcessor.create(settings, field_mapper)

# 3. Create manager with all dependencies
embedding_manager = MultiVectorEmbeddingManager(
    text_processor=text_processor,
    vision_processor=vision_processor,
    settings=settings,
    field_mapper=field_mapper,  # Same instance
)
```

**Benefits:**
- ✅ Single source of truth
- ✅ Reduced memory usage
- ✅ Clear dependency graph
- ✅ Easier to mock for testing

---

## Phase 6: Code Cleanup

### 6.1 Delete Commented Code

**Lines to delete from `src/vector/client/qdrant_client.py`:**

1. **Lines 707-813:** Commented `update_single_anime_vector()` method (107 lines)
2. **Lines 1021-1165:** Commented `update_batch_anime_vectors()` method (145 lines)

**Rationale:**
- These methods mixed vector generation with database operations
- The correct pattern is now established (separation of concerns)
- Git history preserves this code if ever needed for reference
- Removes 252 lines of dead code

**Command:**
```bash
# Before deleting, verify line numbers
grep -n "# async def update_single_anime_vector" src/vector/client/qdrant_client.py
grep -n "# async def update_batch_anime_vectors" src/vector/client/qdrant_client.py
```

### 6.2 Update Imports After Extraction

**New imports in `src/vector/client/qdrant_client.py`:**

```python
from .base import VectorDBClient
from .utils.id_generator import PointIDGenerator
from .utils.vector_validator import VectorValidator
from .utils.filter_builder import FilterBuilder
from ..utils.retry import retry_with_backoff
```

### 6.3 Update `__init__.py` Files

**File:** `src/vector/client/__init__.py`

```python
"""Vector database client module."""

from .base import VectorDBClient
from .qdrant_client import QdrantClient
from .utils.id_generator import PointIDGenerator
from .utils.vector_validator import VectorValidator
from .utils.filter_builder import FilterBuilder

__all__ = [
    "VectorDBClient",
    "QdrantClient",
    "PointIDGenerator",
    "VectorValidator",
    "FilterBuilder",
]
```

**File:** `src/vector/client/utils/__init__.py`

```python
"""Vector client utilities."""

from .id_generator import PointIDGenerator
from .vector_validator import VectorValidator
from .filter_builder import FilterBuilder

__all__ = [
    "PointIDGenerator",
    "VectorValidator",
    "FilterBuilder",
]
```

---

## Phase 7: Testing Strategy

### 7.1 New Tests Required

#### Test Abstract Base Class

**File:** `tests/vector/client/test_base_interface.py`

```python
"""Test VectorDBClient abstract interface."""

import pytest
from src.vector.client.base import VectorDBClient


class TestVectorDBClientInterface:
    """Test abstract base class contract."""

    def test_cannot_instantiate_abstract_class(self):
        """Verify VectorDBClient cannot be instantiated directly."""
        with pytest.raises(TypeError):
            VectorDBClient()

    def test_qdrant_client_implements_interface(self):
        """Verify QdrantClient implements VectorDBClient."""
        from src.vector.client import QdrantClient

        assert issubclass(QdrantClient, VectorDBClient)

    def test_all_abstract_methods_defined(self):
        """Verify all required abstract methods are defined."""
        from src.vector.client.base import VectorDBClient
        import inspect

        abstract_methods = {
            name for name, method in inspect.getmembers(VectorDBClient)
            if getattr(method, "__isabstractmethod__", False)
        }

        expected_methods = {
            "create_collection",
            "delete_collection",
            "collection_exists",
            "add_documents",
            "get_by_id",
            "update_single_vector",
            "update_batch_vectors",
            "search_single_vector",
            "search_multi_vector",
            "health_check",
            "get_stats",
        }

        assert abstract_methods == expected_methods
```

#### Test Extracted Utilities

**File:** `tests/vector/client/utils/test_id_generator.py`

```python
"""Test PointIDGenerator utility."""

import pytest
from src.vector.client.utils import PointIDGenerator


class TestPointIDGenerator:
    """Test ID generation functionality."""

    def test_generate_consistent_ids(self):
        """Verify same input produces same ID."""
        id1 = PointIDGenerator.generate("anime_123")
        id2 = PointIDGenerator.generate("anime_123")

        assert id1 == id2

    def test_generate_different_ids(self):
        """Verify different inputs produce different IDs."""
        id1 = PointIDGenerator.generate("anime_123")
        id2 = PointIDGenerator.generate("anime_456")

        assert id1 != id2

    def test_generate_fixed_length(self):
        """Verify IDs are always 32 characters (MD5 hex)."""
        id1 = PointIDGenerator.generate("short")
        id2 = PointIDGenerator.generate("very_long_identifier_string")

        assert len(id1) == 32
        assert len(id2) == 32

    def test_generate_batch(self):
        """Verify batch generation maintains order."""
        ids = ["anime_1", "anime_2", "anime_3"]
        results = PointIDGenerator.generate_batch(ids)

        assert len(results) == 3
        assert results[0] == PointIDGenerator.generate("anime_1")
        assert results[1] == PointIDGenerator.generate("anime_2")
```

**File:** `tests/vector/client/utils/test_vector_validator.py`

```python
"""Test VectorValidator utility."""

import pytest
from src.vector.client.utils import VectorValidator


class TestVectorValidator:
    """Test vector validation functionality."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return VectorValidator(
            allowed_vector_names=["title_vector", "genre_vector"],
            vector_dimensions={"title_vector": 1024, "genre_vector": 1024},
        )

    def test_valid_vector(self, validator):
        """Verify valid vector passes validation."""
        is_valid, error = validator.validate_vector_update(
            "title_vector",
            [0.1] * 1024,
        )

        assert is_valid
        assert error is None

    def test_invalid_vector_name(self, validator):
        """Verify invalid vector name fails."""
        is_valid, error = validator.validate_vector_update(
            "unknown_vector",
            [0.1] * 1024,
        )

        assert not is_valid
        assert "Invalid vector name" in error

    def test_invalid_dimension(self, validator):
        """Verify dimension mismatch fails."""
        is_valid, error = validator.validate_vector_update(
            "title_vector",
            [0.1] * 512,  # Wrong dimension
        )

        assert not is_valid
        assert "dimension mismatch" in error
```

**File:** `tests/vector/client/utils/test_filter_builder.py`

```python
"""Test FilterBuilder utility."""

import pytest
from src.vector.client.utils import FilterBuilder


class TestFilterBuilder:
    """Test filter construction functionality."""

    def test_exact_match_filter(self):
        """Verify exact match filter construction."""
        filter_dict = {"status": "Currently Airing"}
        result = FilterBuilder.build_filter(filter_dict)

        assert result is not None
        assert len(result.must) == 1

    def test_match_any_filter(self):
        """Verify match-any filter construction."""
        filter_dict = {"genre": ["Action", "Adventure"]}
        result = FilterBuilder.build_filter(filter_dict)

        assert result is not None
        assert len(result.must) == 1

    def test_range_filter(self):
        """Verify range filter construction."""
        filter_dict = {"year": {"gte": 2020, "lte": 2023}}
        result = FilterBuilder.build_filter(filter_dict)

        assert result is not None
        assert len(result.must) == 1

    def test_empty_filter(self):
        """Verify empty filter returns None."""
        result = FilterBuilder.build_filter({})
        assert result is None
```

### 7.2 Update Existing Tests

**Files to update:**
- `tests/vector/client/test_qdrant_client_integration.py` - Add import for base class
- `tests/vector/client/test_qdrant_client_retry.py` - Verify retry still works
- `tests/vector/processors/test_text_processor.py` - Update for async factory
- `tests/vector/processors/test_vision_processor.py` - Update for async factory

**Example update:**

```python
# Old pattern
text_processor = TextProcessor(settings)

# New pattern
field_mapper = AnimeFieldMapper()
text_processor = await TextProcessor.create(settings, field_mapper)
```

### 7.3 Integration Test for Full Flow

**File:** `tests/integration/test_complete_initialization.py`

```python
"""Integration test for complete initialization flow."""

import pytest
from src.config import get_settings
from src.vector.processors import TextProcessor, VisionProcessor
from src.vector.processors.anime_field_mapper import AnimeFieldMapper
from src.vector.processors.embedding_manager import MultiVectorEmbeddingManager
from src.vector.client import QdrantClient


@pytest.mark.asyncio
async def test_full_initialization_flow():
    """Verify all components initialize correctly with shared dependencies."""
    settings = get_settings()

    # Step 1: Create singleton AnimeFieldMapper
    field_mapper = AnimeFieldMapper()

    # Step 2: Create processors with async factories
    text_processor = await TextProcessor.create(settings, field_mapper)
    vision_processor = await VisionProcessor.create(settings, field_mapper)

    # Step 3: Create embedding manager
    embedding_manager = MultiVectorEmbeddingManager(
        text_processor=text_processor,
        vision_processor=vision_processor,
        settings=settings,
        field_mapper=field_mapper,
    )

    # Step 4: Create QdrantClient
    from qdrant_client import AsyncQdrantClient
    async_qdrant = AsyncQdrantClient(url=settings.qdrant_url)
    qdrant_client = await QdrantClient.create(
        settings=settings,
        async_qdrant_client=async_qdrant,
    )

    # Verify all components initialized
    assert text_processor.text_model is not None
    assert vision_processor.vision_model is not None
    assert embedding_manager.field_mapper is field_mapper
    assert text_processor._field_mapper is field_mapper
    assert vision_processor._field_mapper is field_mapper

    # Verify QdrantClient implements interface
    from src.vector.client.base import VectorDBClient
    assert isinstance(qdrant_client, VectorDBClient)
```

---

## Phase 8: Implementation Order

### Step-by-Step Migration Path

#### ✅ Step 1: Create Abstract Base Class (Non-Breaking)
1. Create `src/vector/client/base.py`
2. Define `VectorDBClient` abstract class
3. Write tests in `tests/vector/client/test_base_interface.py`
4. Run tests: `pytest tests/vector/client/test_base_interface.py`

**Status:** ✅ No breaking changes, new file only

---

#### ✅ Step 2: Extract Utilities (Non-Breaking)
1. Create `src/vector/client/utils/` directory
2. Create `id_generator.py` with `PointIDGenerator` class
3. Create `vector_validator.py` with `VectorValidator` class
4. Create `filter_builder.py` with `FilterBuilder` class
5. Create `__init__.py` in utils directory
6. Write unit tests for each utility
7. Run tests: `pytest tests/vector/client/utils/`

**Status:** ✅ No breaking changes, utilities exist alongside old code

---

#### ⚠️ Step 3: Update QdrantClient to Use Utilities (Breaking)
1. Update `QdrantClient` to inherit from `VectorDBClient`
2. Replace `_generate_point_id()` with `PointIDGenerator`
3. Replace `_validate_vector_update()` with `VectorValidator`
4. Replace `_build_filter()` with `FilterBuilder`
5. Update imports in `qdrant_client.py`
6. Update `__init__.py` to export new classes
7. Run all QdrantClient tests: `pytest tests/vector/client/test_qdrant_client*.py`

**Status:** ⚠️ BREAKING - Old private methods removed

**Rollback Strategy:** Keep old methods as deprecated for one version

---

#### ⚠️ Step 4: Add Async Factory Methods to Processors (Breaking)
1. Update `TextProcessor`:
   - Make `__init__` lightweight (no model loading)
   - Add `create()` async factory method
   - Add `_initialize_models()` async method
   - Update to accept `field_mapper` parameter
2. Update `VisionProcessor` (same pattern)
3. Write tests for async factories
4. Run processor tests: `pytest tests/vector/processors/`

**Status:** ⚠️ BREAKING - Old synchronous initialization removed

**Migration Path:**
```python
# Old code
processor = TextProcessor(settings)

# New code
field_mapper = AnimeFieldMapper()
processor = await TextProcessor.create(settings, field_mapper)
```

---

#### ⚠️ Step 5: Consolidate AnimeFieldMapper (Breaking)
1. Update `TextProcessor.__init__` to require `field_mapper` parameter
2. Update `VisionProcessor.__init__` to require `field_mapper` parameter
3. Update `MultiVectorEmbeddingManager.__init__` to require `field_mapper`
4. Remove lazy loading properties from processors
5. Update `src/main.py` to create singleton `field_mapper`
6. Update all scripts (`update_vectors.py`, `reindex_anime_database.py`)
7. Run all tests: `pytest tests/`

**Status:** ⚠️ BREAKING - All initialization code must be updated

**Migration Checklist:**
- [ ] `src/main.py` - FastAPI lifespan
- [ ] `scripts/update_vectors.py`
- [ ] `scripts/reindex_anime_database.py`
- [ ] Any other scripts using processors

---

#### ✅ Step 6: Delete Commented Code (Non-Breaking)
1. Delete lines 707-813 in `qdrant_client.py`
2. Delete lines 1021-1165 in `qdrant_client.py`
3. Run all tests to verify no references exist
4. Commit with clear message: "cleanup: remove commented-out legacy methods"

**Status:** ✅ No breaking changes, removing dead code

---

#### ✅ Step 7: Integration Testing (Verification)
1. Run full test suite: `pytest tests/`
2. Run type checking: `mypy --strict src/`
3. Test manual initialization flow in Python REPL
4. Verify scripts work: `python scripts/update_vectors.py --help`
5. Test FastAPI startup: `python -m src.main`

**Status:** ✅ Verification only

---

### Parallel vs Sequential Work

**Can be done in parallel:**
- ✅ Step 1 (Base class) + Step 2 (Utilities)
- ✅ Step 4 (Async factories) can start after Step 2

**Must be sequential:**
- Step 3 depends on Step 1 and Step 2
- Step 5 depends on Step 4
- Step 6 depends on Step 3
- Step 7 runs after all steps

---

## Phase 9: Future Monorepo Migration

### What Changes During Monorepo Migration?

**File Moves Only (No Logic Changes):**

```bash
# Base class moves to shared library
src/vector/client/base.py
  → libs/vector_db_interface/src/vector_db_interface/base.py

# QdrantClient moves to dedicated library
src/vector/client/qdrant_client.py
  → libs/qdrant_client/src/qdrant_client/client.py

src/vector/client/utils/
  → libs/qdrant_client/src/qdrant_client/utils/

# Processors move to vector processing library
src/vector/processors/
  → libs/vector_processing/src/vector_processing/
```

**Import Changes:**

```python
# Before monorepo
from src.vector.client import QdrantClient
from src.vector.client.base import VectorDBClient

# After monorepo
from qdrant_client import QdrantClient
from vector_db_interface import VectorDBClient
```

**That's it!** Because we've already:
- ✅ Created the base class
- ✅ Extracted utilities
- ✅ Established clear interfaces
- ✅ Removed coupling

The monorepo migration becomes a **file move operation** with import updates.

---

## Success Criteria

### ✅ Phase Complete When:

1. **Base Class:**
   - [ ] `VectorDBClient` abstract class exists in `src/vector/client/base.py`
   - [ ] All abstract methods defined with proper signatures
   - [ ] QdrantClient inherits from VectorDBClient
   - [ ] Tests verify interface contract

2. **Utilities Extracted:**
   - [ ] `PointIDGenerator` exists in `utils/id_generator.py`
   - [ ] `VectorValidator` exists in `utils/vector_validator.py`
   - [ ] `FilterBuilder` exists in `utils/filter_builder.py`
   - [ ] All utilities have unit tests
   - [ ] QdrantClient uses utilities instead of private methods

3. **Async Factories:**
   - [ ] `TextProcessor.create()` async factory implemented
   - [ ] `VisionProcessor.create()` async factory implemented
   - [ ] Model loading happens in async methods
   - [ ] `__init__` is lightweight (no blocking operations)
   - [ ] Tests verify async initialization

4. **AnimeFieldMapper Consolidated:**
   - [ ] Single `field_mapper` instance created in main.py
   - [ ] Passed to all processors and embedding manager
   - [ ] No lazy loading properties remain
   - [ ] All processors use shared instance

5. **Code Cleanup:**
   - [ ] Commented methods deleted (252 lines removed)
   - [ ] Imports updated throughout codebase
   - [ ] `__init__.py` files export new classes
   - [ ] No dead code remains

6. **Testing:**
   - [ ] All existing tests pass
   - [ ] New tests for base class
   - [ ] New tests for utilities
   - [ ] Integration test for complete initialization
   - [ ] Type checking passes: `mypy --strict src/`

7. **Documentation:**
   - [ ] This plan document exists
   - [ ] Docstrings updated in all modified files
   - [ ] Migration examples in this document
   - [ ] Future monorepo structure documented

---

## Risk Mitigation

### Breaking Change Strategy

**Approach:** Incremental migration with rollback capability

1. **Feature flags:** Consider adding `USE_NEW_ARCHITECTURE` flag
2. **Deprecation warnings:** Log warnings before removing old code
3. **Parallel paths:** Keep old code for 1 version, mark deprecated
4. **Comprehensive testing:** 100% test coverage for changed code

### Rollback Plan

If issues arise after Step 3+ (breaking changes):

1. **Git revert:** All changes in single commit for easy revert
2. **Feature toggle:** Disable new architecture via settings
3. **Database compatibility:** Ensure no database schema changes
4. **Documentation:** Clear rollback instructions in commit message

### Testing Checkpoints

**After each step:**
- ✅ Run full test suite
- ✅ Run type checking
- ✅ Manual smoke test
- ✅ Check logs for warnings/errors

**Do NOT proceed** if any checkpoint fails.

---

## Timeline Estimate

| Phase | Estimated Time | Complexity |
|-------|---------------|------------|
| Step 1: Base class | 2 hours | Low |
| Step 2: Extract utilities | 3 hours | Medium |
| Step 3: Update QdrantClient | 2 hours | Medium |
| Step 4: Async factories | 4 hours | High |
| Step 5: Consolidate field mapper | 3 hours | Medium |
| Step 6: Delete commented code | 1 hour | Low |
| Step 7: Integration testing | 2 hours | Medium |
| **Total** | **17 hours** | **~2-3 days** |

**Note:** Timeline assumes sequential work. With parallel execution of Steps 1+2, could reduce to ~1.5-2 days.

---

## Conclusion

This restructuring plan aligns the current codebase with the future monorepo architecture while improving code quality through:

- **Abstraction:** VectorDBClient enables database pluggability
- **Separation of Concerns:** Utilities extracted, processors simplified
- **Async-First:** All initialization is non-blocking
- **Dependency Injection:** Shared components, no hidden state
- **Testability:** Clear interfaces, mockable components
- **Maintainability:** Less code, clearer responsibilities

**Most importantly:** When Phase 4 of the monorepo migration arrives, the changes will be **file moves only**, not logic refactoring. This "sharpen the saw" investment pays dividends during the migration.

---

## Next Steps

1. **Review this plan** with the team
2. **Create GitHub issue** for tracking
3. **Start with Step 1** (base class) - lowest risk
4. **Create feature branch** for all changes
5. **Incremental PRs** for each step (easier review)
6. **Comprehensive testing** at each checkpoint

**Ready to proceed?** Let's start with Step 1! 🚀
