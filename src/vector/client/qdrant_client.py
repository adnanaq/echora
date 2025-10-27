"""Qdrant Vector Database Client for Anime Search

Provides high-performance vector search capabilities optimized for anime data
with advanced filtering, cross-platform ID lookups, and hybrid search.
"""

import asyncio
import hashlib
import logging
from typing import Any, TypeGuard

# fastembed import moved to _init_encoder method for lazy loading
from qdrant_client import QdrantClient as QdrantSDK
from qdrant_client.models import (  # Qdrant optimization models; Multi-vector search models
    BinaryQuantization,
    BinaryQuantizationConfig,
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    HnswConfigDiff,
    MatchAny,
    MatchValue,
    OptimizersConfig,
    OptimizersConfigDiff,
    PayloadSchemaType,
    PointStruct,
    PointVectors,
    Prefetch,
    ProductQuantization,
    QuantizationConfig,
    Range,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    VectorParams,
    WalConfigDiff,
)

from ...config import Settings
from ...models.anime import AnimeEntry
from ..processors.embedding_manager import MultiVectorEmbeddingManager

logger = logging.getLogger(__name__)


def is_float_vector(vector: Any) -> TypeGuard[list[float]]:
    """Type guard to check if vector is a List[float]."""
    return (
        isinstance(vector, list)
        and len(vector) > 0
        and all(isinstance(x, (int, float)) for x in vector)
    )


class QdrantClient:
    """Qdrant client wrapper optimized for anime search operations."""

    def __init__(
        self,
        url: str | None = None,
        collection_name: str | None = None,
        settings: Settings | None = None,
    ):
        """Initialize Qdrant client with FastEmbed and configuration.

        Args:
            url: Qdrant server URL (optional, uses settings if not provided)
            collection_name: Name of the anime collection (optional, uses settings if not provided)
            settings: Configuration settings instance (optional, will import default if not provided)
        """
        # Use provided settings or import default settings
        if settings is None:
            from ...config.settings import Settings

            settings = Settings()

        self.settings = settings
        self.url = url or settings.qdrant_url
        self.collection_name = collection_name or settings.qdrant_collection_name

        # Initialize Qdrant client with API key if provided
        if settings.qdrant_api_key:
            self.client = QdrantSDK(url=self.url, api_key=settings.qdrant_api_key)
        else:
            self.client = QdrantSDK(url=self.url)

        self._distance_metric = settings.qdrant_distance_metric

        # Initialize embedding manager
        self.embedding_manager = MultiVectorEmbeddingManager(settings)

        # Initialize processors
        self._init_processors()

        # Create collection if it doesn't exist
        self._initialize_collection()

    def _init_processors(self) -> None:
        """Initialize embedding processors."""
        try:
            # Import processors
            from ..processors.text_processor import TextProcessor
            from ..processors.vision_processor import VisionProcessor

            # Initialize text processor
            self.text_processor = TextProcessor(self.settings)

            # Initialize vision processor
            self.vision_processor = VisionProcessor(self.settings)

            # Update vector sizes based on modern models
            text_info = self.text_processor.get_model_info()
            vision_info = self.vision_processor.get_model_info()

            self._vector_size = text_info.get("embedding_size", 384)
            self._image_vector_size = vision_info.get("embedding_size", 512)

            logger.info(
                f"Initialized processors - Text: {text_info['model_name']} ({self._vector_size}), "
                f"Vision: {vision_info['model_name']} ({self._image_vector_size})"
            )

        except Exception as e:
            logger.error(f"Failed to initialize processors: {e}")
            raise

    def _initialize_collection(self) -> None:
        """Initialize and validate anime collection with 11-vector architecture and performance optimization."""
        try:
            # Check if collection exists and validate its configuration
            collections = self.client.get_collections().collections
            collection_exists = any(
                col.name == self.collection_name for col in collections
            )

            if not collection_exists:
                # Create collection with current vector architecture
                logger.info(f"Creating optimized collection: {self.collection_name}")
                vectors_config = self._create_multi_vector_config()

                # Validate vector configuration before creation
                self._validate_vector_config(vectors_config)

                # Add performance optimization configurations
                quantization_config = self._create_quantization_config()
                optimizers_config = self._create_optimized_optimizers_config()
                wal_config = self._create_wal_config()

                # Create collection with optimization
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=vectors_config,
                    quantization_config=quantization_config,
                    optimizers_config=optimizers_config,
                    wal_config=wal_config,
                )

                # Configure payload indexing for faster filtering
                if getattr(self.settings, "qdrant_enable_payload_indexing", True):
                    self._setup_payload_indexing()

                logger.info(
                    f"Successfully created collection with {len(vectors_config)} vectors"
                )
            else:
                # Validate existing collection compatibility
                if not self._validate_collection_compatibility():
                    logger.warning(
                        f"Collection {self.collection_name} exists but may have compatibility issues"
                    )
                    logger.info(
                        "Continuing with existing collection configuration for backward compatibility"
                    )
                else:
                    logger.info(
                        f"Collection {self.collection_name} validated successfully"
                    )

        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {e}")
            raise

    def _validate_collection_compatibility(self) -> bool:
        """Validate existing collection compatibility with current vector architecture."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            existing_vectors = collection_info.config.params.vectors

            # Check if collection has expected vector configurations
            expected_vectors = set(self.settings.vector_names.keys())

            if isinstance(existing_vectors, dict):
                existing_vector_names = set(existing_vectors.keys())

                # Check if we have the current semantic vectors
                has_current_vectors = expected_vectors.issubset(existing_vector_names)

                if has_current_vectors:
                    logger.info("Collection has complete vector configuration")
                    return True
                else:
                    logger.warning(
                        f"Collection missing expected vectors. Expected: {expected_vectors}, Found: {existing_vector_names}"
                    )
                    return False
            else:
                logger.warning(
                    "Collection uses single vector configuration, not compatible with multi-vector architecture"
                )
                return False

        except Exception as e:
            logger.error(f"Failed to validate collection compatibility: {e}")
            return False

    def _validate_vector_config(self, vectors_config: dict[str, VectorParams]) -> None:
        """Validate vector configuration before collection creation."""
        if not vectors_config:
            raise ValueError("Vector configuration is empty")

        expected_count = len(self.settings.vector_names)
        actual_count = len(vectors_config)

        if actual_count != expected_count:
            raise ValueError(
                f"Vector count mismatch: expected {expected_count}, got {actual_count}"
            )

        # Validate vector dimensions
        for vector_name, vector_params in vectors_config.items():
            expected_dim = self.settings.vector_names.get(vector_name)
            if expected_dim and vector_params.size != expected_dim:
                raise ValueError(
                    f"Vector {vector_name} dimension mismatch: expected {expected_dim}, got {vector_params.size}"
                )

        logger.info(
            f"Vector configuration validated: {actual_count} vectors with correct dimensions"
        )

    # Distance mapping constant
    _DISTANCE_MAPPING = {
        "cosine": Distance.COSINE,
        "euclid": Distance.EUCLID,
        "dot": Distance.DOT,
    }

    def _create_multi_vector_config(self) -> dict[str, VectorParams]:
        """Create 11-vector configuration with priority-based optimization."""
        distance = self._DISTANCE_MAPPING.get(self._distance_metric, Distance.COSINE)

        # Use new 11-vector architecture from settings
        vector_params = {}
        for vector_name, dimension in self.settings.vector_names.items():
            priority = self._get_vector_priority(vector_name)
            vector_params[vector_name] = VectorParams(
                size=dimension,
                distance=distance,
                hnsw_config=self._get_hnsw_config(priority),
                quantization_config=self._get_quantization_config(priority),
            )

        logger.info(
            f"Created 11-vector configuration with {len(vector_params)} vectors"
        )
        return vector_params

    # NEW: Priority-Based Configuration Methods for Million-Query Optimization

    def _get_quantization_config(self, priority: str) -> QuantizationConfig | None:
        """Get quantization config based on vector priority."""
        config = self.settings.quantization_config.get(priority, {})
        if config.get("type") == "scalar":
            scalar_config = ScalarQuantizationConfig(
                type=ScalarType.INT8, always_ram=config.get("always_ram", False)
            )
            return ScalarQuantization(scalar=scalar_config)
        elif config.get("type") == "binary":
            binary_config = BinaryQuantizationConfig(
                always_ram=config.get("always_ram", False)
            )
            return BinaryQuantization(binary=binary_config)
        return None

    def _get_hnsw_config(self, priority: str) -> HnswConfigDiff:
        """Get HNSW config based on vector priority."""
        config = self.settings.hnsw_config.get(priority, {})
        return HnswConfigDiff(
            ef_construct=config.get("ef_construct", 200), m=config.get("m", 48)
        )

    def _get_vector_priority(self, vector_name: str) -> str:
        """Determine priority level for vector."""
        for priority, vectors in self.settings.vector_priorities.items():
            if vector_name in vectors:
                return str(priority)
        return "medium"  # default

    def _create_optimized_optimizers_config(self) -> OptimizersConfigDiff | None:
        """Create optimized optimizers configuration for million-query scale."""
        try:
            return OptimizersConfigDiff(
                default_segment_number=4,
                indexing_threshold=20000,
                memmap_threshold=self.settings.memory_mapping_threshold_mb * 1024,
            )
        except Exception as e:
            logger.error(f"Failed to create optimized optimizers config: {e}")
            return None

    def _create_quantization_config(
        self,
    ) -> BinaryQuantization | ScalarQuantization | ProductQuantization | None:
        """Create quantization configuration for performance optimization."""
        if not getattr(self.settings, "qdrant_enable_quantization", False):
            return None

        quantization_type = getattr(self.settings, "qdrant_quantization_type", "scalar")
        always_ram = getattr(self.settings, "qdrant_quantization_always_ram", None)

        try:
            if quantization_type == "binary":
                binary_config = BinaryQuantizationConfig(always_ram=always_ram)
                logger.info("Enabling binary quantization for 40x speedup potential")
                return BinaryQuantization(binary=binary_config)
            elif quantization_type == "scalar":
                scalar_config = ScalarQuantizationConfig(
                    type=ScalarType.INT8,  # 8-bit quantization for good balance
                    always_ram=always_ram,
                )
                logger.info("Enabling scalar quantization for memory optimization")
                return ScalarQuantization(scalar=scalar_config)
            elif quantization_type == "product":
                from qdrant_client.models import (
                    CompressionRatio,
                    ProductQuantizationConfig,
                )

                product_config = ProductQuantizationConfig(
                    compression=CompressionRatio.X16
                )
                logger.info("Enabling product quantization for storage optimization")
                return ProductQuantization(product=product_config)
            else:
                logger.warning(f"Unknown quantization type: {quantization_type}")
                return None
        except Exception as e:
            logger.error(f"Failed to create quantization config: {e}")
            return None

    def _create_optimizers_config(self) -> OptimizersConfig | None:
        """Create optimizers configuration for indexing performance."""
        try:
            optimizer_params = {}

            # Task #116: Configure memory mapping threshold
            memory_threshold = getattr(
                self.settings, "qdrant_memory_mapping_threshold", None
            )
            if memory_threshold:
                optimizer_params["memmap_threshold"] = memory_threshold

            # Configure indexing threads if specified
            indexing_threads = getattr(
                self.settings, "qdrant_hnsw_max_indexing_threads", None
            )
            if indexing_threads:
                optimizer_params["indexing_threshold"] = 0  # Start indexing immediately

            if optimizer_params:
                logger.info(f"Applying optimizer configuration: {optimizer_params}")
                return OptimizersConfig(**optimizer_params)
            return None
        except Exception as e:
            logger.error(f"Failed to create optimizers config: {e}")
            return None

    def _create_wal_config(self) -> WalConfigDiff | None:
        """Create Write-Ahead Logging configuration."""
        enable_wal = getattr(self.settings, "qdrant_enable_wal", None)
        if enable_wal is not None:
            try:
                config = WalConfigDiff(wal_capacity_mb=32, wal_segments_ahead=0)
                logger.info(f"WAL configuration: enabled={enable_wal}")
                return config
            except Exception as e:
                logger.error(f"Failed to create WAL config: {e}")
        return None

    def _setup_payload_indexing(self) -> None:
        """Setup payload field indexing for faster filtering.

        Creates indexes only for searchable metadata fields while keeping
        operational data (like enrichment_metadata) non-indexed for storage efficiency.

        Indexed fields include:
        - Core searchable fields: id, title, type, status, episodes, rating, nsfw
        - Categorical fields: genres, tags, demographics, content_warnings
        - Temporal fields: anime_season, duration
        - Platform fields: sources
        - Statistics for numerical filtering: statistics, score

        Non-indexed operational data:
        - enrichment_metadata: Development/debugging data not needed for search
        """
        indexed_fields = getattr(self.settings, "qdrant_indexed_payload_fields", {})
        if not indexed_fields:
            logger.info("No payload indexing configured")
            return

        try:
            logger.info(
                f"Setting up payload indexing for {len(indexed_fields)} searchable fields with optimized types"
            )
            logger.info(
                "Indexed fields enable fast filtering on: core metadata, genres, temporal data, platform stats"
            )
            logger.info(
                "Non-indexed fields (enrichment_metadata) stored for debugging but don't impact search performance"
            )

            # Map string types to PayloadSchemaType enums
            type_mapping = {
                "keyword": PayloadSchemaType.KEYWORD,
                "integer": PayloadSchemaType.INTEGER,
                "float": PayloadSchemaType.FLOAT,
                "bool": PayloadSchemaType.BOOL,
                "text": PayloadSchemaType.TEXT,
                "geo": PayloadSchemaType.GEO,
                "datetime": PayloadSchemaType.DATETIME,
                "uuid": PayloadSchemaType.UUID,
            }

            for field_name, field_type in indexed_fields.items():
                # Get the appropriate schema type
                schema_type = type_mapping.get(
                    field_type.lower(), PayloadSchemaType.KEYWORD
                )

                # Create index for each field with its specific type
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=schema_type,
                )
                logger.debug(
                    f"✓ Created {field_type.upper()} index for field: {field_name}"
                )

            logger.info(
                f"Successfully indexed {len(indexed_fields)} searchable payload fields with optimized types"
            )
            logger.info(
                "Payload optimization complete: type-specific indexing enabled for better filtering performance"
            )

        except Exception as e:
            logger.warning(f"Failed to setup payload indexing: {e}")
            # Don't fail collection creation if indexing fails

    async def health_check(self) -> bool:
        """Check if Qdrant is healthy and reachable."""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            # Simple health check by getting collections
            await loop.run_in_executor(
                None, lambda: self.client.get_collections()
            )
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def get_stats(self) -> dict[str, Any]:
        """Get collection statistics."""
        try:
            loop = asyncio.get_event_loop()

            # Get collection info
            collection_info = await loop.run_in_executor(
                None, lambda: self.client.get_collection(self.collection_name)
            )

            # Count total points
            count_result = await loop.run_in_executor(
                None,
                lambda: self.client.count(
                    collection_name=self.collection_name, count_filter=None, exact=True
                ),
            )

            return {
                "collection_name": self.collection_name,
                "total_documents": count_result.count,
                "vector_size": self._vector_size,
                "distance_metric": "cosine",
                "status": collection_info.status,
                "optimizer_status": collection_info.optimizer_status,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "points_count": collection_info.points_count,
            }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}

    def _generate_point_id(self, anime_id: str) -> str:
        """Generate unique point ID from anime ID."""
        return hashlib.md5(anime_id.encode()).hexdigest()

    async def add_documents(
        self, documents: list[AnimeEntry], batch_size: int = 100
    ) -> bool:
        """Add anime documents to the collection using the 11-vector architecture.

        Args:
            documents: List of AnimeEntry objects
            batch_size: Number of documents to process per batch

        Returns:
            True if successful, False otherwise
        """
        try:
            total_docs = len(documents)
            logger.info(f"Adding {total_docs} documents in batches of {batch_size}")

            for i in range(0, total_docs, batch_size):
                batch_documents = documents[i : i + batch_size]

                # Process batch to get vectors and payloads
                processed_batch = await self.embedding_manager.process_anime_batch(
                    batch_documents
                )

                points = []
                for doc_data in processed_batch:
                    if doc_data["metadata"].get("processing_failed"):
                        logger.warning(
                            f"Skipping failed document: {doc_data['metadata'].get('anime_title')}"
                        )
                        continue

                    point_id = self._generate_point_id(doc_data["payload"]["id"])

                    point = PointStruct(
                        id=point_id,
                        vector=doc_data["vectors"],
                        payload=doc_data["payload"],
                    )
                    points.append(point)

                if points:
                    # Upsert batch to Qdrant
                    self.client.upsert(
                        collection_name=self.collection_name, points=points, wait=True
                    )
                    logger.info(
                        f"Uploaded batch {i // batch_size + 1}/{(total_docs - 1) // batch_size + 1} ({len(points)} points)"
                    )

            logger.info(f"Successfully added {total_docs} documents")
            return True

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False

    def _validate_vector_update(
        self,
        vector_name: str,
        vector_data: list[float],
    ) -> tuple[bool, str | None]:
        """Validate a vector update for correct name and dimensions.

        Args:
            vector_name: Name of the vector to validate
            vector_data: Vector embedding data to validate

        Returns:
            Tuple of (is_valid, error_message)
            - (True, None) if valid
            - (False, error_message) if invalid
        """
        # Check if vector name is valid
        expected_dim = self.settings.vector_names.get(vector_name)
        if expected_dim is None:
            return False, f"Invalid vector name: {vector_name}"

        # Check if data is a valid float vector
        if not is_float_vector(vector_data):
            return False, "Vector data is not a valid float vector"

        # Check dimension matches
        if len(vector_data) != expected_dim:
            return (
                False,
                f"Vector dimension mismatch: expected {expected_dim}, got {len(vector_data)}",
            )

        return True, None

    async def _update_single_vector(
        self,
        anime_id: str,
        vector_name: str,
        vector_data: list[float],
    ) -> bool:
        """Update a single named vector for an existing anime point (low-level internal method).

        This method updates ONLY the specified vector while keeping all other
        vectors unchanged. Useful for selective updates like weekly statistics
        refreshes without re-indexing the entire database.

        Args:
            anime_id: Anime ID (will be hashed to point ID)
            vector_name: Name of vector to update (e.g., "review_vector")
            vector_data: New vector embedding

        Returns:
            True if successful, False otherwise

        Note:
            This is a low-level internal method. For most use cases, prefer the
            high-level update_single_anime_vector() method that auto-generates vectors.
        """
        try:
            # Validate vector update
            is_valid, error_msg = self._validate_vector_update(vector_name, vector_data)
            if not is_valid:
                logger.error(f"Validation failed for {vector_name}: {error_msg}")
                return False

            point_id = self._generate_point_id(anime_id)

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: self.client.update_vectors(
                    collection_name=self.collection_name,
                    points=[
                        PointVectors(id=point_id, vector={vector_name: vector_data})
                    ],
                    wait=True,
                ),
            )

            logger.debug(f"Updated {vector_name} for anime {anime_id}")
            return True

        except Exception as e:
            logger.exception(
                f"Failed to update vector {vector_name} for {anime_id}: {e}"
            )
            return False

    async def update_single_anime_vector(
        self,
        anime_entry: AnimeEntry,
        vector_name: str,
    ) -> dict[str, Any]:
        """Update a single vector for an anime by auto-generating the embedding.

        This is a high-level convenience method that:
        1. Takes an AnimeEntry object
        2. Automatically generates the vector embedding
        3. Updates the vector in Qdrant

        Args:
            anime_entry: AnimeEntry object containing anime data
            vector_name: Name of vector to update (e.g., "title_vector", "genre_vector")

        Returns:
            Dictionary with update result:
                - success: Boolean indicating success/failure
                - anime_id: Anime ID that was updated
                - vector_name: Vector name that was updated
                - error: Error message (only if success=False)
                - generation_failed: Boolean indicating if vector generation failed

        Example:
            >>> anime = AnimeEntry(id="anime_123", title="One Piece", ...)
            >>> result = await client.update_single_anime_vector(
            ...     anime_entry=anime,
            ...     vector_name="title_vector"
            ... )
            >>> if result['success']:
            ...     print(f"Updated {result['vector_name']} for {result['anime_id']}")
            >>> else:
            ...     print(f"Failed: {result['error']}")
        """
        try:
            # Validate vector name first
            if vector_name not in self.settings.vector_names:
                return {
                    "success": False,
                    "anime_id": anime_entry.id,
                    "vector_name": vector_name,
                    "error": f"Invalid vector name: {vector_name}",
                    "generation_failed": False,
                }

            # Generate vector using embedding manager
            gen_results = await self.embedding_manager.process_anime_batch(
                [anime_entry]
            )

            if not gen_results or len(gen_results) == 0:
                return {
                    "success": False,
                    "anime_id": anime_entry.id,
                    "vector_name": vector_name,
                    "error": "Vector generation returned no results",
                    "generation_failed": True,
                }

            gen_result = gen_results[0]
            vectors = gen_result.get("vectors", {})

            # Check if requested vector was generated
            if vector_name not in vectors or not vectors[vector_name]:
                return {
                    "success": False,
                    "anime_id": anime_entry.id,
                    "vector_name": vector_name,
                    "error": f"Vector generation failed for {vector_name}",
                    "generation_failed": True,
                }

            vector_data = vectors[vector_name]

            # Update the vector in Qdrant using low-level method
            success = await self._update_single_vector(
                anime_id=anime_entry.id,
                vector_name=vector_name,
                vector_data=vector_data,
            )

            if success:
                return {
                    "success": True,
                    "anime_id": anime_entry.id,
                    "vector_name": vector_name,
                    "generation_failed": False,
                }
            else:
                return {
                    "success": False,
                    "anime_id": anime_entry.id,
                    "vector_name": vector_name,
                    "error": "Failed to update vector in Qdrant",
                    "generation_failed": False,
                }

        except Exception as e:
            logger.exception(
                f"Failed to update {vector_name} for anime {anime_entry.id}: {e}"
            )
            return {
                "success": False,
                "anime_id": anime_entry.id,
                "vector_name": vector_name,
                "error": str(e),
                "generation_failed": False,
            }

    async def _update_batch_vectors(
        self,
        updates: list[dict[str, Any]],
        dedup_policy: str = "last-wins",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> dict[str, Any]:
        """Update multiple vectors across multiple anime points in a single batch (low-level internal method).

        More efficient than individual updates when updating many points.
        Processes updates in batches for optimal performance.

        Args:
            updates: List of update dictionaries with keys:
                - anime_id: Anime ID to update
                - vector_name: Name of vector to update
                - vector_data: New vector embedding
            dedup_policy: How to handle duplicate (anime_id, vector_name) pairs:
                - "last-wins": Keep last occurrence (default)
                - "first-wins": Keep first occurrence
                - "fail": Raise error on duplicates
                - "warn": Keep last but log warning
            max_retries: Maximum number of retry attempts for transient failures (default: 3)
            retry_delay: Initial delay in seconds between retries, doubles each retry (default: 1.0)

        Returns:
            Dictionary with detailed results:
                - success: Total count of successful updates
                - failed: Total count of failed updates
                - results: List of per-update status dicts with keys:
                    - anime_id: Anime ID
                    - vector_name: Vector name
                    - success: Boolean indicating success/failure
                    - error: Error message (only if success=False)
                - duplicates_removed: Number of duplicates removed (if any)

        Raises:
            ValueError: If dedup_policy is "fail" and duplicates are found

        Note:
            This is a low-level internal method. For most use cases, prefer the
            high-level update_batch_anime_vectors() method that auto-generates vectors from
            AnimeEntry objects
        """
        try:
            if not updates:
                return {
                    "success": 0,
                    "failed": 0,
                    "results": [],
                    "duplicates_removed": 0,
                }

            # Check for duplicates and apply deduplication policy
            seen_keys: dict[tuple, int] = {}  # (anime_id, vector_name) -> first index
            duplicates: list[tuple] = []
            deduplicated_updates: list[dict[str, Any]] = []

            for idx, update in enumerate(updates):
                key = (update["anime_id"], update["vector_name"])

                if key in seen_keys:
                    duplicates.append(key)
                    if dedup_policy == "first-wins":
                        # Skip this duplicate, keep first occurrence
                        continue
                    elif dedup_policy == "last-wins":
                        # Remove previous occurrence, keep this one
                        # Mark for removal by finding it in deduplicated_updates
                        deduplicated_updates = [
                            u
                            for u in deduplicated_updates
                            if not (
                                u["anime_id"] == update["anime_id"]
                                and u["vector_name"] == update["vector_name"]
                            )
                        ]
                    elif dedup_policy == "warn":
                        # Same as last-wins but log warning
                        logger.warning(
                            f"Duplicate update for ({update['anime_id']}, {update['vector_name']}), "
                            f"using last occurrence"
                        )
                        deduplicated_updates = [
                            u
                            for u in deduplicated_updates
                            if not (
                                u["anime_id"] == update["anime_id"]
                                and u["vector_name"] == update["vector_name"]
                            )
                        ]
                    elif dedup_policy == "fail":
                        raise ValueError(
                            f"Duplicate update found for ({update['anime_id']}, {update['vector_name']}). "
                            f"Deduplication policy is 'fail'."
                        )

                seen_keys[key] = idx
                deduplicated_updates.append(update)

            duplicates_removed = len(updates) - len(deduplicated_updates)

            if duplicates and dedup_policy not in ["fail", "warn"]:
                logger.debug(
                    f"Removed {duplicates_removed} duplicate updates "
                    f"(policy: {dedup_policy})"
                )

            # Track detailed results for each update
            results: list[dict[str, Any]] = []

            # Group updates by anime_id to batch multiple vector updates per point
            grouped_updates: dict[str, dict[str, list[float]]] = {}
            # Map to track which updates belong to which anime
            update_mapping: dict[str, list[int]] = {}

            for idx, update in enumerate(deduplicated_updates):
                anime_id = update["anime_id"]
                vector_name = update["vector_name"]
                vector_data = update["vector_data"]

                # Validate update using shared helper
                is_valid, error_msg = self._validate_vector_update(
                    vector_name, vector_data
                )
                if not is_valid:
                    results.append(
                        {
                            "anime_id": anime_id,
                            "vector_name": vector_name,
                            "success": False,
                            "error": error_msg,
                        }
                    )
                    continue

                # Valid update - add to batch
                if anime_id not in grouped_updates:
                    grouped_updates[anime_id] = {}
                    update_mapping[anime_id] = []

                grouped_updates[anime_id][vector_name] = vector_data
                update_mapping[anime_id].append(idx)

            # Create PointVectors for batch update
            point_updates = []
            for anime_id, vectors_dict in grouped_updates.items():
                point_id = self._generate_point_id(anime_id)
                point_updates.append(PointVectors(id=point_id, vector=vectors_dict))

            # Only execute if we have valid updates
            if point_updates:
                # Execute batch update with retry logic for transient failures
                retry_count = 0
                last_error = None

                while retry_count <= max_retries:
                    try:
                        loop = asyncio.get_running_loop()
                        await loop.run_in_executor(
                            None,
                            lambda: self.client.update_vectors(
                                collection_name=self.collection_name,
                                points=point_updates,
                                wait=True,
                            ),
                        )

                        # Success - mark all grouped updates as successful
                        for anime_id in grouped_updates.keys():
                            for vector_name in grouped_updates[anime_id].keys():
                                results.append(
                                    {
                                        "anime_id": anime_id,
                                        "vector_name": vector_name,
                                        "success": True,
                                    }
                                )

                        # Break out of retry loop on success
                        break

                    except Exception as e:
                        last_error = e
                        retry_count += 1

                        # Check if this is a transient error worth retrying
                        error_str = str(e).lower()
                        is_transient = any(
                            keyword in error_str
                            for keyword in [
                                "timeout",
                                "connection",
                                "network",
                                "temporary",
                                "unavailable",
                            ]
                        )

                        if is_transient and retry_count <= max_retries:
                            # Exponential backoff
                            delay = retry_delay * (2 ** (retry_count - 1))
                            logger.warning(
                                f"Transient error on batch update (attempt {retry_count}/{max_retries}): {e}. "
                                f"Retrying in {delay}s..."
                            )
                            await asyncio.sleep(delay)
                        else:
                            # Non-transient error or max retries exceeded
                            if retry_count > max_retries:
                                logger.error(
                                    f"Max retries ({max_retries}) exceeded for batch update. "
                                    f"Last error: {last_error}"
                                )
                            else:
                                logger.error(
                                    f"Non-transient error on batch update: {e}"
                                )

                            # Mark all as failed
                            for anime_id in grouped_updates.keys():
                                for vector_name in grouped_updates[anime_id].keys():
                                    results.append(
                                        {
                                            "anime_id": anime_id,
                                            "vector_name": vector_name,
                                            "success": False,
                                            "error": f"Update failed after {retry_count} attempts: {str(last_error)}",
                                        }
                                    )
                            break

            logger.info(
                f"Batch updated {len(point_updates)} points with {len(deduplicated_updates)} vector updates"
            )

            # Calculate summary counts
            success_count = sum(1 for r in results if r["success"])
            failed_count = len(results) - success_count

            return {
                "success": success_count,
                "failed": failed_count,
                "results": results,
                "duplicates_removed": duplicates_removed,
            }

        except ValueError as e:
            # Deduplication policy violation
            logger.error(f"Deduplication policy error: {e}")
            raise
        except Exception as e:
            logger.exception("Failed to batch update vectors")
            # Return failure for all updates
            results = [
                {
                    "anime_id": update.get("anime_id", "unknown"),
                    "vector_name": update.get("vector_name", "unknown"),
                    "success": False,
                    "error": f"Batch update failed: {str(e)}",
                }
                for update in updates
            ]
            return {
                "success": 0,
                "failed": len(updates),
                "results": results,
                "duplicates_removed": 0,
            }

    async def update_batch_anime_vectors(
        self,
        anime_entries: list[AnimeEntry],
        vector_names: list[str] | None = None,
        batch_size: int = 100,
        progress_callback: Any | None = None,
    ) -> dict[str, Any]:
        """Generate and update vectors for a batch of anime entries with automatic batching.

        This is a high-level method that:
        1. Generates vectors from AnimeEntry objects using the embedding manager
        2. Batches processing for memory efficiency
        3. Updates Qdrant with the generated vectors
        4. Provides progress callbacks for monitoring

        Args:
            anime_entries: List of AnimeEntry objects to process
            vector_names: Optional list of specific vectors to update.
                         If None, generates and updates all 11 vectors.
                         Valid names: title_vector, character_vector, genre_vector,
                         staff_vector, temporal_vector, streaming_vector, related_vector,
                         franchise_vector, episode_vector, image_vector, character_image_vector
            batch_size: Number of anime to process per batch (default: 100).
                       Smaller values use less memory, larger values are faster.
            progress_callback: Optional callback function(current, total, batch_result)
                              called after each batch completes. Useful for progress logging.

        Returns:
            Dictionary with comprehensive statistics:
                - total_anime: Total number of anime processed
                - total_requested_updates: Total updates attempted (anime × vectors)
                - successful_updates: Number of successful vector updates
                - failed_updates: Number of failed vector updates
                - generation_failures: Number of vectors that failed to generate
                - results: List of per-update results with anime_id, vector_name, success, error
                - generation_failures_detail: List of generation failures with details

        Example:
            >>> # Update all vectors for a batch of anime
            >>> result = await client.update_batch_anime_vectors(
            ...     anime_entries=[anime1, anime2],
            ...     vector_names=["title_vector", "genre_vector"]
            ... )
            >>> print(f"Success: {result['successful_updates']}")

            >>> # With progress callback for CLI
            >>> def log_progress(current, total, batch_result):
            ...     print(f"Processed {current}/{total} anime")
            >>> result = await client.update_batch_anime_vectors(
            ...     anime_entries=large_list,
            ...     progress_callback=log_progress
            ... )
        """
        try:
            if not anime_entries:
                return {
                    "total_anime": 0,
                    "total_requested_updates": 0,
                    "successful_updates": 0,
                    "failed_updates": 0,
                    "generation_failures": 0,
                    "results": [],
                    "generation_failures_detail": [],
                }

            total_anime = len(anime_entries)
            all_batch_results: list[dict[str, Any]] = []
            all_generation_failures: list[dict[str, Any]] = []

            # Process in batches for memory efficiency
            for batch_start in range(0, total_anime, batch_size):
                batch_end = min(batch_start + batch_size, total_anime)
                batch = anime_entries[batch_start:batch_end]

                logger.debug(
                    f"Processing batch {batch_start // batch_size + 1}: "
                    f"anime {batch_start + 1}-{batch_end}/{total_anime}"
                )

                # Generate vectors for this batch using embedding manager
                gen_results = await self.embedding_manager.process_anime_batch(batch)

                # Prepare updates and track generation failures
                batch_updates: list[dict[str, Any]] = []

                for i, anime_entry in enumerate(batch):
                    gen_result = gen_results[i]
                    vectors = gen_result.get("vectors", {})

                    # Filter to requested vectors if specified
                    if vector_names:
                        requested_vectors = {
                            k: v for k, v in vectors.items() if k in vector_names
                        }
                        # Track which requested vectors failed to generate
                        for requested_vec in vector_names:
                            if (
                                requested_vec not in vectors
                                or not vectors[requested_vec]
                            ):
                                all_generation_failures.append(
                                    {
                                        "anime_id": anime_entry.id,
                                        "vector_name": requested_vec,
                                        "error": "Vector generation failed or returned None",
                                    }
                                )
                    else:
                        requested_vectors = vectors

                    # Add valid vectors to batch updates
                    for vector_name, vector_data in requested_vectors.items():
                        if vector_data and len(vector_data) > 0:
                            batch_updates.append(
                                {
                                    "anime_id": anime_entry.id,
                                    "vector_name": vector_name,
                                    "vector_data": vector_data,
                                }
                            )

                # Update in Qdrant if we have valid updates
                if batch_updates:
                    batch_result = await self._update_batch_vectors(batch_updates)
                else:
                    batch_result = {"success": 0, "failed": 0, "results": []}
                    logger.warning(
                        f"Batch {batch_start // batch_size + 1} had no valid updates"
                    )

                all_batch_results.append(batch_result)

                # Call progress callback if provided
                if progress_callback:
                    progress_callback(batch_end, total_anime, batch_result)

            # Aggregate all batch results
            num_vectors = len(vector_names) if vector_names else 11
            return self._aggregate_batch_results(
                all_batch_results, all_generation_failures, total_anime, num_vectors
            )

        except Exception as e:
            logger.exception(f"Failed to update anime vectors: {e}")
            raise

    def _aggregate_batch_results(
        self,
        batch_results: list[dict[str, Any]],
        generation_failures: list[dict[str, Any]],
        total_anime: int,
        num_vectors: int,
    ) -> dict[str, Any]:
        """Aggregate results from multiple batches.

        Args:
            batch_results: List of batch update results
            generation_failures: List of generation failure details
            total_anime: Total number of anime processed
            num_vectors: Number of vectors requested per anime

        Returns:
            Aggregated statistics dictionary
        """
        try:
            total_successful = sum(r["success"] for r in batch_results)
            total_failed = sum(r["failed"] for r in batch_results)
            combined_results: list[dict[str, Any]] = []

            for batch_result in batch_results:
                combined_results.extend(batch_result.get("results", []))

            return {
                "total_anime": total_anime,
                "total_requested_updates": total_anime * num_vectors,
                "successful_updates": total_successful,
                "failed_updates": total_failed,
                "generation_failures": len(generation_failures),
                "results": combined_results,
                "generation_failures_detail": generation_failures,
            }

        except Exception as e:
            logger.error(f"Failed to aggregate batch results: {e}")
            return {
                "total_anime": total_anime,
                "total_requested_updates": total_anime * num_vectors,
                "successful_updates": 0,
                "failed_updates": 0,
                "generation_failures": len(generation_failures),
                "results": [],
                "generation_failures_detail": generation_failures,
            }

    def _build_filter(self, filters: dict[str, Any]) -> Filter | None:
        """Build Qdrant filter from filter dictionary.

        Args:
            filters: Dictionary with filter conditions

        Returns:
            Qdrant Filter object
        """
        if not filters:
            return None

        conditions = []

        for key, value in filters.items():
            # Skip None values and empty collections
            if value is None:
                continue
            if isinstance(value, (list, tuple)) and len(value) == 0:
                continue
            if isinstance(value, dict) and len(value) == 0:
                continue

            if isinstance(value, dict):
                # Range filter
                if "gte" in value or "lte" in value or "gt" in value or "lt" in value:
                    conditions.append(FieldCondition(key=key, range=Range(**value)))
                # Match any filter
                elif "any" in value:
                    any_values = value["any"]
                    # Skip empty any values
                    if any_values and len(any_values) > 0:
                        conditions.append(
                            FieldCondition(key=key, match=MatchAny(any=any_values))
                        )
            elif isinstance(value, list):
                # Match any from list - only add if list is not empty
                if len(value) > 0:
                    conditions.append(
                        FieldCondition(key=key, match=MatchAny(any=value))
                    )
            else:
                # Exact match - only add if value is not None and is a valid type
                if value is not None and isinstance(value, (str, int, bool)):
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )

        return Filter(must=conditions) if conditions else None  # type: ignore[arg-type]

    async def get_by_id(self, anime_id: str) -> dict[str, Any] | None:
        """Get anime by ID.

        Args:
            anime_id: The anime ID to retrieve

        Returns:
            Anime data dictionary or None if not found
        """
        try:
            loop = asyncio.get_event_loop()
            point_id = self._generate_point_id(anime_id)

            # Retrieve point by ID
            points = await loop.run_in_executor(
                None,
                lambda: self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=[point_id],
                    with_payload=True,
                    with_vectors=False,
                ),
            )

            if points:
                return dict(points[0].payload) if points[0].payload else {}
            return None

        except Exception as e:
            logger.error(f"Failed to get anime by ID {anime_id}: {e}")
            return None

    async def get_point(self, point_id: str) -> dict[str, Any] | None:
        """Get point by Qdrant point ID including vectors and payload.

        Args:
            point_id: The Qdrant point ID to retrieve

        Returns:
            Point data dictionary with vectors and payload or None if not found
        """
        try:
            loop = asyncio.get_event_loop()

            # Retrieve point by ID with vectors
            points = await loop.run_in_executor(
                None,
                lambda: self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=[point_id],
                    with_vectors=True,
                    with_payload=True,
                ),
            )

            if points:
                point = points[0]
                return {
                    "id": str(point.id),
                    "vector": point.vector,
                    "payload": dict(point.payload) if point.payload else {},
                }
            return None

        except Exception as e:
            logger.error(f"Failed to get point by ID {point_id}: {e}")
            return None

    async def clear_index(self) -> bool:
        """Clear all points from the collection (for fresh re-indexing)."""
        try:
            # Delete and recreate collection for clean state
            delete_success = await self.delete_collection()
            if not delete_success:
                return False

            create_success = await self.create_collection()
            if not create_success:
                return False

            logger.info(f"Cleared and recreated collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear index: {e}")
            return False

    async def delete_collection(self) -> bool:
        """Delete the anime collection (for testing/reset)."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, lambda: self.client.delete_collection(self.collection_name)
            )
            logger.info(f"Deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False

    async def create_collection(self) -> bool:
        """Create the anime collection."""
        try:
            self._initialize_collection()
            return True
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False

    async def search_single_vector(
        self,
        vector_name: str,
        vector_data: list[float],
        limit: int = 10,
        filters: Filter | None = None,
    ) -> list[dict[str, Any]]:
        """Search a single vector with raw similarity scores.

        Args:
            vector_name: Name of the vector to search (e.g., "title_vector")
            vector_data: The query vector (list of floats)
            limit: Maximum number of results to return
            filters: Optional Qdrant filter conditions

        Returns:
            List of search results with raw similarity scores
        """
        try:
            # Direct vector search with raw similarity scores
            response = self.client.search(
                collection_name=self.collection_name,
                query_vector=(vector_name, vector_data),
                limit=limit,
                with_payload=True,
                with_vectors=False,
                query_filter=filters,
            )

            # Convert response to our format with raw similarity scores
            results = []
            for point in response:
                payload = point.payload if point.payload else {}
                result = {
                    "id": str(point.id),
                    "anime_id": str(point.id),
                    "_id": str(point.id),
                    **payload,
                    # Vector similarity score from Qdrant search
                    "similarity_score": point.score,
                }
                results.append(result)

            logger.info(
                f"Single vector search ({vector_name}) returned {len(results)} results with similarity scores"
            )
            return results

        except Exception as e:
            logger.error(f"Single vector search failed: {e}")
            raise

    async def search_multi_vector(
        self,
        vector_queries: list[dict[str, Any]],
        limit: int = 10,
        fusion_method: str = "rrf",
        filters: Filter | None = None,
    ) -> list[dict[str, Any]]:
        """Search across multiple vectors using Qdrant's native multi-vector API.

        Args:
            vector_queries: List of vector query dicts with keys:
                - vector_name: Name of the vector to search (e.g., "title_vector")
                - vector_data: The query vector (list of floats)
                - weight: Optional weight for fusion (default: 1.0)
            limit: Maximum number of results to return
            fusion_method: Fusion algorithm - "rrf" or "dbsf"
            filters: Optional Qdrant filter conditions

        Returns:
            List of search results with fusion scores
        """
        try:
            if not vector_queries:
                raise ValueError("vector_queries cannot be empty")

            # Create prefetch queries for each vector
            prefetch_queries = []
            for query_config in vector_queries:
                vector_name = query_config["vector_name"]
                vector_data = query_config["vector_data"]

                # Create a Prefetch query for this vector
                prefetch_query = Prefetch(
                    using=vector_name,
                    query=vector_data,
                    limit=limit * 2,  # Get more results for better fusion
                    filter=filters,
                )
                prefetch_queries.append(prefetch_query)

            # Determine fusion method
            if fusion_method.lower() == "rrf":
                fusion = Fusion.RRF
            elif fusion_method.lower() == "dbsf":
                fusion = Fusion.DBSF
            else:
                logger.warning(f"Unknown fusion method {fusion_method}, using RRF")
                fusion = Fusion.RRF

            # Create the fusion query
            fusion_query = FusionQuery(fusion=fusion)

            # Execute the multi-vector search using query_points
            response = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=prefetch_queries,
                query=fusion_query,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

            # Convert response to our format
            results = []
            for point in response.points:
                payload = point.payload if point.payload else {}
                result = {
                    "id": str(point.id),
                    "anime_id": str(point.id),
                    "_id": str(point.id),
                    **payload,
                    # Vector similarity score from Qdrant search
                    "similarity_score": point.score,
                }
                results.append(result)

            logger.info(
                f"Multi-vector search returned {len(results)} results using {fusion_method.upper()}"
            )
            return results

        except Exception as e:
            logger.error(f"Multi-vector search failed: {e}")
            raise

    async def search_text_comprehensive(
        self,
        query: str,
        limit: int = 10,
        fusion_method: str = "rrf",
        filters: Filter | None = None,
    ) -> list[dict[str, Any]]:
        """Search across all 12 text vectors using native Qdrant fusion.

        Args:
            query: Text search query
            limit: Maximum number of results
            fusion_method: Fusion algorithm - "rrf" or "dbsf"
            filters: Optional Qdrant filter conditions

        Returns:
            List of search results with comprehensive text similarity scores
        """
        try:
            # Generate text embedding once
            query_embedding = self.embedding_manager.text_processor.encode_text(query)
            if query_embedding is None:
                logger.warning(
                    "Failed to create embedding for comprehensive text search"
                )
                return []

            # All 11 text vectors for comprehensive search
            text_vector_names = [
                "title_vector",
                "character_vector",
                "genre_vector",
                "staff_vector",
                "temporal_vector",
                "streaming_vector",
                "related_vector",
                "franchise_vector",
                "episode_vector",
            ]

            # Create vector queries for all text vectors
            vector_queries = []
            for vector_name in text_vector_names:
                vector_queries.append(
                    {"vector_name": vector_name, "vector_data": query_embedding}
                )

            # Use native multi-vector search
            results = await self.search_multi_vector(
                vector_queries=vector_queries,
                limit=limit,
                fusion_method=fusion_method,
                filters=filters,
            )

            logger.info(
                f"Comprehensive text search returned {len(results)} results across {len(text_vector_names)} vectors"
            )
            return results

        except Exception as e:
            logger.error(f"Comprehensive text search failed: {e}")
            return []

    async def search_visual_comprehensive(
        self,
        image_data: str,
        limit: int = 10,
        fusion_method: str = "rrf",
        filters: Filter | None = None,
    ) -> list[dict[str, Any]]:
        """Search across both image vectors using native Qdrant fusion.

        Args:
            image_data: Base64 encoded image data
            limit: Maximum number of results
            fusion_method: Fusion algorithm - "rrf" or "dbsf"
            filters: Optional Qdrant filter conditions

        Returns:
            List of search results with comprehensive visual similarity scores
        """
        try:
            # Generate image embedding once
            image_embedding = self.embedding_manager.vision_processor.encode_image(
                image_data
            )
            if image_embedding is None:
                logger.error(
                    "Failed to create image embedding for comprehensive visual search"
                )
                return []

            # Both image vectors for comprehensive visual search
            image_vector_names = ["image_vector", "character_image_vector"]

            # Create vector queries for both image vectors
            vector_queries = []
            for vector_name in image_vector_names:
                vector_queries.append(
                    {"vector_name": vector_name, "vector_data": image_embedding}
                )

            # Use native multi-vector search
            results = await self.search_multi_vector(
                vector_queries=vector_queries,
                limit=limit,
                fusion_method=fusion_method,
                filters=filters,
            )

            logger.info(
                f"Comprehensive visual search returned {len(results)} results across {len(image_vector_names)} vectors"
            )
            return results

        except Exception as e:
            logger.error(f"Comprehensive visual search failed: {e}")
            return []

    async def search_complete(
        self,
        query: str,
        image_data: str | None = None,
        limit: int = 10,
        fusion_method: str = "rrf",
        filters: Filter | None = None,
    ) -> list[dict[str, Any]]:
        """Search across all 11 vectors (9 text + 2 image) using native Qdrant fusion.

        Args:
            query: Text search query
            image_data: Optional base64 encoded image data
            limit: Maximum number of results
            fusion_method: Fusion algorithm - "rrf" or "dbsf"
            filters: Optional Qdrant filter conditions

        Returns:
            List of search results with complete multi-modal similarity scores
        """
        try:
            vector_queries = []

            # Generate text embedding for all 12 text vectors
            query_embedding = self.embedding_manager.text_processor.encode_text(query)
            if query_embedding is None:
                logger.warning("Failed to create text embedding for complete search")
            else:
                # All 11 text vectors
                text_vector_names = [
                    "title_vector",
                    "character_vector",
                    "genre_vector",
                    "staff_vector",
                    "temporal_vector",
                    "streaming_vector",
                    "related_vector",
                    "franchise_vector",
                    "episode_vector",
                ]

                for vector_name in text_vector_names:
                    vector_queries.append(
                        {"vector_name": vector_name, "vector_data": query_embedding}
                    )

            # Add image vectors if image provided
            if image_data:
                image_embedding = self.embedding_manager.vision_processor.encode_image(
                    image_data
                )
                if image_embedding is None:
                    logger.warning(
                        "Failed to create image embedding for complete search"
                    )
                else:
                    # Both image vectors
                    image_vector_names = ["image_vector", "character_image_vector"]

                    for vector_name in image_vector_names:
                        vector_queries.append(
                            {"vector_name": vector_name, "vector_data": image_embedding}
                        )

            if not vector_queries:
                logger.error("No valid embeddings generated for complete search")
                return []

            # Use native multi-vector search across all vectors
            results = await self.search_multi_vector(
                vector_queries=vector_queries,
                limit=limit,
                fusion_method=fusion_method,
                filters=filters,
            )

            logger.info(
                f"Complete search returned {len(results)} results across {len(vector_queries)} vectors"
            )
            return results

        except Exception as e:
            logger.error(f"Complete search failed: {e}")
            return []

    async def search_characters(
        self,
        query: str,
        image_data: str | None = None,
        limit: int = 10,
        fusion_method: str = "rrf",
        filters: Filter | None = None,
    ) -> list[dict[str, Any]]:
        """Search specifically for character-related content using character vectors.

        Args:
            query: Text search query focused on characters
            image_data: Optional base64 encoded character image data
            limit: Maximum number of results
            fusion_method: Fusion algorithm - "rrf" or "dbsf"
            filters: Optional Qdrant filter conditions

        Returns:
            List of search results focused on character similarity across text and image
        """
        try:
            vector_queries = []

            # Generate text embedding for character_vector
            query_embedding = self.embedding_manager.text_processor.encode_text(query)
            if query_embedding is None:
                logger.warning("Failed to create text embedding for character search")
            else:
                vector_queries.append(
                    {"vector_name": "character_vector", "vector_data": query_embedding}
                )

            # Generate image embedding for character_image_vector if provided
            if image_data:
                image_embedding = self.embedding_manager.vision_processor.encode_image(
                    image_data
                )
                if image_embedding is None:
                    logger.warning(
                        "Failed to create image embedding for character search"
                    )
                else:
                    vector_queries.append(
                        {
                            "vector_name": "character_image_vector",
                            "vector_data": image_embedding,
                        }
                    )

            if not vector_queries:
                logger.error("No valid embeddings generated for character search")
                return []

            results = await self.search_multi_vector(
                vector_queries=vector_queries,
                limit=limit,
                fusion_method=fusion_method,
                filters=filters,
            )

            search_type = "text+image" if image_data else "text-only"
            logger.info(
                f"Character search ({search_type}) returned {len(results)} results across {len(vector_queries)} vectors"
            )
            return results

        except Exception as e:
            logger.error(f"Character search failed: {e}")
            return []
