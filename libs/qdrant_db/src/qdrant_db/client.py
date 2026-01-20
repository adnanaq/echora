"""Qdrant Vector Database Client for Anime Search

Provides high-performance vector search capabilities optimized for anime data
with advanced filtering, cross-platform ID lookups, and hybrid search.
"""

import hashlib
import logging
from typing import Any, TypeGuard, cast

from common.config import Settings

# fastembed import moved to _init_encoder method for lazy loading
from qdrant_client import AsyncQdrantClient
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
    MultiVectorComparator,
    MultiVectorConfig,
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
from vector_db_interface import VectorDBClient, VectorDocument

from qdrant_db.utils import retry_with_backoff

logger = logging.getLogger(__name__)


def is_float_vector(vector: Any) -> TypeGuard[list[float]]:
    """Type guard to check if vector is a List[float].

    Args:
        vector: Value to check

    Returns:
        True if vector is a non-empty list of numeric values, False otherwise
    """
    return (
        isinstance(vector, list)
        and len(vector) > 0
        and all(isinstance(x, int | float) for x in vector)
    )


class QdrantClient(VectorDBClient):
    """Qdrant client wrapper optimized for anime search operations."""

    def __init__(
        self,
        settings: Settings,
        async_qdrant_client: AsyncQdrantClient,
        url: str | None = None,
        collection_name: str | None = None,
    ):
        """Initialize Qdrant client with injected dependencies and configuration.

        Args:
            settings: Configuration settings instance.
            async_qdrant_client: An initialized AsyncQdrantClient instance from qdrant-client library.
            url: Qdrant server URL (optional, uses settings if not provided)
            collection_name: Name of the anime collection (optional, uses settings if not provided)
        """
        self.settings = settings
        self.url = url or settings.qdrant_url
        self.collection_name = collection_name or settings.qdrant_collection_name

        self.client = async_qdrant_client

        self._distance_metric = settings.qdrant_distance_metric

        # Initialize vector sizes based on settings
        self._vector_size = settings.text_vector_size
        self._image_vector_size = settings.image_vector_size

    @property
    def vector_size(self) -> int:
        """Get the text vector embedding size."""
        return self._vector_size

    @property
    def image_vector_size(self) -> int:
        """Get the image vector embedding size."""
        return self._image_vector_size

    @property
    def distance_metric(self) -> str:
        """Get the distance metric used for vector similarity."""
        return self._distance_metric

    @classmethod
    async def create(
        cls,
        settings: Settings,
        async_qdrant_client: AsyncQdrantClient,
        url: str | None = None,
        collection_name: str | None = None,
    ) -> "QdrantClient":
        """Create and initialize a QdrantClient instance.

        Factory method that creates a QdrantClient instance and initializes the
        collection. This is the recommended way to instantiate QdrantClient as it
        ensures the collection is properly initialized before use.

        Args:
            settings: Configuration settings instance
            async_qdrant_client: An initialized AsyncQdrantClient instance from qdrant-client library
            url: Qdrant server URL (optional, uses settings if not provided)
            collection_name: Name of the anime collection (optional, uses settings if not provided)

        Returns:
            Initialized QdrantClient instance with collection ready

        Raises:
            Exception: If collection initialization fails
        """
        client = cls(settings, async_qdrant_client, url, collection_name)
        await client._initialize_collection()
        return client

    async def _initialize_collection(self) -> None:
        """Initialize and validate anime collection with multi-vector architecture and performance optimization.

        Creates collection if it doesn't exist with optimized configuration including
        quantization, HNSW parameters, and payload indexing. Validates existing collections
        for compatibility with current vector architecture.

        Raises:
            Exception: If collection creation or validation fails
        """
        try:
            # Check if collection exists and validate its configuration
            collections = (await self.client.get_collections()).collections
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
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=vectors_config,
                    quantization_config=quantization_config,
                    optimizers_config=optimizers_config,
                    wal_config=wal_config,
                )

                # Configure payload indexing for faster filtering
                if getattr(self.settings, "qdrant_enable_payload_indexing", True):
                    await self._setup_payload_indexing()

                logger.info(
                    f"Successfully created collection with {len(vectors_config)} vectors"
                )
            else:
                # Validate existing collection compatibility
                if not await self._validate_collection_compatibility():
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

    async def _validate_collection_compatibility(self) -> bool:
        """Validate existing collection compatibility with current vector architecture.

        Checks if existing collection has all expected vectors defined in settings.

        Returns:
            True if collection is compatible with current configuration, False otherwise
        """
        try:
            collection_info = await self.client.get_collection(self.collection_name)
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
        """Validate vector configuration before collection creation.

        Args:
            vectors_config: Dictionary mapping vector names to VectorParams

        Raises:
            ValueError: If configuration is invalid or dimensions don't match
        """
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
        """Create multi-vector configuration with priority-based optimization.

        Generates VectorParams for all vectors defined in settings with appropriate
        HNSW and quantization configurations based on priority levels.
        Vectors listed in settings.multivector_vectors get MultiVectorConfig
        with MAX_SIM comparator for storing multiple vectors per point.

        Returns:
            Dictionary mapping vector names to VectorParams with optimized configurations
        """
        distance = self._DISTANCE_MAPPING.get(self._distance_metric, Distance.COSINE)

        # Get list of vectors that use multivector storage
        multivector_names = getattr(self.settings, "multivector_vectors", [])

        # Use multi-vector architecture from settings
        vector_params = {}
        for vector_name, dimension in self.settings.vector_names.items():
            priority = self._get_vector_priority(vector_name)

            # Build VectorParams kwargs
            params_kwargs: dict[str, Any] = {
                "size": dimension,
                "distance": distance,
                "hnsw_config": self._get_hnsw_config(priority),
                "quantization_config": self._get_quantization_config(priority),
            }

            # Add multivector config for vectors that store multiple vectors per point
            if vector_name in multivector_names:
                params_kwargs["multivector_config"] = MultiVectorConfig(
                    comparator=MultiVectorComparator.MAX_SIM
                )

            vector_params[vector_name] = VectorParams(**params_kwargs)

        logger.info(
            f"Created multi-vector configuration with {len(vector_params)} vectors"
        )
        return vector_params

    # NEW: Priority-Based Configuration Methods for Million-Query Optimization

    def _get_quantization_config(self, priority: str) -> QuantizationConfig | None:
        """Get quantization config based on vector priority.

        Args:
            priority: Priority level string (e.g., "high", "medium", "low")

        Returns:
            QuantizationConfig appropriate for priority level, or None if not configured
        """
        config = self.settings.quantization_config.get(priority, {})
        if config.get("type") == "scalar":
            scalar_config = ScalarQuantizationConfig(
                type=ScalarType.INT8, always_ram=bool(config.get("always_ram", False))
            )
            return ScalarQuantization(scalar=scalar_config)
        elif config.get("type") == "binary":
            binary_config = BinaryQuantizationConfig(
                always_ram=bool(config.get("always_ram", False))
            )
            return BinaryQuantization(binary=binary_config)
        return None

    def _get_hnsw_config(self, priority: str) -> HnswConfigDiff:
        """Get HNSW config based on vector priority.

        Args:
            priority: Priority level string (e.g., "high", "medium", "low")

        Returns:
            HnswConfigDiff with appropriate parameters for priority level
        """
        config = self.settings.hnsw_config.get(priority, {})
        return HnswConfigDiff(
            ef_construct=config.get("ef_construct", 200), m=config.get("m", 48)
        )

    def _get_vector_priority(self, vector_name: str) -> str:
        """Determine priority level for vector.

        Args:
            vector_name: Name of the vector (e.g., "title_vector")

        Returns:
            Priority level string ("high", "medium", or "low"), defaults to "medium"
        """
        for priority, vectors in self.settings.vector_priorities.items():
            if vector_name in vectors:
                return str(priority)
        return "medium"  # default

    def _create_optimized_optimizers_config(self) -> OptimizersConfigDiff | None:
        """Create optimized optimizers configuration for million-query scale.

        Returns:
            OptimizersConfigDiff with production-tuned parameters, or None on error
        """
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
        """Create quantization configuration for performance optimization.

        Returns:
            Quantization config based on settings (binary, scalar, or product), or None if disabled
        """
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
        """Create optimizers configuration for indexing performance.

        Returns:
            OptimizersConfig with memory mapping and indexing settings, or None if not configured
        """
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
        """Create Write-Ahead Logging configuration.

        Returns:
            WalConfigDiff with WAL parameters, or None if not enabled in settings
        """
        enable_wal = getattr(self.settings, "qdrant_enable_wal", None)
        if enable_wal is not None:
            try:
                config = WalConfigDiff(wal_capacity_mb=32, wal_segments_ahead=0)
                logger.info(f"WAL configuration: enabled={enable_wal}")
                return config
            except Exception as e:
                logger.error(f"Failed to create WAL config: {e}")
        return None

    async def _setup_payload_indexing(self) -> None:
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
                await self.client.create_payload_index(
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
        """Check if Qdrant is healthy and reachable.

        Returns:
            True if Qdrant is accessible and responding, False otherwise
        """
        try:
            # Simple health check by getting collections
            await self.client.get_collections()
            return True
        except Exception as e:
            logger.exception(f"Health check failed: {e}")
            return False

    async def get_stats(self) -> dict[str, Any]:
        """Get collection statistics.

        Returns:
            Dictionary containing collection statistics including document count,
            vector configuration, and optimizer status
        """
        try:
            # Get collection info
            collection_info = await self.client.get_collection(self.collection_name)

            # Count total points
            count_result = await self.client.count(
                collection_name=self.collection_name, count_filter=None, exact=True
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

    async def get_collection_info(self) -> Any:
        """Get collection information.

        Returns:
            Collection information from Qdrant
        """
        return await self.client.get_collection(self.collection_name)

    async def scroll(
        self,
        limit: int = 10,
        with_vectors: bool = False,
        offset: Any | None = None,
    ) -> tuple[list[Any], Any | None]:
        """Scroll through collection points.

        Args:
            limit: Maximum number of points to return
            with_vectors: Whether to include vector data
            offset: Pagination offset (point ID to start from)

        Returns:
            Tuple of (list of records, next offset for pagination)
        """
        return await self.client.scroll(
            collection_name=self.collection_name,
            limit=limit,
            with_vectors=with_vectors,
            offset=offset,
        )

    def _generate_point_id(self, anime_id: str) -> str:
        """Generate unique point ID from anime ID.

        Args:
            anime_id: Anime identifier string

        Returns:
            MD5 hash of anime ID as hexadecimal string
        """
        return hashlib.md5(anime_id.encode()).hexdigest()

    async def add_documents(
        self,
        documents: list[VectorDocument],
        batch_size: int = 100,
    ) -> dict[str, Any]:
        """Add documents (with pre-generated vectors) to the collection.

        Args:
            documents: List of VectorDocument objects (provider-agnostic)
            batch_size: Number of documents to process per batch

        Returns:
            Dict with success status and document count
        """
        try:
            # Convert VectorDocument to Qdrant PointStruct
            qdrant_points = [
                PointStruct(
                    id=doc.id,
                    vector=cast(dict[str, Any], doc.vectors),
                    payload=doc.payload,
                )
                for doc in documents
            ]

            total_docs = len(qdrant_points)
            logger.info(
                f"Adding {total_docs} pre-processed documents in batches of {batch_size}"
            )

            for i in range(0, total_docs, batch_size):
                batch_points = qdrant_points[i : i + batch_size]

                if batch_points:
                    # Upsert batch to Qdrant
                    await self.client.upsert(
                        collection_name=self.collection_name,
                        points=batch_points,
                        wait=True,
                    )
                    logger.info(
                        f"Uploaded batch {i // batch_size + 1}/{(total_docs - 1) // batch_size + 1} ({len(batch_points)} points)"
                    )

            logger.info(f"Successfully added {total_docs} documents")
            return {"success": True, "document_count": total_docs}

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return {"success": False, "error": str(e)}

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

    async def update_single_vector(
        self,
        anime_id: str,
        vector_name: str,
        vector_data: list[float],
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> bool:
        """Update a single named vector for an existing anime point.

        This method updates ONLY the specified vector while keeping all other
        vectors unchanged. Useful for selective updates like weekly statistics
        refreshes without re-indexing the entire database.

        Args:
            anime_id: Anime ID (will be hashed to point ID)
            vector_name: Name of vector to update (e.g., "review_vector")
            vector_data: New vector embedding
            max_retries: Maximum number of retry attempts (default: 3)
            retry_delay: Initial delay in seconds between retries (default: 1.0)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate vector update
            is_valid, error_msg = self._validate_vector_update(vector_name, vector_data)
            if not is_valid:
                logger.error(f"Validation failed for {vector_name}: {error_msg}")
                return False

            point_id = self._generate_point_id(anime_id)

            # Define the update operation
            async def _perform_update() -> None:
                await self.client.update_vectors(
                    collection_name=self.collection_name,
                    points=[
                        PointVectors(id=point_id, vector={vector_name: vector_data})
                    ],
                    wait=True,
                )

            # Execute with retry logic
            await retry_with_backoff(
                operation=_perform_update,
                max_retries=max_retries,
                retry_delay=retry_delay,
            )

            logger.debug(f"Updated {vector_name} for anime {anime_id}")
            return True

        except Exception:
            logger.exception(f"Failed to update vector {vector_name} for {anime_id}")
            return False

    async def update_batch_vectors(
        self,
        updates: list[dict[str, Any]],
        dedup_policy: str = "last-wins",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> dict[str, Any]:
        """Update multiple vectors across multiple anime points in a single batch.

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
            seen_keys: dict[
                tuple[str, str], int
            ] = {}  # (anime_id, vector_name) -> first index
            duplicates: list[tuple[str, str]] = []
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
                # Cast needed due to dict invariance: Dict[str, List[float]] -> Dict[str, Union[List[float], ...]]
                point_updates.append(
                    PointVectors(id=point_id, vector=cast(dict[str, Any], vectors_dict))
                )

            # Only execute if we have valid updates
            if point_updates:
                # Define the update operation
                async def _perform_batch_update() -> None:
                    await self.client.update_vectors(
                        collection_name=self.collection_name,
                        points=point_updates,
                        wait=True,
                    )

                try:
                    # Execute batch update with retry logic using retry utility
                    await retry_with_backoff(
                        operation=_perform_batch_update,
                        max_retries=max_retries,
                        retry_delay=retry_delay,
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

                except Exception as e:
                    # Retry failed - mark all as failed
                    logger.exception("Batch update failed after retries")
                    for anime_id in grouped_updates.keys():
                        for vector_name in grouped_updates[anime_id].keys():
                            results.append(
                                {
                                    "anime_id": anime_id,
                                    "vector_name": vector_name,
                                    "success": False,
                                    "error": f"Update failed after {max_retries} retries: {e!s}",
                                }
                            )

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
                "success": total_successful,
                "failed": total_failed,
                "generation_failures": len(generation_failures),
                "results": combined_results,
                "generation_failures_detail": generation_failures,
            }

        except Exception as e:
            logger.error(f"Failed to aggregate batch results: {e}")
            return {
                "total_anime": total_anime,
                "total_requested_updates": total_anime * num_vectors,
                "success": 0,
                "failed": 0,
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
            if isinstance(value, list | tuple) and len(value) == 0:
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
                if value is not None and isinstance(value, str | int | bool):
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )

        return Filter(must=conditions) if conditions else None

    async def get_by_id(
        self, point_id: str, with_vectors: bool = False
    ) -> dict[str, Any] | None:
        """Get anime by ID.

        Args:
            point_id: The anime ID to retrieve
            with_vectors: Whether to include vector data in response

        Returns:
            Anime data dictionary or None if not found
        """
        try:
            internal_point_id = self._generate_point_id(point_id)

            # Retrieve point by ID
            points = await self.client.retrieve(
                collection_name=self.collection_name,
                ids=[internal_point_id],
                with_payload=True,
                with_vectors=with_vectors,
            )

            if points:
                return dict(points[0].payload) if points[0].payload else {}
            return None

        except Exception as e:
            logger.error(f"Failed to get anime by ID {point_id}: {e}")
            return None

    async def get_point(self, point_id: str) -> dict[str, Any] | None:
        """Get point by Qdrant point ID including vectors and payload.

        Args:
            point_id: The Qdrant point ID to retrieve

        Returns:
            Point data dictionary with vectors and payload or None if not found
        """
        try:
            # Retrieve point by ID with vectors
            points = await self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_vectors=True,
                with_payload=True,
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
        """Clear all points from the collection (for fresh re-indexing).

        Returns:
            True if successful, False otherwise
        """
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
        """Delete the collection.

        Returns:
            True if successful, False otherwise
        """
        try:
            await self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False

    async def collection_exists(self) -> bool:
        """Check if the collection exists.

        Returns:
            True if collection exists, False otherwise
        """
        try:
            collections = (await self.client.get_collections()).collections
            return any(col.name == self.collection_name for col in collections)
        except Exception as e:
            logger.error(f"Failed to check collection existence: {e}")
            return False

    async def create_collection(self) -> bool:
        """Create a new collection with configuration from settings.

        Returns:
            True if successful, False otherwise
        """
        try:
            await self._initialize_collection()
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
            response = await self.client.search(
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

    async def search(
        self,
        text_embedding: list[float] | None = None,
        image_embedding: list[float] | None = None,
        entity_type: str | None = None,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Unified search across all entity types.

        Searches text_vector and/or image_vector based on provided embeddings.
        Uses Qdrant's native fusion (RRF) when both embeddings are provided.

        Args:
            text_embedding: Optional text query embedding (1024-dim BGE-M3)
            image_embedding: Optional image query embedding (768-dim OpenCLIP)
            entity_type: Optional filter by type ("anime", "character", "episode")
            limit: Maximum number of results to return
            filters: Additional payload filters

        Returns:
            List of search results with payload and similarity scores
        """
        if not text_embedding and not image_embedding:
            raise ValueError(
                "At least one of text_embedding or image_embedding is required"
            )

        # Build filter conditions
        filter_conditions = filters.copy() if filters else {}
        if entity_type:
            filter_conditions["type"] = entity_type

        qdrant_filter = (
            self._build_filter(filter_conditions) if filter_conditions else None
        )

        # Single vector search
        if text_embedding and not image_embedding:
            return await self.search_single_vector(
                vector_name="text_vector",
                vector_data=text_embedding,
                limit=limit,
                filters=qdrant_filter,
            )

        if image_embedding and not text_embedding:
            return await self.search_single_vector(
                vector_name="image_vector",
                vector_data=image_embedding,
                limit=limit,
                filters=qdrant_filter,
            )

        # Multi-vector fusion search (both embeddings provided)
        vector_queries = [
            {"vector_name": "text_vector", "vector_data": text_embedding},
            {"vector_name": "image_vector", "vector_data": image_embedding},
        ]

        return await self.search_multi_vector(
            vector_queries=vector_queries,
            limit=limit,
            fusion_method="rrf",
            filters=qdrant_filter,
        )

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
            response = await self.client.query_points(
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
