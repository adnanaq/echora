"""Qdrant client implementation with strict request/response contracts.

This module provides a typed wrapper around :class:`qdrant_client.AsyncQdrantClient`
for collection initialization, vector/payload updates, and search operations.
All write and search entry points use explicit contract models and domain errors.
"""

import logging
from typing import Any, TypeGuard, cast

from common.config import QdrantConfig
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
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
    OptimizersConfigDiff,
    OverwritePayloadOperation,
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
    SetPayload,
    SetPayloadOperation,
    VectorParams,
    WalConfigDiff,
)
from vector_db_interface import VectorDBClient, VectorDocument

from qdrant_db.contracts import (
    BatchOperationResult,
    BatchPayloadUpdateItem,
    BatchVectorUpdateItem,
    CollectionStats,
    DedupPolicy,
    PayloadUpdateMode,
    SearchFilterCondition,
    SearchHit,
    SearchRange,
    SearchRequest,
)
from qdrant_db.errors import (
    CollectionCompatibilityError,
    ConfigurationError,
    DuplicateUpdateError,
    PermanentQdrantError,
    ValidationError,
)
from qdrant_db.utils import DuplicateKeyError, deduplicate_items, retry_with_backoff

logger = logging.getLogger(__name__)


def is_float_vector(vector: Any) -> TypeGuard[list[float]]:
    """Check whether a value is a non-empty numeric vector.

    Args:
        vector: Value to validate.

    Returns:
        ``True`` when ``vector`` is a ``list`` containing at least one numeric
        element, otherwise ``False``.
    """
    return (
        isinstance(vector, list)
        and len(vector) > 0
        and all(isinstance(x, int | float) for x in vector)
    )


class QdrantClient(VectorDBClient):
    """Qdrant client wrapper with strict request/response contracts."""

    _DISTANCE_MAPPING = {
        "cosine": Distance.COSINE,
        "euclid": Distance.EUCLID,
        "dot": Distance.DOT,
    }

    def __init__(
        self,
        config: QdrantConfig,
        async_qdrant_client: AsyncQdrantClient,
        url: str | None = None,
        collection_name: str | None = None,
    ):
        """Initialize a strict-contract Qdrant client instance.

        Args:
            config: Runtime Qdrant settings.
            async_qdrant_client: Initialized async Qdrant transport client.
            url: Optional override for Qdrant URL.
            collection_name: Optional override for collection name.
        """
        self.config = config
        self.url = url or config.qdrant_url
        self._collection_name = collection_name or config.qdrant_collection_name
        self.client = async_qdrant_client
        self._distance_metric = config.qdrant_distance_metric

        self._text_vector_name = config.primary_text_vector_name
        self._image_vector_name = config.primary_image_vector_name

        self._vector_size = config.vector_names[self._text_vector_name]
        self._image_vector_size = config.vector_names[self._image_vector_name]

    @property
    def collection_name(self) -> str:
        """Return the configured collection name."""
        return self._collection_name

    @property
    def vector_size(self) -> int:
        """Return the primary text vector size."""
        return self._vector_size

    @property
    def image_vector_size(self) -> int:
        """Return the primary image vector size."""
        return self._image_vector_size

    @property
    def distance_metric(self) -> str:
        """Return the configured distance metric."""
        return self._distance_metric

    @property
    def connection_url(self) -> str:
        """Return the Qdrant connection URL."""
        return self.url

    @property
    def text_vector_name(self) -> str:
        """Return the configured primary text vector name."""
        return self._text_vector_name

    @property
    def image_vector_name(self) -> str:
        """Return the configured primary image vector name."""
        return self._image_vector_name

    @classmethod
    async def create(
        cls,
        config: QdrantConfig,
        async_qdrant_client: AsyncQdrantClient,
        url: str | None = None,
        collection_name: str | None = None,
    ) -> "QdrantClient":
        """Create a client and ensure collection state is initialized.

        Args:
            config: Runtime Qdrant settings.
            async_qdrant_client: Initialized async Qdrant transport client.
            url: Optional override for Qdrant URL.
            collection_name: Optional override for collection name.

        Returns:
            Initialized :class:`QdrantClient`.
        """
        client = cls(config, async_qdrant_client, url, collection_name)
        await client._initialize_collection()
        return client

    async def _initialize_collection(self) -> None:
        """Create collection when missing or validate compatibility when present.

        Raises:
            CollectionCompatibilityError: If existing collection schema is
                incompatible with configured vectors.
        """
        if not await self.collection_exists():
            vectors_config = self._create_multi_vector_config()
            self._validate_vector_config(vectors_config)
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=vectors_config,
                quantization_config=self._create_quantization_config(),
                optimizers_config=self._create_optimized_optimizers_config(),
                wal_config=self._create_wal_config(),
            )
            logger.info("Created collection %s", self.collection_name)
            return

        await self._validate_collection_compatibility()

    async def _validate_collection_compatibility(self) -> None:
        """Validate vector schema compatibility against current configuration.

        Raises:
            CollectionCompatibilityError: If vector names, dimensions, distance
                metric, or multivector mode do not match expectations.
        """
        collection_info = await self.client.get_collection(self.collection_name)
        existing_vectors = collection_info.config.params.vectors

        if not isinstance(existing_vectors, dict):
            raise CollectionCompatibilityError(
                "Collection uses single-vector layout but config expects named vectors"
            )

        expected_distance = self._DISTANCE_MAPPING.get(
            self._distance_metric, Distance.COSINE
        )
        expected_multivectors = set(self.config.multivector_vectors)

        for vector_name, expected_dim in self.config.vector_names.items():
            vector_params = existing_vectors.get(vector_name)
            if vector_params is None:
                raise CollectionCompatibilityError(
                    f"Collection missing required vector: {vector_name}"
                )
            if vector_params.size != expected_dim:
                raise CollectionCompatibilityError(
                    f"Vector {vector_name} size mismatch: expected {expected_dim}, got {vector_params.size}"
                )
            if vector_params.distance != expected_distance:
                raise CollectionCompatibilityError(
                    f"Vector {vector_name} distance mismatch: expected {expected_distance}, got {vector_params.distance}"
                )
            has_multivector = vector_params.multivector_config is not None
            expects_multivector = vector_name in expected_multivectors
            if has_multivector != expects_multivector:
                raise CollectionCompatibilityError(
                    f"Vector {vector_name} multivector mismatch: expected {expects_multivector}, got {has_multivector}"
                )

    def _validate_vector_config(self, vectors_config: dict[str, VectorParams]) -> None:
        """Validate generated vector configuration before collection creation.

        Args:
            vectors_config: Generated mapping of vector names to params.

        Raises:
            ConfigurationError: If generated config is empty, has mismatched
                vector count, or contains unexpected dimensions.
        """
        if not vectors_config:
            raise ConfigurationError("Vector configuration is empty")

        if len(vectors_config) != len(self.config.vector_names):
            raise ConfigurationError(
                "Vector count mismatch between generated config and settings"
            )

        for vector_name, vector_params in vectors_config.items():
            expected_dim = self.config.vector_names.get(vector_name)
            if expected_dim is None:
                raise ConfigurationError(
                    f"Vector {vector_name} is not present in configured vector_names"
                )
            if vector_params.size != expected_dim:
                raise ConfigurationError(
                    f"Vector {vector_name} size mismatch: expected {expected_dim}, got {vector_params.size}"
                )

    def _get_vector_priority(self, vector_name: str) -> str:
        """Resolve priority bucket for a vector name.

        Args:
            vector_name: Vector name configured for the collection.

        Returns:
            Priority label used by HNSW/quantization lookup.
        """
        for priority, vectors in self.config.vector_priorities.items():
            if vector_name in vectors:
                return str(priority)
        return "medium"

    def _get_hnsw_config(self, priority: str) -> HnswConfigDiff:
        """Build HNSW configuration for a priority class.

        Args:
            priority: Priority bucket key.

        Returns:
            Configured HNSW diff model.
        """
        hnsw_cfg = self.config.hnsw_config.get(priority, {})
        return HnswConfigDiff(
            ef_construct=hnsw_cfg.get("ef_construct", 200),
            m=hnsw_cfg.get("m", 48),
        )

    def _get_quantization_config(self, priority: str) -> QuantizationConfig | None:
        """Build per-vector quantization config for a priority class.

        Args:
            priority: Priority bucket key.

        Returns:
            Quantization config model when configured; otherwise ``None``.
        """
        quant_cfg = self.config.quantization_config.get(priority, {})
        if quant_cfg.get("type") == "scalar":
            return ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    always_ram=bool(quant_cfg.get("always_ram", False)),
                )
            )
        if quant_cfg.get("type") == "binary":
            return BinaryQuantization(
                binary=BinaryQuantizationConfig(
                    always_ram=bool(quant_cfg.get("always_ram", False))
                )
            )
        return None

    def _create_multi_vector_config(self) -> dict[str, VectorParams]:
        """Create named-vector configuration for collection creation.

        Returns:
            Mapping of vector name to :class:`VectorParams`.
        """
        distance = self._DISTANCE_MAPPING.get(self._distance_metric, Distance.COSINE)
        multivector_names = set(self.config.multivector_vectors)

        vector_params: dict[str, VectorParams] = {}
        for vector_name, dimension in self.config.vector_names.items():
            priority = self._get_vector_priority(vector_name)
            params_kwargs: dict[str, Any] = {
                "size": dimension,
                "distance": distance,
                "hnsw_config": self._get_hnsw_config(priority),
                "quantization_config": self._get_quantization_config(priority),
            }
            if vector_name in multivector_names:
                params_kwargs["multivector_config"] = MultiVectorConfig(
                    comparator=MultiVectorComparator.MAX_SIM
                )
            vector_params[vector_name] = VectorParams(**params_kwargs)

        return vector_params

    def _create_optimized_optimizers_config(self) -> OptimizersConfigDiff | None:
        """Create optimizer tuning parameters for collection creation.

        Returns:
            Optimizer config when construction succeeds, otherwise ``None``.
        """
        try:
            return OptimizersConfigDiff(
                default_segment_number=4,
                indexing_threshold=20000,
                memmap_threshold=self.config.memory_mapping_threshold_mb * 1024,
            )
        except Exception:
            logger.exception("Failed to create optimizers config")
            return None

    def _create_quantization_config(
        self,
    ) -> BinaryQuantization | ScalarQuantization | ProductQuantization | None:
        """Create global collection quantization config from settings.

        Returns:
            One quantization config model or ``None`` when disabled/invalid.
        """
        if not self.config.qdrant_enable_quantization:
            return None

        quantization_type = self.config.qdrant_quantization_type
        always_ram = self.config.qdrant_quantization_always_ram

        try:
            if quantization_type == "binary":
                return BinaryQuantization(
                    binary=BinaryQuantizationConfig(always_ram=always_ram)
                )
            if quantization_type == "scalar":
                return ScalarQuantization(
                    scalar=ScalarQuantizationConfig(
                        type=ScalarType.INT8,
                        always_ram=always_ram,
                    )
                )
            if quantization_type == "product":
                from qdrant_client.models import CompressionRatio, ProductQuantizationConfig

                return ProductQuantization(
                    product=ProductQuantizationConfig(compression=CompressionRatio.X16)
                )
            return None
        except Exception:
            logger.exception("Failed to create quantization config")
            return None

    def _create_wal_config(self) -> WalConfigDiff | None:
        """Create WAL config from settings.

        Returns:
            WAL config when enabled and valid; otherwise ``None``.
        """
        if not self.config.qdrant_enable_wal:
            return None
        try:
            return WalConfigDiff(wal_capacity_mb=32, wal_segments_ahead=0)
        except Exception:
            logger.exception("Failed to create WAL config")
            return None

    async def setup_payload_indexes(self) -> None:
        """Create configured payload indexes for filterable metadata fields."""
        indexed_fields = self.config.qdrant_indexed_payload_fields
        if not indexed_fields:
            return

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
            await self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=type_mapping.get(field_type.lower(), PayloadSchemaType.KEYWORD),
            )

    async def health_check(self) -> bool:
        """Check Qdrant reachability using a lightweight collection request.

        Returns:
            ``True`` when Qdrant responds, else ``False``.
        """
        try:
            await self.client.get_collections()
            return True
        except Exception:
            logger.exception("Health check failed")
            return False

    async def get_stats(self) -> dict[str, Any]:
        """Return normalized collection statistics.

        Returns:
            Dictionary generated from :class:`CollectionStats`.

        Raises:
            PermanentQdrantError: If stats retrieval fails.
        """
        try:
            collection_info = await self.client.get_collection(self.collection_name)
            count_result = await self.client.count(
                collection_name=self.collection_name,
                count_filter=None,
                exact=True,
            )
            stats = CollectionStats(
                collection_name=self.collection_name,
                total_documents=count_result.count,
                vector_size=self._vector_size,
                distance_metric=self._distance_metric,
                indexed_vectors_count=collection_info.indexed_vectors_count,
                points_count=collection_info.points_count,
            )
            return stats.model_dump()
        except Exception as error:
            logger.exception("Failed to get stats")
            raise PermanentQdrantError("Failed to retrieve collection stats") from error

    async def get_collection_info(self) -> Any:
        """Return raw collection info object from Qdrant."""
        return await self.client.get_collection(self.collection_name)

    async def scroll(
        self,
        limit: int = 10,
        with_vectors: bool = False,
        offset: Any | None = None,
    ) -> tuple[list[Any], Any | None]:
        """Scroll points from collection.

        Args:
            limit: Max number of points to return.
            with_vectors: Include vector payloads when ``True``.
            offset: Scroll cursor from previous call.

        Returns:
            Tuple of points list and next offset cursor.
        """
        return await self.client.scroll(
            collection_name=self.collection_name,
            limit=limit,
            with_vectors=with_vectors,
            offset=offset,
        )

    async def add_documents(
        self,
        documents: list[VectorDocument],
        batch_size: int = 100,
    ) -> BatchOperationResult:
        """Upsert documents in batches.

        Args:
            documents: Documents with vectors/payload to upsert.
            batch_size: Number of points per upsert request.

        Returns:
            Batch operation summary.

        Raises:
            ValidationError: If ``batch_size`` is invalid.
            PermanentQdrantError: If Qdrant upsert fails.
        """
        if batch_size <= 0:
            raise ValidationError("batch_size must be >= 1")

        if not documents:
            return BatchOperationResult(total=0, successful=0, failed=0)

        points = [
            PointStruct(
                id=doc.id,
                vector=cast(dict[str, Any], doc.vectors),
                payload=doc.payload,
            )
            for doc in documents
        ]

        try:
            for start in range(0, len(points), batch_size):
                await self.client.upsert(
                    collection_name=self.collection_name,
                    points=points[start : start + batch_size],
                    wait=True,
                )
        except Exception as error:
            raise PermanentQdrantError("Failed to upsert documents") from error

        return BatchOperationResult(
            total=len(points),
            successful=len(points),
            failed=0,
        )

    def _validate_vector_update(
        self,
        vector_name: str,
        vector_data: list[float] | list[list[float]],
    ) -> None:
        """Validate vector update payload against configured schema.

        Args:
            vector_name: Vector field to update.
            vector_data: Single-vector or multivector payload.

        Raises:
            ValidationError: If name, type, or dimensions are invalid.
        """
        expected_dim = self.config.vector_names.get(vector_name)
        if expected_dim is None:
            raise ValidationError(f"Invalid vector name: {vector_name}")

        is_multivector = vector_name in set(self.config.multivector_vectors)

        if is_multivector:
            if not isinstance(vector_data, list) or len(vector_data) == 0:
                raise ValidationError("Multivector data must be a non-empty list")
            for idx, element in enumerate(vector_data):
                if not is_float_vector(element):
                    raise ValidationError(
                        f"Multivector element {idx} is not a valid float vector"
                    )
                if len(element) != expected_dim:
                    raise ValidationError(
                        f"Multivector element {idx} dimension mismatch: expected {expected_dim}, got {len(element)}"
                    )
            return

        if not is_float_vector(vector_data):
            raise ValidationError("Vector data is not a valid float vector")
        if len(vector_data) != expected_dim:
            raise ValidationError(
                f"Vector dimension mismatch: expected {expected_dim}, got {len(vector_data)}"
            )

    async def update_vectors(
        self,
        updates: list[BatchVectorUpdateItem],
        dedup_policy: DedupPolicy = "last-wins",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> BatchOperationResult:
        """Batch update point vectors with deduplication and retries.

        Args:
            updates: Requested vector updates.
            dedup_policy: Duplicate handling strategy.
            max_retries: Retry attempts for transient failures.
            retry_delay: Initial retry backoff delay in seconds.

        Returns:
            Batch operation summary.

        Raises:
            DuplicateUpdateError: If duplicates exist under ``fail`` policy.
            ValidationError: If any update payload is invalid.
            PermanentQdrantError: If Qdrant update fails.
        """
        if not updates:
            return BatchOperationResult(total=0, successful=0, failed=0)

        try:
            deduplicated_updates, duplicates_removed = deduplicate_items(
                items=updates,
                key_fn=lambda item: (item.point_id, item.vector_name),
                dedup_policy=dedup_policy,
            )
        except DuplicateKeyError as error:
            key = cast(tuple[str, str], error.key)
            raise DuplicateUpdateError(
                f"Duplicate update found for ({key[0]}, {key[1]})"
            ) from error

        grouped: dict[str, dict[str, list[float] | list[list[float]]]] = {}
        for update in deduplicated_updates:
            self._validate_vector_update(update.vector_name, update.vector_data)
            grouped.setdefault(update.point_id, {})[update.vector_name] = update.vector_data

        point_updates = [
            PointVectors(id=point_id, vector=cast(dict[str, Any], vectors))
            for point_id, vectors in grouped.items()
        ]

        async def _perform_update() -> None:
            await self.client.update_vectors(
                collection_name=self.collection_name,
                points=point_updates,
                wait=True,
            )

        try:
            await retry_with_backoff(
                operation=_perform_update,
                max_retries=max_retries,
                retry_delay=retry_delay,
            )
        except Exception as error:
            raise PermanentQdrantError("Failed to batch update vectors") from error

        total = len(deduplicated_updates)
        return BatchOperationResult(
            total=total,
            successful=total,
            failed=0,
            duplicates_removed=duplicates_removed,
        )

    def _validate_payload_update(self, payload: dict[str, Any]) -> None:
        """Validate payload update dictionary.

        Args:
            payload: Payload patch or replacement payload.

        Raises:
            ValidationError: If payload is not a non-empty dict.
        """
        if not isinstance(payload, dict):
            raise ValidationError("Payload must be a dictionary")
        if not payload:
            raise ValidationError("Payload must not be empty")

    async def update_payload(
        self,
        updates: list[BatchPayloadUpdateItem],
        mode: PayloadUpdateMode = "merge",
        dedup_policy: DedupPolicy = "last-wins",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> BatchOperationResult:
        """Batch update payload fields for points.

        Args:
            updates: Payload update items.
            mode: ``merge`` or ``overwrite`` update mode.
            dedup_policy: Duplicate handling strategy.
            max_retries: Retry attempts for transient failures.
            retry_delay: Initial retry backoff delay in seconds.

        Returns:
            Batch operation summary.

        Raises:
            ValidationError: If mode or payload inputs are invalid.
            DuplicateUpdateError: If duplicates exist under ``fail`` policy.
            PermanentQdrantError: If Qdrant batch update fails.
        """
        if mode not in {"merge", "overwrite"}:
            raise ValidationError("mode must be merge or overwrite")

        if not updates:
            return BatchOperationResult(total=0, successful=0, failed=0)

        for update in updates:
            self._validate_payload_update(update.payload)

        try:
            deduplicated_updates, duplicates_removed = deduplicate_items(
                items=updates,
                key_fn=lambda item: item.point_id,
                dedup_policy=dedup_policy,
            )
        except DuplicateKeyError as error:
            raise DuplicateUpdateError(
                f"Duplicate payload update found for point_id={error.key}"
            ) from error

        operations: list[SetPayloadOperation | OverwritePayloadOperation] = []
        for update in deduplicated_updates:
            payload_update = SetPayload(
                payload=update.payload,
                points=[update.point_id],
                key=update.key if mode == "merge" else None,
            )
            if mode == "merge":
                operations.append(SetPayloadOperation(set_payload=payload_update))
            else:
                operations.append(
                    OverwritePayloadOperation(overwrite_payload=payload_update)
                )

        async def _perform_update() -> None:
            await self.client.batch_update_points(
                collection_name=self.collection_name,
                update_operations=operations,
                wait=True,
            )

        try:
            await retry_with_backoff(
                operation=_perform_update,
                max_retries=max_retries,
                retry_delay=retry_delay,
            )
        except Exception as error:
            raise PermanentQdrantError("Failed to batch update payload") from error

        total = len(deduplicated_updates)
        return BatchOperationResult(
            total=total,
            successful=total,
            failed=0,
            duplicates_removed=duplicates_removed,
        )

    def _build_filter(self, filters: list[SearchFilterCondition]) -> Filter | None:
        """Convert contract filter conditions into Qdrant filter model.

        Args:
            filters: Typed filter conditions.

        Returns:
            Qdrant filter model or ``None`` when no filters are provided.
        """
        if not filters:
            return None

        conditions: list[FieldCondition] = []
        for condition in filters:
            if condition.operator == "eq":
                conditions.append(
                    FieldCondition(
                        key=condition.field,
                        match=MatchValue(value=condition.value),
                    )
                )
                continue

            if condition.operator == "in":
                values = cast(list[Any], condition.value)
                conditions.append(
                    FieldCondition(
                        key=condition.field,
                        match=MatchAny(any=values),
                    )
                )
                continue

            if condition.operator == "range":
                range_value = cast(SearchRange, condition.value)
                conditions.append(
                    FieldCondition(
                        key=condition.field,
                        range=Range(**range_value.model_dump(exclude_none=True)),
                    )
                )

        return Filter(must=conditions) if conditions else None

    async def get_by_id(
        self,
        point_id: str,
        with_vectors: bool = False,
    ) -> dict[str, Any] | None:
        """Fetch payload for a single point id.

        Args:
            point_id: Point identifier.
            with_vectors: Include vectors in underlying fetch request.

        Returns:
            Payload dictionary when found, else ``None``.
        """
        points = await self.client.retrieve(
            collection_name=self.collection_name,
            ids=[point_id],
            with_payload=True,
            with_vectors=with_vectors,
        )
        if not points:
            return None
        return dict(points[0].payload) if points[0].payload else {}

    async def get_point(self, point_id: str) -> dict[str, Any] | None:
        """Fetch raw point info with vector and payload.

        Args:
            point_id: Point identifier.

        Returns:
            Normalized point dictionary or ``None`` when missing.
        """
        points = await self.client.retrieve(
            collection_name=self.collection_name,
            ids=[point_id],
            with_vectors=True,
            with_payload=True,
        )
        if not points:
            return None
        point = points[0]
        return {
            "id": str(point.id),
            "vector": point.vector,
            "payload": dict(point.payload) if point.payload else {},
        }

    async def clear_index(self) -> bool:
        """Delete and recreate the configured collection.

        Returns:
            ``True`` when both delete and create succeed.
        """
        deleted = await self.delete_collection()
        if not deleted:
            return False
        return await self.create_collection()

    async def delete_collection(self) -> bool:
        """Delete the configured collection.

        Returns:
            ``True`` when deletion succeeds, otherwise ``False``.
        """
        try:
            await self.client.delete_collection(self.collection_name)
        except Exception:
            logger.exception("Failed to delete collection")
            return False
        return True

    async def collection_exists(self) -> bool:
        """Check whether the configured collection exists.

        Returns:
            ``True`` when collection is present, else ``False``.
        """
        try:
            collections = (await self.client.get_collections()).collections
        except Exception:
            logger.exception("Failed to check collection existence")
            return False
        return any(col.name == self.collection_name for col in collections)

    async def create_collection(self) -> bool:
        """Ensure collection exists and is compatible.

        Returns:
            Always ``True`` when initialization completes without exception.
        """
        await self._initialize_collection()
        return True

    async def _search_single_vector(
        self,
        vector_name: str,
        vector_data: list[float],
        limit: int,
        filters: Filter | None,
    ) -> list[SearchHit]:
        """Run a single-vector query and normalize hits.

        Args:
            vector_name: Named vector field for search.
            vector_data: Query vector values.
            limit: Max hit count.
            filters: Optional Qdrant filter.

        Returns:
            List of normalized search hits.
        """
        response = await self.client.query_points(
            collection_name=self.collection_name,
            query=vector_data,
            using=vector_name,
            limit=limit,
            with_payload=True,
            with_vectors=False,
            query_filter=filters,
        )
        return [
            SearchHit(
                id=str(point.id),
                payload=dict(point.payload) if point.payload else {},
                score=float(point.score),
            )
            for point in response.points
        ]

    async def _search_multi_vector(
        self,
        text_embedding: list[float],
        image_embedding: list[float],
        limit: int,
        filters: Filter | None,
        fusion_method: str,
    ) -> list[SearchHit]:
        """Run multimodal query using Query API prefetch + fusion.

        Args:
            text_embedding: Text query embedding.
            image_embedding: Image query embedding.
            limit: Max hit count.
            filters: Optional Qdrant filter.
            fusion_method: Fusion mode (``rrf`` or ``dbsf``).

        Returns:
            List of normalized search hits.
        """
        prefetch_queries = [
            Prefetch(
                using=self._text_vector_name,
                query=text_embedding,
                limit=limit * 2,
                filter=filters,
            ),
            Prefetch(
                using=self._image_vector_name,
                query=image_embedding,
                limit=limit * 2,
                filter=filters,
            ),
        ]

        if fusion_method == "dbsf":
            fusion = Fusion.DBSF
        else:
            fusion = Fusion.RRF

        response = await self.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetch_queries,
            query=FusionQuery(fusion=fusion),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        return [
            SearchHit(
                id=str(point.id),
                payload=dict(point.payload) if point.payload else {},
                score=float(point.score),
            )
            for point in response.points
        ]

    async def search(self, request: SearchRequest) -> list[SearchHit]:
        """Execute strict-contract search request.

        Args:
            request: Typed search request model.

        Returns:
            List of normalized search hits.

        Raises:
            ValidationError: If request implies invalid fusion combination.
        """
        filter_conditions = list(request.filters)
        if request.entity_type:
            filter_conditions.append(
                SearchFilterCondition(
                    field="entity_type",
                    operator="eq",
                    value=request.entity_type,
                )
            )
        qdrant_filter = self._build_filter(filter_conditions)

        if request.text_embedding and not request.image_embedding:
            return await self._search_single_vector(
                vector_name=self._text_vector_name,
                vector_data=request.text_embedding,
                limit=request.limit,
                filters=qdrant_filter,
            )

        if request.image_embedding and not request.text_embedding:
            return await self._search_single_vector(
                vector_name=self._image_vector_name,
                vector_data=request.image_embedding,
                limit=request.limit,
                filters=qdrant_filter,
            )

        if request.text_embedding is None or request.image_embedding is None:
            raise ValidationError(
                "Both text_embedding and image_embedding are required for fusion search"
            )

        return await self._search_multi_vector(
            text_embedding=request.text_embedding,
            image_embedding=request.image_embedding,
            limit=request.limit,
            filters=qdrant_filter,
            fusion_method=request.fusion_method,
        )
