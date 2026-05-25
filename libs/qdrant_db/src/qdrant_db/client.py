"""Qdrant client implementation with strict request/response contracts.

This module provides a typed wrapper around :class:`qdrant_client.AsyncQdrantClient`
for collection initialization, vector/payload updates, and search operations.
All write and search entry points use explicit contract models and domain errors.
"""

import logging
import time
from typing import Any, Protocol, cast

from common.config import QdrantConfig
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Filter,
    Fusion,
    FusionQuery,
    OverwritePayloadOperation,
    PointStruct,
    PointVectors,
    Prefetch,
    Rrf,
    RrfQuery,
    SetPayload,
    SetPayloadOperation,
    SparseVector,
)
from vector_db_interface import SearchHit, VectorDBClient, VectorDocument

from qdrant_db.collection.manager import QdrantCollectionManager
from qdrant_db.contracts import (
    BatchOperationResult,
    BatchPayloadUpdateItem,
    BatchVectorUpdateItem,
    DedupPolicy,
    PayloadUpdateMode,
    SearchFilterCondition,
    SearchRequest,
)
from qdrant_db.errors import (
    DuplicateUpdateError,
    PermanentQdrantError,
    ValidationError,
)
from qdrant_db.normalizer import VectorNormalizer
from qdrant_db.query_builder import (
    build_filter,
    build_prefetch_queries,
    build_sparse_query,
)
from qdrant_db.utils import DuplicateKeyError, deduplicate_items, retry_with_backoff

logger = logging.getLogger(__name__)


class _Telemetry(Protocol):
    DB_QUERY_DURATION: Any
    DB_ERRORS: Any



class QdrantClient(VectorDBClient):
    """Qdrant client wrapper with strict request/response contracts."""

    def __init__(
        self,
        config: QdrantConfig,
        async_qdrant_client: AsyncQdrantClient,
        url: str | None = None,
        collection_name: str | None = None,
        telemetry: _Telemetry | None = None,
    ):
        """Initialize a strict-contract Qdrant client instance.

        Args:
            config: Runtime Qdrant settings.
            async_qdrant_client: Initialized async Qdrant transport client.
            url: Optional override for Qdrant URL.
            collection_name: Optional override for collection name.
            telemetry: Optional telemetry registry for recording metrics.
        """
        self.config = config
        self.url = url or config.qdrant_url
        self._collection_name = collection_name or config.qdrant_collection_name
        self._async_client = async_qdrant_client
        self._telemetry = telemetry
        self._distance_metric = config.qdrant_distance_metric

        self._text_vector_name = config.primary_text_vector_name
        self._image_vector_name = config.primary_image_vector_name
        self._sparse_vector_names = set(config.sparse_vector_names)
        self._primary_sparse_vector_name = config.primary_sparse_vector_name
        self._prefetch_limit_multiplier = config.prefetch_limit_multiplier
        self._rrf_k = config.rrf_k

        self._vector_size = config.vector_names[self._text_vector_name]
        self._image_vector_size = config.vector_names[self._image_vector_name]

        self._normalizer = VectorNormalizer(
            sparse_vector_names=self._sparse_vector_names,
            multivector_vectors=set(config.multivector_vectors),
            vector_names=config.vector_names,
        )
        self._collection_manager = QdrantCollectionManager(
            config=config,
            async_client=async_qdrant_client,
            collection_name=self._collection_name,
        )

    @property
    def collection_name(self) -> str:
        """Return the configured collection name."""
        return self._collection_name

    @property
    def vector_size(self) -> int:
        """Return the primary text vector size."""
        return self._vector_size

    @property
    def indexed_fields(self) -> frozenset[str]:
        """Return the set of payload field names that have a Qdrant index."""
        return frozenset(self.config.qdrant_indexed_payload_fields.keys())

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

    @property
    def sparse_vector_name(self) -> str:
        """Return the configured primary sparse vector name."""
        return self._primary_sparse_vector_name

    @classmethod
    async def create(
        cls,
        config: QdrantConfig,
        async_qdrant_client: AsyncQdrantClient,
        url: str | None = None,
        collection_name: str | None = None,
        telemetry: _Telemetry | None = None,
    ) -> "QdrantClient":
        """Create a client and ensure collection state is initialized.

        Args:
            config: Runtime Qdrant settings.
            async_qdrant_client: Initialized async Qdrant transport client.
            url: Optional override for Qdrant URL.
            collection_name: Optional override for collection name.
            telemetry: Optional telemetry registry for recording metrics.

        Returns:
            Initialized :class:`QdrantClient`.
        """
        client = cls(config, async_qdrant_client, url, collection_name, telemetry)
        await client._collection_manager.initialize_collection()
        return client

    async def health_check(self) -> bool:
        """Check Qdrant reachability using a lightweight collection request.

        Returns:
            ``True`` when Qdrant responds, else ``False``.
        """
        try:
            await self._async_client.get_collections()
            return True
        except Exception:
            logger.exception("Health check failed")
            return False

    async def get_stats(self) -> dict[str, Any]:
        """Return full collection statistics and schema.

        Merges the complete Qdrant ``CollectionInfo`` response with local
        config fields not present in the SDK response. The result includes
        all nested config (HNSW, optimizer, WAL, quantization, payload schema,
        vector and sparse vector definitions) alongside operational counts.

        Returns:
            Dictionary with all ``CollectionInfo`` fields plus
            ``collection_name``, ``total_documents``, ``vector_size``, and
            ``distance_metric``.

        Raises:
            PermanentQdrantError: If stats retrieval fails.
        """
        try:
            collection_info = await self._async_client.get_collection(
                self.collection_name
            )
            count_result = await self._async_client.count(
                collection_name=self.collection_name,
                count_filter=None,
                exact=True,
            )
            stats = collection_info.model_dump()
            stats.update(
                {
                    "collection_name": self.collection_name,
                    "total_documents": count_result.count,
                    "vector_size": self._vector_size,
                    "distance_metric": self._distance_metric,
                }
            )
            return stats
        except Exception as error:
            logger.exception("Failed to get stats")
            raise PermanentQdrantError("Failed to retrieve collection stats") from error

    async def scroll(
        self,
        limit: int = 10,
        with_vectors: bool = False,
        offset: Any | None = None,
        scroll_filter: list[SearchFilterCondition] | None = None,
    ) -> tuple[list[Any], Any | None]:
        """Scroll points from collection.

        Args:
            limit: Max number of points to return.
            with_vectors: Include vector payloads when ``True``.
            offset: Scroll cursor from previous call.
            scroll_filter: Optional filter conditions to narrow results.

        Returns:
            Tuple of points list and next offset cursor.
        """
        return await self._async_client.scroll(
            collection_name=self.collection_name,
            limit=limit,
            with_vectors=with_vectors,
            offset=offset,
            scroll_filter=build_filter(scroll_filter) if scroll_filter else None,
        )

    async def add_documents(
        self,
        documents: list[VectorDocument],
        batch_size: int = 100,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> BatchOperationResult:
        """Upsert documents in batches with per-batch retry.

        Each batch is retried independently on transient failures. If a batch
        fails permanently after all retries, earlier batches are already
        committed to Qdrant. Because upserts are idempotent, re-running the
        full call is safe to recover from a partial commit.

        Args:
            documents: Documents with vectors/payload to upsert.
            batch_size: Number of points per upsert request.
            max_retries: Retry attempts per batch for transient failures.
            retry_delay: Initial backoff delay in seconds (doubles each retry).

        Returns:
            Batch operation summary.

        Raises:
            ValidationError: If ``batch_size`` is invalid.
            PermanentQdrantError: If any batch fails after all retries.
        """
        if batch_size <= 0:
            raise ValidationError("batch_size must be >= 1")

        if not documents:
            return BatchOperationResult(total=0, successful=0, failed=0)

        def _normalize_point_vectors(
            vectors: dict[str, Any],
        ) -> dict[str, list[float] | list[list[float]] | SparseVector]:
            normalized_vectors: dict[
                str, list[float] | list[list[float]] | SparseVector
            ] = {}
            for vector_name, vector_data in vectors.items():
                normalized_vectors[vector_name] = (
                    self._normalizer.normalize_vector_payload(
                        vector_name=vector_name,
                        vector_data=vector_data,
                    )
                )
            return normalized_vectors

        for start in range(0, len(documents), batch_size):
            batch_points = [
                PointStruct(
                    id=doc.id,
                    vector=cast(
                        dict[str, Any], _normalize_point_vectors(doc.vectors)
                    ),
                    payload=doc.payload,
                )
                for doc in documents[start : start + batch_size]
            ]

            async def _upsert(pts: list[PointStruct] = batch_points) -> None:
                await self._async_client.upsert(
                    collection_name=self.collection_name,
                    points=pts,
                    wait=True,
                )

            try:
                await retry_with_backoff(
                    operation=_upsert,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                )
            except Exception as error:
                raise PermanentQdrantError("Failed to upsert documents") from error

        return BatchOperationResult(
            total=len(documents),
            successful=len(documents),
            failed=0,
        )

    async def update_vectors(  # ty: ignore[invalid-method-override]
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

        grouped: dict[
            str, dict[str, list[float] | list[list[float]] | SparseVector]
        ] = {}
        for update in deduplicated_updates:
            grouped.setdefault(update.point_id, {})[update.vector_name] = (
                self._normalizer.normalize_vector_payload(
                    vector_name=update.vector_name,
                    vector_data=update.vector_data,
                )
            )

        point_updates = [
            PointVectors(id=point_id, vector=cast(dict[str, Any], vectors))
            for point_id, vectors in grouped.items()
        ]

        async def _perform_update() -> None:
            await self._async_client.update_vectors(
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

    async def update_payload(  # ty: ignore[invalid-method-override]
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
            self._normalizer.validate_payload_update(update.payload)

        try:
            deduplicated_updates, duplicates_removed = deduplicate_items(
                items=updates,
                key_fn=(
                    (lambda item: (item.point_id, item.key))
                    if mode == "merge"
                    else (lambda item: item.point_id)
                ),
                dedup_policy=dedup_policy,
            )
        except DuplicateKeyError as error:
            raise DuplicateUpdateError(
                f"Duplicate payload update found for key={error.key}"
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
            await self._async_client.batch_update_points(
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

    async def get_by_id(
        self,
        point_id: str,
        with_vectors: bool = False,
    ) -> dict[str, Any] | None:
        """Fetch a single point by ID, returning id, payload, and optionally vectors.

        Args:
            point_id: Point identifier.
            with_vectors: When ``True``, include vectors in the response.

        Returns:
            Dictionary with ``id``, ``payload``, and (if requested) ``vector``
            when found, else ``None``.
        """
        points = await self._async_client.retrieve(
            collection_name=self.collection_name,
            ids=[point_id],
            with_payload=True,
            with_vectors=with_vectors,
        )
        if not points:
            return None
        point = points[0]
        result: dict[str, Any] = {
            "id": str(point.id),
            "payload": dict(point.payload) if point.payload else {},
        }
        if with_vectors:
            result["vector"] = point.vector
        return result

    async def clear_index(self) -> bool:
        """Delete and recreate the collection, returning ``True`` on success."""
        return await self._collection_manager.clear_index()

    async def delete_collection(self) -> bool:
        """Delete the collection, returning ``True`` on success."""
        return await self._collection_manager.delete_collection()

    async def collection_exists(self) -> bool:
        """Return ``True`` when the configured collection is present."""
        return await self._collection_manager.collection_exists()

    async def create_collection(self) -> bool:
        """Ensure the collection exists and is compatible, returning ``True`` on success."""
        return await self._collection_manager.create_collection()

    async def setup_payload_indexes(self) -> None:
        """Create configured payload indexes for filterable metadata fields."""
        await self._collection_manager.setup_payload_indexes()

    def _emit_telemetry(self, emit: Any, operation: str) -> None:
        # Telemetry must never mask real exceptions or fail search paths.
        try:
            emit()
        except Exception:
            logger.debug("Telemetry emission failed for %s", operation, exc_info=True)

    async def _search_single_vector(
        self,
        vector_name: str,
        vector_data: list[float] | SparseVector,
        limit: int,
        filters: Filter | None,
        score_threshold: float | None = None,
    ) -> list[SearchHit]:
        """Run a single-vector query and normalize hits.

        Args:
            vector_name: Named vector field for search.
            vector_data: Query vector values.
            limit: Max hit count.
            filters: Optional Qdrant filter.
            score_threshold: Optional minimum score cutoff applied server-side.

        Returns:
            List of normalized search hits.
        """
        _start = time.perf_counter()
        try:
            response = await self._async_client.query_points(
                collection_name=self.collection_name,
                query=vector_data,
                using=vector_name,
                limit=limit,
                with_payload=True,
                with_vectors=False,
                query_filter=filters,
                score_threshold=score_threshold,
        )
            if self._telemetry:
                _elapsed = time.perf_counter() - _start
                _tel = self._telemetry
                self._emit_telemetry(
                    lambda: _tel.DB_QUERY_DURATION.record(
                        _elapsed, {"operation": "search_single_vector"}
                    ),
                    "search_single_vector.duration",
                )
            return [
                SearchHit(
                    id=str(point.id),
                    payload=dict(point.payload) if point.payload else {},
                    score=float(point.score),
                )
                for point in response.points
            ]
        except Exception:
            if self._telemetry:
                _tel = self._telemetry
                self._emit_telemetry(
                    lambda: _tel.DB_ERRORS.add(1, {"operation": "search_single_vector"}),
                    "search_single_vector.error",
                )
            raise

    async def _search_fusion(
        self,
        prefetch_queries: list[Prefetch],
        limit: int,
        fusion_method: str,
        score_threshold: float | None = None,
    ) -> list[SearchHit]:
        """Run fused multi-query search using Query API prefetch + fusion.

        Args:
            prefetch_queries: Prepared prefetch branches for each query signal.
            limit: Max hit count.
            fusion_method: Fusion mode (``rrf`` or ``dbsf``).
            score_threshold: Optional minimum score cutoff applied server-side.

        Returns:
            List of normalized search hits.
        """
        if fusion_method == "dbsf":
            query = FusionQuery(fusion=Fusion.DBSF)
        else:
            query = RrfQuery(rrf=Rrf(k=self._rrf_k))

        _start = time.perf_counter()
        try:
            response = await self._async_client.query_points(
                collection_name=self.collection_name,
                prefetch=prefetch_queries,
                query=query,
                limit=limit,
                with_payload=True,
                with_vectors=False,
                score_threshold=score_threshold,
            )
            if self._telemetry:
                _elapsed = time.perf_counter() - _start
                _tel = self._telemetry
                self._emit_telemetry(
                    lambda: _tel.DB_QUERY_DURATION.record(
                        _elapsed, {"operation": "search_fusion"}
                    ),
                    "search_fusion.duration",
                )
            return [
                SearchHit(
                    id=str(point.id),
                    payload=dict(point.payload) if point.payload else {},
                    score=float(point.score),
                )
                for point in response.points
            ]
        except Exception:
            if self._telemetry:
                _tel = self._telemetry
                self._emit_telemetry(
                    lambda: _tel.DB_ERRORS.add(1, {"operation": "search_fusion"}),
                    "search_fusion.error",
                )
            raise

    async def search(self, request: SearchRequest) -> list[SearchHit]:
        """Execute strict-contract search request.

        Returns raw vector search results with vector similarity scores only.
        Reranking should be applied at the application layer if needed.

        Args:
            request: Typed search request model.

        Returns:
            List of normalized search hits with vector similarity scores.

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
        qdrant_filter = build_filter(filter_conditions)

        active_embeddings: list[tuple[str, list[float] | SparseVector]] = [
            (name, vec)
            for name, vec in (
                (self._text_vector_name, request.text_embedding),
                (self._image_vector_name, request.image_embedding),
                (
                    self._primary_sparse_vector_name,
                    build_sparse_query(request.sparse_embedding)
                    if request.sparse_embedding
                    else None,
                ),
            )
            if vec is not None
        ]

        has_expansions = bool(request.expanded_text_embeddings)
        if len(active_embeddings) == 1 and not has_expansions:
            vector_name, vector_data = active_embeddings[0]
            return await self._search_single_vector(
                vector_name=vector_name,
                vector_data=vector_data,
                limit=request.limit,
                filters=qdrant_filter,
                score_threshold=request.score_threshold,
            )

        prefetch_queries = build_prefetch_queries(
            request=request,
            text_vector_name=self._text_vector_name,
            image_vector_name=self._image_vector_name,
            sparse_vector_name=self._primary_sparse_vector_name,
            qdrant_filter=qdrant_filter,
            prefetch_limit=request.limit * self._prefetch_limit_multiplier,
        )

        if len(prefetch_queries) < 2:
            raise ValidationError(
                "Fusion search requires at least two embeddings from text/image/sparse"
            )

        return await self._search_fusion(
            prefetch_queries=prefetch_queries,
            limit=request.limit,
            fusion_method=request.fusion_method,
            score_threshold=request.score_threshold,
        )
