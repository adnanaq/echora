"""Collection lifecycle management — creation, deletion, compatibility, and indexing."""

import logging

from common.config import QdrantConfig
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Datatype, PayloadSchemaType, VectorParams
from qdrant_db.collection.schema_builder import (
    _DATATYPE_MAPPING,
    _DISTANCE_MAPPING,
    build_optimizers_config,
    build_quantization_config,
    build_sparse_vector_config,
    build_vector_config,
    build_wal_config,
)
from qdrant_db.errors import CollectionCompatibilityError
from qdrant_db.tracing import qdrant_span

logger = logging.getLogger(__name__)


class QdrantCollectionManager:
    """Owns all collection I/O: creation, deletion, compatibility checks, and indexing."""

    def __init__(
        self,
        config: QdrantConfig,
        async_client: AsyncQdrantClient,
        collection_name: str,
    ) -> None:
        """Initialize the collection manager.

        Args:
            config: Qdrant runtime settings.
            async_client: Initialized async Qdrant transport client.
            collection_name: Name of the collection to manage.
        """
        self._config = config
        self._async_client = async_client
        self._collection_name = collection_name

    async def initialize_collection(self) -> None:
        """Create collection when missing or validate compatibility when present.

        Raises:
            CollectionCompatibilityError: If existing collection schema is
                incompatible with configured vectors.
        """
        if not await self.collection_exists():
            vectors_config = build_vector_config(self._config)
            try:
                with qdrant_span("create_collection", self._collection_name):
                    await self._async_client.create_collection(
                        collection_name=self._collection_name,
                        vectors_config=vectors_config,
                        sparse_vectors_config=build_sparse_vector_config(self._config),
                        quantization_config=build_quantization_config(self._config),
                        optimizers_config=build_optimizers_config(self._config),
                        wal_config=build_wal_config(self._config),
                    )
            except UnexpectedResponse as exc:
                if (
                    exc.status_code in (400, 409)
                    and b"already exists" in exc.content.lower()
                ):
                    # Concurrent instance created the collection between our existence
                    # check and create — validate compatibility and proceed.
                    logger.info(
                        f"Collection {self._collection_name} created concurrently, validating compatibility"
                    )
                    await self._validate_compatibility()
                    return
                raise

            if self._config.qdrant_enable_payload_indexing:
                await self.setup_payload_indexes()

            logger.info(f"Created collection {self._collection_name}")
            return

        await self._validate_compatibility()

    async def create_collection(self) -> bool:
        """Ensure collection exists and is compatible.

        Returns:
            Always ``True`` when initialization completes without exception.
        """
        await self.initialize_collection()
        return True

    async def delete_collection(self) -> bool:
        """Delete the configured collection.

        Returns:
            ``True`` when deletion succeeds, otherwise ``False``.
        """
        try:
            with qdrant_span("delete_collection", self._collection_name):
                await self._async_client.delete_collection(self._collection_name)
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
            with qdrant_span("collection_exists", self._collection_name):
                return await self._async_client.collection_exists(self._collection_name)
        except Exception:
            logger.exception("Failed to check collection existence")
            return False

    async def clear_index(self) -> bool:
        """Delete and recreate the configured collection.

        Returns:
            ``True`` when both delete and create succeed.
        """
        deleted = await self.delete_collection()
        if not deleted:
            return False
        return await self.create_collection()

    async def setup_payload_indexes(self) -> None:
        """Create configured payload indexes for filterable metadata fields."""
        indexed_fields = self._config.qdrant_indexed_payload_fields
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
            with qdrant_span("create_payload_index", self._collection_name):
                await self._async_client.create_payload_index(
                    collection_name=self._collection_name,
                    field_name=field_name,
                    field_schema=type_mapping.get(
                        field_type.lower(), PayloadSchemaType.KEYWORD
                    ),
                )

    async def _validate_compatibility(self) -> None:
        """Validate vector schema compatibility against current configuration.

        Raises:
            CollectionCompatibilityError: If vector names, dimensions, distance
                metric, datatype, or multivector mode do not match expectations.
        """
        with qdrant_span("get_collection", self._collection_name):
            collection_info = await self._async_client.get_collection(
                self._collection_name
            )
        existing_vectors = collection_info.config.params.vectors

        if not isinstance(existing_vectors, dict):
            raise CollectionCompatibilityError(
                "Collection uses single-vector layout but config expects named vectors"
            )

        expected_distance = _DISTANCE_MAPPING[self._config.qdrant_distance_metric]
        expected_datatype = (
            _DATATYPE_MAPPING[self._config.qdrant_vector_datatype]
            if self._config.qdrant_vector_datatype is not None
            else Datatype.FLOAT32
        )
        expected_multivectors = set(self._config.multivector_vectors)

        for vector_name, expected_dim in self._config.vector_names.items():
            self._validate_vector_params(
                vector_name=vector_name,
                vector_params=existing_vectors.get(vector_name),
                expected_dim=expected_dim,
                expected_distance=expected_distance,
                expected_datatype=expected_datatype,
                expects_multivector=vector_name in expected_multivectors,
            )

        existing_sparse_vectors = getattr(
            collection_info.config.params, "sparse_vectors", None
        )
        if not isinstance(existing_sparse_vectors, dict):
            raise CollectionCompatibilityError(
                "Collection missing sparse vector configuration"
            )
        missing_sparse_vectors = set(self._config.sparse_vector_names) - set(
            existing_sparse_vectors.keys()
        )
        if missing_sparse_vectors:
            raise CollectionCompatibilityError(
                "Collection missing required sparse vectors: "
                + ", ".join(sorted(missing_sparse_vectors))
            )

    def _validate_vector_params(
        self,
        vector_name: str,
        vector_params: VectorParams | None,
        expected_dim: int,
        expected_distance: object,
        expected_datatype: Datatype,
        expects_multivector: bool,
    ) -> None:
        """Validate one named dense vector against expected schema parameters.

        Args:
            vector_name: Named vector field.
            vector_params: Existing vector parameters from Qdrant, if present.
            expected_dim: Expected vector dimension.
            expected_distance: Expected Qdrant distance metric.
            expected_datatype: Expected Qdrant vector storage datatype.
            expects_multivector: Whether multivector mode is expected.

        Raises:
            CollectionCompatibilityError: If the vector is missing or mismatches
                size, distance, datatype, or multivector mode.
        """
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
        stored_datatype = vector_params.datatype or Datatype.FLOAT32
        if stored_datatype != expected_datatype:
            raise CollectionCompatibilityError(
                f"Vector {vector_name} datatype mismatch: expected {expected_datatype}, got {stored_datatype}"
            )
        has_multivector = vector_params.multivector_config is not None
        if has_multivector != expects_multivector:
            raise CollectionCompatibilityError(
                f"Vector {vector_name} multivector mismatch: expected {expects_multivector}, got {has_multivector}"
            )
