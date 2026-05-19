"""Collection lifecycle management — creation, deletion, compatibility, and indexing."""

import logging

from common.config import QdrantConfig
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PayloadSchemaType

from qdrant_db.collection.schema_builder import (
    build_optimizers_config,
    build_quantization_config,
    build_sparse_vector_config,
    build_vector_config,
    build_wal_config,
    validate_vector_config,
)
from qdrant_db.errors import CollectionCompatibilityError

logger = logging.getLogger(__name__)


class QdrantCollectionManager:
    """Owns all collection I/O: creation, deletion, compatibility checks, and indexing."""

    _DISTANCE_MAPPING = {
        "cosine": Distance.COSINE,
        "euclid": Distance.EUCLID,
        "dot": Distance.DOT,
    }

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
            validate_vector_config(self._config, vectors_config)
            await self._async_client.create_collection(
                collection_name=self._collection_name,
                vectors_config=vectors_config,
                sparse_vectors_config=build_sparse_vector_config(self._config),
                quantization_config=build_quantization_config(self._config),
                optimizers_config=build_optimizers_config(self._config),
                wal_config=build_wal_config(self._config),
            )
            logger.info("Created collection %s", self._collection_name)
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
            collections = (await self._async_client.get_collections()).collections
        except Exception:
            logger.exception("Failed to check collection existence")
            return False
        return any(col.name == self._collection_name for col in collections)

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
                metric, or multivector mode do not match expectations.
        """
        collection_info = await self._async_client.get_collection(self._collection_name)
        existing_vectors = collection_info.config.params.vectors

        if not isinstance(existing_vectors, dict):
            raise CollectionCompatibilityError(
                "Collection uses single-vector layout but config expects named vectors"
            )

        expected_distance = self._DISTANCE_MAPPING.get(
            self._config.qdrant_distance_metric, Distance.COSINE
        )
        expected_multivectors = set(self._config.multivector_vectors)

        for vector_name, expected_dim in self._config.vector_names.items():
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
