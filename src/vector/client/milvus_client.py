import asyncio
import hashlib
import logging
from typing import Any, Dict, List, Optional

from qdrant_client.models import Filter  # Using Qdrant's Filter for now, will abstract later
from pymilvus import (  # type: ignore
    Collection, 
    CollectionSchema, 
    DataType, 
    FieldSchema, 
    connections, 
    utility
)

from ...config import Settings
from ...models.anime import AnimeEntry
from ..processors.embedding_manager import MultiVectorEmbeddingManager
from .vector_db_client import VectorDBClient

logger = logging.getLogger(__name__)


class MilvusClient(VectorDBClient):
    """Milvus Vector Database Client for Anime Search.

    Implements the VectorDBClient interface for Milvus.
    """

    def __init__(
        self,
        embedding_manager: MultiVectorEmbeddingManager,
        url: Optional[str] = None,
        collection_name: Optional[str] = None,
        settings: Optional[Settings] = None,
    ):
        super().__init__(url, collection_name, settings)

        if settings is None:
            from ...config.settings import Settings
            settings = Settings()

        self.settings = settings
        self.embedding_manager = embedding_manager

        # Milvus connection details (will be configurable in settings later)
        self.host = "localhost" # settings.milvus_host
        self.port = "19530" # settings.milvus_port
        self.collection_name = collection_name or "anime_milvus_collection" # settings.milvus_collection_name

        # Connect to Milvus
        connections.connect(host=self.host, port=self.port)
        logger.info(f"Connected to Milvus at {self.host}:{self.port}")

        # Initialize collection
        self._initialize_collection()

        # Update vector sizes based on embedding manager's models
        text_info = self.embedding_manager.text_processor.get_model_info()
        vision_info = self.embedding_manager.vision_processor.get_model_info()

        self._vector_size = text_info.get("embedding_size", 384)
        self._image_vector_size = vision_info.get("embedding_size", 512)

        logger.info(
            f"MilvusClient initialized - Text: {text_info['model_name']} ({self._vector_size}), "
            f"Vision: {vision_info['model_name']} ({self._image_vector_size})"
        )

    def _initialize_collection(self) -> None:
        """Initialize Milvus collection."""
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            logger.info(f"Milvus collection '{self.collection_name}' already exists.")
        else:
            # Define fields for the collection
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=256),
                FieldSchema(name="anime_id", dtype=DataType.VARCHAR, max_length=256),
            ]

            # Dynamically add vector fields based on settings
            for vector_name, dim in self.settings.vector_names.items():
                fields.append(FieldSchema(name=vector_name, dtype=DataType.FLOAT_VECTOR, dim=dim))

            # Add payload field for metadata
            fields.append(FieldSchema(name="payload", dtype=DataType.JSON))

            schema = CollectionSchema(fields, description="Anime Vector Database")
            self.collection = Collection(name=self.collection_name, schema=schema)
            logger.info(f"Created Milvus collection '{self.collection_name}' with {len(self.settings.vector_names)} vector fields.")

        # Load collection into memory
        self.collection.load()

    async def health_check(self) -> bool:
        """Check if Milvus is healthy and reachable."""
        try:
            # A simple way to check connection is to list collections
            # This will raise an exception if connection is not active
            await asyncio.to_thread(utility.list_collections)
            return True
        except Exception as e:
            logger.error(f"Milvus health check failed: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            # Ensure collection is loaded
            self.collection.load()
            # Get number of entities
            num_entities = await asyncio.to_thread(self.collection.num_entities)
            return {
                "collection_name": self.collection_name,
                "total_documents": num_entities,
                "vector_size": self._vector_size, # Assuming text_vector size
                "distance_metric": "cosine", # Milvus supports various, will be configurable
                "status": "loaded",
                "indexed_vectors_count": num_entities, # Simplified for now
                "points_count": num_entities,
            }
        except Exception as e:
            logger.error(f"Failed to get Milvus stats: {e}")
            return {"error": str(e)}

    def _generate_point_id(self, anime_id: str) -> str:
        """Generate unique point ID from anime ID."""
        return hashlib.md5(anime_id.encode()).hexdigest()

    async def add_documents(
        self,
        documents: List[AnimeEntry],
        batch_size: int = 100,
    ) -> bool:
        """Add anime documents to the collection."""
        try:
            total_docs = len(documents)
            logger.info(f"Adding {total_docs} documents to Milvus in batches of {batch_size}")

            for i in range(0, total_docs, batch_size):
                batch_documents = documents[i : i + batch_size]

                # Process batch to get vectors and payloads
                processed_batch = await self.embedding_manager.process_anime_batch(
                    batch_documents
                )

                entities = []
                for doc_data in processed_batch:
                    if doc_data["metadata"].get("processing_failed"):
                        logger.warning(
                            f"Skipping failed document: {doc_data["metadata"].get("anime_title")}"
                        )
                        continue

                    point_id = self._generate_point_id(doc_data["payload"]["id"])
                    
                    entity_data = {
                        "id": point_id,
                        "anime_id": doc_data["payload"]["id"],
                        "payload": doc_data["payload"], # Milvus stores JSON directly
                    }

                    # Dynamically add all vector fields
                    for vector_name, dim in self.settings.vector_names.items():
                        entity_data[vector_name] = doc_data["vectors"].get(vector_name, [0.0] * dim)

                    entities.append(entity_data)
                
                if entities:
                    # Insert entities into Milvus
                    await asyncio.to_thread(self.collection.insert, entities)
                    logger.info(
                        f"Uploaded batch {i//batch_size + 1}/{(total_docs-1)//batch_size + 1} ({len(entities)} entities)"
                    )
            
            # After inserting, you usually need to call flush and create an index
            await asyncio.to_thread(self.collection.flush)
            self._create_milvus_index()

            logger.info(f"Successfully added {total_docs} documents to Milvus")
            return True

        except Exception as e:
            logger.error(f"Failed to add documents to Milvus: {e}")
            return False

    def _create_milvus_index(self) -> None:
        """Create Milvus index for vector fields."""
        # Check if index already exists
        if not self.collection.has_index():
            # Define common index parameters (can be made configurable)
            index_params = {
                "metric_type": "COSINE",  # Will be configurable
                "index_type": "IVF_FLAT",  # Will be configurable
                "params": {"nlist": 128},  # Will be configurable
            }

            # Create index for each vector field dynamically
            for vector_name in self.settings.vector_names.keys():
                self.collection.create_index(field_name=vector_name, index_params=index_params)
                logger.info(f"Created index for '{vector_name}' in '{self.collection_name}'.")

            self.collection.load()

    async def update_single_vector(
        self,
        anime_id: str,
        vector_name: str,
        vector_data: List[float],
    ) -> bool:
        """Update a single named vector for an existing anime point."""
        logger.warning("Milvus does not support direct single vector updates like Qdrant. "
                       "This operation will re-insert the entity, which might be inefficient.")
        try:
            # Milvus update is more like a delete + insert for specific fields
            # For now, we'll fetch the existing entity, update the vector, and re-insert.
            # This is a simplified approach and needs optimization for production.
            existing_entity = await self.get_by_id(anime_id)
            if not existing_entity:
                logger.error(f"Anime ID {anime_id} not found for update.")
                return False
            
            # Assuming vector_name maps directly to a field in Milvus
            # This needs a more robust mapping for the 11-vector architecture
            milvus_field_name = vector_name # This needs to be mapped correctly

            # Prepare new entity data
            new_entity_data = {
                "id": self._generate_point_id(anime_id),
                "anime_id": anime_id,
                "payload": existing_entity, # Existing payload
            }
            # Update the specific vector field
            new_entity_data[milvus_field_name] = vector_data

            # Delete existing entity
            await asyncio.to_thread(self.collection.delete, expr=f"anime_id == \"{anime_id}\" ")
            await asyncio.to_thread(self.collection.flush)

            # Insert updated entity
            await asyncio.to_thread(self.collection.insert, [new_entity_data])
            await asyncio.to_thread(self.collection.flush)
            self._create_milvus_index() # Re-create index after flush

            logger.debug(f"Updated {vector_name} for anime {anime_id} in Milvus")
            return True

        except Exception as e:
            logger.exception(
                f"Failed to update vector {vector_name} for {anime_id} in Milvus: {e}"
            )
            return False

    async def update_batch_vectors(
        self,
        updates: List[Dict[str, Any]],
        dedup_policy: str = "last-wins",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> Dict[str, Any]:
        """Update multiple vectors across multiple anime points in a single batch."""
        logger.warning("Milvus batch vector update is not optimized and will perform delete+insert for each affected entity.")
        results: List[Dict[str, Any]] = []
        success_count = 0
        failed_count = 0

        # Group updates by anime_id
        grouped_updates: Dict[str, Dict[str, Any]] = {}
        for update in updates:
            anime_id = update["anime_id"]
            if anime_id not in grouped_updates:
                grouped_updates[anime_id] = {"vectors": {}, "payload": None}
            grouped_updates[anime_id]["vectors"].update(update["vector_data"])
        
        for anime_id, data in grouped_updates.items():
            try:
                existing_entity = await self.get_by_id(anime_id)
                if not existing_entity:
                    raise ValueError(f"Anime ID {anime_id} not found for batch update.")
                
                # Merge existing payload with new vector data
                updated_payload = existing_entity.copy()
                # This part needs careful mapping of vector_name to Milvus field names
                # For now, assuming 'title_vector' and 'image_vector' are the main ones
                if "title_vector" in data["vectors"]:
                    updated_payload["text_vector"] = data["vectors"]["title_vector"]
                if "image_vector" in data["vectors"]:
                    updated_payload["image_vector"] = data["vectors"]["image_vector"]
                
                # Re-insert logic (delete + insert)
                point_id = self._generate_point_id(anime_id)
                await asyncio.to_thread(self.collection.delete, expr=f"anime_id == \"{anime_id}\" ")
                await asyncio.to_thread(self.collection.flush)

                new_entity_data = {
                    "id": point_id,
                    "anime_id": anime_id,
                    "text_vector": updated_payload.get("text_vector", [0.0] * self._vector_size),
                    "image_vector": updated_payload.get("image_vector", [0.0] * self._image_vector_size),
                    "payload": updated_payload,
                }
                await asyncio.to_thread(self.collection.insert, [new_entity_data])
                await asyncio.to_thread(self.collection.flush)
                self._create_milvus_index()

                for vector_name in data["vectors"].keys():
                    results.append({"anime_id": anime_id, "vector_name": vector_name, "success": True})
                success_count += len(data["vectors"])

            except Exception as e:
                logger.error(f"Batch update failed for anime {anime_id}: {e}")
                for vector_name in data["vectors"].keys():
                    results.append({"anime_id": anime_id, "vector_name": vector_name, "success": False, "error": str(e)})
                failed_count += len(data["vectors"])

        return {
            "success": success_count,
            "failed": failed_count,
            "results": results,
            "duplicates_removed": 0, # Not handling deduplication here
        }

    async def update_anime_vectors(
        self,
        anime_entries: List[AnimeEntry],
        vector_names: Optional[List[str]] = None,
        batch_size: int = 100,
        progress_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Generate and update vectors for anime entries with automatic batching."""
        # This method can largely reuse the logic from QdrantClient, as it relies on embedding_manager
        # and then calls add_documents or update_batch_vectors.
        # For Milvus, we'll simplify and call add_documents for now.
        logger.warning("Milvus update_anime_vectors currently re-adds documents. Optimization needed.")
        return {"success": 0, "failed": len(anime_entries), "results": [], "generation_failures": 0}

    async def get_by_id(self, anime_id: str) -> Optional[Dict[str, Any]]:
        """Get anime by ID."""
        try:
            # Milvus query to retrieve entity by anime_id
            expr = f"anime_id == \"{anime_id}\""
            res = await asyncio.to_thread(self.collection.query, expr=expr, output_fields=["payload"])
            if res:
                # Milvus query returns a list of dicts, each dict is an entity
                # The 'payload' field contains the original payload dict
                return res[0]["payload"]
            return None
        except Exception as e:
            logger.error(f"Failed to get anime by ID {anime_id} from Milvus: {e}")
            return None

    async def get_point(self, point_id: str) -> Optional[Dict[str, Any]]:
        """Get point by Milvus primary key ID including vectors and payload."""
        try:
            # Milvus query to retrieve entity by primary key 'id'
            expr = f"id == \"{point_id}\""
            # Request all fields including vector fields
            output_fields = [field.name for field in self.collection.schema.fields if field.dtype != DataType.FLOAT_VECTOR] + \
                            [field.name for field in self.collection.schema.fields if field.dtype == DataType.FLOAT_VECTOR]
            res = await asyncio.to_thread(self.collection.query, expr=expr, output_fields=output_fields)
            if res:
                entity = res[0]
                # Reconstruct the Qdrant-like structure
                return {
                    "id": entity["id"],
                    "vector": {
                        "text_vector": entity.get("text_vector"),
                        "image_vector": entity.get("image_vector"),
                        # Add other vectors here
                    },
                    "payload": entity.get("payload", {}),
                }
            return None
        except Exception as e:
            logger.error(f"Failed to get point by ID {point_id} from Milvus: {e}")
            return None

    async def clear_index(self) -> bool:
        """Clear all points from the collection (for fresh re-indexing)."""
        try:
            await self.delete_collection()
            await self.create_collection()
            logger.info(f"Cleared and recreated Milvus collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear Milvus index: {e}")
            return False

    async def delete_collection(self) -> bool:
        """Delete the Milvus collection."""
        try:
            if utility.has_collection(self.collection_name):
                await asyncio.to_thread(utility.drop_collection, self.collection_name)
                logger.info(f"Deleted Milvus collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete Milvus collection: {e}")
            return False

    async def create_collection(self) -> bool:
        """Create the Milvus collection."""
        try:
            self._initialize_collection()
            return True
        except Exception as e:
            logger.error(f"Failed to create Milvus collection: {e}")
            return False

    async def search_single_vector(
        self,
        vector_name: str,
        vector_data: List[float],
        limit: int = 10,
        filters: Optional[Filter] = None, # Qdrant Filter, needs abstraction
    ) -> List[Dict[str, Any]]:
        """Search a single vector with raw similarity scores."""
        try:
            # Milvus search parameters
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            # Map vector_name to Milvus field name
            milvus_vector_field = vector_name # This needs proper mapping

            # Convert Qdrant Filter to Milvus expression (simplified for now)
            expr = self._convert_qdrant_filter_to_milvus_expr(filters)

            res = await asyncio.to_thread(
                self.collection.search,
                data=[vector_data],
                anns_field=milvus_vector_field,
                param=search_params,
                limit=limit,
                expr=expr,
                output_fields=["anime_id", "payload"], # Include payload for results
            )

            results = []
            for hit in res[0]: # res[0] contains hits for the first query
                payload = hit.entity.get("payload", {}) # Get payload from entity
                results.append({
                    "id": hit.id,
                    "anime_id": hit.entity.get("anime_id"),
                    "_id": hit.id,
                    **payload,
                    "similarity_score": hit.distance, # Milvus returns distance, convert to similarity if needed
                })
            
            logger.info(
                f"Milvus single vector search ({vector_name}) returned {len(results)} results"
            )
            return results

        except Exception as e:
            logger.error(f"Milvus single vector search failed: {e}")
            raise

    def _convert_qdrant_filter_to_milvus_expr(self, qdrant_filter: Optional[Filter]) -> str:
        """Converts Qdrant Filter object to Milvus expression string."""
        if not qdrant_filter or not qdrant_filter.must:
            return ""
        
        milvus_conditions = []
        for condition in qdrant_filter.must:
            if isinstance(condition, FieldCondition):
                key = condition.key
                if condition.match:
                    if condition.match.any:
                        # Milvus 'in' operator for multiple values
                        values = [f"\"{v}\"" if isinstance(v, str) else str(v) for v in condition.match.any]
                        milvus_conditions.append(f"{key} in [{', '.join(values)}] ")
                    elif condition.match.value is not None:
                        # Exact match
                        value = f"\"{condition.match.value}\"" if isinstance(condition.match.value, str) else str(condition.match.value)
                        milvus_conditions.append(f"{key} == {value}")
                elif condition.range:
                    # Range conditions
                    range_parts = []
                    if condition.range.gte is not None: range_parts.append(f"{key} >= {condition.range.gte}")
                    if condition.range.lte is not None: range_parts.append(f"{key} <= {condition.range.lte}")
                    if condition.range.gt is not None: range_parts.append(f"{key} > {condition.range.gt}")
                    if condition.range.lt is not None: range_parts.append(f"{key} < {condition.range.lt}")
                    milvus_conditions.append(" and ".join(range_parts))
        
        return " and ".join(milvus_conditions)

    async def search_multi_vector(
        self,
        vector_queries: List[Dict[str, Any]],
        limit: int = 10,
        fusion_method: str = "rrf", # Milvus doesn't have native RRF/DBSF like Qdrant
        filters: Optional[Filter] = None,
    ) -> List[Dict[str, Any]]:
        """Search across multiple vectors using Milvus (simulated fusion)."""
        logger.warning("Milvus does not have native multi-vector fusion like Qdrant. Simulating fusion.")
        
        # This will require multiple individual searches and then re-ranking/fusion logic
        # For now, we'll just do a single vector search as a placeholder.
        if not vector_queries:
            return []
        
        # Take the first query as a placeholder
        first_query = vector_queries[0]
        return await self.search_single_vector(
            vector_name=first_query["vector_name"],
            vector_data=first_query["vector_data"],
            limit=limit,
            filters=filters,
        )

    async def search_text_comprehensive(
        self,
        query: str,
        limit: int = 10,
        fusion_method: str = "rrf",
        filters: Optional[Filter] = None,
    ) -> List[Dict[str, Any]]:
        """Search across all text vectors using Milvus (simulated fusion)."""
        logger.warning("Milvus comprehensive text search is simulated. Optimization needed.")
        query_embedding = self.embedding_manager.text_processor.encode_text(query)
        if query_embedding is None:
            return []
        
        # For now, use 'text_vector' as the primary text vector in Milvus
        return await self.search_single_vector(
            vector_name="text_vector", # Needs mapping to actual Milvus field
            vector_data=query_embedding,
            limit=limit,
            filters=filters,
        )

    async def search_visual_comprehensive(
        self,
        image_data: str,
        limit: int = 10,
        fusion_method: str = "rrf",
        filters: Optional[Filter] = None,
    ) -> List[Dict[str, Any]]:
        """Search across both image vectors using Milvus (simulated fusion)."""
        logger.warning("Milvus comprehensive visual search is simulated. Optimization needed.")
        image_embedding = self.embedding_manager.vision_processor.encode_image(image_data)
        if image_embedding is None:
            return []
        
        # For now, use 'image_vector' as the primary image vector in Milvus
        return await self.search_single_vector(
            vector_name="image_vector", # Needs mapping to actual Milvus field
            vector_data=image_embedding,
            limit=limit,
            filters=filters,
        )

    async def search_complete(
        self,
        query: str,
        image_data: Optional[str] = None,
        limit: int = 10,
        fusion_method: str = "rrf",
        filters: Optional[Filter] = None,
    ) -> List[Dict[str, Any]]:
        """Search across all vectors (text + image) using Milvus (simulated fusion)."""
        logger.warning("Milvus complete search is simulated. Optimization needed.")
        # This will combine text and image searches and then fuse results
        # For now, just perform a text search as a placeholder
        return await self.search_text_comprehensive(query, limit, fusion_method, filters)

    async def search_characters(
        self,
        query: str,
        image_data: Optional[str] = None,
        limit: int = 10,
        fusion_method: str = "rrf",
        filters: Optional[Filter] = None,
    ) -> List[Dict[str, Any]]:
        """Search specifically for character-related content using character vectors."""
        logger.warning("Milvus character search is simulated. Optimization needed.")
        # For now, just perform a text search as a placeholder
        return await self.search_text_comprehensive(query, limit, fusion_method, filters)
