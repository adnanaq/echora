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

        # Milvus connection details
        self.host = settings.milvus_host
        self.port = settings.milvus_port
        self.collection_name = collection_name or settings.milvus_collection_name
        self.metric_type = settings.milvus_metric_type
        self.index_type = settings.milvus_index_type
        self.nlist = settings.milvus_nlist
        self.nprobe = settings.milvus_nprobe

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
            # Define common index parameters from settings
            index_params = {
                "metric_type": self.metric_type,
                "index_type": self.index_type,
                "params": {"nlist": self.nlist},
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
        """Update a single named vector for an existing anime point.

        Milvus does not support direct in-place updates of specific vector fields.
        This operation is implemented as a delete-and-insert of the entire entity.
        """
        try:
            # 1. Fetch the existing entity
            existing_entity_data = await self.get_point(self._generate_point_id(anime_id))
            if not existing_entity_data:
                logger.error(f"Anime ID {anime_id} not found for update.")
                return False
            
            # 2. Prepare the updated entity data
            updated_entity = existing_entity_data["payload"].copy() # Start with existing payload
            updated_entity["id"] = existing_entity_data["id"]
            updated_entity["anime_id"] = anime_id

            # Update the specific vector field
            # Ensure the vector_name exists in our schema
            if vector_name not in self.settings.vector_names:
                logger.error(f"Unknown vector name: {vector_name}")
                return False
            updated_entity[vector_name] = vector_data

            # Also ensure all other vector fields are present, even if unchanged
            for vn, dim in self.settings.vector_names.items():
                if vn not in updated_entity:
                    # Try to get from existing vectors, or use zero vector
                    existing_vec = existing_entity_data["vector"].get(vn)
                    updated_entity[vn] = existing_vec if existing_vec is not None else [0.0] * dim

            # 3. Delete the old entity
            await asyncio.to_thread(self.collection.delete, expr=f"anime_id == \"{anime_id}\"")
            await asyncio.to_thread(self.collection.flush)

            # 4. Insert the new (updated) entity
            # Milvus insert expects a list of entities, each entity is a dict
            entity_to_insert = {}
            for field in self.collection.schema.fields:
                if field.name in updated_entity:
                    entity_to_insert[field.name] = updated_entity[field.name]
                elif field.name == "payload": # Special handling for payload
                    entity_to_insert[field.name] = updated_entity["payload"]
                else: # Ensure all fields are present, even if empty
                    entity_to_insert[field.name] = None # Or a default value

            await asyncio.to_thread(self.collection.insert, [entity_to_insert])
            await asyncio.to_thread(self.collection.flush)
            # No need to re-create index after flush, just ensure it's loaded
            self.collection.load()

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
        """Update multiple vectors across multiple anime points in a single batch.

        Milvus does not support direct in-place batch updates of specific vector fields.
        This operation is implemented as a batch delete-and-insert of affected entities.
        """
        results: List[Dict[str, Any]] = []
        success_count = 0
        failed_count = 0

        # Group updates by anime_id
        grouped_updates: Dict[str, Dict[str, Any]] = {}
        for update in updates:
            anime_id = update["anime_id"]
            vector_name = update["vector_name"]
            vector_data = update["vector_data"]

            if anime_id not in grouped_updates:
                grouped_updates[anime_id] = {"vectors": {}, "original_payload": None}
            grouped_updates[anime_id]["vectors"][vector_name] = vector_data
        
        entities_to_delete_exprs = []
        entities_to_insert = []

        for anime_id, data in grouped_updates.items():
            try:
                # 1. Fetch the existing entity
                existing_entity_data = await self.get_point(self._generate_point_id(anime_id))
                if not existing_entity_data:
                    raise ValueError(f"Anime ID {anime_id} not found for batch update.")
                
                # 2. Prepare the updated entity data
                updated_entity = existing_entity_data["payload"].copy() # Start with existing payload
                updated_entity["id"] = existing_entity_data["id"]
                updated_entity["anime_id"] = anime_id

                # Update specific vector fields
                for vector_name, vector_data in data["vectors"].items():
                    if vector_name not in self.settings.vector_names:
                        logger.warning(f"Unknown vector name {vector_name} for anime {anime_id}. Skipping.")
                        continue
                    updated_entity[vector_name] = vector_data
                
                # Ensure all other vector fields are present, even if unchanged
                for vn, dim in self.settings.vector_names.items():
                    if vn not in updated_entity: # If not updated in this batch
                        existing_vec = existing_entity_data["vector"].get(vn)
                        updated_entity[vn] = existing_vec if existing_vec is not None else [0.0] * dim

                # 3. Add to batch delete and insert lists
                entities_to_delete_exprs.append(f"anime_id == \"{anime_id}\"")
                
                entity_to_insert = {}
                for field in self.collection.schema.fields:
                    if field.name in updated_entity:
                        entity_to_insert[field.name] = updated_entity[field.name]
                    elif field.name == "payload":
                        entity_to_insert[field.name] = updated_entity["payload"]
                    else:
                        entity_to_insert[field.name] = None
                entities_to_insert.append(entity_to_insert)

                for vector_name in data["vectors"].keys():
                    results.append({"anime_id": anime_id, "vector_name": vector_name, "success": True})
                success_count += len(data["vectors"])

            except Exception as e:
                logger.error(f"Batch update failed for anime {anime_id}: {e}")
                for vector_name in data["vectors"].keys():
                    results.append({"anime_id": anime_id, "vector_name": vector_name, "success": False, "error": str(e)})
                failed_count += len(data["vectors"])
        
        # Perform batch delete and insert operations
        if entities_to_delete_exprs:
            await asyncio.to_thread(self.collection.delete, expr=" or ".join(entities_to_delete_exprs))
            await asyncio.to_thread(self.collection.flush)
        
        if entities_to_insert:
            await asyncio.to_thread(self.collection.insert, entities_to_insert)
            await asyncio.to_thread(self.collection.flush)
            self.collection.load()

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
        """Generate and update vectors for anime entries with automatic batching.

        This method generates vectors from AnimeEntry objects using the embedding manager
        and then updates Milvus with the generated vectors.
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
            all_batch_results: List[Dict[str, Any]] = []
            all_generation_failures: List[Dict[str, Any]] = []

            # Process in batches for memory efficiency
            for batch_start in range(0, total_anime, batch_size):
                batch_end = min(batch_start + batch_size, total_anime)
                batch = anime_entries[batch_start:batch_end]

                logger.debug(
                    f"Processing batch {batch_start//batch_size + 1}: "
                    f"anime {batch_start + 1}-{batch_end}/{total_anime}"
                )

                # Generate vectors for this batch using embedding manager
                gen_results = await self.embedding_manager.process_anime_batch(batch)

                # Prepare updates and track generation failures
                batch_updates: List[Dict[str, Any]] = []

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

                # Update in Milvus if we have valid updates
                if batch_updates:
                    batch_result = await self.update_batch_vectors(batch_updates)
                else:
                    batch_result = {"success": 0, "failed": 0, "results": []}
                    logger.warning(
                        f"Batch {batch_start//batch_size + 1} had no valid updates"
                    )

                all_batch_results.append(batch_result)

                # Call progress callback if provided
                if progress_callback:
                    progress_callback(batch_end, total_anime, batch_result)

            # Aggregate all batch results
            num_vectors = len(vector_names) if vector_names else len(self.settings.vector_names)
            return self._aggregate_batch_results(
                all_batch_results, all_generation_failures, total_anime, num_vectors
            )

        except Exception as e:
            logger.exception(f"Failed to update anime vectors in Milvus: {e}")
            raise

    def _aggregate_batch_results(
        self,
        batch_results: List[Dict[str, Any]],
        generation_failures: List[Dict[str, Any]],
        total_anime: int,
        num_vectors: int,
    ) -> Dict[str, Any]:
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
            combined_results: List[Dict[str, Any]] = []

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
            search_params = {"metric_type": self.metric_type, "params": {"nprobe": self.nprobe}}
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
        if not qdrant_filter:
            return ""

        milvus_conditions = []

        # Handle 'must' conditions (AND logic)
        if qdrant_filter.must:
            must_conditions = []
            for condition in qdrant_filter.must:
                milvus_expr = self._convert_field_condition_to_milvus_expr(condition)
                if milvus_expr: must_conditions.append(milvus_expr)
            if must_conditions: milvus_conditions.append(f"({" and ".join(must_conditions)})")

        # Handle 'should' conditions (OR logic)
        if qdrant_filter.should:
            should_conditions = []
            for condition in qdrant_filter.should:
                milvus_expr = self._convert_field_condition_to_milvus_expr(condition)
                if milvus_expr: should_conditions.append(milvus_expr)
            if should_conditions: milvus_conditions.append(f"({" or ".join(should_conditions)})")

        # Handle 'must_not' conditions (NOT logic)
        if qdrant_filter.must_not:
            must_not_conditions = []
            for condition in qdrant_filter.must_not:
                milvus_expr = self._convert_field_condition_to_milvus_expr(condition)
                if milvus_expr: must_not_conditions.append(f"not ({milvus_expr})")
            if must_not_conditions: milvus_conditions.append(f"({" and ".join(must_not_conditions)})")

        return " and ".join(milvus_conditions)

    def _convert_field_condition_to_milvus_expr(self, condition: Any) -> Optional[str]:
        """Converts a single Qdrant FieldCondition to Milvus expression string."""
        # Assuming condition is a FieldCondition from qdrant_client.models
        if not hasattr(condition, "key"): return None

        key = condition.key

        if hasattr(condition, "match") and condition.match:
            if hasattr(condition.match, "any") and condition.match.any:
                values = [f"\"{v}\"" if isinstance(v, str) else str(v) for v in condition.match.any]
                return f"{key} in [{', '.join(values)}]"
            elif hasattr(condition.match, "value") and condition.match.value is not None:
                value = f"\"{condition.match.value}\"" if isinstance(condition.match.value, str) else str(condition.match.value)
                return f"{key} == {value}"
        elif hasattr(condition, "range") and condition.range:
            range_parts = []
            if condition.range.gte is not None: range_parts.append(f"{key} >= {condition.range.gte}")
            if condition.range.lte is not None: range_parts.append(f"{key} <= {condition.range.lte}")
            if condition.range.gt is not None: range_parts.append(f"{key} > {condition.range.gt}")
            if condition.range.lt is not None: range_parts.append(f"{key} < {condition.range.lt}")
            return " and ".join(range_parts)
        
        return None

    async def search_multi_vector(
        self,
        vector_queries: List[Dict[str, Any]],
        limit: int = 10,
        fusion_method: str = "rrf", # Milvus doesn't have native RRF/DBSF like Qdrant
        filters: Optional[Filter] = None,
    ) -> List[Dict[str, Any]]:
        """Search across multiple vectors using Milvus (simulated fusion)."""
        if not vector_queries:
            return []

        # Check if all query vectors are the same
        first_vector_data = vector_queries[0]["vector_data"]
        all_vectors_same = all(q["vector_data"] == first_vector_data for q in vector_queries)

        if all_vectors_same:
            logger.info("Performing multi-field search in Milvus (all query vectors are the same).")
            # Use Milvus's multi-anns_field capability
            anns_fields = [q["vector_name"] for q in vector_queries]
            query_vector = first_vector_data

            search_params = {"metric_type": self.metric_type, "params": {"nprobe": self.nprobe}}
            expr = self._convert_qdrant_filter_to_milvus_expr(filters)

            res = await asyncio.to_thread(
                self.collection.search,
                data=[query_vector],
                anns_field=anns_fields,
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
                f"Milvus multi-field search returned {len(results)} results across {len(anns_fields)} fields"
            )
            return results

        else:
            logger.warning("Milvus does not have native multi-vector fusion like Qdrant. Simulating fusion with RRF.")
            all_results: Dict[str, Dict[str, Any]] = {}
            # Perform individual searches for each vector query
            for query_config in vector_queries:
                vector_name = query_config["vector_name"]
                vector_data = query_config["vector_data"]

                # Perform single vector search
                single_search_results = await self.search_single_vector(
                    vector_name=vector_name,
                    vector_data=vector_data,
                    limit=limit * 2, # Fetch more results for better fusion
                    filters=filters,
                )

                # Aggregate results for fusion
                for rank, result in enumerate(single_search_results):
                    anime_id = result["anime_id"]
                    if anime_id not in all_results:
                        all_results[anime_id] = {"anime_id": anime_id, "scores": [], "payload": result}
                    
                    # Store score and rank for fusion
                    all_results[anime_id]["scores"].append({"score": result["similarity_score"], "rank": rank, "vector_name": vector_name})

            # Apply fusion method (RRF for now)
            fused_results = self._apply_rrf_fusion(list(all_results.values()), limit)

            logger.info(
                f"Milvus multi-vector search returned {len(fused_results)} results using simulated {fusion_method.upper()}"
            )
            return fused_results

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
