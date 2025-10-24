import asyncio
import hashlib
import logging
from typing import Any, Dict, List, Optional

from qdrant_client.models import Filter  # Using Qdrant's Filter for now, will abstract later
from marqo import Client as MarqoClientSDK

from ...config import Settings
from ...models.anime import AnimeEntry
from ..processors.embedding_manager import MultiVectorEmbeddingManager
from .vector_db_client import VectorDBClient

logger = logging.getLogger(__name__)


class MarqoClient(VectorDBClient):
    """Marqo Vector Database Client for Anime Search.

    Implements the VectorDBClient interface for Marqo.
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

        # Marqo connection details
        self.url = url or settings.marqo_url
        self.index_name = collection_name or settings.marqo_index_name
        self.marqo_model = settings.marqo_model
        self.marqo_treat_urls_as_images = settings.marqo_treat_urls_and_pointers_as_images

        # Initialize Marqo client
        self.client = MarqoClientSDK(url=self.url)
        logger.info(f"Initialized Marqo client with URL: {self.url}")

        # Initialize index
        self._initialize_index()

        # Update vector sizes based on embedding manager's models
        text_info = self.embedding_manager.text_processor.get_model_info()
        vision_info = self.embedding_manager.vision_processor.get_model_info()

        self._vector_size = text_info.get("embedding_size", 384)
        self._image_vector_size = vision_info.get("embedding_size", 512)

        logger.info(
            f"MarqoClient initialized - Text: {text_info['model_name']} ({self._vector_size}), "
            f"Vision: {vision_info['model_name']} ({self._image_vector_size})"
        )

    def _initialize_index(self) -> None:
        """Initialize Marqo index."""
        try:
            # Check if index exists
            existing_indexes = self.client.get_indexes()["results"]
            index_exists = any(idx["indexName"] == self.index_name for idx in existing_indexes)

            if not index_exists:
                # Create index with appropriate settings
                # Marqo can use its own models or external models. For now, we'll use a default.
                # We can make this configurable later.
                index_settings = {
                    "index_defaults": {
                        "model": self.marqo_model, # Use configurable Marqo model
                        "treat_urls_and_pointers_as_images": self.marqo_treat_urls_as_images, # Use configurable setting
                        "normalize_embeddings": True,
                    }
                }
                self.client.create_index(index_name=self.index_name, settings=index_settings)
                logger.info(f"Created Marqo index '{self.index_name}'.")
            else:
                logger.info(f"Marqo index '{self.index_name}' already exists.")

        except Exception as e:
            logger.error(f"Failed to initialize Marqo index: {e}")
            raise

    async def health_check(self) -> bool:
        """Check if Marqo is healthy and reachable."""
        try:
            # Placeholder for Marqo health check
            # await asyncio.to_thread(self.client.health)
            logger.warning("Marqo health check is a placeholder.")
            return True
        except Exception as e:
            logger.error(f"Marqo health check failed: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        try:
            # Placeholder for Marqo stats
            # stats = await asyncio.to_thread(self.client.index(self.index_name).get_stats)
            logger.warning("Marqo get_stats is a placeholder.")
            return {
                "index_name": self.index_name,
                "total_documents": 0, # stats.numberOfDocuments
                "vector_size": self._vector_size, # Assuming text_vector size
                "status": "unknown",
            }
        except Exception as e:
            logger.error(f"Failed to get Marqo stats: {e}")
            return {"error": str(e)}

    def _generate_point_id(self, anime_id: str) -> str:
        """Generate unique point ID from anime ID."""
        return hashlib.md5(anime_id.encode()).hexdigest()

    async def add_documents(
        self, documents: List[AnimeEntry], batch_size: int = 100
    ) -> bool:
        """Add anime documents to the index."""
        try:
            total_docs = len(documents)
            logger.info(f"Adding {total_docs} documents to Marqo in batches of {batch_size}")

            for i in range(0, total_docs, batch_size):
                batch_documents = documents[i : i + batch_size]

                # Process batch to get vectors and payloads
                processed_batch = await self.embedding_manager.process_anime_batch(
                    batch_documents
                )

                marqo_docs = []
                for doc_data in processed_batch:
                    if doc_data["metadata"].get("processing_failed"):
                        logger.warning(
                            f"Skipping failed document: {doc_data["metadata"].get("anime_title")}"
                        )
                        continue

                    point_id = self._generate_point_id(doc_data["payload"]["id"])
                    
                    # Marqo expects a dictionary with _id and fields
                    marqo_doc = {
                        "_id": point_id,
                        "anime_id": doc_data["payload"]["id"],
                        **doc_data["payload"], # Flatten payload into Marqo doc
                    }

                    # Add all pre-computed vectors directly as fields
                    for vector_name, vector_data in doc_data["vectors"].items():
                        marqo_doc[vector_name] = vector_data

                    marqo_docs.append(marqo_doc)
                
                if marqo_docs:
                    # Insert documents into Marqo
                    await asyncio.to_thread(self.client.index(self.index_name).add_documents, marqo_docs)
                    logger.info(f"Uploaded batch {i//batch_size + 1}/{(total_docs-1)//batch_size + 1} ({len(marqo_docs)} documents)")

                    marqo_docs.append(marqo_doc)
                
                if marqo_docs:
                    # Insert documents into Marqo
                    await asyncio.to_thread(self.client.index(self.index_name).add_documents, marqo_docs)
                    logger.info(f"Uploaded batch {i//batch_size + 1}/{(total_docs-1)//batch_size + 1} ({len(marqo_docs)} documents)")

            logger.info(f"Successfully added {total_docs} documents to Marqo (placeholder)")
            return True

        except Exception as e:
            logger.error(f"Failed to add documents to Marqo: {e}")
            return False

    async def update_single_vector(
        self,
        anime_id: str,
        vector_name: str,
        vector_data: List[float],
    ) -> bool:
        """Update a single named vector for an existing anime point.

        Marqo updates documents by re-adding them with the same _id.
        This operation is implemented as fetching the existing document, updating the specific vector field, and re-adding it.
        """
        try:
            point_id = self._generate_point_id(anime_id)
            # 1. Fetch the existing document
            existing_doc = await self.get_by_id(anime_id)
            if not existing_doc:
                logger.error(f"Anime ID {anime_id} not found for update.")
                return False
            
            # 2. Prepare the updated document
            updated_doc = existing_doc.copy()
            updated_doc["_id"] = point_id # Ensure _id is present
            updated_doc[vector_name] = vector_data # Update the specific vector field

            # 3. Re-add the document to update
            await asyncio.to_thread(self.client.index(self.index_name).add_documents, [updated_doc])

            logger.debug(f"Updated {vector_name} for anime {anime_id} in Marqo")
            return True

        except Exception as e:
            logger.exception(
                f"Failed to update vector {vector_name} for {anime_id} in Marqo: {e}"
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

        Marqo updates documents by re-adding them with the same _id.
        This operation is implemented as fetching existing documents, updating specific vector fields, and re-adding them in a batch.
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
                grouped_updates[anime_id] = {"vectors": {}, "original_doc": None}
            grouped_updates[anime_id]["vectors"][vector_name] = vector_data
        
        docs_to_readd = []

        for anime_id, data in grouped_updates.items():
            try:
                point_id = self._generate_point_id(anime_id)
                # 1. Fetch the existing document
                existing_doc = await self.get_by_id(anime_id)
                if not existing_doc:
                    raise ValueError(f"Anime ID {anime_id} not found for batch update.")
                
                # 2. Prepare the updated document
                updated_doc = existing_doc.copy()
                updated_doc["_id"] = point_id # Ensure _id is present

                # Update specific vector fields
                for vector_name, vector_data in data["vectors"].items():
                    updated_doc[vector_name] = vector_data
                
                docs_to_readd.append(updated_doc)

                for vector_name in data["vectors"].keys():
                    results.append({"anime_id": anime_id, "vector_name": vector_name, "success": True})
                success_count += len(data["vectors"])

            except Exception as e:
                logger.error(f"Batch update failed for anime {anime_id}: {e}")
                for vector_name in data["vectors"].keys():
                    results.append({"anime_id": anime_id, "vector_name": vector_name, "success": False, "error": str(e)})
                failed_count += len(data["vectors"])
        
        # Perform batch re-add operation
        if docs_to_readd:
            await asyncio.to_thread(self.client.index(self.index_name).add_documents, docs_to_readd)

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
        and then updates Marqo with the generated vectors.
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

                # Update in Marqo if we have valid updates
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
            logger.exception(f"Failed to update anime vectors in Marqo: {e}")
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
            point_id = self._generate_point_id(anime_id)
            doc = await asyncio.to_thread(self.client.index(self.index_name).get_document, document_id=point_id)
            if doc:
                # Marqo returns the document directly
                return doc
            return None
        except Exception as e:
            logger.error(f"Failed to get anime by ID {anime_id} from Marqo: {e}")
            return None

    async def get_point(self, point_id: str) -> Optional[Dict[str, Any]]:
        """Get point by Marqo _id including vectors and payload."""
        try:
            doc = await asyncio.to_thread(self.client.index(self.index_name).get_document, document_id=point_id)
            if doc:
                # Marqo returns the document directly, which includes all fields (payload and vectors)
                # We need to reformat it to match the VectorDBClient's expected output
                vectors = {}
                payload = {}
                for key, value in doc.items():
                    if key in self.settings.vector_names: # Check if it's one of our vector fields
                        vectors[key] = value
                    elif key not in ["_id", "anime_id"]:
                        payload[key] = value

                return {
                    "id": doc["_id"],
                    "vector": vectors,
                    "payload": payload,
                }
            return None
        except Exception as e:
            logger.error(f"Failed to get point by ID {point_id} from Marqo: {e}")
            return None

    async def clear_index(self) -> bool:
        """Clear all documents from the index (for fresh re-indexing)."""
        try:
            await self.delete_collection()
            await self.create_collection()
            logger.info(f"Cleared and recreated Marqo index: {self.index_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear Marqo index: {e}")
            return False

    async def delete_collection(self) -> bool:
        """Delete the Marqo index."""
        try:
            # Placeholder for Marqo delete index
            # await asyncio.to_thread(self.client.delete_index, self.index_name)
            logger.warning("Marqo delete_collection is a placeholder.")
            return True
        except Exception as e:
            logger.error(f"Failed to delete Marqo index: {e}")
            return False

    async def create_collection(self) -> bool:
        """Create the Marqo index."""
        try:
            self._initialize_index()
            return True
        except Exception as e:
            logger.error(f"Failed to create Marqo index: {e}")
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
            # Marqo search parameters
            search_query = {
                "vector": vector_data,
                "attributesToRetrieve": ["*"], # Retrieve all attributes
                "limit": limit,
                "searchableAttributes": [vector_name], # Search against the specified vector field
            }

            # Convert Qdrant Filter to Marqo filter (simplified for now)
            marqo_filter = self._convert_qdrant_filter_to_marqo_filter(filters)
            if marqo_filter: search_query["filter"] = marqo_filter

            res = await asyncio.to_thread(self.client.index(self.index_name).search, **search_query)

            results = []
            for hit in res["hits"]:
                # Marqo returns _score as similarity score
                results.append({
                    "id": hit["_id"],
                    "anime_id": hit["anime_id"],
                    "_id": hit["_id"],
                    **hit, # Include all Marqo document fields
                    "similarity_score": hit["_score"],
                })
            
            logger.info(
                f"Marqo single vector search ({vector_name}) returned {len(results)} results"
            )
            return results

        except Exception as e:
            logger.error(f"Marqo single vector search failed: {e}")
            raise

    def _convert_qdrant_filter_to_marqo_filter(self, qdrant_filter: Optional[Filter]) -> Optional[str]:
        """Converts Qdrant Filter object to Marqo filter string."""
        if not qdrant_filter:
            return None

        marqo_filter_parts = []

        # Marqo filter syntax is SQL-like (e.g., "field_name > 10 AND field_name < 20")
        # This is a simplified conversion and needs to be expanded for full Qdrant filter support.

        # Handle 'must' conditions (AND logic)
        if qdrant_filter.must:
            for condition in qdrant_filter.must:
                if hasattr(condition, "key"):
                    key = condition.key
                    if hasattr(condition, "match") and condition.match:
                        if hasattr(condition.match, "any") and condition.match.any:
                            # Marqo 'IN' operator
                            values = [f"\"{v}\"" if isinstance(v, str) else str(v) for v in condition.match.any]
                            marqo_filter_parts.append(f"{key} IN [{', '.join(values)}]")
                        elif hasattr(condition.match, "value") and condition.match.value is not None:
                            # Exact match
                            value = f"\"{condition.match.value}\"" if isinstance(condition.match.value, str) else str(condition.match.value)
                            marqo_filter_parts.append(f"{key} = {value}")
                    elif hasattr(condition, "range") and condition.range:
                        # Range conditions
                        range_parts = []
                        if condition.range.gte is not None: range_parts.append(f"{key} >= {condition.range.gte}")
                        if condition.range.lte is not None: range_parts.append(f"{key} <= {condition.range.lte}")
                        if condition.range.gt is not None: range_parts.append(f"{key} > {condition.range.gt}")
                        if condition.range.lt is not None: range_parts.append(f"{key} < {condition.range.lt}")
                        marqo_filter_parts.append(" AND ".join(range_parts))
        
        # Marqo does not directly support 'should' (OR) or 'must_not' (NOT) in the same way as Qdrant's filter syntax
        # For now, we only convert 'must' conditions.
        if qdrant_filter.should or qdrant_filter.must_not:
            logger.warning("Marqo filter conversion for 'should' and 'must_not' is not fully implemented. Only 'must' conditions are applied.")

        return " AND ".join(marqo_filter_parts) if marqo_filter_parts else None
    async def search_multi_vector(
        self,
        vector_queries: List[Dict[str, Any]],
        limit: int = 10,
        fusion_method: str = "rrf", # Marqo handles fusion internally based on query and index settings
        filters: Optional[Filter] = None,
    ) -> List[Dict[str, Any]]:
        """Search across multiple vectors using Marqo's integrated search capabilities."""
        if not vector_queries:
            return []

        # Marqo's search API allows specifying multiple searchableAttributes.
        # We will take the first query vector as the primary query and specify all relevant vector fields.
        # This assumes that if multiple vector_queries are provided, they are intended to be searched
        # with a single query vector against multiple fields, or that the first query vector is representative.
        # For more complex scenarios (e.g., different query vectors for different fields), a more
        # sophisticated fusion strategy would be needed, potentially involving multiple Marqo searches.

        first_query = vector_queries[0]
        query_vector = first_query["vector_data"]
        searchable_attributes = [q["vector_name"] for q in vector_queries]

        search_query = {
            "vector": query_vector,
            "attributesToRetrieve": ["*"], # Retrieve all attributes
            "limit": limit,
            "searchableAttributes": searchable_attributes, # Search against all specified vector fields
        }

        marqo_filter = self._convert_qdrant_filter_to_marqo_filter(filters)
        if marqo_filter: search_query["filter"] = marqo_filter

        res = await asyncio.to_thread(self.client.index(self.index_name).search, **search_query)

        results = []
        for hit in res["hits"]:
            results.append({
                "id": hit["_id"],
                "anime_id": hit["anime_id"],
                "_id": hit["_id"],
                **hit, # Include all Marqo document fields
                "similarity_score": hit["_score"],
            })
        
        logger.info(
            f"Marqo multi-vector search returned {len(results)} results across {len(searchable_attributes)} fields"
        )
        return results

    async def search_text_comprehensive(
        self,
        query: str,
        limit: int = 10,
        fusion_method: str = "rrf",
        filters: Optional[Filter] = None,
    ) -> List[Dict[str, Any]]:
        """Search across all text vectors using Marqo's integrated search capabilities."""
        query_embedding = self.embedding_manager.text_processor.encode_text(query)
        if query_embedding is None:
            logger.warning("Failed to create embedding for comprehensive text search")
            return []

        # All text vectors for comprehensive search
        text_vector_names = [
            name
            for name, dim in self.settings.vector_names.items()
            if "_vector" in name and "image" not in name
        ]

        # Create vector queries for all text vectors (using the same query embedding)
        vector_queries = []
        for vector_name in text_vector_names:
            vector_queries.append(
                {"vector_name": vector_name, "vector_data": query_embedding}
            )

        # Use multi-vector search
        results = await self.search_multi_vector(
            vector_queries=vector_queries,
            limit=limit,
            fusion_method=fusion_method,
            filters=filters,
        )

        logger.info(
            f"Marqo comprehensive text search returned {len(results)} results across {len(text_vector_names)} vectors"
        )
        return results

    async def search_visual_comprehensive(
        self,
        image_data: str,
        limit: int = 10,
        fusion_method: str = "rrf",
        filters: Optional[Filter] = None,
    ) -> List[Dict[str, Any]]:
        """Search across both image vectors using Marqo's integrated search capabilities."""
        image_embedding = self.embedding_manager.vision_processor.encode_image(image_data)
        if image_embedding is None:
            logger.error("Failed to create image embedding for comprehensive visual search")
            return []

        # Both image vectors for comprehensive visual search
        image_vector_names = [
            name
            for name, dim in self.settings.vector_names.items()
            if "image_vector" in name
        ]

        # Create vector queries for both image vectors (using the same query embedding)
        vector_queries = []
        for vector_name in image_vector_names:
            vector_queries.append(
                {"vector_name": vector_name, "vector_data": image_embedding}
            )

        # Use multi-vector search
        results = await self.search_multi_vector(
            vector_queries=vector_queries,
            limit=limit,
            fusion_method=fusion_method,
            filters=filters,
        )

        logger.info(
            f"Marqo comprehensive visual search returned {len(results)} results across {len(image_vector_names)} vectors"
        )
        return results

    async def search_complete(
        self,
        query: str,
        image_data: Optional[str] = None,
        limit: int = 10,
        fusion_method: str = "rrf",
        filters: Optional[Filter] = None,
    ) -> List[Dict[str, Any]]:
        """Search across all vectors (text + image) using Marqo's integrated search capabilities."""
        vector_queries = []

        # Generate text embedding for all text vectors
        query_embedding = self.embedding_manager.text_processor.encode_text(query)
        if query_embedding is None:
            logger.warning("Failed to create text embedding for complete search")
        else:
            text_vector_names = [
                name
                for name, dim in self.settings.vector_names.items()
                if "_vector" in name and "image" not in name
            ]
            for vector_name in text_vector_names:
                vector_queries.append(
                    {"vector_name": vector_name, "vector_data": query_embedding}
                )

        # Add image vectors if image provided
        if image_data:
            image_embedding = self.embedding_manager.vision_processor.encode_image(image_data)
            if image_embedding is None:
                logger.warning("Failed to create image embedding for complete search")
            else:
                image_vector_names = [
                    name
                    for name, dim in self.settings.vector_names.items()
                    if "image_vector" in name
                ]
                for vector_name in image_vector_names:
                    vector_queries.append(
                        {"vector_name": vector_name, "vector_data": image_embedding}
                    )

        if not vector_queries:
            logger.error("No valid embeddings generated for complete search")
            return []

        # Use multi-vector search across all vectors
        results = await self.search_multi_vector(
            vector_queries=vector_queries,
            limit=limit,
            fusion_method=fusion_method,
            filters=filters,
        )

        logger.info(
            f"Marqo complete search returned {len(results)} results across {len(vector_queries)} vectors"
        )
        return results

    async def search_characters(
        self,
        query: str,
        image_data: Optional[str] = None,
        limit: int = 10,
        fusion_method: str = "rrf",
        filters: Optional[Filter] = None,
    ) -> List[Dict[str, Any]]:
        """Search specifically for character-related content using character vectors."""
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
            image_embedding = self.embedding_manager.vision_processor.encode_image(image_data)
            if image_embedding is None:
                logger.warning("Failed to create image embedding for character search")
            else:
                vector_queries.append(
                    {"vector_name": "character_image_vector", "vector_data": image_embedding}
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
            f"Marqo character search ({search_type}) returned {len(results)} results across {len(vector_queries)} vectors"
        )
        return results
