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

        # Marqo connection details (will be configurable in settings later)
        self.url = url or "http://localhost:8882" # settings.marqo_url
        self.index_name = collection_name or "anime_marqo_index" # settings.marqo_index_name

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
                        "model": self.embedding_manager.text_processor.model_name, # Use our text model as a hint
                        "treat_urls_and_pointers_as_images": True, # Enable Marqo to embed images from URLs
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

                    # Marqo can embed text and image content directly
                    # We need to map our 11 vectors to Marqo-friendly fields.
                    # For text vectors, we can combine relevant text content.
                    # For image vectors, we can provide image URLs.

                    # Example: Combine text content for a general text field
                    marqo_doc["combined_text_content"] = " ".join(
                        str(doc_data["vectors"].get(vec_name, ""))
                        for vec_name in self.settings.vector_names.keys()
                        if "_vector" in vec_name and "image" not in vec_name
                    )

                    # Example: Provide image URLs for Marqo to embed
                    # Assuming image_vector and character_image_vector are derived from URLs
                    # This requires the original image URLs to be available in the payload
                    if "image_vector" in doc_data["vectors"] and doc_data["payload"].get("image_url"):
                        marqo_doc["image_url_field"] = doc_data["payload"]["image_url"]
                    
                    # Add all pre-embedded vectors as separate fields if Marqo supports it
                    # Marqo can also take pre-computed vectors if the index is configured for it.
                    # For now, we'll rely on Marqo's internal embedding for simplicity.
                    # If we want to use our pre-computed 11 vectors, we'd need to configure Marqo
                    # to accept external vectors for specific fields.

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
        """Update a single named vector for an existing anime point."""
        logger.warning("Marqo update_single_vector is a placeholder. Marqo updates documents by re-adding them.")
        # Marqo updates documents by re-adding them with the same _id
        # This needs to fetch the existing document, update the relevant field, and re-add.
        return False

    async def update_batch_vectors(
        self,
        updates: List[Dict[str, Any]],
        dedup_policy: str = "last-wins",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> Dict[str, Any]:
        """Update multiple vectors across multiple anime points in a single batch."""
        logger.warning("Marqo update_batch_vectors is a placeholder.")
        return {"success": 0, "failed": len(updates), "results": [], "duplicates_removed": 0}

    async def update_anime_vectors(
        self,
        anime_entries: List[AnimeEntry],
        vector_names: Optional[List[str]] = None,
        batch_size: int = 100,
        progress_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Generate and update vectors for anime entries with automatic batching."""
        logger.warning("Marqo update_anime_vectors is a placeholder.")
        return {"success": 0, "failed": len(anime_entries), "results": [], "generation_failures": 0}

    async def get_by_id(self, anime_id: str) -> Optional[Dict[str, Any]]:
        """Get anime by ID."""
        try:
            # Placeholder for Marqo get_by_id
            # doc = await asyncio.to_thread(self.client.index(self.index_name).get_document, self._generate_point_id(anime_id))
            logger.warning("Marqo get_by_id is a placeholder.")
            return None
        except Exception as e:
            logger.error(f"Failed to get anime by ID {anime_id} from Marqo: {e}")
            return None

    async def get_point(self, point_id: str) -> Optional[Dict[str, Any]]:
        """Get point by Marqo _id including vectors and payload."""
        try:
            # Placeholder for Marqo get_point
            # doc = await asyncio.to_thread(self.client.index(self.index_name).get_document, point_id)
            logger.warning("Marqo get_point is a placeholder.")
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
            # Placeholder for Marqo search
            # results = await asyncio.to_thread(self.client.index(self.index_name).search,
            #                                   q=vector_data, searchable_attributes=[vector_name], limit=limit)
            logger.warning("Marqo search_single_vector is a placeholder.")
            return []
        except Exception as e:
            logger.error(f"Marqo search_single_vector failed: {e}")
            raise

    async def search_multi_vector(
        self,
        vector_queries: List[Dict[str, Any]],
        limit: int = 10,
        fusion_method: str = "rrf",
        filters: Optional[Filter] = None,
    ) -> List[Dict[str, Any]]:
        """Search across multiple vectors using Marqo (simulated fusion)."""
        logger.warning("Marqo search_multi_vector is a placeholder.")
        return []

    async def search_text_comprehensive(
        self,
        query: str,
        limit: int = 10,
        fusion_method: str = "rrf",
        filters: Optional[Filter] = None,
    ) -> List[Dict[str, Any]]:
        """Search across all text vectors using Marqo (simulated fusion)."""
        logger.warning("Marqo search_text_comprehensive is a placeholder.")
        return []

    async def search_visual_comprehensive(
        self,
        image_data: str,
        limit: int = 10,
        fusion_method: str = "rrf",
        filters: Optional[Filter] = None,
    ) -> List[Dict[str, Any]]:
        """Search across both image vectors using Marqo (simulated fusion)."""
        logger.warning("Marqo search_visual_comprehensive is a placeholder.")
        return []

    async def search_complete(
        self,
        query: str,
        image_data: Optional[str] = None,
        limit: int = 10,
        fusion_method: str = "rrf",
        filters: Optional[Filter] = None,
    ) -> List[Dict[str, Any]]:
        """Search across all vectors (text + image) using Marqo (simulated fusion)."""
        logger.warning("Marqo search_complete is a placeholder.")
        return []

    async def search_characters(
        self,
        query: str,
        image_data: Optional[str] = None,
        limit: int = 10,
        fusion_method: str = "rrf",
        filters: Optional[Filter] = None,
    ) -> List[Dict[str, Any]]:
        """Search specifically for character-related content using character vectors."""
        logger.warning("Marqo search_characters is a placeholder.")
        return []
