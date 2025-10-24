import abc
from typing import Any, Dict, List, Optional

from qdrant_client.models import Filter

from ...models.anime import AnimeEntry


class VectorDBClient(abc.ABC):
    """Abstract Base Class for Vector Database Clients.

    Defines the common interface for interacting with any vector database,
    allowing for modularity and easy switching between different providers.
    """

    @abc.abstractmethod
    def __init__(
        self,
        url: Optional[str] = None,
        collection_name: Optional[str] = None,
        settings: Optional[Any] = None,  # Use Any to avoid circular dependency with Settings
    ):
        """Initialize the vector database client."""
        pass

    @abc.abstractmethod
    async def health_check(self) -> bool:
        """Check if the vector database is healthy and reachable."""
        pass

    @abc.abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        pass

    @abc.abstractmethod
    async def add_documents(
        self, documents: List[AnimeEntry], batch_size: int = 100
    ) -> bool:
        """Add anime documents to the collection."""
        pass

    @abc.abstractmethod
    async def update_single_vector(
        self,
        anime_id: str,
        vector_name: str,
        vector_data: List[float],
    ) -> bool:
        """Update a single named vector for an existing anime point."""
        pass

    @abc.abstractmethod
    async def update_batch_vectors(
        self,
        updates: List[Dict[str, Any]],
        dedup_policy: str = "last-wins",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> Dict[str, Any]:
        """Update multiple vectors across multiple anime points in a single batch."""
        pass

    @abc.abstractmethod
    async def update_anime_vectors(
        self,
        anime_entries: List[AnimeEntry],
        vector_names: Optional[List[str]] = None,
        batch_size: int = 100,
        progress_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Generate and update vectors for anime entries with automatic batching."""
        pass

    @abc.abstractmethod
    async def get_by_id(self, anime_id: str) -> Optional[Dict[str, Any]]:
        """Get anime by ID."""
        pass

    @abc.abstractmethod
    async def get_point(self, point_id: str) -> Optional[Dict[str, Any]]:
        """Get point by Qdrant point ID including vectors and payload."""
        pass

    @abc.abstractmethod
    async def clear_index(self) -> bool:
        """Clear all points from the collection (for fresh re-indexing)."""
        pass

    @abc.abstractmethod
    async def delete_collection(self) -> bool:
        """Delete the anime collection."""
        pass

    @abc.abstractmethod
    async def create_collection(self) -> bool:
        """Create the anime collection."""
        pass

    @abc.abstractmethod
    async def search_single_vector(
        self,
        vector_name: str,
        vector_data: List[float],
        limit: int = 10,
        filters: Optional[Filter] = None,
    ) -> List[Dict[str, Any]]:
        """Search a single vector with raw similarity scores."""
        pass

    @abc.abstractmethod
    async def search_multi_vector(
        self,
        vector_queries: List[Dict[str, Any]],
        limit: int = 10,
        fusion_method: str = "rrf",
        filters: Optional[Filter] = None,
    ) -> List[Dict[str, Any]]:
        """Search across multiple vectors using native multi-vector API."""
        pass

    @abc.abstractmethod
    async def search_text_comprehensive(
        self,
        query: str,
        limit: int = 10,
        fusion_method: str = "rrf",
        filters: Optional[Filter] = None,
    ) -> List[Dict[str, Any]]:
        """Search across all text vectors using native fusion."""
        pass

    @abc.abstractmethod
    async def search_visual_comprehensive(
        self,
        image_data: str,
        limit: int = 10,
        fusion_method: str = "rrf",
        filters: Optional[Filter] = None,
    ) -> List[Dict[str, Any]]:
        """Search across both image vectors using native fusion."""
        pass

    @abc.abstractmethod
    async def search_complete(
        self,
        query: str,
        image_data: Optional[str] = None,
        limit: int = 10,
        fusion_method: str = "rrf",
        filters: Optional[Filter] = None,
    ) -> List[Dict[str, Any]]:
        """Search across all vectors (text + image) using native fusion."""
        pass

    @abc.abstractmethod
    async def search_characters(
        self,
        query: str,
        image_data: Optional[str] = None,
        limit: int = 10,
        fusion_method: str = "rrf",
        filters: Optional[Filter] = None,
    ) -> List[Dict[str, Any]]:
        """Search specifically for character-related content using character vectors."""
        pass
