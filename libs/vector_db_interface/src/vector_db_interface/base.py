"""Abstract base class for vector database clients."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client.models import PointStruct, ScoredPoint


class VectorDBClient(ABC):
    """Abstract base class for vector database operations.

    All vector database clients must implement this interface to ensure
    consistent behavior across different vector database providers.
    """

    # ==================== Collection Management ====================

    @abstractmethod
    async def create_collection(
        self,
        collection_name: str,
        vector_config: Dict[str, Any],
        **kwargs: Any,
    ) -> bool:
        """Create a new collection with the specified vector configuration."""
        pass

    @abstractmethod
    async def delete_collection(self, collection_name: str) -> bool:
        """Delete an existing collection."""
        pass

    @abstractmethod
    async def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        pass

    # ==================== Document Operations ====================

    @abstractmethod
    async def add_documents(
        self,
        documents: List[PointStruct],
        batch_size: int = 100,
    ) -> Dict[str, Any]:
        """Add documents to the collection in batches."""
        pass

    @abstractmethod
    async def get_by_id(
        self,
        point_id: str,
        with_vectors: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a document by its ID."""
        pass

    # ==================== Vector Update Operations ====================

    @abstractmethod
    async def update_single_vector(
        self,
        point_id: str,
        vector_name: str,
        vector_data: List[float],
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> bool:
        """Update a single named vector for an existing point."""
        pass

    @abstractmethod
    async def update_batch_vectors(
        self,
        updates: List[Dict[str, Any]],
        dedup_policy: str = "last-wins",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> Dict[str, Any]:
        """Update multiple vectors across multiple points in a batch."""
        pass

    # ==================== Search Operations ====================

    @abstractmethod
    async def search_single_vector(
        self,
        vector_name: str,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[ScoredPoint]:
        """Search using a single vector."""
        pass

    @abstractmethod
    async def search_multi_vector(
        self,
        vector_queries: List[Dict[str, Any]],
        fusion_algorithm: str = "rrf",
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[ScoredPoint, Dict[str, float]]]:
        """Search using multiple vectors with score fusion."""
        pass

    # ==================== Health & Statistics ====================

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the database connection is healthy."""
        pass

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        pass
