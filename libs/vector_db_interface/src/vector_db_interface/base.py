"""Abstract base class for vector database clients."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class VectorDocument:
    """Provider-agnostic representation of a document with vectors.

    Attributes:
        id: Unique identifier for the document
        vectors: Named vectors for multi-vector search (e.g., {"text": [...], "image": [...]})
        payload: Metadata and searchable fields
    """
    id: str
    vectors: Dict[str, List[float]]
    payload: Dict[str, Any]


class VectorDBClient(ABC):
    """Abstract base class for vector database operations.

    All vector database clients must implement this interface to ensure
    consistent behavior across different vector database providers.
    """

    # ==================== Collection Management ====================

    @abstractmethod
    async def create_collection(self) -> bool:
        """Create a new collection with configuration from settings."""
        pass

    @abstractmethod
    async def delete_collection(self) -> bool:
        """Delete the collection."""
        pass

    @abstractmethod
    async def collection_exists(self) -> bool:
        """Check if the collection exists."""
        pass

    # ==================== Document Operations ====================

    @abstractmethod
    async def add_documents(
        self,
        documents: List[VectorDocument],
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

    # ==================== Health & Statistics ====================

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the database connection is healthy."""
        pass

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        pass
