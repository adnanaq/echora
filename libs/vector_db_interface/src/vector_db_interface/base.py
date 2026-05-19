"""Abstract base class for vector database clients."""

from abc import abstractmethod

from vector_db_interface.interfaces.collection import CollectionManager
from vector_db_interface.interfaces.document import DocumentReader, DocumentWriter
from vector_db_interface.interfaces.monitor import CollectionMonitor
from vector_db_interface.interfaces.search import VectorSearcher
from vector_db_interface.types import SparseVectorData, VectorDocument

__all__ = ["VectorDBClient", "VectorDocument", "SparseVectorData"]


class VectorDBClient(CollectionManager, DocumentWriter, DocumentReader, VectorSearcher, CollectionMonitor):
    """Composite ABC for vector database operations.

    Composes all focused interfaces. Callers that only need a subset
    can depend on the individual ABCs (CollectionManager, VectorSearcher, etc.)
    instead of this full suite.
    """

    # ==================== Connection & Configuration ====================

    @property
    @abstractmethod
    def collection_name(self) -> str:
        """Name of the active collection/index."""
        ...

    @property
    @abstractmethod
    def connection_url(self) -> str:
        """Database connection URL."""
        ...

    @property
    @abstractmethod
    def vector_size(self) -> int:
        """Primary (text) vector dimension."""
        ...

    @property
    @abstractmethod
    def image_vector_size(self) -> int:
        """Image vector dimension."""
        ...

    @property
    @abstractmethod
    def distance_metric(self) -> str:
        """Distance metric for similarity (cosine, euclid, dot)."""
        ...
