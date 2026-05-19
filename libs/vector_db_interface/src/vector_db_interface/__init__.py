"""Vector database interface - Abstract base class for all vector DB implementations."""

from vector_db_interface.base import VectorDBClient
from vector_db_interface.types import SparseVectorData, VectorDocument
from vector_db_interface.interfaces import (
    CollectionManager,
    CollectionMonitor,
    DocumentReader,
    DocumentWriter,
    VectorSearcher,
)

__all__ = [
    "VectorDBClient",
    "VectorDocument",
    "SparseVectorData",
    "CollectionManager",
    "DocumentWriter",
    "DocumentReader",
    "VectorSearcher",
    "CollectionMonitor",
]
