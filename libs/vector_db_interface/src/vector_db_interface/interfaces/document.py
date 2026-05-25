"""Document read and write interfaces."""

from abc import ABC, abstractmethod
from typing import Any

from vector_db_interface.types import VectorDocument


class DocumentWriter(ABC):
    """Owns document upsert and batch update operations."""

    @abstractmethod
    async def add_documents(
        self,
        documents: list[VectorDocument],
        batch_size: int = 100,
    ) -> Any:
        """Upsert documents to the collection in batches."""
        ...

    @abstractmethod
    async def update_vectors(
        self,
        updates: list[Any],
        dedup_policy: str = "last-wins",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> Any:
        """Batch update vectors for existing points."""
        ...

    @abstractmethod
    async def update_payload(
        self,
        updates: list[Any],
        mode: str = "merge",
        dedup_policy: str = "last-wins",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> Any:
        """Batch update payload for existing points."""
        ...


class DocumentReader(ABC):
    """Owns point retrieval and scroll operations."""

    @abstractmethod
    async def get_by_id(
        self,
        point_id: str,
        with_vectors: bool = False,
    ) -> dict[str, Any] | None:
        """Retrieve a document by its ID."""
        ...
