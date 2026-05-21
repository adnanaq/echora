"""Collection lifecycle interface."""

from abc import ABC, abstractmethod


class CollectionManager(ABC):
    """Owns collection create/delete/existence checks."""

    @abstractmethod
    async def create_collection(self) -> bool:
        """Create a new collection with configuration from settings."""
        ...

    @abstractmethod
    async def delete_collection(self) -> bool:
        """Delete the collection."""
        ...

    @abstractmethod
    async def collection_exists(self) -> bool:
        """Check if the collection exists."""
        ...
