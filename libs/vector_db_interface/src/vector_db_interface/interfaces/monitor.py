"""Collection monitoring and health interface."""

from abc import ABC, abstractmethod
from typing import Any


class CollectionMonitor(ABC):
    """Owns health checks and statistics."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the database connection is healthy."""
        ...

    @abstractmethod
    async def get_stats(self) -> dict[str, Any]:
        """Get database statistics."""
        ...
