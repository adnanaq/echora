"""Vector search interface."""

from abc import ABC, abstractmethod
from typing import Any


class VectorSearcher(ABC):
    """Owns vector search operations."""

    @abstractmethod
    async def search(self, request: Any) -> list[Any]:
        """Run vector search with a strict request contract."""
        ...
