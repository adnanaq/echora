"""Vector search interface."""

from abc import ABC, abstractmethod
from typing import Any

from vector_db_interface.types import SearchHit


class VectorSearcher(ABC):
    """Owns vector search operations."""

    @abstractmethod
    async def search(self, request: Any) -> list[SearchHit]:
        """Run vector search with a strict request contract."""
        ...
