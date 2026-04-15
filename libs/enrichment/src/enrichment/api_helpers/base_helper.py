from abc import ABC, abstractmethod
from typing import Any


class BaseEnrichmentHelper(ABC):
    """Abstract base class for all anime enrichment service helpers."""

    @abstractmethod
    async def fetch_all(
        self,
        ids: dict[str, str],
        offline_data: dict[str, Any],
        temp_dir: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Fetch all available data for a specific anime from the service.

        Args:
            ids: Dictionary of validated platform IDs/URLs.
            offline_data: The original offline anime metadata.
            temp_dir: Optional directory for intermediate JSONL storage.

        Returns:
            Service-specific data dictionary or None if fetch failed.
        """
        pass

    async def close(self) -> None:
        """Close any open sessions or resources."""
        pass

    async def __aenter__(self) -> "BaseEnrichmentHelper":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        await self.close()
        return False
