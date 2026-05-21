from abc import ABC, abstractmethod
from collections.abc import Mapping
from copy import deepcopy
from types import TracebackType
from typing import Any


def normalize_enrichment_payload(
    data: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return helper results in a consistent normalized payload shape."""
    if data is None:
        return None

    if any(key in data for key in ("anime", "episodes", "characters", "extras")):
        anime = data.get("anime")
        episodes = data.get("episodes")
        characters = data.get("characters")
        extras = data.get("extras")

        if anime is None:
            anime = deepcopy(data)
            anime.pop("episodes", None)
            anime.pop("characters", None)
            anime.pop("extras", None)

        if anime is not None and not isinstance(anime, dict):
            anime = None
        if not isinstance(episodes, list):
            episodes = []
        if not isinstance(characters, list):
            characters = []
        if not isinstance(extras, Mapping):
            extras = {}

        return {
            "anime": deepcopy(anime) if anime is not None else None,
            "episodes": episodes,
            "characters": characters,
            "extras": dict(extras),
        }

    episodes = data.get("episodes")
    characters = data.get("characters")
    extras = data.get("extras")

    anime = deepcopy(data)
    anime.pop("episodes", None)
    anime.pop("characters", None)
    anime.pop("extras", None)

    if not isinstance(episodes, list):
        episodes = []
    if not isinstance(characters, list):
        characters = []
    if not isinstance(extras, Mapping):
        extras = {}

    return {
        "anime": anime or None,
        "episodes": episodes,
        "characters": characters,
        "extras": dict(extras),
    }


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

        Most implementations derive their lookup key from ``ids`` (e.g. ``ids["anilist_url"]``).
        Search-based services with no platform ID (e.g. AnimSchedule) may instead use
        ``offline_data["title"]`` and validate against ``offline_data["sources"]``.

        Args:
            ids: Dictionary of validated platform IDs/URLs.
            offline_data: The original offline anime metadata.
            temp_dir: Optional directory for intermediate JSONL storage.

        Returns:
            Normalized payload with ``anime``, ``episodes``, ``characters``,
            and ``extras`` keys, or None if fetch failed.
        """
        pass

    async def close(self) -> None:
        """Close any open sessions or resources."""
        pass

    async def __aenter__(self) -> "BaseEnrichmentHelper":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException],
        exc_val: BaseException,
        exc_tb: TracebackType,
    ) -> bool:
        await self.close()
        return False
