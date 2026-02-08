from __future__ import annotations

"""High-level MAL enrichment helper (via MAL API).

This module orchestrates sequential requests across multiple MAL endpoints for a
single anime (full payload, episodes, characters). Requests are made sequentially to
respect the shared quota across endpoints. A shared, pre-request limiter is applied
at the HTTP client layer.
"""

from typing import Any

from common.models.mal_models import (
    MalAnimeCharacterEntry,
    MalAnimeEpisode,
    MalAnimeFull,
    MalCharacterFull,
)
from http_cache.instance import http_cache_manager as _cache_manager

from enrichment.api_helpers.mal_client import MalClient
from enrichment.api_helpers.mal_rate_limiter import MalRateLimiter


class MalEnrichmentHelper:
    """Fetch and map MAL API payloads (via MAL) for an anime.

    Responsibilities:
    - Own or borrow an HTTP session.
    - Call MAL endpoints sequentially.
    - Map raw JSON to validated Mal Pydantic models.

    Args:
        anime_id: MyAnimeList anime id (stringified).
        session: Optional aiohttp-style session to reuse connection pooling/caching.
        limiter: Optional limiter override. Defaults to the shared limiter used by
            `MalClient`.
    """

    def __init__(
        self,
        anime_id: str,
        *,
        session: Any | None = None,
        limiter: MalRateLimiter | Any | None = None,
    ) -> None:
        self.anime_id = anime_id
        self._owns_session = session is None
        self._session = session or _cache_manager.get_aiohttp_session("mal")
        self._client = MalClient(session=self._session, limiter=limiter)

    async def close(self) -> None:
        """Close the underlying HTTP session if this helper created it."""
        if self._owns_session and self._session:
            await self._session.close()

    async def __aenter__(self) -> MalEnrichmentHelper:
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit async context manager and close owned resources."""
        await self.close()
        return False

    async def fetch_anime(self) -> dict[str, Any] | None:
        """Fetch the `/anime/{id}/full` payload.

        Returns:
            A dict of anime fields validated against MalAnimeFull, or None on failure.
        """
        url = f"https://api.jikan.moe/v4/anime/{self.anime_id}/full"
        payload = await self._client.get_json(url)
        if payload and isinstance(payload.get("data"), dict):
            # Validate with Pydantic and return as JSON-serializable dict
            model = MalAnimeFull.model_validate(payload["data"])
            return model.model_dump(mode="json")
        return None

    async def fetch_characters_basic(self) -> list[dict[str, Any]]:
        """Fetch the basic character list for the anime.

        Returns:
            A list of validated character entry dicts. Empty on failure.
        """
        url = f"https://api.jikan.moe/v4/anime/{self.anime_id}/characters"
        payload = await self._client.get_json(url)
        data = payload.get("data") if payload else None
        if isinstance(data, list):
            return [
                MalAnimeCharacterEntry.model_validate(item).model_dump() for item in data
            ]
        return []

    async def fetch_episode_detail(self, episode_id: int) -> dict[str, Any] | None:
        """Fetch a single episode detail payload.

        Args:
            episode_id: 1-based episode number to fetch.

        Returns:
            A mapped episode dict or None if the request failed.
        """
        url = f"https://api.jikan.moe/v4/anime/{self.anime_id}/episodes/{episode_id}"
        payload = await self._client.get_json(url)
        if not payload or "data" not in payload:
            return None
        ep_data = payload["data"] if isinstance(payload["data"], dict) else {}
        if ep_data:
            model = MalAnimeEpisode.model_validate(ep_data)
            result = model.model_dump(mode="json")
            # Maintain episode_number for pipeline compatibility
            result["episode_number"] = episode_id
            return result
        return None

    async def fetch_episodes(self, episode_count: int) -> list[dict[str, Any]]:
        """Fetch episode details sequentially for episode numbers 1..episode_count.

        Args:
            episode_count: Number of episodes to fetch.

        Returns:
            A list of mapped episode dicts (failed episodes are skipped).
        """
        episodes: list[dict[str, Any]] = []
        for eid in range(1, episode_count + 1):
            item = await self.fetch_episode_detail(eid)
            if item:
                episodes.append(item)
        return episodes

    async def fetch_character_detail(
        self, character_basic: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Fetch detailed character data for one basic character entry.

        Args:
            character_basic: One element from `fetch_characters_basic()`.

        Returns:
            The full character data from /characters/{id}/full as a dictionary.
        """
        character = character_basic.get("character") or {}
        character_id = character.get("mal_id")
        if not isinstance(character_id, int):
            return None

        # Fetch the FULL character payload directly
        url = f"https://api.jikan.moe/v4/characters/{character_id}/full"
        payload = await self._client.get_json(url)
        if not payload or "data" not in payload:
            return None
        char_data = payload["data"] if isinstance(payload["data"], dict) else {}
        if char_data:
            # Validate with MalCharacterFull and return as JSON-serializable dict
            model = MalCharacterFull.model_validate(char_data)
            return model.model_dump(mode="json")
        return None

    async def fetch_characters_detailed(
        self, characters_basic: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Fetch detailed character data sequentially for a list of basic entries.

        Args:
            characters_basic: Output of `fetch_characters_basic()`.

        Returns:
            A list of mapped character dicts (failed characters are skipped).
        """
        characters: list[dict[str, Any]] = []
        for item in characters_basic:
            detail = await self.fetch_character_detail(item)
            if detail:
                characters.append(detail)
        return characters

