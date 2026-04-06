#!/usr/bin/env python3
"""Kitsu data fetcher for the AI enrichment pipeline."""

import asyncio
import logging
import os
import sys
from contextlib import nullcontext
from types import TracebackType
from typing import Any, cast

import aiohttp
from common.models.anime import ThemeEntry
from common.utils.jsonl_utils import append_jsonl as _append_jsonl
from common.utils.jsonl_utils import write_jsonl as _write_jsonl
from common.utils.retry import retry_with_backoff
from http_cache.instance import http_cache_manager as _cache_manager

from ..exceptions import ServiceNotFoundError
from ..mappers.kitsu_mapper import (
    anime_from_kitsu,
    character_from_kitsu,
    episode_from_kitsu,
)
from .kitsu.kitsu_models import (
    KitsuAnime,
    KitsuAnimeographyEntry,
    KitsuCategory,
    KitsuCharacter,
    KitsuCharacterVoice,
    KitsuEpisode,
    KitsuGenre,
    KitsuMediaCharacter,
    KitsuPerson,
)

logger = logging.getLogger(__name__)


class KitsuEnrichmentHelper:
    """Helper for Kitsu data fetching in AI enrichment pipeline."""

    def __init__(self) -> None:
        self.base_url = "https://kitsu.io/api/edge"

    async def _make_request(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        *,
        session: Any | None = None,
    ) -> dict[str, Any]:
        """Perform a GET request to a Kitsu API endpoint.

        Args:
            endpoint: API path appended to base_url (e.g. ``/anime/1``).
            params: Optional query parameters.
            session: Optional aiohttp session to reuse.

        Returns:
            Parsed JSON response body including a ``_from_cache`` bool key.

        Raises:
            ServiceNotFoundError: If the server returns 404.
            aiohttp.ClientResponseError: For other non-200 HTTP status codes.
        """
        headers = {
            "Accept": "application/vnd.api+json",
            "Content-Type": "application/vnd.api+json",
        }
        url = f"{self.base_url}{endpoint}"

        async def _execute() -> dict[str, Any]:
            ctx = (
                nullcontext(session)
                if session is not None
                else _cache_manager.get_aiohttp_session(
                    "kitsu", timeout=aiohttp.ClientTimeout(total=30)
                )
            )
            async with ctx as active_session:
                async with cast(aiohttp.ClientSession, active_session).get(
                    url, headers=headers, params=params
                ) as response:
                    if response.status == 404:
                        raise ServiceNotFoundError(  # noqa: TRY003
                            f"not found: {endpoint}", service="kitsu"
                        )
                    if response.status != 200:
                        response.raise_for_status()
                    payload = cast(dict[str, Any], await response.json())
                    from_cache = bool(getattr(response, "from_cache", False))
                    payload["_from_cache"] = from_cache
                    logger.debug(f"[Kitsu] {'CACHE' if from_cache else 'LIVE '} {response.url}")
                    return payload

        return await retry_with_backoff(
            operation=_execute,
            max_retries=3,
            retry_delay=1.0,
            is_transient_error=lambda e: isinstance(e, aiohttp.ClientError)
            and not isinstance(e, aiohttp.ClientResponseError),
        )

    async def _fetch_all_pages(
        self,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        session: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Paginate a Kitsu collection endpoint, returning all items across pages.

        Args:
            endpoint: API path for a collection resource.
            params: Extra query parameters merged with pagination params.
            session: Optional aiohttp session to reuse across page requests.

        Returns:
            All resource objects from the collection, or a partial list on error.
        """
        items, _ = await self._paginate(endpoint, params=params, session=session)
        return items

    async def _paginate(
        self,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        session: Any | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Paginate a Kitsu collection endpoint, collecting items and included resources.

        Args:
            endpoint: API path for a collection resource.
            params: Extra query parameters merged with pagination params.
            session: Optional aiohttp session to reuse across page requests.

        Returns:
            Tuple of (all data items, all included items) across all pages.
        """
        all_items: list[dict[str, Any]] = []
        all_included: list[dict[str, Any]] = []
        page = 0
        page_size = 20
        base_params = dict(params or {})
        try:
            while True:
                page_params = {
                    **base_params,
                    "page[limit]": page_size,
                    "page[offset]": page * page_size,
                }
                response = await self._make_request(
                    endpoint, page_params, session=session
                )
                if not response or "data" not in response:
                    break
                items = response["data"]
                if not items:
                    break
                all_items.extend(items)
                all_included.extend(response.get("included", []))
                count = response.get("meta", {}).get("count", 0)
                if len(all_items) >= count or len(items) < page_size:
                    break
                page += 1
                if not response.get("_from_cache", False):
                    await asyncio.sleep(0.1)
        except Exception:
            logger.exception(f"Kitsu pagination failed for {endpoint}")
        return all_items, all_included

    async def get_anime_by_id(
        self, anime_id: int, *, session: Any | None = None
    ) -> dict[str, Any] | None:
        """Fetch a single anime resource by its Kitsu ID."""
        try:
            response = await self._make_request(f"/anime/{anime_id}", session=session)
            return response.get("data") if response else None
        except ServiceNotFoundError:
            logger.warning(f"Kitsu anime not found: ID {anime_id}")
            return None
        except aiohttp.ClientError:
            logger.exception(f"Kitsu get_anime_by_id failed for ID {anime_id}")
            return None

    async def get_anime_episodes(
        self, anime_id: int, *, session: Any | None = None
    ) -> list[dict[str, Any]]:
        """Fetch all episode resources for a Kitsu anime ID."""
        return await self._fetch_all_pages(
            f"/anime/{anime_id}/episodes", session=session
        )

    async def get_anime_categories(
        self, anime_id: int, *, session: Any | None = None
    ) -> list[dict[str, Any]]:
        """Fetch all category resources for a Kitsu anime ID."""
        return await self._fetch_all_pages(
            f"/anime/{anime_id}/categories", session=session
        )

    async def get_anime_genres(
        self, anime_id: int, *, session: Any | None = None
    ) -> list[dict[str, Any]]:
        """Fetch all genre resources for a Kitsu anime ID."""
        return await self._fetch_all_pages(f"/anime/{anime_id}/genres", session=session)

    async def get_anime_characters(
        self, anime_id: int, *, session: Any | None = None
    ) -> list[KitsuMediaCharacter]:
        """Fetch all mediaCharacters for a Kitsu anime ID with characters sideloaded.

        Uses ``GET /anime/{id}/characters?include=character`` — the newer endpoint
        with broader coverage than the legacy ``/anime-characters`` endpoint.

        Returns:
            List of ``KitsuMediaCharacter`` with ``.character`` populated from included[].
        """
        items, included = await self._paginate(
            f"/anime/{anime_id}/characters",
            params={"include": "character"},
            session=session,
        )
        char_map: dict[str, dict[str, Any]] = {
            i["id"]: i for i in included if i.get("type") == "characters"
        }
        media_chars: list[KitsuMediaCharacter] = []
        for item in items:
            char = KitsuMediaCharacter.model_validate(item)
            char_id = (
                item.get("relationships", {})
                .get("character", {})
                .get("data", {})
                .get("id")
            )
            if char_id and char_id in char_map:
                char.character = KitsuCharacter.model_validate(char_map[char_id])
            media_chars.append(char)
        return media_chars

    async def get_character_voices(
        self, media_char_id: str, *, session: Any | None = None
    ) -> list[KitsuCharacterVoice]:
        """Fetch all voice actor records for one mediaCharacter resource.

        Uses ``GET /media-characters/{id}/voices?include=person``.

        Returns:
            List of ``KitsuCharacterVoice`` with ``.person`` populated from included[].
        """
        items, included = await self._paginate(
            f"/media-characters/{media_char_id}/voices",
            params={"include": "person"},
            session=session,
        )
        person_map: dict[str, dict[str, Any]] = {
            i["id"]: i for i in included if i.get("type") == "people"
        }
        voices: list[KitsuCharacterVoice] = []
        for item in items:
            voice = KitsuCharacterVoice.model_validate(item)
            person_id = (
                item.get("relationships", {})
                .get("person", {})
                .get("data", {})
                .get("id")
            )
            if person_id and person_id in person_map:
                voice.person = KitsuPerson.model_validate(person_map[person_id])
            voices.append(voice)
        return voices

    async def get_character_animeography(
        self, char_id: str, *, session: Any | None = None
    ) -> list[KitsuAnimeographyEntry]:
        """Fetch all media appearances for a Kitsu character ID.

        Uses ``GET /characters/{id}/media-characters?include=media``.

        Returns:
            List of ``KitsuAnimeographyEntry`` with ``.media`` and ``.media_type`` populated.
        """
        items, included = await self._paginate(
            f"/characters/{char_id}/media-characters",
            params={"include": "media"},
            session=session,
        )
        media_map: dict[str, dict[str, Any]] = {i["id"]: i for i in included}
        entries: list[KitsuAnimeographyEntry] = []
        for item in items:
            entry = KitsuAnimeographyEntry.model_validate(item)
            rel = item.get("relationships", {}).get("media", {}).get("data", {})
            media_id = rel.get("id")
            media_type = rel.get("type")
            entry.media_type = media_type
            if media_id and media_id in media_map:
                entry.media = KitsuAnime.model_validate(media_map[media_id])
            entries.append(entry)
        return entries

    async def fetch_anime(
        self,
        anime_id: int,
        *,
        output_path: str | None = None,
        session: Any | None = None,
    ) -> dict[str, Any] | None:
        """Fetch and map a Kitsu anime to the canonical Anime dict.

        Args:
            anime_id: Kitsu integer anime identifier.
            output_path: If provided, write result as JSONL to this path.
            session: Optional aiohttp session to reuse.

        Returns:
            Canonical anime dict, or ``None`` if the anime was not found.
        """
        logger.info(f"[Kitsu] Fetching anime ID {anime_id} …")
        async with (
            _cache_manager.get_aiohttp_session(
                "kitsu", timeout=aiohttp.ClientTimeout(total=30)
            )
            if session is None
            else nullcontext(session)
        ) as active_session:  # type: ignore[attr-defined]
            anime_raw, genre_raw, category_raw = await asyncio.gather(
                self.get_anime_by_id(anime_id, session=active_session),
                self.get_anime_genres(anime_id, session=active_session),
                self.get_anime_categories(anime_id, session=active_session),
                return_exceptions=True,
            )

        if not anime_raw or isinstance(anime_raw, Exception):
            return None

        genres: list[str] = []
        if not isinstance(genre_raw, Exception):
            genres = [
                name
                for g in genre_raw  # type: ignore[union-attr]
                if (name := KitsuGenre.model_validate(g).attributes.name) is not None
            ]

        themes: list[ThemeEntry] = []
        if not isinstance(category_raw, Exception):
            for c in category_raw:  # type: ignore[union-attr]
                cat = KitsuCategory.model_validate(c)
                if cat.attributes.title:
                    themes.append(
                        ThemeEntry(
                            name=cat.attributes.title,
                            description=cat.attributes.description or None,
                        )
                    )

        anime_model = KitsuAnime.model_validate(anime_raw)  # type: ignore[arg-type]
        anime_model.genres = genres
        anime_model.themes = themes
        result = anime_from_kitsu(anime_model)
        logger.info(f"[Kitsu] Anime done: {result.get('title', anime_id)}")

        if output_path:
            _write_jsonl(output_path, [result])
        return result

    async def fetch_episodes(
        self,
        anime_id: int,
        *,
        anime_slug: str | None = None,
        output_path: str | None = None,
        session: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch and map all Kitsu episodes to canonical Episode dicts.

        Args:
            anime_id: Kitsu integer anime identifier.
            anime_slug: Kitsu anime slug used to build episode source URLs.
            output_path: If provided, write results as JSONL to this path.
            session: Optional aiohttp session to reuse.

        Returns:
            List of canonical episode dicts.
        """
        logger.info(f"[Kitsu] Fetching episodes for anime ID {anime_id} …")
        raw_episodes = await self.get_anime_episodes(anime_id, session=session)
        total = len(raw_episodes)
        results: list[dict[str, Any]] = []
        for i, ep_raw in enumerate(raw_episodes, 1):
            ep = episode_from_kitsu(
                KitsuEpisode.model_validate(ep_raw), anime_slug=anime_slug
            )
            logger.debug(f"[Kitsu] Episode {i}/{total}: {ep.get('title', '?')}")
            results.append(ep)
            if output_path:
                _append_jsonl(output_path, ep)
        return results

    async def fetch_characters(
        self,
        anime_id: int,
        *,
        output_path: str | None = None,
        session: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch and map all Kitsu characters (with voices + animeography) to canonical dicts.

        For each character:
        1. ``GET /anime/{id}/characters?include=character``
        2. Concurrently per character: voices + animeography (semaphore-limited to 5).

        Args:
            anime_id: Kitsu integer anime identifier.
            output_path: If provided, write results as JSONL to this path.
            session: Optional aiohttp session to reuse across all sub-requests.

        Returns:
            List of canonical character dicts.
        """
        logger.info(f"[Kitsu] Fetching characters for anime ID {anime_id} …")
        media_chars = await self.get_anime_characters(anime_id, session=session)
        resolved = [char for char in media_chars if char.character is not None]
        total = len(resolved)
        logger.info(f"[Kitsu] {total} character(s) to process")

        sem = asyncio.Semaphore(5)

        async def _fetch_one(char: KitsuMediaCharacter) -> dict[str, Any] | None:
            if char.character is None:
                return None
            char_id = char.character.id
            char_name = (
                char.character.attributes.canonicalName
                or char.character.attributes.name
                or char_id
            )
            async with sem:
                voices, animeography = await asyncio.gather(
                    self.get_character_voices(char.id, session=session),
                    self.get_character_animeography(char_id, session=session),
                    return_exceptions=True,
                )
            if isinstance(voices, Exception):
                logger.warning(f"Voices fetch failed for mediaChar {char.id}: {voices}")
                voices = []
            if isinstance(animeography, Exception):
                logger.warning(
                    f"Animeography fetch failed for char {char_id}: {animeography}"
                )
                animeography = []
            char.voices = voices  # type: ignore[assignment]
            char.animeography = animeography  # type: ignore[assignment]
            try:
                result = character_from_kitsu(char)
            except Exception:
                logger.exception(f"character_from_kitsu failed for mediaChar {char.id}")
                return None
            logger.debug(f"[Kitsu] Character {char_name} ({total} total)")
            if output_path:
                _append_jsonl(output_path, result)
            return result

        char_results = await asyncio.gather(*[_fetch_one(char) for char in resolved])
        return [r for r in char_results if r is not None]

    async def fetch_all(
        self,
        anime_id: int,
        output_dir: str | None = None,
    ) -> dict[str, Any] | None:
        """Fetch canonical anime, episodes, and characters from Kitsu.

        Episodes and genres/categories share a session. Characters are fetched
        separately (each requires per-character sub-requests).

        Args:
            anime_id: Kitsu integer anime identifier.
            output_dir: If provided, write JSONL files to this directory:
                ``kitsu_anime.jsonl``, ``kitsu_episodes.jsonl``, ``kitsu_characters.jsonl``.

        Returns:
            ``{"anime": dict, "episodes": list, "characters": list}`` or ``None`` on failure.
        """
        try:
            anime_path = (
                os.path.join(output_dir, "kitsu_anime.jsonl") if output_dir else None
            )
            episodes_path = (
                os.path.join(output_dir, "kitsu_episodes.jsonl") if output_dir else None
            )
            characters_path = (
                os.path.join(output_dir, "kitsu_characters.jsonl")
                if output_dir
                else None
            )

            async with _cache_manager.get_aiohttp_session(
                "kitsu", timeout=aiohttp.ClientTimeout(total=30)
            ) as session:
                # Fetch anime first so we can extract the slug for episode URLs.
                canonical_anime = await self.fetch_anime(
                    anime_id, output_path=anime_path, session=session
                )

                if not canonical_anime:
                    logger.warning(f"Kitsu returned no anime for ID {anime_id}")
                    return None

                sources = canonical_anime.get("sources", [])
                anime_slug = sources[0].rsplit("/", 1)[-1] if sources else None

                canonical_episodes, canonical_characters = await asyncio.gather(
                    self.fetch_episodes(
                        anime_id, anime_slug=anime_slug, output_path=episodes_path, session=session
                    ),
                    self.fetch_characters(anime_id, output_path=characters_path, session=session),
                    return_exceptions=True,
                )

            if isinstance(canonical_episodes, Exception):
                logger.error(
                    f"Kitsu episodes fetch failed for ID {anime_id}: {canonical_episodes}"
                )
                canonical_episodes = []
            if isinstance(canonical_characters, Exception):
                logger.error(
                    f"Kitsu characters fetch failed for ID {anime_id}: {canonical_characters}"
                )
                canonical_characters = []
        except Exception:
            logger.exception(f"Kitsu fetch_all failed for ID {anime_id}")
            return None
        else:
            return {
                "anime": canonical_anime,
                "episodes": canonical_episodes,
                "characters": canonical_characters,
            }

    async def close(self) -> None:
        """No-op — sessions are created per-request and managed by the cache manager."""

    async def __aenter__(self) -> "KitsuEnrichmentHelper":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        return False



async def main() -> int:
    """CLI entry point: fetch Kitsu anime data and write to a JSON file."""
    if len(sys.argv) != 3:
        print("Usage: python kitsu_helper.py <anime_id> <output_file>", file=sys.stderr)
        return 1

    try:
        anime_id = int(sys.argv[1])
        output_file = sys.argv[2]

        helper = KitsuEnrichmentHelper()
        data = await helper.fetch_all(anime_id)

        if data:
            import json as _json

            with open(output_file, "w", encoding="utf-8") as f:
                _json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"Data for {anime_id} saved to {output_file}")
            return 0
        else:
            print(f"Could not fetch data for {anime_id}", file=sys.stderr)
            return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    sys.exit(asyncio.run(main()))
