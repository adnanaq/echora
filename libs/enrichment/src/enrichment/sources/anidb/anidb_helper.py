#!/usr/bin/env python3
"""AniDB enrichment helper — XML API for anime, episodes, and characters.

Usage:
    # Fetch anime metadata
    python -m enrichment.sources.anidb.anidb_helper anime https://anidb.net/anime/69 anidb_anime.json

    # Fetch episodes
    python -m enrichment.sources.anidb.anidb_helper episodes https://anidb.net/anime/69 anidb_episodes.json

    # Fetch characters (live JSONL, cancel after N characters)
    python -m enrichment.sources.anidb.anidb_helper characters https://anidb.net/anime/69 anidb_characters.jsonl

    # Fetch all (anime + episodes + characters)
    python -m enrichment.sources.anidb.anidb_helper all https://anidb.net/anime/69 output_dir/
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import re

import aiohttp
from common.utils.jsonl_utils import append_jsonl
from enrichment.sources.anidb.anidb_character_crawler import fetch_anidb_characters
from enrichment.sources.anidb.anidb_mapper import (
    anime_from_anidb,
    character_from_anidb,
    episode_from_anidb,
)
from enrichment.sources.anidb.anidb_models import AniDBAnime
from enrichment.sources.anidb.anidb_xml_parser import parse_anime_xml
from enrichment.sources.base.base_helper import (
    BaseEnrichmentHelper,
    normalize_enrichment_payload,
)
from enrichment.sources.base.exceptions import ServiceBlockedError, ServiceNetworkError
from enrichment.sources.base.utils import sanitize_output_path
from http_cache.instance import http_cache_manager as _cache_manager

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states for AniDB API protection."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class AniDBRequestMetrics:
    """Metrics for tracking AniDB API health and rate-limit compliance."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    consecutive_failures: int = 0
    last_request_time: float = 0
    last_error_time: float = 0
    current_interval: float = 2.0

    @property
    def success_rate(self) -> float:
        """Return successful requests as a percentage of total requests."""
        if self.total_requests == 0:
            return 100.0
        return (self.successful_requests / self.total_requests) * 100.0

    @property
    def error_rate(self) -> float:
        """Return failed requests as a percentage of total requests."""
        return 100.0 - self.success_rate


class AniDBHelper(BaseEnrichmentHelper):
    """AniDB XML API enrichment helper.

    Handles HTTP transport, adaptive rate limiting, and circuit breaking.
    Delegates XML parsing to ``anidb_xml_parser`` and field mapping to
    ``anidb_mapper``.
    """

    def __init__(
        self, client_name: str | None = None, client_version: str | None = None
    ) -> None:
        """Initialise the AniDB helper with client metadata and resilience config.

        Args:
            client_name: Client identifier sent to AniDB. Defaults to the
                ``ANIDB_CLIENT`` environment variable or ``"animeenrichment"``.
            client_version: Client version sent to AniDB. Defaults to the
                ``ANIDB_CLIENTVER`` environment variable or ``"1.0"``.
        """
        self.base_url = "http://api.anidb.net:9001/httpapi"
        self.client_name = client_name or os.getenv("ANIDB_CLIENT", "animeenrichment")
        self.client_version = client_version or os.getenv("ANIDB_CLIENTVER", "1.0")

        self.session = None
        self._session_created_at: float = 0.0
        self._session_max_age = 300

        self.min_request_interval = float(os.getenv("ANIDB_MIN_REQUEST_INTERVAL", "2.0"))
        self.max_request_interval = float(os.getenv("ANIDB_MAX_REQUEST_INTERVAL", "10.0"))
        self.error_cooldown_base = float(os.getenv("ANIDB_ERROR_COOLDOWN_BASE", "5.0"))
        self.max_retries = int(os.getenv("ANIDB_MAX_RETRIES", "3"))

        self.circuit_breaker_threshold = int(os.getenv("ANIDB_CIRCUIT_BREAKER_THRESHOLD", "5"))
        self.circuit_breaker_timeout = float(os.getenv("ANIDB_CIRCUIT_BREAKER_TIMEOUT", "300"))
        self.circuit_breaker_state = CircuitBreakerState.CLOSED
        self.circuit_breaker_opened_at = 0.0

        self.metrics = AniDBRequestMetrics()
        self._request_lock = asyncio.Lock()

    # =========================================================================
    # PUBLIC INTERFACE (BaseEnrichmentHelper contract)
    # =========================================================================

    async def fetch_all(
        self,
        ids: dict[str, str],
        offline_data: dict[str, Any],
        temp_dir: str | None = None,
        *,
        fetch_characters: bool = True,
        fetch_episodes: bool = True,
    ) -> dict[str, Any] | None:
        """Fetch all AniDB data for a single anime.

        Args:
            ids: Platform ID map. Must contain ``anidb_url``
            (e.g. ``"https://anidb.net/anime/69"``).
            offline_data: Original offline anime metadata from the seed database.
            temp_dir: Optional directory for intermediate JSONL output files.
            fetch_characters: When False, skip character fetching.
            fetch_episodes: When False, skip episode fetching.

        Returns:
            Dict with keys ``anime``, ``episodes``, and ``characters`` passed
            through ``normalize_enrichment_payload``, or None when the anime
            fetch fails.
        """
        anidb_url = ids.get("anidb_url")
        if not anidb_url:
            return None

        anime_output_path = os.path.join(temp_dir, "anidb_anime.jsonl") if temp_dir else None
        episodes_output_path = os.path.join(temp_dir, "anidb_episodes.jsonl") if temp_dir else None
        characters_output_path = os.path.join(temp_dir, "anidb_characters.jsonl") if temp_dir else None

        logger.info(f"Fetching AniDB data for: {anidb_url}")
        anime_dict, anime_model = await self._fetch_anime(anidb_url, output_path=anime_output_path)
        if not anime_dict or not anime_model:
            return None

        logger.info(f"AniDB anime fetched: {anime_dict.get('title', anidb_url)}")

        episodes_data: list[dict[str, Any]] = []
        if fetch_episodes:
            try:
                episodes_data = await self._fetch_episodes(
                    anime_model, output_path=episodes_output_path
                )
            except Exception as e:
                logger.warning(f"Episode fetch failed, continuing without episodes: {e}")

        characters_data: list[dict[str, Any]] = []
        if fetch_characters:
            try:
                characters_data = await self._fetch_characters(
                    anime_model, output_path=characters_output_path
                )
            except Exception as e:
                logger.warning(f"Character fetch failed, continuing without characters: {e}")

        logger.info(f"AniDB episodes fetched: {len(episodes_data)}")
        logger.info(f"AniDB characters fetched: {len(characters_data)}")

        return normalize_enrichment_payload({
            "anime": anime_dict,
            "episodes": episodes_data,
            "characters": characters_data,
        })

    # =========================================================================
    # PROTECTED FETCH METHODS
    # =========================================================================

    async def _fetch_anime(
        self,
        anidb_url: str,
        *,
        output_path: str | None = None,
    ) -> tuple[dict[str, Any] | None, AniDBAnime | None]:
        """Fetch XML, parse to AniDBAnime model, and map to canonical dict.

        Args:
            anidb_url: Original AniDB URL (e.g. ``"https://anidb.net/anime/69"``).
                The numeric ID is extracted from it for the API call, and the
                URL itself is passed to the mapper as the canonical source URL.
            output_path: If provided, write the canonical anime dict as a
                JSONL line to this file.

        Returns:
            Tuple of (canonical anime dict, AniDBAnime model). Both elements
            are None on fetch or parse failure.
        """
        match = re.search(r"/anime/(\d+)", anidb_url)
        if not match:
            logger.warning(f"Could not extract AniDB ID from URL: {anidb_url}")
            return None, None
        anidb_id = int(match.group(1))

        xml_content = await self._fetch_xml(anidb_id)
        if not xml_content:
            return None, None
        try:
            anime_model = parse_anime_xml(xml_content)
        except ValueError:
            logger.exception(f"XML parse failed for AniDB ID {anidb_id}")
            return None, None
        anime_dict = anime_from_anidb(anime_model, anidb_url=anidb_url)
        if output_path:
            append_jsonl(output_path, anime_dict)
        return anime_dict, anime_model

    async def _fetch_episodes(
        self,
        anime_model: AniDBAnime,
        *,
        output_path: str | None = None,
    ) -> list[dict[str, Any]]:
        """Map regular episodes from the parsed anime model (no network calls).

        Args:
            anime_model: Parsed AniDBAnime containing all episode elements.
            output_path: If provided, each mapped episode is appended as a
                JSONL line to this file.

        Returns:
            List of canonical episode dicts (regular episodes only;
            specials, credits, trailers, and parodies are excluded).
        """
        episodes = [
            ep
            for e in anime_model.episodes
            if (ep := episode_from_anidb(e)) is not None
        ]
        if output_path:
            for episode in episodes:
                append_jsonl(output_path, episode)
        return episodes

    async def _fetch_characters(
        self,
        anime_model: AniDBAnime,
        *,
        output_path: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch and map character data for the given anime.

        Args:
            anime_model: Parsed AniDBAnime containing character elements.
            output_path: If provided, each mapped character is appended as a
                JSONL line to this file.

        Returns:
            List of canonical character dicts.
        """
        xml_by_id = {c.id: c for c in anime_model.characters if c.id is not None}
        char_ids = list(xml_by_id.keys())

        characters: list[dict[str, Any]] = []

        async for char_id, page in fetch_anidb_characters(char_ids):
            xml_char = xml_by_id[char_id]
            char_dict = character_from_anidb(xml_char, page_data=page)
            characters.append(char_dict)
            if output_path:
                append_jsonl(output_path, char_dict)

        for char_model in anime_model.characters:
            if char_model.id is None:
                char_dict = character_from_anidb(char_model, page_data=None)
                characters.append(char_dict)
                if output_path:
                    append_jsonl(output_path, char_dict)

        return characters

    async def _fetch_xml(self, anidb_id: int) -> str | None:
        """Fetch the raw XML response from AniDB HTTP API.

        Args:
            anidb_id: AniDB numeric anime ID.

        Returns:
            Raw XML string, or None on network failure.
        """
        try:
            params = {"request": "anime", "aid": anidb_id}
            return await self._make_request(params)
        except Exception:
            logger.exception(f"Failed to fetch XML for AniDB ID {anidb_id}")
            return None

    # =========================================================================
    # CIRCUIT BREAKER
    # =========================================================================

    async def _check_circuit_breaker(self) -> bool:
        """Check whether the circuit breaker allows a request.

        Transitions from OPEN to HALF_OPEN once the timeout has elapsed.

        Returns:
            True if requests are allowed (CLOSED or HALF_OPEN), False if
            the circuit is OPEN and the timeout has not expired.
        """
        current_time = time.time()
        if self.circuit_breaker_state == CircuitBreakerState.OPEN:
            if current_time - self.circuit_breaker_opened_at > self.circuit_breaker_timeout:
                self.circuit_breaker_state = CircuitBreakerState.HALF_OPEN
                logger.info("Circuit breaker moved to HALF_OPEN state")
                return True
            remaining = self.circuit_breaker_timeout - (
                current_time - self.circuit_breaker_opened_at
            )
            logger.warning(f"Circuit breaker OPEN — blocking request. {remaining:.1f}s remaining")
            return False
        return True

    def _update_circuit_breaker(self, success: bool) -> None:
        """Update circuit breaker state based on the outcome of a request.

        On success, resets the consecutive failure count and transitions
        HALF_OPEN → CLOSED. On failure, increments the failure count and
        may open the circuit.

        Args:
            success: True if the request succeeded, False otherwise.
        """
        if success:
            if self.circuit_breaker_state == CircuitBreakerState.HALF_OPEN:
                self.circuit_breaker_state = CircuitBreakerState.CLOSED
                logger.info("Circuit breaker CLOSED — service recovered")
            self.metrics.consecutive_failures = 0
        else:
            self.metrics.consecutive_failures += 1
            if self.circuit_breaker_state == CircuitBreakerState.HALF_OPEN:
                self.circuit_breaker_state = CircuitBreakerState.OPEN
                self.circuit_breaker_opened_at = time.time()
                logger.warning("Circuit breaker OPEN from HALF_OPEN")
            elif (
                self.circuit_breaker_state == CircuitBreakerState.CLOSED
                and self.metrics.consecutive_failures >= self.circuit_breaker_threshold
            ):
                self.circuit_breaker_state = CircuitBreakerState.OPEN
                self.circuit_breaker_opened_at = time.time()
                logger.error(
                    f"Circuit breaker OPENED after {self.metrics.consecutive_failures} failures"
                )

    async def reset_circuit_breaker(self) -> bool:
        """Manually reset the circuit breaker to CLOSED state.

        Returns:
            True if the state was changed, False if it was already CLOSED.
        """
        if self.circuit_breaker_state != CircuitBreakerState.CLOSED:
            old_state = self.circuit_breaker_state
            self.circuit_breaker_state = CircuitBreakerState.CLOSED
            self.circuit_breaker_opened_at = 0.0
            self.metrics.consecutive_failures = 0
            logger.info(f"Circuit breaker manually reset from {old_state.value} to CLOSED")
            return True
        return False

    # =========================================================================
    # RATE LIMITING
    # =========================================================================

    async def _adaptive_rate_limit(self, is_retry: bool = False) -> None:
        """Apply adaptive rate limiting with exponential back-off on errors.

        Args:
            is_retry: When True, multiplies the calculated interval by 1.5
                to add extra delay for retry attempts.
        """
        current_time = time.time()
        time_since_last = current_time - self.metrics.last_request_time

        if self.metrics.consecutive_failures > 0:
            error_multiplier = min(2**self.metrics.consecutive_failures, 8)
            adaptive_interval = min(
                self.error_cooldown_base * error_multiplier, self.max_request_interval
            )
        else:
            adaptive_interval = self.min_request_interval

        if is_retry:
            adaptive_interval *= 1.5

        if time_since_last < adaptive_interval:
            wait_time = adaptive_interval - time_since_last
            logger.info(f"Rate limiting: waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)

        self.metrics.last_request_time = time.time()
        self.metrics.current_interval = adaptive_interval

    # =========================================================================
    # HTTP SESSION
    # =========================================================================

    async def _ensure_session_health(self) -> None:
        """Ensure an active HTTP session exists, recreating it if expired."""
        current_time = time.time()
        if (
            self.session is None
            or current_time - self._session_created_at > self._session_max_age
        ):
            if self.session:
                await self.session.close()

            headers = {
                "Accept-Encoding": "gzip, deflate",
                "User-Agent": f"{self.client_name}/{self.client_version}",
                "Accept": "application/xml, text/xml",
                "Connection": "keep-alive",
                "Cache-Control": "no-cache",
            }
            connector = aiohttp.TCPConnector(
                limit=2,
                limit_per_host=1,
                ttl_dns_cache=300,
                use_dns_cache=True,
                keepalive_timeout=60,
                enable_cleanup_closed=True,
            )
            self.session = _cache_manager.get_aiohttp_session(
                "anidb",
                timeout=aiohttp.ClientTimeout(total=60, connect=30),
                headers=headers,
                connector=connector,
            )
            self._session_created_at = current_time

    async def _make_request(self, params: dict[str, Any]) -> str | None:
        """Serialise requests through a lock and delegate to retry logic.

        Args:
            params: AniDB API query parameters.

        Returns:
            Decoded response content, or None on failure.
        """
        async with self._request_lock:
            return await self._make_request_with_retry(params)

    async def _make_request_with_retry(self, params: dict[str, Any]) -> str | None:
        """Make a request with exponential back-off retry and circuit breaker checks.

        Args:
            params: AniDB API query parameters.

        Returns:
            Decoded response content, or None if all attempts fail.

        Raises:
            ServiceNetworkError: After all retry attempts are exhausted.
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                is_retry = attempt > 0
                if not await self._check_circuit_breaker():
                    return None

                await self._adaptive_rate_limit(is_retry=is_retry)
                await self._ensure_session_health()

                result = await self._make_single_request(params, attempt)
                if result is not None:
                    self.metrics.successful_requests += 1
                    self._update_circuit_breaker(success=True)
                    return result
                else:
                    self.metrics.failed_requests += 1
                    self._update_circuit_breaker(success=False)
                    if attempt < self.max_retries:
                        wait = (2**attempt) + (time.time() % 1)
                        logger.warning(f"Request failed, retrying in {wait:.2f}s (attempt {attempt + 1})")
                        await asyncio.sleep(wait)

            except Exception as e:
                last_exception = e
                self.metrics.failed_requests += 1
                self._update_circuit_breaker(success=False)
                if attempt < self.max_retries:
                    wait = (2**attempt) + (time.time() % 1)
                    logger.warning(f"Request exception, retrying in {wait:.2f}s: {e}")
                    await asyncio.sleep(wait)
                else:
                    logger.exception(f"Request failed after {self.max_retries + 1} attempts: {e}")
            finally:
                self.metrics.total_requests += 1

        if last_exception:
            raise ServiceNetworkError(service="anidb", cause=last_exception)
        return None

    async def _make_single_request(
        self, params: dict[str, Any], attempt: int
    ) -> str | None:
        """Make a single HTTP request to the AniDB API.

        Handles gzip decompression and AniDB-specific error status codes
        (503 service unavailable, 555 banned).

        Args:
            params: AniDB API query parameters.
            attempt: Zero-based attempt number (used for debug logging).

        Returns:
            Decoded XML string on success, or None on error responses.

        Raises:
            RuntimeError: If the session has not been initialised.
            ServiceBlockedError: On HTTP 555 (banned/rate-limit violation).
        """
        request_params = {
            **params,
            "client": self.client_name,
            "clientver": self.client_version,
            "protover": os.getenv("ANIDB_PROTOVER", "1"),
        }

        if self.session is None:
            raise RuntimeError("Session not initialized")

        async with self.session.get(self.base_url, params=request_params) as response:
            if response.status == 200:
                content = await response.read()
                if content.startswith(b"\x1f\x8b"):
                    content = gzip.decompress(content)

                text = self._decode_content(content)
                if text and not text.strip().startswith("<error"):
                    return text
                if text and "<error" in text:
                    logger.warning(f"AniDB error response: {text[:200]}")
                return None

            elif response.status == 503:
                logger.warning("AniDB service unavailable (503)")
                self.metrics.last_error_time = time.time()
                return None

            elif response.status == 555:
                self.metrics.last_error_time = time.time()
                self.circuit_breaker_state = CircuitBreakerState.OPEN
                self.circuit_breaker_opened_at = time.time()
                raise ServiceBlockedError(
                    "banned/blocked (555) — serious rate limit violation",
                    service="anidb",
                )

            else:
                logger.warning(f"AniDB HTTP {response.status}")
                self.metrics.last_error_time = time.time()
                return None

    def _decode_content(self, content: bytes) -> str | None:
        """Decode raw response bytes using UTF-8 with latin-1 fallback.

        Args:
            content: Raw bytes from the HTTP response body.

        Returns:
            Decoded string, or None if all encodings fail.
        """
        for encoding in ("utf-8", "latin-1"):
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue
        logger.error("Failed to decode AniDB response content")
        return None

    # =========================================================================
    # CONTEXT MANAGER
    # =========================================================================

    async def close(self) -> None:
        """Close and release the internal HTTP session."""
        if self.session:
            try:
                await self.session.close()
            except Exception as e:
                logger.warning(f"Error closing AniDB session: {e}")
            finally:
                self.session = None
                self._session_created_at = 0


# =============================================================================
# CLI TEST DRIVER
# =============================================================================


def _write_json(path: str, data: object) -> None:  # pragma: no cover
    safe = sanitize_output_path(path)
    with open(safe, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved to {safe}")


async def main() -> int:  # pragma: no cover
    """CLI entrypoint for inspecting AniDB data fetching."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Fetch data from AniDB HTTP API")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_anime = sub.add_parser("anime", help="Fetch anime metadata")
    p_anime.add_argument("anidb_url", help="AniDB anime URL (e.g. https://anidb.net/anime/69)")
    p_anime.add_argument("output_file", help="Output JSON file")
    p_anime.add_argument(
        "--save-xml",
        type=str,
        nargs="?",
        const="",
        default=None,
        help="Also save raw XML (default: anidb_{id}_raw.xml)",
    )

    p_eps = sub.add_parser("episodes", help="Fetch regular episodes")
    p_eps.add_argument("anidb_url", help="AniDB anime URL (e.g. https://anidb.net/anime/69)")
    p_eps.add_argument("output_file", help="Output JSON file")

    p_chars = sub.add_parser("characters", help="Fetch character data")
    p_chars.add_argument("anidb_url", help="AniDB anime URL (e.g. https://anidb.net/anime/69)")
    p_chars.add_argument("output_file", help="Output JSONL file (written live as each character is fetched)")

    p_all = sub.add_parser("all", help="Fetch anime, episodes, and characters")
    p_all.add_argument("anidb_url", help="AniDB anime URL (e.g. https://anidb.net/anime/69)")
    p_all.add_argument("output_dir", help="Directory to write anidb_anime.json, anidb_episodes.json, anidb_characters.json")

    args = parser.parse_args()
    helper = AniDBHelper()

    try:
        anime_dict, anime_model = await helper._fetch_anime(args.anidb_url)
        if not anime_dict or not anime_model:
            logger.error(f"No data returned for: {args.anidb_url}")
            return 1

        if args.cmd == "anime":
            if args.save_xml is not None:
                match = re.search(r"/anime/(\d+)", args.anidb_url)
                if match:
                    xml_response = await helper._fetch_xml(int(match.group(1)))
                    if xml_response:
                        xml_path = args.save_xml or f"anidb_{match.group(1)}_raw.xml"
                        with open(sanitize_output_path(xml_path), "w", encoding="utf-8") as f:
                            f.write(xml_response)
                        logger.info(f"Raw XML saved to {xml_path}")
            _write_json(args.output_file, anime_dict)

        elif args.cmd == "episodes":
            episodes = await helper._fetch_episodes(anime_model)
            _write_json(args.output_file, episodes)

        elif args.cmd == "characters":
            await helper._fetch_characters(anime_model, output_path=args.output_file)

        elif args.cmd == "all":
            os.makedirs(args.output_dir, exist_ok=True)
            episodes = await helper._fetch_episodes(anime_model)
            characters = await helper._fetch_characters(anime_model)
            _write_json(os.path.join(args.output_dir, "anidb_anime.json"), anime_dict)
            _write_json(os.path.join(args.output_dir, "anidb_episodes.json"), episodes)
            _write_json(os.path.join(args.output_dir, "anidb_characters.json"), characters)

        return 0

    except KeyboardInterrupt:
        return 1
    finally:
        await helper.close()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(asyncio.run(main()))
