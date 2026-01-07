#!/usr/bin/env python3
"""AniDB Helper for AI Enrichment Integration.

Provides helper functions to fetch AniDB data using XML API for AI
enrichment pipeline with production-level rate limiting and error handling.
"""

import argparse
import asyncio
import gzip
import json
import logging
import os
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from enum import Enum
from types import TracebackType
from typing import Any

import aiohttp
from http_cache.instance import http_cache_manager as _cache_manager

from enrichment.crawlers.anidb_character_crawler import fetch_anidb_character
from enrichment.crawlers.utils import sanitize_output_path

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states for AniDB API protection."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Service unavailable, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class AniDBRequestMetrics:
    """Metrics for tracking AniDB API health and compliance.

    Attributes:
        total_requests: Total number of requests made.
        successful_requests: Number of successful requests.
        failed_requests: Number of failed requests.
        consecutive_failures: Current streak of consecutive failures.
        last_request_time: Unix timestamp of last request.
        last_error_time: Unix timestamp of last error.
        current_interval: Current adaptive request interval in seconds.
    """

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    consecutive_failures: int = 0
    last_request_time: float = 0
    last_error_time: float = 0
    current_interval: float = 2.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage.

        Returns:
            Success rate from 0.0 to 100.0.
        """
        if self.total_requests == 0:
            return 100.0
        return (self.successful_requests / self.total_requests) * 100.0

    @property
    def error_rate(self) -> float:
        """Calculate error rate as percentage.

        Returns:
            Error rate from 0.0 to 100.0.
        """
        return 100.0 - self.success_rate


class AniDBEnrichmentHelper:
    """Enhanced AniDB XML API helper with production-level features.

    Provides rate limiting, session management, circuit breaker pattern,
    and comprehensive request metrics for robust AniDB API integration.

    Attributes:
        base_url: AniDB HTTP API endpoint URL.
        client_name: Client identifier sent to AniDB.
        client_version: Client version sent to AniDB.
        session: Active aiohttp session for requests.
        metrics: Request metrics for health monitoring.
        circuit_breaker_state: Current circuit breaker state.
    """

    # XML namespace constants
    XML_LANG_NAMESPACE = "{http://www.w3.org/XML/1998/namespace}lang"

    # Language code normalization mapping
    LANG_NORMALIZATION = {"x-jat": "romaji"}

    def __init__(
        self, client_name: str | None = None, client_version: str | None = None
    ):
        """Initialize the AniDB enrichment helper.

        Configures client metadata, session policy, rate limiting, retry
        behavior, circuit breaker, and request metrics.

        Args:
            client_name: Client identifier sent to AniDB. If None, uses
                the ANIDB_CLIENT environment variable or defaults to
                "animeenrichment".
            client_version: Client version sent to AniDB. If None, uses
                the ANIDB_CLIENTVER environment variable or defaults to
                "1.0".
        """
        self.base_url = "http://api.anidb.net:9001/httpapi"

        # Client configuration
        self.client_name = client_name or os.getenv("ANIDB_CLIENT", "animeenrichment")
        self.client_version = client_version or os.getenv("ANIDB_CLIENTVER", "1.0")

        # Session management
        self.session = None
        self._session_created_at: float = 0.0
        self._session_max_age = 300  # Recreate session every 5 minutes

        # Enhanced rate limiting configuration
        self.min_request_interval = float(
            os.getenv("ANIDB_MIN_REQUEST_INTERVAL", "2.0")
        )
        self.max_request_interval = float(
            os.getenv("ANIDB_MAX_REQUEST_INTERVAL", "10.0")
        )
        self.error_cooldown_base = float(os.getenv("ANIDB_ERROR_COOLDOWN_BASE", "5.0"))
        self.max_retries = int(os.getenv("ANIDB_MAX_RETRIES", "3"))

        # Circuit breaker configuration
        self.circuit_breaker_threshold = int(
            os.getenv("ANIDB_CIRCUIT_BREAKER_THRESHOLD", "5")
        )
        self.circuit_breaker_timeout = float(
            os.getenv("ANIDB_CIRCUIT_BREAKER_TIMEOUT", "300")
        )
        self.circuit_breaker_state = CircuitBreakerState.CLOSED
        self.circuit_breaker_opened_at = 0.0

        # Request tracking and metrics
        self.metrics = AniDBRequestMetrics()
        self._request_lock = asyncio.Lock()  # Ensure request serialization

        logger.info("AniDB helper initialized with enhanced features:")
        logger.info(
            f"  - Rate limiting: {self.min_request_interval}s-{self.max_request_interval}s"
        )
        logger.info(f"  - Circuit breaker: {self.circuit_breaker_threshold} failures")
        logger.info(f"  - Max retries: {self.max_retries}")

    async def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker allows requests.

        Transitions from OPEN to HALF_OPEN state after timeout expires.

        Returns:
            True if requests are allowed, False if blocked.
        """
        current_time = time.time()

        if self.circuit_breaker_state == CircuitBreakerState.OPEN:
            # Check if timeout has passed
            if (
                current_time - self.circuit_breaker_opened_at
                > self.circuit_breaker_timeout
            ):
                self.circuit_breaker_state = CircuitBreakerState.HALF_OPEN
                logger.info("Circuit breaker moved to HALF_OPEN state")
                return True
            else:
                remaining = self.circuit_breaker_timeout - (
                    current_time - self.circuit_breaker_opened_at
                )
                logger.warning(
                    f"Circuit breaker OPEN - blocking request. {remaining:.1f}s remaining"
                )
                return False

        return True  # CLOSED or HALF_OPEN allows requests

    def _update_circuit_breaker(self, success: bool) -> None:
        """Update circuit breaker state based on request result.

        On success, transitions HALF_OPEN to CLOSED and resets failure count.
        On failure, increments failure count and may open the circuit.

        Args:
            success: Whether the request succeeded.
        """
        if success:
            if self.circuit_breaker_state == CircuitBreakerState.HALF_OPEN:
                self.circuit_breaker_state = CircuitBreakerState.CLOSED
                logger.info("Circuit breaker moved to CLOSED state - service recovered")
            self.metrics.consecutive_failures = 0
        else:
            self.metrics.consecutive_failures += 1

            if self.circuit_breaker_state == CircuitBreakerState.HALF_OPEN:
                self.circuit_breaker_state = CircuitBreakerState.OPEN
                self.circuit_breaker_opened_at = time.time()
                logger.warning(
                    "Circuit breaker moved back to OPEN state from HALF_OPEN"
                )
            elif (
                self.circuit_breaker_state == CircuitBreakerState.CLOSED
                and self.metrics.consecutive_failures >= self.circuit_breaker_threshold
            ):
                self.circuit_breaker_state = CircuitBreakerState.OPEN
                self.circuit_breaker_opened_at = time.time()
                logger.error(
                    f"Circuit breaker OPENED after {self.metrics.consecutive_failures} consecutive failures"
                )

    async def _adaptive_rate_limit(self, is_retry: bool = False) -> None:
        """Apply adaptive rate limiting with error-aware delays.

        Calculates delay based on consecutive failures using exponential
        backoff. Adds extra delay for retry attempts.

        Args:
            is_retry: Whether this is a retry attempt. Defaults to False.
        """
        current_time = time.time()
        time_since_last = current_time - self.metrics.last_request_time

        # Calculate adaptive interval based on recent errors
        if self.metrics.consecutive_failures > 0:
            # Exponential backoff for errors
            error_multiplier = min(2**self.metrics.consecutive_failures, 8)
            adaptive_interval = min(
                self.error_cooldown_base * error_multiplier, self.max_request_interval
            )
        else:
            adaptive_interval = self.metrics.current_interval

        # Add extra delay for retries
        if is_retry:
            adaptive_interval *= 1.5

        if time_since_last < adaptive_interval:
            wait_time = adaptive_interval - time_since_last
            logger.info(
                f"Adaptive rate limiting: waiting {wait_time:.2f}s (interval: {adaptive_interval:.2f}s)"
            )
            await asyncio.sleep(wait_time)

        self.metrics.last_request_time = time.time()
        self.metrics.current_interval = adaptive_interval

    async def _ensure_session_health(self) -> None:
        """Ensure an active HTTP session exists and recreate if expired.

        Creates a new aiohttp session via the cache manager if the current
        session is absent or older than the configured maximum age.
        """
        current_time = time.time()

        # Check if session needs recreation
        if (
            self.session is None
            or current_time - self._session_created_at > self._session_max_age
        ):
            if self.session:
                logger.debug("Recreating AniDB session (max age reached)")
                await self.session.close()

            # Create new session with optimized settings
            headers = {
                "Accept-Encoding": "gzip, deflate",
                "User-Agent": f"{self.client_name}/{self.client_version}",
                "Accept": "application/xml, text/xml",
                "Connection": "keep-alive",  # Better connection reuse
                "Cache-Control": "no-cache",  # Prevent caching issues
            }

            connector = aiohttp.TCPConnector(
                limit=2,  # Small connection pool
                limit_per_host=1,  # Single connection to AniDB
                ttl_dns_cache=300,  # DNS cache for 5 minutes
                use_dns_cache=True,
                keepalive_timeout=60,  # Keep connections alive
                enable_cleanup_closed=True,
            )

            self.session = _cache_manager.get_aiohttp_session(
                "anidb",
                timeout=aiohttp.ClientTimeout(total=60, connect=30),
                headers=headers,
                connector=connector,
            )

            self._session_created_at = current_time
            logger.debug("Created new AniDB session with enhanced settings")

    async def _make_request_with_retry(self, params: dict[str, Any]) -> str | None:
        """Make request with enhanced retry logic and error handling.

        Implements exponential backoff with jitter and circuit breaker checks.

        Args:
            params: Request parameters dictionary.

        Returns:
            Response content if successful, None otherwise.
        """
        # Try request with retries
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                is_retry = attempt > 0

                # Check circuit breaker
                if not await self._check_circuit_breaker():
                    return None

                # Apply adaptive rate limiting
                await self._adaptive_rate_limit(is_retry=is_retry)

                # Ensure session health
                await self._ensure_session_health()

                # Make the actual request
                result = await self._make_single_request(params, attempt)

                if result is not None:
                    # Success - update metrics and circuit breaker
                    self.metrics.successful_requests += 1
                    self._update_circuit_breaker(success=True)

                    if is_retry:
                        logger.info(f"Request succeeded on attempt {attempt + 1}")

                    return result
                else:
                    # Request failed but no exception
                    self.metrics.failed_requests += 1
                    self._update_circuit_breaker(success=False)

                    if attempt < self.max_retries:
                        wait_time = (2**attempt) + (
                            time.time() % 1
                        )  # Exponential backoff with jitter
                        logger.warning(
                            f"Request failed, retrying in {wait_time:.2f}s (attempt {attempt + 1}/{self.max_retries})"
                        )
                        await asyncio.sleep(wait_time)

            except Exception as e:
                last_exception = e
                self.metrics.failed_requests += 1
                self._update_circuit_breaker(success=False)

                if attempt < self.max_retries:
                    wait_time = (2**attempt) + (
                        time.time() % 1
                    )  # Exponential backoff with jitter
                    logger.warning(
                        f"Request exception, retrying in {wait_time:.2f}s: {e} (attempt {attempt + 1}/{self.max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"Request failed after {self.max_retries + 1} attempts: {e}"
                    )

            finally:
                self.metrics.total_requests += 1

        # All retries exhausted
        if last_exception:
            logger.error(
                f"Request failed permanently after {self.max_retries + 1} attempts: {last_exception}"
            )

        return None

    async def _make_single_request(
        self, params: dict[str, Any], attempt: int
    ) -> str | None:
        """Make a single request attempt to the AniDB API.

        Handles gzip decompression and various HTTP error codes including
        503 (service unavailable) and 555 (banned).

        Args:
            params: Request parameters dictionary.
            attempt: Current attempt number (0-indexed).

        Returns:
            Decoded response content if successful, None otherwise.

        Raises:
            RuntimeError: If session is not initialized.
        """
        # Add required client parameters
        request_params = params.copy()
        request_params.update(
            {
                "client": self.client_name,
                "clientver": self.client_version,
                "protover": os.getenv("ANIDB_PROTOVER", "1"),
            }
        )

        logger.debug(
            f"AniDB request attempt {attempt + 1}: {self.base_url} with params: {request_params}"
        )

        if self.session is None:
            raise RuntimeError("Session not initialized")
        async with self.session.get(self.base_url, params=request_params) as response:
            logger.debug(f"AniDB response status: {response.status}")

            if response.status == 200:
                # Handle successful response
                content = await response.read()
                logger.debug(f"Response content length: {len(content)} bytes")

                # Handle gzip compression
                if content.startswith(b"\x1f\x8b"):
                    logger.debug("Decompressing gzipped content")
                    try:
                        content = gzip.decompress(content)
                        logger.debug(
                            f"Decompressed content length: {len(content)} bytes"
                        )
                    except Exception as e:
                        logger.error(f"Failed to decompress gzipped content: {e}")
                        raise

                # Decode content with multiple encoding fallbacks
                text_content = self._decode_content(content)
                if text_content and not text_content.strip().startswith("<error"):
                    logger.debug(
                        f"Successfully decoded response: {text_content[:100]}..."
                    )
                    return text_content
                elif text_content and "<error" in text_content:
                    logger.warning(
                        f"AniDB returned error response: {text_content[:200]}"
                    )
                    return None
                else:
                    logger.error("Failed to decode response content")
                    return None

            elif response.status == 503:
                logger.warning("AniDB service unavailable (503) - will retry")
                self.metrics.last_error_time = time.time()
                return None

            elif response.status == 555:
                logger.error(
                    "AniDB banned/blocked (555) - serious rate limit violation"
                )
                self.metrics.last_error_time = time.time()
                # Force circuit breaker open for ban scenarios
                self.circuit_breaker_state = CircuitBreakerState.OPEN
                self.circuit_breaker_opened_at = time.time()
                return None

            else:
                logger.warning(f"AniDB API error: HTTP {response.status}")
                error_content = await response.text()
                logger.debug(f"Error response: {error_content[:200]}")
                self.metrics.last_error_time = time.time()
                return None

    def _decode_content(self, content: bytes) -> str | None:
        """Decode response content with multiple encoding fallbacks.

        Tries utf-8, latin-1, cp1252, and iso-8859-1 encodings in order.

        Args:
            content: Raw bytes to decode.

        Returns:
            Decoded string if successful, None if all encodings fail.
        """
        encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]

        for encoding in encodings:
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue

        logger.error(f"Failed to decode content with any encoding: {encodings}")
        return None

    async def _make_request(self, params: dict[str, Any]) -> str | None:
        """Make a thread-safe request with serialization lock.

        Args:
            params: Request parameters dictionary.

        Returns:
            Response content if successful, None otherwise.
        """
        async with self._request_lock:
            return await self._make_request_with_retry(params)

    def _validate_anime_xml(self, root: ET.Element) -> bool:
        """Validate anime XML structure for critical fields.

        Checks for required root element, id attribute, and critical
        elements (type, episodecount, titles). Logs warnings for
        missing optional elements.

        Args:
            root: Root XML element to validate.

        Returns:
            True if structure is valid, False if root element or id attribute
                is invalid.
        """
        # Check root element
        if root.tag != "anime":
            logger.error(f"Invalid root element: expected 'anime', got '{root.tag}'")
            return False

        # Check required attribute
        if root.get("id") is None:
            logger.error("Missing required 'id' attribute on <anime> element")
            return False

        # Check critical elements exist
        critical_elements = ["type", "episodecount", "titles"]
        for elem_name in critical_elements:
            if root.find(elem_name) is None:
                logger.warning(f"Missing critical element: <{elem_name}>")

        # Validate titles structure
        titles_elem = root.find("titles")
        if titles_elem is not None:
            title_elements = titles_elem.findall("title")
            if not title_elements:
                logger.warning("No <title> elements found in <titles>")

            # Check for at least one main title
            main_titles = [t for t in title_elements if t.get("type") == "main"]
            if not main_titles:
                logger.warning("No main title found in <titles>")

        # Validate episodes structure if present
        episodes_elem = root.find("episodes")
        if episodes_elem is not None:
            for episode in episodes_elem.findall("episode"):
                if episode.get("id") is None:
                    logger.warning("Episode missing 'id' attribute")
                if episode.find("epno") is None:
                    logger.warning(
                        f"Episode {episode.get('id', 'unknown')} missing <epno>"
                    )

        return True

    async def _parse_anime_xml(self, xml_content: str) -> dict[str, Any]:
        """Parse anime XML response into structured data.

        Extracts all anime metadata including titles, tags, ratings,
        categories, creators, characters, episodes, and related anime.

        Args:
            xml_content: Raw XML string from AniDB API.

        Returns:
            Structured anime data dictionary. Returns empty dict if parsing
                fails.
        """
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {str(e)}")
            return {}

        # Validate XML structure
        if not self._validate_anime_xml(root):
            logger.error("XML validation failed - structure may have changed")
            # Continue parsing but log the issue

        # Extract basic anime information
        type_elem = root.find("type")
        episodecount_elem = root.find("episodecount")
        startdate_elem = root.find("startdate")
        enddate_elem = root.find("enddate")
        description_elem = root.find("description")
        url_elem = root.find("url")
        picture_elem = root.find("picture")

        anime_data: dict[str, Any] = {
            "anidb_id": root.get("id"),
            "type": type_elem.text if type_elem is not None else None,
            "episodes": (
                int(episodecount_elem.text)
                if episodecount_elem is not None
                and episodecount_elem.text
                and episodecount_elem.text.isdigit()
                else 0
            ),
            "start_date": startdate_elem.text if startdate_elem is not None else None,
            "end_date": enddate_elem.text if enddate_elem is not None else None,
            "synopsis": (
                description_elem.text if description_elem is not None else None
            ),
            "url": url_elem.text if url_elem is not None else None,
            "cover": (
                f"https://cdn-eu.anidb.net/images/main/{picture_elem.text}"
                if picture_elem is not None and picture_elem.text
                else None
            ),
            "title": None,
            "title_english": None,
            "title_japanese": None,
            "synonyms": [],
        }

        # Extract titles
        titles_element = root.find("titles")
        if titles_element is not None:
            for title in titles_element.findall("title"):
                title_type = title.get("type", "unknown")
                lang = title.get("xml:lang") or title.get(
                    self.XML_LANG_NAMESPACE, "unknown"
                )

                if title_type == "main":
                    anime_data["title"] = title.text
                elif title_type == "official":
                    if lang == "en":
                        anime_data["title_english"] = title.text
                    elif lang == "ja":
                        anime_data["title_japanese"] = title.text
                elif title_type in ["synonym", "short"]:
                    if title.text:
                        anime_data["synonyms"].append(title.text)

        # Extract tags (simple list of names)
        tags_element = root.find("tags")
        anime_data["tags"] = [
            name.text
            for tag in (tags_element.findall("tag") if tags_element is not None else [])
            if (name := tag.find("name")) is not None and name.text
        ]

        # Extract ratings (map permanent to statistics)
        ratings_element = root.find("ratings")
        statistics = {}
        if ratings_element is not None:
            permanent = ratings_element.find("permanent")
            if permanent is not None:
                statistics["score"] = float(permanent.text) if permanent.text else None
                statistics["scored_by"] = int(permanent.get("count", 0))
        anime_data["statistics"] = statistics

        # Extract categories/genres
        categories_element = root.find("categories")
        anime_data["categories"] = [
            {
                "id": cat.get("id"),
                "name": name.text,
                "weight": int(cat.get("weight", 0)),
                "hentai": cat.get("hentai") == "true",
            }
            for cat in (
                categories_element.findall("category")
                if categories_element is not None
                else []
            )
            if (name := cat.find("name")) is not None
        ]

        # Extract creator information
        creators_element = root.find("creators")
        anime_data["creators"] = [
            {
                "id": creator.get("id"),
                "name": creator.text,
                "type": creator.get("type"),
            }
            for creator in (
                creators_element.findall("name") if creators_element is not None else []
            )
        ]

        # Extract characters
        characters_element = root.find("characters")
        characters = []
        if characters_element is not None:
            for character in characters_element.findall("character"):
                char_data = await self._parse_character_xml(character)
                characters.append(char_data)

        anime_data["character_details"] = characters

        # Extract episodes
        episodes_element = root.find("episodes")
        episodes = []
        if episodes_element is not None:
            for episode_elem in episodes_element.findall("episode"):
                episodes.append(self._parse_episode_xml(episode_elem))
        anime_data["episode_details"] = episodes

        # Extract related anime
        anime_data["related_anime"] = (
            [
                {
                    "url": f"https://anidb.net/anime/{elem.get('id')}",
                    "relation": elem.get("type"),
                    "title": elem.text,
                }
                for elem in related_anime_element.findall("anime")
            ]
            if (related_anime_element := root.find("relatedanime")) is not None
            else []
        )

        # Extract external links from resources
        external_links: dict[str, str | None] = {
            "official_website": None,
            "wikipedia_en": None,
            "wikipedia_jp": None,
        }
        resources_element = root.find("resources")
        if resources_element is not None:
            # Use dict for O(1) type lookup instead of if-elif chain
            resource_handlers: dict[str, tuple[str, str, str | None]] = {
                "4": ("official_website", "url", None),
                "6": ("wikipedia_en", "identifier", "https://en.wikipedia.org/wiki/"),
                "7": ("wikipedia_jp", "identifier", "https://ja.wikipedia.org/wiki/"),
            }
            for resource in resources_element.findall("resource"):
                resource_type = resource.get("type")
                if resource_type not in resource_handlers:
                    continue
                key, tag, url_prefix = resource_handlers[resource_type]
                # Get first externalentity's value
                entity = resource.find("externalentity")
                if entity is not None:
                    value_elem = entity.find(tag)
                    if value_elem is not None and value_elem.text:
                        external_links[key] = (
                            f"{url_prefix}{value_elem.text}"
                            if url_prefix
                            else value_elem.text
                        )
        anime_data["external_links"] = external_links

        return anime_data

    def _parse_episode_xml(self, episode_element: ET.Element) -> dict[str, Any]:
        """Parse episode XML element into structured data.

        Extracts episode metadata from embedded anime response including
        episode number, type, length, air date, rating, titles, and
        streaming links.

        Args:
            episode_element: Episode XML element from anime response.

        Returns:
            Structured episode data with id, episode_number, episode_type,
                length, air_date, rating, titles, and streaming.
        """
        epno_elem = episode_element.find("epno")
        length_elem = episode_element.find("length")
        airdate_elem = episode_element.find("airdate")
        rating_elem = episode_element.find("rating")
        summary_elem = episode_element.find("summary")

        # Parse episode type first to determine episode_number parsing
        episode_type: int | None = None
        if epno_elem is not None:
            type_str = epno_elem.get("type")
            if type_str is not None:
                episode_type = int(type_str)

        # Parse episode_number: convert to int if episode_type is 1 (regular episodes)
        episode_number: int | str | None = None
        if epno_elem is not None and epno_elem.text:
            if episode_type == 1:
                # Regular episode - parse as int
                try:
                    episode_number = int(epno_elem.text)
                except ValueError:
                    episode_number = (
                        epno_elem.text
                    )  # Keep as string if conversion fails
            else:
                # Special, OP, ED, etc. - keep as string
                episode_number = epno_elem.text

        # Parse episode ID safely
        ep_id_str = episode_element.get("id")
        ep_id: int | None = int(ep_id_str) if ep_id_str else None

        episode_data: dict[str, Any] = {
            "id": ep_id,
            "update": episode_element.get("update"),
            "episode_number": episode_number,
            "episode_type": episode_type,
            "length": (
                int(length_elem.text)
                if length_elem is not None and length_elem.text
                else None
            ),
            "air_date": airdate_elem.text if airdate_elem is not None else None,
            "rating": (
                float(rating_elem.text)
                if rating_elem is not None and rating_elem.text
                else None
            ),
            "rating_votes": (
                int(rating_elem.get("votes", 0)) if rating_elem is not None else 0
            ),
            "summary": summary_elem.text if summary_elem is not None else None,
        }

        # Extract episode titles
        ep_titles: dict[str, str | None] = {}
        for ep_title in episode_element.findall("title"):
            ep_lang = ep_title.get("xml:lang") or ep_title.get(
                self.XML_LANG_NAMESPACE, "unknown"
            )
            if ep_title.text:
                # Normalize language codes (e.g., x-jat → romaji)
                normalized_lang = self.LANG_NORMALIZATION.get(ep_lang, ep_lang)
                ep_titles[normalized_lang] = ep_title.text
        episode_data["titles"] = ep_titles

        # Extract episode streaming links from resources
        streaming: dict[str, str] = {}
        ep_resources_element = episode_element.find("resources")
        if ep_resources_element is not None:
            for ep_resource_elem in ep_resources_element.findall("resource"):
                resource_type = ep_resource_elem.get("type")

                # Type 28 = Crunchyroll
                if resource_type == "28":
                    external_entity_elem = ep_resource_elem.find("externalentity")
                    if external_entity_elem is not None:
                        identifier_elem = external_entity_elem.find("identifier")
                        if identifier_elem is not None and identifier_elem.text:
                            streaming["crunchyroll"] = (
                                f"https://www.crunchyroll.com/watch/{identifier_elem.text}"
                            )

        episode_data["streaming"] = streaming

        return episode_data

    async def _parse_character_xml(self, character: ET.Element) -> dict[str, Any]:
        """Parse character XML element into structured data.

        Extracts character metadata and enriches with web-scraped details
        from AniDB character pages. Normalizes character type values.

        Args:
            character: Character XML element to parse.

        Returns:
            Structured character data including id, type, name, gender,
                description, rating, picture, voice_actor, and enriched details
                from web scraping.
        """
        # Normalize character type: "main character in" -> "Main", "secondary character in" -> "Secondary", "appears in" -> "Minor"
        raw_type = character.get("type")
        char_type = None
        if raw_type:
            if "main character" in raw_type.lower():
                char_type = "Main"
            elif "secondary character" in raw_type.lower():
                char_type = "Secondary"
            elif "appears in" in raw_type.lower():
                char_type = "Minor"
            else:
                char_type = raw_type  # Keep original if unknown pattern

        char_data: dict[str, Any] = {
            "id": character.get("id"),
            "type": char_type,
            "update": character.get("update"),
        }

        # Get character details from XML
        name_element = character.find("name")
        if name_element is not None:
            char_data["name_main"] = name_element.text

        gender_element = character.find("gender")
        if gender_element is not None:
            char_data["gender"] = gender_element.text

        char_type_element = character.find("charactertype")
        if char_type_element is not None:
            char_data["character_type"] = char_type_element.text
            char_data["character_type_id"] = char_type_element.get("id")

        description_element = character.find("description")
        if description_element is not None:
            char_data["description"] = description_element.text

        # Character rating
        rating_element = character.find("rating")
        if rating_element is not None:
            char_data["rating"] = (
                float(rating_element.text) if rating_element.text else None
            )
            char_data["rating_votes"] = int(rating_element.get("votes", 0))

        # Character picture
        picture_element = character.find("picture")
        if picture_element is not None:
            char_data["picture"] = picture_element.text

        # Voice actor
        seiyuu_element = character.find("seiyuu")
        if seiyuu_element is not None:
            char_data["voice_actor"] = {
                "name": seiyuu_element.text,
                "id": seiyuu_element.get("id"),
                "picture": seiyuu_element.get("picture"),
            }

        # Fetch detailed character information from AniDB website
        character_id = character.get("id")
        if character_id:
            try:
                detailed_char_data = await fetch_anidb_character(int(character_id))
                if detailed_char_data:
                    # Merge detailed data into char_data
                    char_data.update(detailed_char_data)
                    logger.info(f"Enriched character {character_id} with detailed data")
            except Exception as e:
                logger.warning(
                    f"Failed to fetch detailed data for character {character_id}: {e}"
                )

        return char_data

    async def get_anime_by_id(self, anidb_id: int) -> dict[str, Any] | None:
        """Get anime information by AniDB ID.

        Args:
            anidb_id: AniDB anime ID.

        Returns:
            Parsed anime data if successful, None if not found or on error.
        """
        try:
            params = {"request": "anime", "aid": anidb_id}
            xml_response = await self._make_request(params)

            if not xml_response:
                logger.warning(f"No response for AniDB ID: {anidb_id}")
                return None

            if "<error" in xml_response:
                logger.warning(
                    f"Error response for AniDB ID {anidb_id}: {xml_response[:200]}"
                )
                return None

            # Log first 200 chars of response for debugging
            logger.info(f"AniDB response preview: {xml_response[:200]}")

            return await self._parse_anime_xml(xml_response)
        except Exception as e:
            logger.error(f"Failed to fetch anime by AniDB ID {anidb_id}: {e}")
            return None

    async def fetch_all_data(self, anidb_id: int) -> dict[str, Any] | None:
        """Fetch comprehensive AniDB data for an anime by ID.

        Args:
            anidb_id: The AniDB anime ID.

        Returns:
            Comprehensive AniDB data including metadata, characters, episodes,
                and related anime. None if not found.
        """
        try:
            anime_data = await self.get_anime_by_id(anidb_id)
            if not anime_data:
                logger.warning(f"No AniDB data found for ID: {anidb_id}")
                return None

            logger.info(f"Successfully fetched AniDB data for ID: {anidb_id}")
            return anime_data

        except Exception as e:
            logger.error(f"Error in fetch_all_data for AniDB ID {anidb_id}: {e}")
            return None

    async def reset_circuit_breaker(self) -> bool:
        """Manually reset circuit breaker to CLOSED state.

        Admin function to force recovery after issues are resolved.

        Returns:
            True if state was changed, False if already CLOSED.
        """
        if self.circuit_breaker_state != CircuitBreakerState.CLOSED:
            old_state = self.circuit_breaker_state
            self.circuit_breaker_state = CircuitBreakerState.CLOSED
            self.metrics.consecutive_failures = 0
            logger.info(
                f"Circuit breaker manually reset from {old_state.value} to CLOSED"
            )
            return True
        return False

    async def close(self) -> None:
        """Close and clean up the helper's internal HTTP session.

        Closes the active session if it exists, clears the session
        reference, and resets the session creation timestamp.
        """
        if self.session:
            try:
                await self.session.close()
                logger.debug("AniDB session closed successfully")
            except Exception as e:
                logger.warning(f"Error closing AniDB session: {e}")
            finally:
                self.session = None
                self._session_created_at = 0

    async def __aenter__(self) -> "AniDBEnrichmentHelper":
        """Enter asynchronous context and return the helper instance.

        Returns:
            The helper instance for use within an async with block.
        """
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """Close the HTTP session when exiting async context.

        Awaits close() to release network resources.

        Args:
            exc_type: Exception type if an exception was raised.
            exc_val: Exception value if an exception was raised.
            exc_tb: Exception traceback if an exception was raised.

        Returns:
            False to not suppress exceptions raised within the context.
        """
        await self.close()
        return False


async def main() -> int:
    """Command-line test driver for AniDB data fetching.

    Parses command-line options (--anidb-id, --output), fetches anime data
    by ID using AniDBEnrichmentHelper, and saves the result to the specified
    output path.

    Returns:
        Exit code where 0 indicates success and 1 indicates failure, no data
            found, or interruption.
    """
    parser = argparse.ArgumentParser(description="Test AniDB data fetching")
    parser.add_argument(
        "--anidb-id", type=int, required=True, help="AniDB ID to fetch"
    )
    parser.add_argument(
        "--output", type=str, default="test_anidb_output.json", help="Output file path"
    )
    parser.add_argument(
        "--save-xml",
        type=str,
        nargs="?",
        const="",  # Use empty string to trigger default naming
        default=None,  # None when flag not provided
        help="Save raw XML response (default: anidb_{id}_raw.xml in repo root, or specify custom path)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    helper = AniDBEnrichmentHelper()

    try:
        # Fetch raw XML if requested (optional, non-blocking)
        if args.save_xml is not None:
            try:
                params = {"request": "anime", "aid": args.anidb_id}
                xml_response = await helper._make_request(params)
                if xml_response:
                    # Use default filename if flag provided without path
                    xml_path = (
                        args.save_xml
                        if args.save_xml
                        else f"anidb_{args.anidb_id}_raw.xml"
                    )
                    safe_xml_path = sanitize_output_path(xml_path)
                    with open(safe_xml_path, "w", encoding="utf-8") as f:
                        f.write(xml_response)
                    logger.info(f"Raw XML saved to {safe_xml_path}")
                else:
                    logger.warning("Failed to fetch XML for --save-xml (continuing anyway)")
            except Exception as e:
                logger.warning(f"Failed to save XML: {e} (continuing anyway)")

        # Fetch data by ID
        anime_data = await helper.fetch_all_data(args.anidb_id)

        if anime_data:
            # Save to file (sanitize path to prevent traversal attacks)
            safe_path = sanitize_output_path(args.output)
            with open(safe_path, "w", encoding="utf-8") as f:
                json.dump(anime_data, f, indent=2, ensure_ascii=False)
            return 0
        else:
            logger.error("No data found")
            return 1

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 1
    except Exception:
        logger.exception("Main execution failed")
        return 1
    finally:
        await helper.close()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(asyncio.run(main()))
