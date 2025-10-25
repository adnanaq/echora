#!/usr/bin/env python3
"""
AniDB Helper for AI Enrichment Integration

Helper function to fetch AniDB data using XML API for AI enrichment pipeline.
"""

import argparse
import asyncio
import gzip
import hashlib
import json
import logging
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

import aiohttp
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states for AniDB API protection."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Service unavailable, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class AniDBRequestMetrics:
    """Metrics for tracking AniDB API health and compliance."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    consecutive_failures: int = 0
    last_request_time: float = 0
    last_error_time: float = 0
    current_interval: float = 2.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_requests == 0:
            return 100.0
        return (self.successful_requests / self.total_requests) * 100.0

    @property
    def error_rate(self) -> float:
        """Calculate error rate as percentage."""
        return 100.0 - self.success_rate


class AniDBEnrichmentHelper:
    """Enhanced AniDB XML API helper with production-level rate limiting and session management."""

    def __init__(
        self, client_name: str | None = None, client_version: str | None = None
    ):
        """Initialize AniDB enrichment helper with enhanced reliability features."""
        self.base_url = "http://api.anidb.net:9001/httpapi"

        # Client configuration
        self.client_name = client_name or os.getenv("ANIDB_CLIENT", "animeenrichment")
        self.client_version = client_version or os.getenv("ANIDB_CLIENTVER", "1.0")

        # Session management
        self.session = None
        self._session_created_at = 0
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
        self.circuit_breaker_opened_at = 0

        # Request tracking and metrics
        self.metrics = AniDBRequestMetrics()
        self.recent_requests: set[str] = set()  # Track recent request fingerprints
        self._request_lock = asyncio.Lock()  # Ensure request serialization

        logger.info("AniDB helper initialized with enhanced features:")
        logger.info(
            f"  - Rate limiting: {self.min_request_interval}s-{self.max_request_interval}s"
        )
        logger.info(f"  - Circuit breaker: {self.circuit_breaker_threshold} failures")
        logger.info(f"  - Max retries: {self.max_retries}")

    def _generate_request_fingerprint(self, params: dict[str, Any]) -> str:
        """Generate fingerprint for request deduplication."""
        # Create a consistent hash of request parameters
        param_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        return hashlib.md5(param_str.encode()).hexdigest()

    async def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker allows requests."""
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
        """Update circuit breaker state based on request result."""
        if success:
            if self.circuit_breaker_state == CircuitBreakerState.HALF_OPEN:
                self.circuit_breaker_state = CircuitBreakerState.CLOSED
                logger.info("Circuit breaker moved to CLOSED state - service recovered")
            self.metrics.consecutive_failures = 0
        else:
            self.metrics.consecutive_failures += 1

            if (
                self.circuit_breaker_state == CircuitBreakerState.CLOSED
                and self.metrics.consecutive_failures >= self.circuit_breaker_threshold
            ):
                self.circuit_breaker_state = CircuitBreakerState.OPEN
                self.circuit_breaker_opened_at = int(time.time())
                logger.error(
                    f"Circuit breaker OPENED after {self.metrics.consecutive_failures} consecutive failures"
                )

    async def _adaptive_rate_limit(self, is_retry: bool = False) -> None:
        """Enhanced rate limiting with adaptive delays and error recovery."""
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
        """Ensure session is healthy and recreate if needed."""
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

            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60, connect=30),
                headers=headers,
                connector=connector,
            )

            self._session_created_at = current_time
            logger.debug("Created new AniDB session with enhanced settings")

    async def _make_request_with_retry(self, params: dict[str, Any]) -> str | None:
        """Make request with enhanced retry logic and error handling."""
        # Generate request fingerprint for deduplication
        fingerprint = self._generate_request_fingerprint(params)

        # Check for recent duplicate requests
        if fingerprint in self.recent_requests:
            logger.warning(
                f"Skipping duplicate request (fingerprint: {fingerprint[:8]}...)"
            )
            return None

        # Add to recent requests (with cleanup of old entries)
        self.recent_requests.add(fingerprint)
        if len(self.recent_requests) > 1000:  # Prevent memory growth
            # Remove oldest half of entries (simple cleanup)
            old_requests = list(self.recent_requests)[:500]
            self.recent_requests -= set(old_requests)

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
        """Make a single request attempt to the AniDB API."""
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
                self.circuit_breaker_opened_at = int(time.time())
                return None

            else:
                logger.warning(f"AniDB API error: HTTP {response.status}")
                error_content = await response.text()
                logger.debug(f"Error response: {error_content[:200]}")
                self.metrics.last_error_time = time.time()
                return None

    def _decode_content(self, content: bytes) -> str | None:
        """Decode response content with multiple encoding fallbacks."""
        encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]

        for encoding in encodings:
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue

        logger.error(f"Failed to decode content with any encoding: {encodings}")
        return None

    async def _make_request(self, params: dict[str, Any]) -> str | None:
        """Thread-safe request method with serialization lock."""
        async with self._request_lock:
            return await self._make_request_with_retry(params)

    def _parse_anime_xml(self, xml_content: str) -> dict[str, Any]:
        """Parse anime XML response into structured data."""
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {str(e)}")
            return {}

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
            "episodecount": (
                episodecount_elem.text if episodecount_elem is not None else None
            ),
            "startdate": startdate_elem.text if startdate_elem is not None else None,
            "enddate": enddate_elem.text if enddate_elem is not None else None,
            "description": (
                description_elem.text if description_elem is not None else None
            ),
            "url": url_elem.text if url_elem is not None else None,
            "picture": picture_elem.text if picture_elem is not None else None,
        }

        # Extract titles
        titles_element = root.find("titles")
        titles: dict[str, str | list[str] | None] = {}
        if titles_element is not None:
            for title in titles_element.findall("title"):
                title_type = title.get("type", "unknown")
                lang = title.get("xml:lang", "unknown")

                if title_type == "main":
                    titles["main"] = title.text
                elif title_type == "official":
                    if lang == "en":
                        titles["english"] = title.text
                    elif lang == "ja":
                        titles["japanese"] = title.text
                elif title_type == "synonym":
                    if "synonyms" not in titles:
                        titles["synonyms"] = []
                    if title.text:
                        synonyms = cast(list[str], titles["synonyms"])
                        synonyms.append(title.text)
        anime_data["titles"] = titles

        # Extract tags
        tags_element = root.find("tags")
        tags = []
        if tags_element is not None:
            for tag in tags_element.findall("tag"):
                name_element = tag.find("name")
                description_element = tag.find("description")
                if name_element is not None:
                    tag_data = {
                        "id": tag.get("id"),
                        "name": name_element.text,
                        "count": int(tag.get("count", 0)),
                        "weight": int(tag.get("weight", 0)),
                    }
                    if description_element is not None:
                        tag_data["description"] = description_element.text
                    tags.append(tag_data)
        anime_data["tags"] = tags

        # Extract ratings
        ratings_element = root.find("ratings")
        ratings = {}
        if ratings_element is not None:
            permanent = ratings_element.find("permanent")
            temporary = ratings_element.find("temporary")
            review = ratings_element.find("review")

            if permanent is not None:
                ratings["permanent"] = {
                    "value": float(permanent.text) if permanent.text else None,
                    "count": int(permanent.get("count", 0)),
                }
            if temporary is not None:
                ratings["temporary"] = {
                    "value": float(temporary.text) if temporary.text else None,
                    "count": int(temporary.get("count", 0)),
                }
            if review is not None:
                ratings["review"] = {
                    "value": float(review.text) if review.text else None,
                    "count": int(review.get("count", 0)),
                }
        anime_data["ratings"] = ratings

        # Extract categories/genres
        categories_element = root.find("categories")
        categories = []
        if categories_element is not None:
            for category in categories_element.findall("category"):
                name_element = category.find("name")
                if name_element is not None:
                    categories.append(
                        {
                            "id": category.get("id"),
                            "name": name_element.text,
                            "weight": int(category.get("weight", 0)),
                            "hentai": category.get("hentai") == "true",
                        }
                    )
        anime_data["categories"] = categories

        # Extract creator information
        creators_element = root.find("creators")
        creators = []
        if creators_element is not None:
            for creator in creators_element.findall("name"):
                creators.append(
                    {
                        "id": creator.get("id"),
                        "name": creator.text,
                        "type": creator.get("type"),
                    }
                )
        anime_data["creators"] = creators

        # Extract characters
        characters_element = root.find("characters")
        characters = []
        if characters_element is not None:
            for character in characters_element.findall("character"):
                char_data: dict[str, Any] = {
                    "id": character.get("id"),
                    "type": character.get("type"),
                    "update": character.get("update"),
                }

                # Get character details
                name_element = character.find("name")
                if name_element is not None:
                    char_data["name"] = name_element.text

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

                # Voice actor (seiyuu)
                seiyuu_element = character.find("seiyuu")
                if seiyuu_element is not None:
                    char_data["seiyuu"] = {
                        "name": seiyuu_element.text,
                        "id": seiyuu_element.get("id"),
                        "picture": seiyuu_element.get("picture"),
                    }

                characters.append(char_data)

        anime_data["characters"] = characters

        return anime_data

    def _parse_episode_xml(self, xml_content: str) -> dict[str, Any]:
        """Parse episode XML response into structured data."""
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {str(e)}")
            return {}

        epno_elem = root.find("epno")
        length_elem = root.find("length")
        airdate_elem = root.find("airdate")
        rating_elem = root.find("rating")
        votes_elem = root.find("votes")
        summary_elem = root.find("summary")

        episode_data: dict[str, Any] = {
            "anidb_id": root.get("id"),
            "anime_id": root.get("aid"),
            "episode_number": epno_elem.text if epno_elem is not None else None,
            "length": (
                int(length_elem.text)
                if length_elem is not None and length_elem.text
                else None
            ),
            "airdate": airdate_elem.text if airdate_elem is not None else None,
            "rating": (
                float(rating_elem.text)
                if rating_elem is not None and rating_elem.text
                else None
            ),
            "votes": (
                int(votes_elem.text)
                if votes_elem is not None and votes_elem.text
                else None
            ),
            "summary": summary_elem.text if summary_elem is not None else None,
        }

        # Extract episode titles
        titles: dict[str, str | list[dict[str, str]] | None] = {}
        for title in root.findall("title"):
            lang = title.get("xml:lang") or title.get(
                "{http://www.w3.org/XML/1998/namespace}lang", "unknown"
            )
            if lang == "en":
                titles["english"] = title.text
            elif lang == "ja":
                titles["japanese"] = title.text
            elif lang == "x-jat":
                titles["romaji"] = title.text
            else:
                if "other" not in titles:
                    titles["other"] = []
                if title.text:
                    other_titles = cast(list[dict[str, str]], titles["other"])
                    other_titles.append(
                        {"lang": lang or "unknown", "title": title.text}
                    )
        episode_data["titles"] = titles

        return episode_data

    async def get_anime_by_id(self, anidb_id: int) -> dict[str, Any] | None:
        """Get anime information by AniDB ID."""
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

            return self._parse_anime_xml(xml_response)
        except Exception as e:
            logger.error(f"Failed to fetch anime by AniDB ID {anidb_id}: {e}")
            return None

    async def get_episode_by_id(self, episode_id: int) -> dict[str, Any] | None:
        """Get episode information by AniDB episode ID."""
        try:
            params = {"request": "episode", "eid": episode_id}
            xml_response = await self._make_request(params)

            if not xml_response or "<error" in xml_response:
                logger.warning(f"No data or error for AniDB episode ID: {episode_id}")
                return None

            return self._parse_episode_xml(xml_response)
        except Exception as e:
            logger.error(f"Failed to fetch episode by AniDB ID {episode_id}: {e}")
            return None

    async def search_anime_by_name(
        self, anime_name: str
    ) -> list[dict[str, Any]] | None:
        """Search anime by name using AniDB API."""
        try:
            params = {"request": "anime", "aname": anime_name}
            xml_response = await self._make_request(params)

            if not xml_response or "<error" in xml_response:
                logger.warning(f"No search results for: {anime_name}")
                return None

            # AniDB search returns single anime, not a list
            anime_data = self._parse_anime_xml(xml_response)
            return [anime_data] if anime_data else None
        except Exception as e:
            logger.error(f"Failed to search anime by name '{anime_name}': {e}")
            return None

    async def fetch_all_data(self, anidb_id: int) -> dict[str, Any] | None:
        """
        Fetch comprehensive AniDB data for an anime by AniDB ID.

        Args:
            anidb_id: The AniDB anime ID

        Returns:
            Dict containing comprehensive AniDB data or None if not found
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

    def get_health_status(self) -> dict[str, Any]:
        """Get comprehensive health and performance metrics."""
        current_time = time.time()

        return {
            "session_health": {
                "session_active": self.session is not None,
                "session_age": (
                    current_time - self._session_created_at if self.session else 0
                ),
                "session_max_age": self._session_max_age,
            },
            "circuit_breaker": {
                "state": self.circuit_breaker_state.value,
                "consecutive_failures": self.metrics.consecutive_failures,
                "threshold": self.circuit_breaker_threshold,
                "opened_at": self.circuit_breaker_opened_at,
                "time_until_retry": (
                    max(
                        0,
                        self.circuit_breaker_timeout
                        - (current_time - self.circuit_breaker_opened_at),
                    )
                    if self.circuit_breaker_state == CircuitBreakerState.OPEN
                    else 0
                ),
            },
            "rate_limiting": {
                "current_interval": self.metrics.current_interval,
                "min_interval": self.min_request_interval,
                "max_interval": self.max_request_interval,
                "time_since_last_request": current_time
                - self.metrics.last_request_time,
                "ready_for_request": (current_time - self.metrics.last_request_time)
                >= self.metrics.current_interval,
            },
            "request_metrics": {
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "failed_requests": self.metrics.failed_requests,
                "success_rate": self.metrics.success_rate,
                "error_rate": self.metrics.error_rate,
                "last_error_time": self.metrics.last_error_time,
            },
            "deduplication": {
                "recent_requests_tracked": len(self.recent_requests),
                "max_tracked_requests": 1000,
            },
            "configuration": {
                "client_name": self.client_name,
                "client_version": self.client_version,
                "max_retries": self.max_retries,
                "error_cooldown_base": self.error_cooldown_base,
            },
        }

    async def reset_circuit_breaker(self) -> bool:
        """Manually reset circuit breaker (admin function)."""
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
        """Close the HTTP session with cleanup."""
        if self.session:
            try:
                await self.session.close()
                logger.debug("AniDB session closed successfully")
            except Exception as e:
                logger.warning(f"Error closing AniDB session: {e}")
            finally:
                self.session = None
                self._session_created_at = 0


async def main() -> None:
    """Main function for testing AniDB data fetching."""
    parser = argparse.ArgumentParser(description="Test AniDB data fetching")
    parser.add_argument("--anidb-id", type=int, help="AniDB ID to fetch")
    parser.add_argument("--search-name", type=str, help="Search anime by name")
    parser.add_argument(
        "--output", type=str, default="test_anidb_output.json", help="Output file path"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    helper = AniDBEnrichmentHelper()

    try:
        if args.anidb_id:
            # Fetch data by ID
            anime_data = await helper.fetch_all_data(args.anidb_id)
        elif args.search_name:
            # Search by name
            search_results = await helper.search_anime_by_name(args.search_name)
            anime_data = search_results[0] if search_results else None
        else:
            logger.error("Must provide either --anidb-id or --search-name")
            return

        if anime_data:
            # Save to file
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(anime_data, f, indent=2, ensure_ascii=False)
        else:
            logger.error("No data found")

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
    finally:
        await helper.close()


if __name__ == "__main__":
    asyncio.run(main())
