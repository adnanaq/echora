"""Shared page fetcher interface and raw response helpers for providers."""

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar

from shared_proto.v1.anime_pb2 import EntityType
from shared_proto.v1.merge_pb2 import PageKey, PageResponse, StoredPage


@dataclass(frozen=True, slots=True)
class FetchedPage:
    """Raw provider page responses and not-found status returned by a fetcher."""

    page_key: PageKey
    responses: tuple[PageResponse, ...]
    is_not_found: bool = False

    def to_stored_page(self, fetched_at: datetime | None = None) -> StoredPage:
        """Convert this fetched page into a StoredPage protobuf message.

        Args:
            fetched_at: UTC timestamp when the fetch completed. Defaults to now.

        Returns:
            Populated StoredPage protobuf message.
        """
        resolved_time = fetched_at or datetime.now(UTC)
        return StoredPage(
            page_key=self.page_key,
            fetched_at=resolved_time,
            is_dead=False,
            responses=self.responses,
            is_not_found=self.is_not_found,
        )


class PageFetcher(ABC):
    """Abstract base class for provider fetchers returning raw page responses."""

    provider: ClassVar[str]
    supported_entity_types: ClassVar[frozenset[EntityType]]

    @abstractmethod
    async def fetch_page(
        self,
        page_key: PageKey,
        *,
        max_age: timedelta | None = None,
    ) -> FetchedPage:
        """Fetch raw responses for a single provider page key.

        Args:
            page_key: Target provider page identifier.
            max_age: Optional cache freshness bound; zero forces a live fetch.

        Returns:
            FetchedPage holding raw PageResponse messages or not-found status.
        """

    async def close(self) -> None:
        """Release any open transport sessions held by this fetcher."""


def build_cache_headers(max_age: timedelta | None) -> dict[str, str]:
    """Build RFC 9111 Cache-Control request headers for a max_age bound.

    Args:
        max_age: Maximum acceptable cached response age, or None for default.

    Returns:
        Request headers dictionary for Cache-Control, or an empty dictionary.
    """
    if max_age is None:
        return {}
    seconds = int(max_age.total_seconds())
    if seconds <= 0:
        return {"Cache-Control": "no-cache"}
    return {"Cache-Control": f"max-age={seconds}"}


def validate_page_key(
    page_key: PageKey,
    expected_provider: str,
    supported_types: frozenset[EntityType],
    *,
    numeric_address: bool = False,
) -> None:
    """Validate that a page key matches a fetcher's provider, type, and address rules.

    Args:
        page_key: Candidate provider page key.
        expected_provider: Required provider slug.
        supported_types: Allowed entity types for this fetcher.
        numeric_address: When True, require an ASCII digit-only address.

    Raises:
        ValueError: When provider, entity_type, or address violates rules.
    """
    if page_key.provider != expected_provider:
        error_message = (
            f"Expected provider {expected_provider!r}, got {page_key.provider!r}"
        )
        raise ValueError(error_message)
    if page_key.entity_type not in supported_types:
        error_message = (
            f"Unsupported entity_type {page_key.entity_type} for {expected_provider}"
        )
        raise ValueError(error_message)
    address = page_key.address
    if not address or "/" in address or any(char.isspace() for char in address):
        error_message = f"Invalid page address {address!r} for {expected_provider}"
        raise ValueError(error_message)
    if numeric_address and not (address.isascii() and address.isdigit()):
        error_message = (
            f"Provider {expected_provider} requires a numeric address, got {address!r}"
        )
        raise ValueError(error_message)


async def read_response_bytes(response: Any) -> tuple[bytes, str]:
    """Read raw body bytes and normalized media type from an HTTP response.

    Args:
        response: Open aiohttp client response.

    Returns:
        Tuple of raw response bytes and MIME content type string.
    """
    raw_body = await response.read()
    raw_header = response.headers.get("Content-Type", "application/json")
    content_type = raw_header.split(";", maxsplit=1)[0].strip() or "application/json"
    return raw_body, content_type


def index_responses_by_name(
    responses: Sequence[PageResponse],
) -> dict[str, PageResponse]:
    """Index stored page responses by their request_name.

    Args:
        responses: Sequence of PageResponse messages from a stored page.

    Returns:
        Mapping from request_name to PageResponse.
    """
    return {response.request_name: response for response in responses}


def parse_json_response(response: PageResponse) -> dict[str, Any]:
    """Decode a JSON PageResponse body into a Python dictionary.

    Args:
        response: PageResponse holding UTF-8 JSON bytes.

    Returns:
        Decoded JSON object as a dictionary.

    Raises:
        ValueError: When the decoded JSON payload is not a dictionary.
    """
    decoded = json.loads(response.body.decode("utf-8"))
    if not isinstance(decoded, dict):
        error_message = f"Expected JSON object in {response.request_name!r}, got {type(decoded).__name__}"
        raise TypeError(error_message)
    return decoded


def collect_paginated_responses(
    responses: Sequence[PageResponse],
    prefix: str,
) -> list[dict[str, Any]]:
    """Decode and return paginated JSON responses matching a request_name prefix.

    Args:
        responses: Sequence of PageResponse messages from a stored page.
        prefix: Request name prefix such as ``"characters_page_"``.

    Returns:
        List of decoded JSON dictionaries ordered by page suffix number.
    """
    matching = [
        response for response in responses if response.request_name.startswith(prefix)
    ]

    def _page_sort_key(page_response: PageResponse) -> int:
        suffix = page_response.request_name.removeprefix(prefix)
        return int(suffix) if suffix.isdigit() else 0

    matching.sort(key=_page_sort_key)
    return [parse_json_response(response) for response in matching]
