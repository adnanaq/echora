"""Exceptions for enrichment pipeline API helpers."""

from __future__ import annotations

from typing import Any


class AniListAPIError(Exception):
    """Base exception for AniList API errors."""


class AniListRateLimitError(AniListAPIError):
    """Raised when AniList API rate limit is exhausted after retries."""

    def __init__(self, attempts: int) -> None:
        """Initialize rate limit error with retry attempt count.

        Args:
            attempts: Number of retry attempts exhausted.
        """
        self.attempts = attempts
        super().__init__(f"AniList rate limit exceeded after {attempts} retry attempts")


class AniListGraphQLError(AniListAPIError):
    """Raised when AniList API returns GraphQL errors in response."""

    def __init__(self, errors: list[dict[str, Any]]) -> None:
        """Initialize GraphQL error with error details.

        Args:
            errors: GraphQL error objects from response.
        """
        self.errors = errors
        super().__init__(f"AniList GraphQL errors: {errors}")


class AniListNetworkError(AniListAPIError):
    """Raised when AniList API request fails due to network or JSON decode errors."""

    def __init__(self, cause: BaseException | str) -> None:
        """Initialize network error with underlying cause.

        Args:
            cause: The underlying exception or error description.
        """
        self.cause = cause
        super().__init__(f"AniList API request failed: {cause}")

    @classmethod
    def exhausted_retries(cls) -> "AniListNetworkError":
        """Sentinel for the unreachable post-retry fallback path."""
        return cls("exhausted retries unexpectedly")
