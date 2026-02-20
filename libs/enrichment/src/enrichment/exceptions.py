"""Exceptions for enrichment pipeline API helpers."""


class AniListAPIError(Exception):
    """Base exception for AniList API errors."""

    pass


class AniListRateLimitError(AniListAPIError):
    """Raised when AniList API rate limit is exhausted after retries."""

    pass


class AniListGraphQLError(AniListAPIError):
    """Raised when AniList API returns GraphQL errors in response."""

    pass


class AniListNetworkError(AniListAPIError):
    """Raised when AniList API request fails due to network or JSON decode errors."""

    pass
