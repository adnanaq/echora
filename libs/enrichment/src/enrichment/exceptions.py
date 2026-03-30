"""Typed exception hierarchy for enrichment pipeline service errors."""

from __future__ import annotations

from typing import Any


class EnrichmentError(Exception):
    """Base for all enrichment pipeline errors."""


class ServiceError(EnrichmentError):
    """Error from a specific external service."""

    def __init__(self, message: str, *, service: str) -> None:
        self.service = service
        super().__init__(f"[{service}] {message}")


class ServiceRateLimitedError(ServiceError):
    """429 / rate limit exhausted after retries."""

    def __init__(self, *, service: str, attempts: int) -> None:
        self.attempts = attempts
        super().__init__(f"rate limit exceeded after {attempts} attempts", service=service)


class ServiceNetworkError(ServiceError):
    """Transient network failure (DNS, connection reset, 5xx) after retries."""

    def __init__(self, *, service: str, cause: BaseException | str) -> None:
        self.cause = cause
        super().__init__(f"network error: {cause}", service=service)


class ServiceNotFoundError(ServiceError):
    """404 — the requested resource does not exist."""


class ServiceBlockedError(ServiceError):
    """WAF / IP block that could not be recovered."""


class ServiceParseError(ServiceError):
    """Malformed response that failed validation/parsing."""

    def __init__(self, *, service: str, cause: BaseException | str) -> None:
        self.cause = cause
        super().__init__(f"parse error: {cause}", service=service)


class AniListGraphQLError(ServiceError):
    """AniList GraphQL-level errors — protocol-specific, not retryable."""

    def __init__(self, errors: list[dict[str, Any]]) -> None:
        self.errors = errors
        super().__init__(f"GraphQL errors: {errors}", service="anilist")
