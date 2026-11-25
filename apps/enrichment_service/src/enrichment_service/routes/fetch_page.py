"""FetchPage gRPC route and provider fetcher registry for enrichment_service."""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import TYPE_CHECKING

import grpc
from enrichment.sources.anidb.anidb_fetcher import AniDBFetcher
from enrichment.sources.anilist.anilist_fetcher import AniListFetcher
from enrichment.sources.anime_planet.anime_planet_fetcher import AnimePlanetFetcher
from enrichment.sources.animeschedule.animeschedule_fetcher import (
    AnimeScheduleFetcher,
)
from enrichment.sources.anisearch.anisearch_fetcher import AniSearchFetcher
from enrichment.sources.base.exceptions import (
    ServiceBlockedError,
    ServiceNetworkError,
    ServiceNotConfiguredError,
    ServiceParseError,
    ServiceRateLimitedError,
)
from enrichment.sources.base.page_fetcher import PageFetcher
from enrichment.sources.kitsu.kitsu_fetcher import KitsuFetcher
from enrichment.sources.mal.mal_fetcher import MalFetcher
from enrichment.sources.syoboi.syoboi_fetcher import SyoboiFetcher
from enrichment_proto.v1 import enrichment_service_pb2

if TYPE_CHECKING:
    from ..runtime import EnrichmentRuntime

logger = logging.getLogger(__name__)


def build_page_fetchers(runtime: EnrichmentRuntime) -> dict[str, PageFetcher]:
    """Instantiate the provider PageFetcher registry from runtime dependencies.

    Args:
        runtime: Initialized enrichment service runtime.

    Returns:
        Mapping from provider slug to its PageFetcher implementation.
    """
    return {
        "anilist": AniListFetcher(runtime.cache),
        "kitsu": KitsuFetcher(runtime.cache),
        "mal": MalFetcher(),
        "anime_planet": AnimePlanetFetcher(),
        "anisearch": AniSearchFetcher(),
        "anidb": AniDBFetcher(runtime.enrichment, runtime.cache),
        "animeschedule": AnimeScheduleFetcher(
            runtime.cache, runtime.enrichment.animeschedule_token
        ),
        "syoboi": SyoboiFetcher(runtime.cache),
    }


def _read_max_age(request: enrichment_service_pb2.FetchPageRequest) -> timedelta | None:
    """Convert an optional protobuf Duration into a timedelta.

    Args:
        request: Incoming FetchPageRequest protobuf message.

    Returns:
        Timedelta freshness bound, or None when max_age is unset.
    """
    if not request.HasField("max_age"):
        return None
    return timedelta(
        seconds=request.max_age.seconds,
        microseconds=request.max_age.nanos // 1000,
    )


async def fetch_page(
    runtime: EnrichmentRuntime,
    request: enrichment_service_pb2.FetchPageRequest,
    context: grpc.aio.ServicerContext,
) -> enrichment_service_pb2.FetchPageResponse:
    """Handle FetchPage RPC by fetching raw provider responses without mapping.

    Args:
        runtime: Enrichment service runtime dependencies.
        request: FetchPageRequest carrying target page_key and optional max_age.
        context: Active gRPC servicer context for status reporting.

    Returns:
        FetchPageResponse wrapping the resulting StoredPage.
    """
    if not request.HasField("page_key"):
        await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "page_key is required")

    page_key = request.page_key
    fetchers = runtime.page_fetchers or build_page_fetchers(runtime)
    fetcher = fetchers.get(page_key.provider)
    if fetcher is None:
        await context.abort(
            grpc.StatusCode.INVALID_ARGUMENT,
            f"unknown provider {page_key.provider!r}",
        )
    assert fetcher is not None  # noqa: S101

    try:
        fetched = await fetcher.fetch_page(page_key, max_age=_read_max_age(request))
    except ValueError as exc:
        await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
    except (ServiceRateLimitedError, ServiceBlockedError) as exc:
        await context.abort(grpc.StatusCode.UNAVAILABLE, str(exc))
    except ServiceNotConfiguredError as exc:
        await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
    except (ServiceNetworkError, ServiceParseError) as exc:
        await context.abort(grpc.StatusCode.UNAVAILABLE, str(exc))

    return enrichment_service_pb2.FetchPageResponse(page=fetched.to_stored_page())
