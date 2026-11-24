"""Seed-cleaning gRPC route handlers for enrichment_service."""

from __future__ import annotations

from pathlib import Path

import grpc
from enrichment_proto.v1 import enrichment_service_pb2
from shared_proto.v1 import review_cases_pb2

from ..runtime import EnrichmentRuntime


async def start_seed_cleaning(
    runtime: EnrichmentRuntime,
    request: enrichment_service_pb2.StartSeedCleaningRequest,
    context: grpc.aio.ServicerContext,
) -> review_cases_pb2.RunState:
    """Start a background seed-cleaning run when none is active.

    Args:
        runtime: Initialized runtime dependencies.
        request: gRPC request specifying seed file, data dir, and fetch flags.
        context: Active gRPC servicer context for status reporting.

    Returns:
        Current ``RunState`` marked as running.
    """
    if not request.seed_file:
        await context.abort(
            grpc.StatusCode.INVALID_ARGUMENT,
            "seed_file must be provided",
        )
    if not request.data_dir:
        await context.abort(
            grpc.StatusCode.INVALID_ARGUMENT,
            "data_dir must be provided",
        )
    if runtime.seed_cleaning_runner is None:
        await context.abort(
            grpc.StatusCode.FAILED_PRECONDITION,
            "seed_cleaning_runner is not configured",
        )
    return runtime.seed_cleaning_runner.start_run(
        seed_file=Path(request.seed_file),
        data_dir=Path(request.data_dir),
        no_fetch=request.no_fetch,
        refetch_stored=request.refetch_stored,
    )


async def get_seed_cleaning_status(
    runtime: EnrichmentRuntime,
    request: enrichment_service_pb2.GetSeedCleaningStatusRequest,
    context: grpc.aio.ServicerContext,
) -> review_cases_pb2.RunState:
    """Report whether a seed-cleaning run is active and its last exit code.

    Args:
        runtime: Initialized runtime dependencies.
        request: gRPC request specifying the seed-cleaning data directory.
        context: Active gRPC servicer context for status reporting.

    Returns:
        Current ``RunState`` protobuf message.
    """
    if not request.data_dir:
        await context.abort(
            grpc.StatusCode.INVALID_ARGUMENT,
            "data_dir must be provided",
        )
    if runtime.seed_cleaning_runner is None:
        await context.abort(
            grpc.StatusCode.FAILED_PRECONDITION,
            "seed_cleaning_runner is not configured",
        )
    return runtime.seed_cleaning_runner.read_run_state(Path(request.data_dir))
