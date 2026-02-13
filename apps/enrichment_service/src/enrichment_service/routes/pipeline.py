"""Pipeline gRPC handler for enrichment_service."""

from __future__ import annotations

import json
import logging

import grpc

from enrichment_proto.v1 import enrichment_service_pb2

from ..pipeline_runner import run_pipeline_and_write_artifact
from ..runtime import EnrichmentRuntime
from .shared import error

logger = logging.getLogger(__name__)


async def run_pipeline(
    runtime: EnrichmentRuntime,
    request: enrichment_service_pb2.RunPipelineRequest,
    context: grpc.aio.ServicerContext,
) -> enrichment_service_pb2.RunPipelineResponse:
    """Execute enrichment pipeline request and return artifact metadata.

    Args:
        runtime: Initialized runtime dependencies.
        request: RunPipeline RPC request payload.
        context: gRPC request context.

    Returns:
        Pipeline result payload or structured error.
    """
    del context
    try:
        file_path = request.file_path or runtime.default_file_path
        index = request.index if request.HasField("index") else None
        if request.index < 0:
            index = None

        output_path, result = await run_pipeline_and_write_artifact(
            file_path=file_path,
            index=index,
            title=request.title or None,
            agent_dir=request.agent_dir or None,
            skip_services=list(request.skip_services) or None,
            only_services=list(request.only_services) or None,
            output_dir=runtime.output_dir,
        )
        return enrichment_service_pb2.RunPipelineResponse(
            success=True,
            output_path=output_path,
            result_json=json.dumps(result, ensure_ascii=False),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("RunPipeline RPC failed")
        return enrichment_service_pb2.RunPipelineResponse(
            success=False,
            error=error("RUN_PIPELINE_FAILED", str(exc)),
        )
