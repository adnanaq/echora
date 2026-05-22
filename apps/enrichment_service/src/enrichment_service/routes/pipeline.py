"""Pipeline gRPC handler for enrichment_service."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import grpc
from common.grpc.error_details import build_error_details as error
from enrichment_proto.v1 import enrichment_service_pb2
from observability import registry
from opentelemetry import trace

from ..pipeline_runner import run_pipeline_and_write_artifact
from ..runtime import EnrichmentRuntime

logger = logging.getLogger(__name__)

_SAFE_AGENT_DIR_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")


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
    # AioServerInterceptor handles RPC-level tracing, duration, and error metrics.
    # This handler records pipeline-specific execution metrics and span events.
    current_span = trace.get_current_span()
    try:
        file_path = request.file_path or runtime.default_file_path
        resolved = Path(file_path).resolve()
        allowed = Path(runtime.default_file_path).resolve().parent
        if not resolved.is_relative_to(allowed):
            return enrichment_service_pb2.RunPipelineResponse(
                success=False,
                error=error(
                    "INVALID_FILE_PATH",
                    "file_path outside allowed directory",
                    retryable=False,
                ),
            )
        agent_dir = request.agent_dir or None
        if agent_dir is not None and not _SAFE_AGENT_DIR_RE.match(agent_dir):
            return enrichment_service_pb2.RunPipelineResponse(
                success=False,
                error=error(
                    "INVALID_AGENT_DIR",
                    "agent_dir must be a single path component containing only"
                    " alphanumerics, hyphens, or underscores",
                    retryable=False,
                ),
            )

        index = request.index if request.HasField("index") else None
        if index is not None and index < 0:
            index = None

        current_span.add_event(
            "validation.complete",
            {"agent_dir": agent_dir or "", "has_index": index is not None},
        )

        _pipeline_start = time.perf_counter()
        try:
            output_path, result, _payload = await run_pipeline_and_write_artifact(
                file_path=file_path,
                index=index,
                title=request.title or None,
                agent_dir=agent_dir,
                skip_services=list(request.skip_services),
                only_services=list(request.only_services),
                output_dir=runtime.output_dir,
            )
            _elapsed = time.perf_counter() - _pipeline_start
            registry.PIPELINE_RUNS.add(1, {"status": "success"})
            registry.PIPELINE_DURATION.record(_elapsed, {"status": "success"})
        except Exception:
            _elapsed = time.perf_counter() - _pipeline_start
            registry.PIPELINE_RUNS.add(1, {"status": "error"})
            registry.PIPELINE_DURATION.record(_elapsed, {"status": "error"})
            raise

        current_span.add_event("pipeline.complete", {"output_path": output_path})
        return enrichment_service_pb2.RunPipelineResponse(
            success=True,
            output_path=output_path,
            result_json=json.dumps(result, ensure_ascii=False),
        )
    except Exception as exc:
        logger.exception("RunPipeline RPC failed")
        return enrichment_service_pb2.RunPipelineResponse(
            success=False,
            error=error("RUN_PIPELINE_FAILED", str(exc)),
        )
