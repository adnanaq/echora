#!/usr/bin/env python3
"""Run the vector service with per-stage timing, to find where searches wait.

A diagnostic run, not for production: it wraps the service's own functions
and prints, every 10 s to stderr, the mean, p50 and p95 of:

- ``search handler``: one search from start to finish inside the service
- ``model wait per search`` / ``qdrant wait per search``: from queueing a
  search's text or query to getting its result (queue time plus the call)
- ``model batch call`` / ``qdrant batch call`` and their batch sizes
- ``model encode in thread``: the model call inside its worker thread
- ``event loop lag``: how late a 10 ms timer fires; high values mean the
  event loop thread is saturated

It reads the same environment variables as the service. Run it outside Docker:

    PYTHONPATH=$(ls -d libs/*/src apps/*/src | tr '\\n' ':') \\
      .venv/bin/python benchmarks/vector_service/diagnostics/run_timed_vector_service.py
"""

import asyncio
import statistics
import sys
import time
from collections import defaultdict
from typing import Any
from unittest.mock import patch

import grpc
from common.utils import request_batcher
from vector_processing.embedding_models.text import flagembedding_model
from vector_proto.v1 import vector_search_pb2
from vector_service import main as service_main
from vector_service.routes import search as search_route
from vector_service.runtime import VectorRuntime

REPORT_SECONDS = 10
samples: dict[str, list[float]] = defaultdict(list)


def kind_of(batcher: request_batcher.RequestBatcher[Any, Any]) -> str:
    batch_function_name = getattr(batcher._process_batch, "__qualname__", "")
    return "model" if "encode" in batch_function_name else "qdrant"


original_submit = request_batcher.RequestBatcher.submit
original_process = request_batcher.RequestBatcher._process_and_deliver
original_search = search_route.search
original_encode = flagembedding_model.FlagEmbeddingModel.encode_with_sparse


async def timed_submit(
    self: request_batcher.RequestBatcher[Any, Any], item: Any
) -> Any:
    started = time.perf_counter()
    try:
        return await original_submit(self, item)
    finally:
        samples[f"{kind_of(self)} wait per search"].append(
            time.perf_counter() - started
        )


async def timed_process(
    self: request_batcher.RequestBatcher[Any, Any], batch: list[Any]
) -> None:
    started = time.perf_counter()
    try:
        return await original_process(self, batch)
    finally:
        samples[f"{kind_of(self)} batch call"].append(time.perf_counter() - started)
        samples[f"{kind_of(self)} batch size"].append(len(batch) / 1000)


async def timed_search(
    runtime: VectorRuntime,
    request: vector_search_pb2.SearchRequest,
    context: grpc.aio.ServicerContext,
) -> vector_search_pb2.SearchResponse:
    started = time.perf_counter()
    try:
        return await original_search(runtime, request, context)
    finally:
        samples["search handler"].append(time.perf_counter() - started)


def timed_encode(self: flagembedding_model.FlagEmbeddingModel, texts: list[str]) -> Any:
    started = time.perf_counter()
    try:
        return original_encode(self, texts)
    finally:
        samples["model encode in thread"].append(time.perf_counter() - started)


async def loop_lag_monitor() -> None:
    while True:
        started = time.perf_counter()
        await asyncio.sleep(0.01)
        samples["event loop lag"].append(time.perf_counter() - started - 0.01)


async def reporter() -> None:
    while True:
        await asyncio.sleep(REPORT_SECONDS)
        lines = []
        for name in sorted(samples):
            values = sorted(samples[name])
            samples[name] = []
            if not values:
                continue
            p95 = values[min(len(values) - 1, int(0.95 * len(values)))]
            lines.append(
                f"  {name:24} n={len(values):6d} mean={statistics.mean(values) * 1000:7.1f}"
                f" p50={statistics.median(values) * 1000:7.1f} p95={p95 * 1000:7.1f}"
            )
        print(
            f"[{time.strftime('%H:%M:%S')}]\n" + "\n".join(lines),
            file=sys.stderr,
            flush=True,
        )


async def serve_with_timing() -> None:
    asyncio.create_task(loop_lag_monitor())
    asyncio.create_task(reporter())
    await service_main.serve()


def main() -> None:
    with (
        patch.object(request_batcher.RequestBatcher, "submit", timed_submit),
        patch.object(
            request_batcher.RequestBatcher, "_process_and_deliver", timed_process
        ),
        patch.object(search_route, "search", timed_search),
        patch.object(
            flagembedding_model.FlagEmbeddingModel, "encode_with_sparse", timed_encode
        ),
    ):
        asyncio.run(serve_with_timing())


if __name__ == "__main__":
    main()
