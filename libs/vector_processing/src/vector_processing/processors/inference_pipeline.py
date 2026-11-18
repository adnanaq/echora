"""Shared async inference, caching, and telemetry pipeline for processors."""

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from typing import Any, Literal

from opentelemetry import metrics as otel_metrics
from opentelemetry import trace as otel_trace

from ..cache import EmbeddingCache

logger = logging.getLogger(__name__)

type EmbeddingModality = Literal["text", "image"]

_tracer = otel_trace.get_tracer("echora.vector_processing")
_meter = otel_metrics.get_meter("echora.vector_processing")
_embedding_duration = _meter.create_histogram(
    "echora_embedding_duration_seconds",
    unit="s",
    description="Embedding model inference duration in seconds",
)
_embedding_cache_lookups = _meter.create_counter(
    "echora_embedding_cache_total",
    description="Embedding cache lookups by result (hit/miss) and modality",
)


async def fetch_cached_embeddings(
    cache: EmbeddingCache | None,
    model_name: str,
    input_hashes: Sequence[str | None],
    modality: EmbeddingModality,
) -> tuple[list[list[float] | None], list[int]]:
    """Fetch cached embeddings for hashed inputs and record hit/miss metrics.

    Args:
        cache: Optional embedding cache backend.
        model_name: Embedding model identifier used as part of cache keys.
        input_hashes: Content SHA-256 hashes aligned with inputs, or ``None``
            when an input could not be hashed.
        modality: Embedding modality for OpenTelemetry metrics (``"text"`` or ``"image"``).

    Returns:
        Tuple of ``(cached_embeddings, uncached_positions)`` aligned with
        ``input_hashes``.
    """
    total_inputs = len(input_hashes)
    cached_embeddings: list[list[float] | None] = [None] * total_inputs
    if cache is None:
        return cached_embeddings, list(range(total_inputs))

    valid_hashes = [digest for digest in input_hashes if digest is not None]
    valid_hash_indices = [
        idx for idx, digest in enumerate(input_hashes) if digest is not None
    ]
    if valid_hashes:
        batch_cached = await cache.get_batch(model_name, valid_hashes)
        for batch_idx, input_idx in enumerate(valid_hash_indices):
            cached_embeddings[input_idx] = batch_cached[batch_idx]

    uncached_positions = [
        idx for idx in range(total_inputs) if cached_embeddings[idx] is None
    ]
    cache_hits = total_inputs - len(uncached_positions)
    if cache_hits:
        _embedding_cache_lookups.add(
            cache_hits, {"result": "hit", "modality": modality}
        )
    if uncached_positions:
        _embedding_cache_lookups.add(
            len(uncached_positions), {"result": "miss", "modality": modality}
        )
    return cached_embeddings, uncached_positions


async def run_cached_batch_inference[InputItem](
    inputs: Sequence[InputItem],
    input_hashes: Sequence[str | None],
    cached_embeddings: list[list[float] | None],
    uncached_positions: list[int],
    encode_fn: Callable[[list[InputItem]], Sequence[list[float] | None]],
    cache: EmbeddingCache | None,
    model_name: str,
    semaphore: asyncio.Semaphore,
    span_name: str,
    span_attributes: dict[str, Any],
    modality: EmbeddingModality,
    failure_message: str,
) -> bool:
    """Encode uncached batch items under tracing and semaphore control and cache results.

    Args:
        inputs: Full sequence of valid batch inputs.
        input_hashes: Content hashes aligned with ``inputs``.
        cached_embeddings: Mutable list of embeddings updated in place on success.
        uncached_positions: Indices into ``inputs`` missing from cache.
        encode_fn: Synchronous model encoding callable accepting uncached inputs.
        cache: Optional embedding cache backend.
        model_name: Embedding model identifier for cache writes.
        semaphore: Concurrency limiter guarding model inference.
        span_name: OpenTelemetry span name for the batch operation.
        span_attributes: OpenTelemetry span attributes for the batch operation.
        modality: Embedding modality for inference duration metrics (``"text"`` or ``"image"``).
        failure_message: Log message emitted when encoding raises an exception.

    Returns:
        True when inference succeeded, False when inference raised an exception.
    """
    uncached_inputs = [inputs[idx] for idx in uncached_positions]
    with _tracer.start_as_current_span(span_name, attributes=span_attributes):
        try:
            async with semaphore:
                start_time = time.perf_counter()
                encoded_embeddings = await asyncio.to_thread(encode_fn, uncached_inputs)
                _embedding_duration.record(
                    time.perf_counter() - start_time, {"modality": modality}
                )
        except Exception:
            logger.exception(failure_message)
            return False

        embeddings_to_cache: dict[str, list[float]] = {}
        for encoded_idx, pos in enumerate(uncached_positions):
            embedding = encoded_embeddings[encoded_idx]
            cached_embeddings[pos] = embedding
            input_hash = input_hashes[pos] if pos < len(input_hashes) else None
            if cache is not None and embedding is not None and input_hash is not None:
                embeddings_to_cache[input_hash] = embedding

        if cache is not None and embeddings_to_cache:
            await cache.set_batch(model_name, embeddings_to_cache)

    return True


async def run_cached_single_inference[InputItem](
    single_input: InputItem,
    input_hash: str | None,
    encode_fn: Callable[[list[InputItem]], list[list[float]]],
    cache: EmbeddingCache | None,
    model_name: str,
    semaphore: asyncio.Semaphore,
    span_name: str,
    span_attributes: dict[str, Any],
    modality: EmbeddingModality,
    failure_message: str,
) -> list[float] | None:
    """Encode a single uncached input under tracing and semaphore control and cache it.

    Args:
        single_input: Input value to encode.
        input_hash: Optional SHA-256 content hash for cache write-back.
        encode_fn: Synchronous model encoding callable accepting a single-item list.
        cache: Optional embedding cache backend.
        model_name: Embedding model identifier for cache writes.
        semaphore: Concurrency limiter guarding model inference.
        span_name: OpenTelemetry span name for the single-item operation.
        span_attributes: OpenTelemetry span attributes for the operation.
        modality: Embedding modality for inference duration metrics (``"text"`` or ``"image"``).
        failure_message: Log message emitted when encoding raises an exception.

    Returns:
        Encoded embedding vector, or None if inference failed or returned empty.
    """
    with _tracer.start_as_current_span(span_name, attributes=span_attributes):
        try:
            async with semaphore:
                start_time = time.perf_counter()
                embeddings = await asyncio.to_thread(encode_fn, [single_input])
                _embedding_duration.record(
                    time.perf_counter() - start_time, {"modality": modality}
                )
        except Exception:
            logger.exception(failure_message)
            return None
        else:
            if not embeddings:
                return None
            embedding = embeddings[0]
            if cache is not None and input_hash is not None:
                await cache.set(model_name, input_hash, embedding)
            return embedding
