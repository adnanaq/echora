#!/usr/bin/env python3
"""Time the BGE-M3 model on its own, outside the service, for batching choices.

For batches of search texts in the load test's query mix, prints per
combination of runner, batch size, chunk size and token budget: the median
time per batch, texts/s, model passes per batch, and the lowest dense cosine
against one eager pass (vectors should agree within fp16 rounding).

Runners:

- ``eager``: the model as the service runs it
- ``compiled``: ``torch.compile`` on the transformer
- ``cuda_graphs``: ``torch.compile`` with ``mode="reduce-overhead"``; each pass
  is padded to fixed sizes (rows to 8/16/32/64/128/256, tokens to a multiple
  of 16) so recorded graphs can be replayed

``--busy-python-thread`` repeats every timing with a second Python thread busy
about 45% of the time, as the service's event loop is under load, to show how
much Python's GIL slows a model call. ``--phases`` splits one batch's time into
tokenizing, the model pass and converting results.

Needs the GPU the service would use. Run from the repository root:

    ./pants run benchmarks/vector_service/diagnostics/time_model_passes.py -- \\
      --batch-sizes 41 64 --chunk-sizes 256 32 --token-budgets 0 256 512

The ``compiled`` and ``cuda_graphs`` runners build GPU code with Triton, which
needs a Python with its C headers; the one Pants uses has none, so run those
with the project's own Python:

    PYTHONPATH=$(printf '%s:' libs/*/src apps/*/src) .venv/bin/python -m \\
      benchmarks.vector_service.diagnostics.time_model_passes --runners compiled
"""

import argparse
import functools
import random
import statistics
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from vector_processing.embedding_models.text.flagembedding_model import (
    FlagEmbeddingModel,
)

from benchmarks.vector_service.toolkit.fixed_shapes import fixed_row_count
from benchmarks.vector_service.toolkit.query_mix import (
    DEFAULT_WEIGHTS,
    load_queries,
    parse_query_weights,
    sample_query_mix,
)

FIXED_ROW_SIZES = (8, 16, 32, 64, 128, 256)
FIXED_LENGTH_STEP = 16
RUNNERS = ("eager", "compiled", "cuda_graphs")


@dataclass(frozen=True)
class Combination:
    batch_size: int
    chunk_size: int
    token_budget: int


class PassCounter:
    """Counts model passes by wrapping the model's per-chunk call."""

    def __init__(self, model: FlagEmbeddingModel) -> None:
        self.passes = 0
        run_chunk = model._run_model

        def counted(chunk: list[dict[str, Any]], **options: Any) -> Any:
            self.passes += 1
            return run_chunk(chunk, **options)

        vars(model)["_run_model"] = counted


def use_fixed_shapes(model: FlagEmbeddingModel) -> None:
    """Pad every pass to a fixed row count and a multiple of 16 tokens."""
    tokenizer = model._tokenizer
    vars(tokenizer)["pad"] = functools.partial(
        tokenizer.pad, pad_to_multiple_of=FIXED_LENGTH_STEP
    )
    run_chunk = model._run_model

    def padded(chunk: list[dict[str, Any]], **options: Any) -> Any:
        rows = chunk + [chunk[-1]] * (
            fixed_row_count(len(chunk), FIXED_ROW_SIZES) - len(chunk)
        )
        dense, weights = run_chunk(rows, **options)
        return dense[: len(chunk)], weights[: len(chunk)]

    vars(model)["_run_model"] = padded


def load_model(runner: str) -> FlagEmbeddingModel:
    model = FlagEmbeddingModel("BAAI/bge-m3")
    if runner == "compiled":
        model._encoder.model = torch.compile(model._encoder.model, dynamic=True)
    elif runner == "cuda_graphs":
        torch._dynamo.config.cache_size_limit = 256
        model._encoder.model = torch.compile(
            model._encoder.model, mode="reduce-overhead", dynamic=False
        )
        use_fixed_shapes(model)
    return model


def encode_dense(model: FlagEmbeddingModel, texts: list[str]) -> np.ndarray:
    dense, _ = model.encode_with_sparse(texts)
    return np.asarray(dense, dtype=np.float32)


def median_batch_ms(model: FlagEmbeddingModel, batches: list[list[str]]) -> float:
    times = []
    for batch in batches:
        torch.cuda.synchronize()
        started = time.perf_counter()
        model.encode_with_sparse(batch)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - started) * 1000)
    return statistics.median(times)


def keep_python_busy(stop: threading.Event) -> None:
    while not stop.is_set():
        end = time.perf_counter() + 0.001
        while time.perf_counter() < end:
            pass
        time.sleep(0.0012)


def with_busy_python_thread(measure: Callable[[], float]) -> float:
    stop = threading.Event()
    thread = threading.Thread(target=keep_python_busy, args=(stop,))
    thread.start()
    try:
        return measure()
    finally:
        stop.set()
        thread.join()


def lowest_cosine(
    model: FlagEmbeddingModel, batches: list[list[str]], references: list[np.ndarray]
) -> float:
    return min(
        float(np.min(np.sum(encode_dense(model, batch) * reference, axis=1)))
        for batch, reference in zip(batches, references, strict=True)
    )


def measure_combination(
    model: FlagEmbeddingModel,
    counter: PassCounter,
    batches: list[list[str]],
    combination: Combination,
    busy_python_thread: bool,
) -> dict[str, float]:
    model._chunk_size = combination.chunk_size
    model._max_tokens_per_pass = combination.token_budget
    for batch in batches[:3]:
        model.encode_with_sparse(batch)
    counter.passes = 0
    alone = median_batch_ms(model, batches)
    result = {"alone_ms": alone, "passes": counter.passes / len(batches)}
    if busy_python_thread:
        result["busy_ms"] = with_busy_python_thread(
            lambda: median_batch_ms(model, batches)
        )
    return result


def print_phases(model: FlagEmbeddingModel, batches: list[list[str]]) -> None:
    """Median milliseconds of tokenizing, the model pass and converting, one pass per batch."""
    from FlagEmbedding.utils.tokenizer_compat import pad_with_compat

    phases: dict[str, list[float]] = {"tokenize": [], "model": [], "convert": []}
    for texts in batches:
        started = time.perf_counter()
        tokenized = model._tokenizer(
            texts, truncation=True, max_length=model._max_length
        )
        examples = [
            {key: tokenized[key][i] for key in tokenized} for i in range(len(texts))
        ]
        inputs = pad_with_compat(
            model._tokenizer, examples, padding=True, return_tensors="pt"
        )
        inputs = inputs.to(model._device)
        torch.cuda.synchronize()
        tokenized_at = time.perf_counter()
        with torch.no_grad():
            outputs = model._encoder(inputs, return_dense=True, return_sparse=True)
        torch.cuda.synchronize()
        model_at = time.perf_counter()
        weights = outputs["sparse_vecs"].squeeze(-1).float().cpu().numpy()
        outputs["dense_vecs"].float().cpu().numpy().tolist()
        for row, token_ids in zip(weights, inputs["input_ids"].tolist(), strict=True):
            model._token_weights(row, token_ids)
        phases["tokenize"].append((tokenized_at - started) * 1000)
        phases["model"].append((model_at - tokenized_at) * 1000)
        phases["convert"].append((time.perf_counter() - model_at) * 1000)
    split = ", ".join(
        f"{name} {statistics.median(values):.1f} ms" for name, values in phases.items()
    )
    print(f"one pass per batch of {len(batches[0])}: {split}")


def print_row(
    runner: str, combination: Combination, result: dict[str, float], cosine: float
) -> None:
    alone = result["alone_ms"]
    busy = result.get("busy_ms")
    busy_text = (
        f" | busy thread {busy:6.1f} ms {combination.batch_size / busy * 1000:5.0f}/s"
        if busy
        else ""
    )
    print(
        f"{runner:11} batch {combination.batch_size:3} chunk {combination.chunk_size:3} "
        f"budget {combination.token_budget:4} | {alone:6.1f} ms "
        f"{combination.batch_size / alone * 1000:5.0f} texts/s | passes {result['passes']:4.1f} "
        f"| lowest cosine {cosine:.5f}{busy_text}",
        flush=True,
    )


def run_runner(
    runner: str, args: argparse.Namespace, batches_by_size: dict[int, list[list[str]]]
) -> None:
    model = load_model(runner)
    counter = PassCounter(model)
    for batch_size, batches in batches_by_size.items():
        model._chunk_size, model._max_tokens_per_pass = 256, 0
        references = [encode_dense(model, batch) for batch in batches[:5]]
        for chunk_size in args.chunk_sizes:
            for token_budget in args.token_budgets:
                combination = Combination(batch_size, chunk_size, token_budget)
                result = measure_combination(
                    model, counter, batches, combination, args.busy_python_thread
                )
                print_row(
                    runner,
                    combination,
                    result,
                    lowest_cosine(model, batches[:5], references),
                )
    if args.phases and runner == "eager":
        for batches in batches_by_size.values():
            print_phases(model, batches)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runners", nargs="+", choices=RUNNERS, default=["eager"])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[41, 64])
    parser.add_argument("--chunk-sizes", nargs="+", type=int, default=[256, 32])
    parser.add_argument("--token-budgets", nargs="+", type=int, default=[0, 256, 512])
    parser.add_argument(
        "--rounds", type=int, default=30, help="batches timed per combination"
    )
    parser.add_argument(
        "--query-mix", default=DEFAULT_WEIGHTS, help="short,title,long weights"
    )
    parser.add_argument("--seed", type=int, default=54)
    parser.add_argument("--busy-python-thread", action="store_true")
    parser.add_argument("--phases", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    queries = load_queries()
    weights = parse_query_weights(args.query_mix)
    random_generator = random.Random(args.seed)  # noqa: S311 - repeatable sample, not security
    batches_by_size = {
        size: [
            sample_query_mix(queries, weights, size, random_generator)
            for _ in range(args.rounds)
        ]
        for size in args.batch_sizes
    }
    for runner in args.runners:
        run_runner(runner, args, batches_by_size)


if __name__ == "__main__":
    main()
