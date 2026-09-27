#!/usr/bin/env python3
"""Summarize a py-spy profile of the vector service by thread and by area.

Reads the folded stacks written by ``benchmarks/vector_service/diagnostics/profile_service.sh``
(``py-spy record --format raw``) and prints:

- how busy the event loop thread and the other threads (model and gRPC
  worker threads) were: samples divided by the samples a fully busy thread
  would produce (py-spy records nothing for an idle thread)
- where the event loop's time went, grouped into areas of the code
- the functions most often on top of each stack, per thread group

Usage: uv run python benchmarks/vector_service/diagnostics/summarize_profile.py PROFILE [--seconds 25] [--rate 200]
"""

import argparse
from collections import Counter
from pathlib import Path

EVENT_LOOP_ROOT = "_run_module_as_main"
EVENT_LOOP_AREAS = (
    ("type_inspector", "Qdrant client: inference check"),
    ("qdrant_client/", "Qdrant client"),
    ("grpc/", "gRPC"),
    ("request_batcher", "request batcher"),
    ("routes/search.py", "search route"),
    ("text_processor", "text processor"),
    ("opentelemetry", "OpenTelemetry"),
    ("asyncio/", "asyncio"),
)
WORKER_AREAS = (
    ("tokenization", "tokenizer"),
    ("torch/", "model forward pass"),
    ("transformers/", "model forward pass"),
    ("_convert_to_numpy", "copying results off the GPU"),
    ("flagembedding_model", "embedding code"),
    ("grpc/", "gRPC"),
)


def area_of(joined_stack: str, areas: tuple[tuple[str, str], ...]) -> str:
    return next((name for key, name in areas if key in joined_stack), "other")


def read_profile(
    path: Path,
) -> tuple[Counter[str], Counter[str], Counter[str], Counter[str]]:
    loop_areas: Counter[str] = Counter()
    loop_leaves: Counter[str] = Counter()
    worker_areas: Counter[str] = Counter()
    worker_leaves: Counter[str] = Counter()
    for line in path.read_text().splitlines():
        stack, _, count_text = line.rpartition(" ")
        frames = stack.split(";")
        count = int(count_text)
        joined = ";".join(frames)
        leaf = frames[-1].split(" (")[0] + " (" + frames[-1].rsplit("/", 1)[-1]
        if frames[0].startswith(EVENT_LOOP_ROOT):
            loop_areas[area_of(joined, EVENT_LOOP_AREAS)] += count
            loop_leaves[leaf] += count
        else:
            worker_areas[area_of(joined, WORKER_AREAS)] += count
            worker_leaves[leaf] += count
    return loop_areas, loop_leaves, worker_areas, worker_leaves


def print_group(title: str, areas: Counter[str], leaves: Counter[str]) -> None:
    total = sum(areas.values())
    if not total:
        return
    print(f"-- {title}")
    for name, count in areas.most_common():
        print(f"  {count / total:6.1%}  {name}")
    print("  top functions:")
    for name, count in leaves.most_common(8):
        print(f"  {count / total:6.1%}  {name[:100]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("profile", type=Path)
    parser.add_argument("--seconds", type=float, default=25.0)
    parser.add_argument(
        "--rate", type=int, default=200, help="py-spy samples per second"
    )
    arguments = parser.parse_args()
    loop_areas, loop_leaves, worker_areas, worker_leaves = read_profile(
        arguments.profile
    )
    full_thread = arguments.seconds * arguments.rate
    print(
        f"event loop thread busy {sum(loop_areas.values()) / full_thread:.0%};"
        f" other threads together {sum(worker_areas.values()) / full_thread:.0%}"
        " of one thread"
    )
    print_group("event loop thread", loop_areas, loop_leaves)
    print_group("other threads", worker_areas, worker_leaves)


if __name__ == "__main__":
    main()
