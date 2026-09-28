#!/usr/bin/env python3
"""Measure what an image search costs Qdrant, on the image test collection.

Sends random unit image vectors as queries to the environment's image
collection (built by ``test_data/build_image_load_test_collection.py``) the
way the service's image branch does: the image vector, ``limit`` candidates,
optionally filtered by ``entity_type``, IDs only. The image vector has no HNSW
index (MaxSim cannot use one), so every search scores every image vector that
passes the filter. For each case (no filter, anime only, characters only) it
prints:

- latency of searches sent one at a time (p50 / p95)
- Qdrant CPU per search, from its container's cgroup counters
- searches/s with ``--in-flight`` searches running at once

Run from the repository root:

    ./pants run benchmarks/vector_service/quality/measure_image_search.py
"""

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from common.config.qdrant_config import QdrantConfig
from qdrant_client import QdrantClient, models

from benchmarks.vector_service.toolkit.resources import ContainerCgroupReader
from benchmarks.vector_service.toolkit.settings import load_environment

ENTITY_TYPES = (None, "anime", "character")


@dataclass(frozen=True)
class ImageSearch:
    client: QdrantClient
    collection: str
    vector_name: str
    limit: int

    def run(self, image: list[float], entity_type: str | None) -> None:
        condition = (
            models.Filter(
                must=[
                    models.FieldCondition(
                        key="entity_type", match=models.MatchValue(value=entity_type)
                    )
                ]
            )
            if entity_type
            else None
        )
        self.client.query_points(
            self.collection,
            query=image,
            using=self.vector_name,
            query_filter=condition,
            limit=self.limit,
            with_payload=False,
        )


def random_images(count: int, dimensions: int, seed: int) -> list[list[float]]:
    images = np.random.default_rng(seed).standard_normal((count, dimensions))
    images /= np.linalg.norm(images, axis=1, keepdims=True)
    return images.astype(np.float32).tolist()


def latency_ms(
    search: ImageSearch, images: list[list[float]], entity_type: str | None
) -> tuple[float, float]:
    times = []
    for image in images:
        started = time.perf_counter()
        search.run(image, entity_type)
        times.append((time.perf_counter() - started) * 1000)
    times.sort()
    return statistics.median(times), times[int(0.95 * (len(times) - 1))]


def throughput(
    search: ImageSearch,
    images: list[list[float]],
    entity_type: str | None,
    in_flight: int,
    cpu: ContainerCgroupReader,
) -> tuple[float, float | None]:
    """Searches/s and Qdrant CPU milliseconds per search."""
    cpu_before = cpu.cpu_seconds()
    started = time.perf_counter()
    with ThreadPoolExecutor(in_flight) as pool:
        list(pool.map(lambda image: search.run(image, entity_type), images))
    elapsed = time.perf_counter() - started
    cpu_after = cpu.cpu_seconds()
    cpu_ms = (
        (cpu_after - cpu_before) * 1000 / len(images)
        if cpu_before is not None and cpu_after is not None
        else None
    )
    return len(images) / elapsed, cpu_ms


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--environment", default="laptop")
    parser.add_argument(
        "--set", action="append", default=[], help="override, e.g. qdrant.url=..."
    )
    parser.add_argument("--searches", type=int, default=300, help="searches per case")
    parser.add_argument("--in-flight", type=int, default=16)
    parser.add_argument(
        "--limit", type=int, default=100, help="candidates, as the service's prefetch"
    )
    parser.add_argument("--seed", type=int, default=54)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    environment = load_environment(args.environment, args.set)
    config = QdrantConfig()
    vector_name = config.primary_image_vector_name
    client = QdrantClient(
        url=environment.qdrant.url,
        api_key=environment.qdrant.api_key(),
        prefer_grpc=True,
    )
    search = ImageSearch(
        client, environment.collections.image_load, vector_name, args.limit
    )
    cpu = ContainerCgroupReader(environment.qdrant.container)
    images = random_images(args.searches, config.vector_names[vector_name], args.seed)
    for image in images[:20]:
        search.run(image, None)
    print(f"{search.collection}: {client.count(search.collection).count} points")
    for entity_type in ENTITY_TYPES:
        p50, p95 = latency_ms(
            search, images[: max(20, args.searches // 3)], entity_type
        )
        rate, cpu_ms = throughput(search, images, entity_type, args.in_flight, cpu)
        cpu_text = f"{cpu_ms:6.1f} ms" if cpu_ms is not None else "unknown"
        print(
            f"filter {entity_type or 'none':9} | one at a time p50 {p50:6.1f} ms p95 {p95:6.1f} ms "
            f"| {args.in_flight} in flight {rate:6.0f} searches/s | Qdrant CPU per search {cpu_text}",
            flush=True,
        )


if __name__ == "__main__":
    main()
