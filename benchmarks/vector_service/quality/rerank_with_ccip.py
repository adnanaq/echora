#!/usr/bin/env python3
"""Rerank image search candidates with CCIP and compare the accuracy.

CCIP (deepghs, ``dghs-imgutils``) is trained to tell whether two single-
character anime images show the same character. It compares images with its
own learned difference, not cosine, so it cannot be the Qdrant vector; here it
reorders the candidates each search returned. Each candidate entity is scored
by its closest image (smallest CCIP difference), as MaxSim does.

Input is the file written by ``measure_image_two_stage.py
--export-candidates``: per query, the candidates of today's search and of the
two-stage search with the average and the first-image main vector. For each,
prints how often the right entity is first and in the top 10, before and after
CCIP reranking, for all queries and per entity type.

``dghs-imgutils`` needs numpy 1.x, which does not build on the project's
Python, so this runs in its own throwaway environment; ONNX Runtime uses the
CUDA libraries of the project's ``.venv`` for the GPU:

    LD_LIBRARY_PATH=$(find .venv/lib/python3.14/site-packages/nvidia -maxdepth 3 \\
      -type d -name lib | tr '\\n' ':') \\
    uv run --no-project --python 3.12 --with "dghs-imgutils>=0.19.0" \\
      --with onnxruntime-gpu python -m \\
      benchmarks.vector_service.quality.rerank_with_ccip candidates.json
"""

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from benchmarks.vector_service.toolkit.candidate_rerank import (
    hit_rates,
    rerank_by_best_image,
)

# Not from toolkit.settings, which needs Python 3.14 (this runs on 3.12).
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
IMAGE_CACHE = REPOSITORY_ROOT / "cache" / "images"
TOP = 10


def cached_path(url: str) -> Path:
    return (
        IMAGE_CACHE / f"{hashlib.blake2b(url.encode(), digest_size=16).hexdigest()}.jpg"
    )


def ccip_features(
    urls: Sequence[str], store: Path, batch_size: int = 32
) -> dict[str, np.ndarray]:
    """CCIP features per URL, reusing the ones saved in ``store``."""
    from imgutils.metrics import ccip_batch_extract_features

    saved = dict(np.load(store)) if store.exists() else {}
    missing = [url for url in urls if url not in saved]
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        features = ccip_batch_extract_features([str(cached_path(url)) for url in batch])
        saved.update(zip(batch, features, strict=True))
        if (start // batch_size) % 20 == 0:
            print(f"CCIP features: {start + len(batch)}/{len(missing)}", flush=True)
    if missing:
        np.savez(store, **saved)
    return {url: saved[url] for url in urls}


def query_differences(
    query_url: str, image_urls: Sequence[str], features: dict[str, np.ndarray]
) -> dict[str, float]:
    """CCIP difference between the query image and each image."""
    from imgutils.metrics import ccip_batch_differences

    differences = ccip_batch_differences(
        [features[query_url]] + [features[url] for url in image_urls]
    )
    return dict(zip(image_urls, differences[0, 1:].tolist(), strict=True))


def report(
    label: str, before: list[list[str]], after: list[list[str]], targets: list[str]
) -> None:
    first_before, top_before = hit_rates(before, targets, TOP)
    first_after, top_after = hit_rates(after, targets, TOP)
    print(
        f"{label:10} right entity 1st {first_before:6.1%} → {first_after:6.1%}   "
        f"in top {TOP} {top_before:6.1%} → {top_after:6.1%}",
        flush=True,
    )


def compare(
    data: dict, features: dict[str, np.ndarray], entity_type: str | None
) -> None:
    rows = [row for row in data["queries"] if entity_type in (None, row["entity_type"])]
    if not rows:
        return
    print(
        f"\nqueries: {entity_type or 'all'} ({len(rows)}); search → with CCIP reranking"
    )
    entities = data["entities"]
    targets = [row["key"] for row in rows]
    for method in ("today", "average", "first"):
        before, after = [], []
        for row in rows:
            candidates = row["candidates"][method]
            images = sorted({url for key in candidates for url in entities[key]})
            differences = query_differences(row["image"], images, features)
            before.append(candidates)
            after.append(rerank_by_best_image(candidates, entities, differences))
        report(method, before, after, targets)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "candidates",
        type=Path,
        help="file from measure_image_two_stage.py --export-candidates",
    )
    parser.add_argument(
        "--features-store",
        type=Path,
        default=None,
        help="default: next to the candidates file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    data = json.loads(args.candidates.read_text())
    urls = sorted(
        {row["image"] for row in data["queries"]}
        | {url for images in data["entities"].values() for url in images}
    )
    store = args.features_store or args.candidates.with_name("ccip_features.npz")
    features = ccip_features(urls, store)
    print(
        f"{len(data['queries'])} queries, {len(data['entities'])} candidate entities, {len(urls)} images"
    )
    for entity_type in (None, "anime", "character"):
        compare(data, features, entity_type)


if __name__ == "__main__":
    main()
