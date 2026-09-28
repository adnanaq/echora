#!/usr/bin/env python3
"""Keep the GPU busy the way indexing would, to see how it slows search.

Indexing embeds whole records: document-length texts through BGE-M3 and
images through OpenCLIP. This runs both models in a loop on the same GPU the
service would use, without writing anything anywhere, and prints how many
texts and images it encodes per interval. Start it, then run a steady search
load against the service and compare with the same load without it.

Document texts are made by joining search queries from the load test's query
file to about ``--document-tokens`` tokens (real records vary); images come
from the image downloader's cache (``cache/images``).

Run from the repository root with the project's Python (the service's
libraries on the path):

    PYTHONPATH=$(printf '%s:' libs/*/src apps/*/src) .venv/bin/python -m \\
      benchmarks.vector_service.diagnostics.simulate_indexing_load --seconds 300
"""

import argparse
import random
import time
from pathlib import Path

from PIL import Image
from vector_processing.embedding_models.text.flagembedding_model import (
    FlagEmbeddingModel,
)
from vector_processing.embedding_models.vision.openclip_model import OpenClipModel

from benchmarks.vector_service.toolkit.query_mix import load_queries

IMAGE_CACHE = Path("cache/images")


class NoCachedImagesError(RuntimeError):
    def __init__(self) -> None:
        super().__init__(
            f"no images in {IMAGE_CACHE}; index some anime first to fill it"
        )


def document_texts(queries: list[str], count: int, words: int, seed: int) -> list[str]:
    """Texts of at least ``words`` words, each joined from random ``queries``."""
    random_generator = random.Random(seed)  # noqa: S311 - test texts, not security
    documents = []
    for _ in range(count):
        parts: list[str] = []
        while sum(len(part.split()) for part in parts) < words:
            parts.append(random_generator.choice(queries))
        documents.append(". ".join(parts))
    return documents


def cached_images(count: int) -> list[Image.Image]:
    paths = sorted(IMAGE_CACHE.glob("*.jpg"))[:count]
    return [Image.open(path).convert("RGB") for path in paths]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("--text-batch", type=int, default=32)
    parser.add_argument("--image-batch", type=int, default=8)
    parser.add_argument(
        "--document-tokens", type=int, default=200, help="about 1.3 tokens per word"
    )
    parser.add_argument("--report-seconds", type=float, default=20)
    parser.add_argument("--seed", type=int, default=54)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    text_model = FlagEmbeddingModel("BAAI/bge-m3")
    image_model = OpenClipModel("ViT-L-14/laion2b_s32b_b82k")
    queries = [text for texts in load_queries().values() for text in texts]
    documents = document_texts(queries, 256, int(args.document_tokens / 1.3), args.seed)
    images = cached_images(64)
    if not images:
        raise NoCachedImagesError
    started = reported = time.perf_counter()
    texts_done = images_done = 0
    position = 0
    print(
        f"indexing load started: text batches of {args.text_batch}, image batches of {args.image_batch}",
        flush=True,
    )
    while time.perf_counter() - started < args.seconds:
        text_batch = [
            documents[(position + i) % len(documents)] for i in range(args.text_batch)
        ]
        image_batch = [
            images[(position + i) % len(images)] for i in range(args.image_batch)
        ]
        text_model.encode_with_sparse(text_batch)
        image_model.encode_image(image_batch)
        texts_done += len(text_batch)
        images_done += len(image_batch)
        position += 1
        now = time.perf_counter()
        if now - reported >= args.report_seconds:
            print(
                f"{now - started:6.0f} s: {texts_done / (now - reported):5.0f} texts/s, "
                f"{images_done / (now - reported):4.0f} images/s",
                flush=True,
            )
            texts_done = images_done = 0
            reported = now


if __name__ == "__main__":
    main()
