#!/usr/bin/env python3
"""Compare today's image search with a two-stage search, on real images.

Today an image search compares the query with every stored image of every
point (the image vector is a MaxSim multivector with no HNSW index). The
two-stage search gives each point one extra single vector, its "main image",
which can have an index: stage 1 finds candidates by that vector, stage 2
compares the query with every image of those candidates only. Two main
vectors are compared: the average of a point's images, and its first image.

Test data: real images from the local enrichment data (``temp/`` agent folders
and ``assets/seed_data/anime_database.json``) that are in the image cache
(fill it with ``test_data/download_images.py``). Images are grouped by entity
(``toolkit/image_entities.py``); for every entity with two or more images one
image is held out as the query, as if a user uploaded another picture of that
anime or character, and the rest are stored. Entities with one image are
stored as other points the search must rank below the right one.

Prints, per search and candidate count: how often the right entity is first
and in the top 10, and how much of today's top 10 the two-stage search keeps.

Run from the repository root with the project's Python (needs the GPU):

    PYTHONPATH=$(printf '%s:' libs/*/src apps/*/src) .venv/bin/python -m \\
      benchmarks.vector_service.quality.measure_image_two_stage
"""

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from common.config.qdrant_config import QdrantConfig
from qdrant_client import QdrantClient, models
from qdrant_db.collection.schema_builder import (
    build_optimizers_config,
    build_vector_config,
    get_hnsw_config,
    get_per_vector_quantization_config,
)

from benchmarks.vector_service.quality.image_embedders import (
    SERVICE_MODEL,
    embed_images,
    embeddings_store,
)
from benchmarks.vector_service.test_data.download_images import image_urls
from benchmarks.vector_service.toolkit.image_entities import (
    ImageEntity,
    collect_agent_entities,
    collect_database_entities,
    hold_out_queries,
)
from benchmarks.vector_service.toolkit.result_agreement import compare_top_results
from benchmarks.vector_service.toolkit.settings import REPOSITORY_ROOT, load_environment

IMAGE_CACHE = REPOSITORY_ROOT / "cache" / "images"
MAIN_VECTORS = ("image_main_average", "image_main_first")
TOP = 10


def cached_path(url: str) -> Path:
    return (
        IMAGE_CACHE / f"{hashlib.blake2b(url.encode(), digest_size=16).hexdigest()}.jpg"
    )


def load_entities(agent_root: Path, database_file: Path) -> list[ImageEntity]:
    def is_cached(url: str) -> bool:
        return cached_path(url).exists()

    folders = sorted(path for path in agent_root.iterdir() if path.is_dir())
    return collect_agent_entities(folders, is_cached) + collect_database_entities(
        database_file, is_cached
    )


def extra_entities(
    files: Sequence[Path], entities: Sequence[ImageEntity]
) -> list[ImageEntity]:
    """One single-image entity per cached image URL in ``files``, never a query.

    Used to make the collection bigger with realistic wrong answers (e.g. the
    offline database's anime covers). Images that already belong to a test
    entity are skipped, so no duplicate picture competes with the right one.
    """
    taken = {url for entity in entities for url in entity.urls}
    urls: set[str] = set()
    for path in files:
        urls |= image_urls(path.read_text(errors="ignore"), excluded_hosts=())
    return [
        ImageEntity(
            f"extra:{hashlib.blake2b(url.encode(), digest_size=8).hexdigest()}",
            "anime",
            (url,),
        )
        for url in sorted(urls - taken)
        if cached_path(url).exists()
    ]


def main_vectors(images: np.ndarray) -> dict[str, list[float]]:
    average = images.mean(axis=0)
    return {
        "image_main_average": (average / np.linalg.norm(average)).tolist(),
        "image_main_first": images[0].tolist(),
    }


def create_collection(
    client: QdrantClient, collection: str, config: QdrantConfig, dimensions: int
) -> str:
    image_vector = config.primary_image_vector_name
    main_params = models.VectorParams(
        size=dimensions,
        distance=models.Distance.COSINE,
        hnsw_config=get_hnsw_config(config, "high"),
        quantization_config=get_per_vector_quantization_config(config, "high"),
    )
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection,
        vectors_config={
            image_vector: build_vector_config(config)[image_vector].model_copy(
                update={"size": dimensions}
            ),
            **dict.fromkeys(MAIN_VECTORS, main_params),
        },
        optimizers_config=build_optimizers_config(config),
    )
    return image_vector


def fill_collection(
    client: QdrantClient,
    collection: str,
    image_vector: str,
    stored: dict[str, tuple[str, ...]],
    types: dict[str, str],
    embeddings: dict[str, np.ndarray],
) -> dict[int, str]:
    """Upload one point per entity; return point id → entity key."""
    keys = sorted(stored)
    points = []
    for point_id, key in enumerate(keys, start=1):
        images = np.stack([embeddings[url] for url in stored[key]])
        vectors = {image_vector: images.tolist(), **main_vectors(images)}
        points.append(
            models.PointStruct(
                id=point_id,
                vector=vectors,
                payload={"key": key, "entity_type": types[key]},
            )
        )
    client.upload_points(collection, points, batch_size=128, wait=True)
    return dict(enumerate(keys, start=1))


def search_ids(
    client: QdrantClient,
    collection: str,
    image_vector: str,
    query: list[float],
    main: str | None,
    candidates: int,
    limit: int = TOP,
) -> list[int]:
    prefetch = (
        models.Prefetch(query=query, using=main, limit=candidates) if main else None
    )
    hits = client.query_points(
        collection, prefetch=prefetch, query=query, using=image_vector, limit=limit
    ).points
    return [int(hit.id) for hit in hits]


def hit_rates(results: list[list[int]], targets: list[int]) -> tuple[float, float]:
    first = sum(
        ids[:1] == [target] for ids, target in zip(results, targets, strict=True)
    )
    top = sum(target in ids for ids, target in zip(results, targets, strict=True))
    return first / len(targets), top / len(targets)


def report(
    label: str, results: list[list[int]], reference: list[list[int]], targets: list[int]
) -> None:
    first, top = hit_rates(results, targets)
    agreement = compare_top_results(reference, results)
    print(
        f"{label:34} right entity 1st {first:6.1%}  in top {TOP} {top:6.1%}  "
        f"keeps today's top {TOP}: {agreement.overlap:6.1%} (identical {agreement.identical}/{agreement.total})",
        flush=True,
    )


def rank_of(scores: np.ndarray, index: int) -> int:
    return int((scores > scores[index]).sum()) + 1


def quantile_text(ranks: Sequence[int], total: int) -> str:
    return ", ".join(
        f"{share:.0%} within {int(np.quantile(ranks, share))} ({np.quantile(ranks, share) / total:.2%})"
        for share in (0.5, 0.8, 0.9, 0.95, 0.99)
    )


def today_ranks(
    keys: list[str],
    stored: dict[str, tuple[str, ...]],
    queries: list[tuple[str, str]],
    embeddings: dict[str, np.ndarray],
) -> list[int]:
    """The right entity's rank in today's search: each entity scored by its best image."""
    images = np.stack([embeddings[url] for key in keys for url in stored[key]])
    starts = np.cumsum([0] + [len(stored[key]) for key in keys[:-1]])
    position = {key: index for index, key in enumerate(keys)}
    return [
        rank_of(np.maximum.reduceat(images @ embeddings[url], starts), position[key])
        for key, url in queries
    ]


def stage_one_ranks(
    stored: dict[str, tuple[str, ...]],
    queries: list[tuple[str, str]],
    embeddings: dict[str, np.ndarray],
) -> None:
    """Where the right entity ranks by each main vector alone (exact), as quantiles.

    Stage 2 can only return what stage 1 found, so this shows how many
    candidates stage 1 needs: for all queries, and for the queries today's
    search already answers (right entity in its top 10), which are the ones a
    two-stage search could lose. The share of the collection carries over to
    a larger one only roughly.
    """
    keys = sorted(stored)
    position = {key: index for index, key in enumerate(keys)}
    answered = [rank <= TOP for rank in today_ranks(keys, stored, queries, embeddings)]
    print(
        f"today's search has the right entity in its top {TOP} for {sum(answered)} of {len(queries)} queries"
    )
    for main in MAIN_VECTORS:
        vectors = np.array(
            [
                main_vectors(np.stack([embeddings[url] for url in stored[key]]))[main]
                for key in keys
            ]
        )
        ranks = [
            rank_of(vectors @ embeddings[url], position[key]) for key, url in queries
        ]
        kept = [
            rank
            for rank, is_answered in zip(ranks, answered, strict=True)
            if is_answered
        ]
        name = main.removeprefix("image_main_")
        print(
            f"stage 1 rank, {name} main, all queries: {quantile_text(ranks, len(keys))}",
            flush=True,
        )
        print(
            f"stage 1 rank, {name} main, queries today answers: {quantile_text(kept, len(keys))}",
            flush=True,
        )


def compare_searches(
    client: QdrantClient,
    collection: str,
    image_vector: str,
    queries: list[tuple[str, str]],
    types: dict[str, str],
    keys_to_ids: dict[str, int],
    embeddings: dict[str, np.ndarray],
    candidate_counts: Sequence[int],
) -> None:
    """Today's search against each two-stage variant, for all queries and per entity type."""
    for entity_type in (None, "anime", "character"):
        chosen = [
            (key, url) for key, url in queries if entity_type in (None, types[key])
        ]
        if not chosen:
            continue
        print(f"\nqueries: {entity_type or 'all'} ({len(chosen)})")
        vectors = [embeddings[url].tolist() for _, url in chosen]
        targets = [keys_to_ids[key] for key, _ in chosen]
        today = [
            search_ids(client, collection, image_vector, vector, None, 0)
            for vector in vectors
        ]
        report("today: every image compared", today, today, targets)
        for main in MAIN_VECTORS:
            for candidates in candidate_counts:
                results = [
                    search_ids(
                        client, collection, image_vector, vector, main, candidates
                    )
                    for vector in vectors
                ]
                report(
                    f"{main.removeprefix('image_main_')} main, {candidates} candidates",
                    results,
                    today,
                    targets,
                )


@dataclass(frozen=True)
class CandidateSearch:
    client: QdrantClient
    collection: str
    image_vector: str
    limit: int
    prefetch: int


def export_candidates(
    path: Path,
    search: CandidateSearch,
    queries: list[tuple[str, str]],
    stored: dict[str, tuple[str, ...]],
    types: dict[str, str],
    embeddings: dict[str, np.ndarray],
    ids_to_keys: dict[int, str],
) -> None:
    """Write each query's candidate entities per search, with their stored images.

    For a second model to rerank elsewhere: today's search, and the two-stage
    search with each main vector, each returning ``search.limit`` entities.
    """
    methods = {
        "today": None,
        "average": "image_main_average",
        "first": "image_main_first",
    }
    rows = []
    listed: set[str] = set()
    for key, url in queries:
        vector = embeddings[url].tolist()
        candidates = {
            name: [
                ids_to_keys[point_id]
                for point_id in search_ids(
                    search.client,
                    search.collection,
                    search.image_vector,
                    vector,
                    main,
                    search.prefetch,
                    search.limit,
                )
            ]
            for name, main in methods.items()
        }
        listed.update(key for keys in candidates.values() for key in keys)
        rows.append(
            {
                "key": key,
                "entity_type": types[key],
                "image": url,
                "candidates": candidates,
            }
        )
    entities = {
        key: list(stored[key]) for key in sorted(listed | {key for key, _ in queries})
    }
    path.write_text(json.dumps({"entities": entities, "queries": rows}))
    print(
        f"wrote candidates for {len(rows)} queries, {len(entities)} entities, to {path}",
        flush=True,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--environment", default="laptop")
    parser.add_argument(
        "--set", action="append", default=[], help="override, e.g. qdrant.url=..."
    )
    parser.add_argument("--agent-root", type=Path, default=REPOSITORY_ROOT / "temp")
    parser.add_argument(
        "--database-file",
        type=Path,
        default=REPOSITORY_ROOT / "assets/seed_data/anime_database.json",
    )
    parser.add_argument("--candidates", nargs="+", type=int, default=[10, 20, 50, 100])
    parser.add_argument(
        "--extra-images",
        nargs="*",
        type=Path,
        default=[],
        help="files whose cached image URLs become one-image wrong answers",
    )
    parser.add_argument("--seed", type=int, default=54)
    parser.add_argument(
        "--image-model",
        default=SERVICE_MODEL,
        help="openclip:<architecture>/<pretrained> or hf-clip:<repo>",
    )
    parser.add_argument(
        "--export-candidates",
        type=Path,
        help="write each query's candidates (today, average, first) for reranking elsewhere",
    )
    parser.add_argument("--export-limit", type=int, default=50)
    parser.add_argument("--export-prefetch", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    environment = load_environment(args.environment, args.set)
    entities = load_entities(args.agent_root, args.database_file)
    stored, queries = hold_out_queries(entities, args.seed)
    extras = extra_entities(args.extra_images, entities)
    stored.update({entity.key: entity.urls for entity in extras})
    entities = entities + extras
    types = {entity.key: entity.entity_type for entity in entities}
    urls = sorted({url for entity in entities for url in entity.urls})
    store = embeddings_store(environment.results_dir, args.image_model)
    embeddings = embed_images(urls, args.image_model, store)
    print(
        f"{len(stored)} entities ({len(extras)} extra one-image), {len(urls)} images, "
        f"{len(queries)} queries",
        flush=True,
    )
    config = QdrantConfig()
    client = QdrantClient(
        url=environment.qdrant.url, api_key=environment.qdrant.api_key(), timeout=300
    )
    collection = environment.collections.image_accuracy
    dimensions = len(next(iter(embeddings.values())))
    image_vector = create_collection(client, collection, config, dimensions)
    ids_to_keys = fill_collection(
        client, collection, image_vector, stored, types, embeddings
    )
    keys_to_ids = {key: point_id for point_id, key in ids_to_keys.items()}
    stage_one_ranks(stored, queries, embeddings)
    compare_searches(
        client,
        collection,
        image_vector,
        queries,
        types,
        keys_to_ids,
        embeddings,
        args.candidates,
    )
    if args.export_candidates:
        search = CandidateSearch(
            client, collection, image_vector, args.export_limit, args.export_prefetch
        )
        export_candidates(
            args.export_candidates,
            search,
            queries,
            stored,
            types,
            embeddings,
            ids_to_keys,
        )


if __name__ == "__main__":
    main()
