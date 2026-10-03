#!/usr/bin/env python3
"""Build a collection of image vectors at production size, for image search cost.

The load test collection has no image vectors (building it must not download
images), so image search is measured on its own collection (default
``anime_image_load_test``). It holds only the image vector, created with the
service's own schema functions, so it has production's settings: a multivector
of OpenCLIP-sized vectors compared with MaxSim, int8 quantization kept in RAM,
and no HNSW index (``m=0``; HNSW cannot index MaxSim). Each point carries its
``entity_type``, indexed, since image searches can be filtered by it.

Point counts default to the load test collection's anime and characters
(episodes have no images). Each anime gets 1 to 6 images and each character 1
to 2, spread evenly, close to the real data (about 3.5 and 1.7). The vectors
are random unit vectors: an unindexed search scores every image vector
whatever it holds, so its cost does not depend on the content. Accuracy cannot
be measured here.

Run from the repository root:

    ./pants run benchmarks/vector_service/test_data/build_image_load_test_collection.py
"""

import argparse
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from common.config.qdrant_config import QdrantConfig
from qdrant_client import QdrantClient, models
from qdrant_db.collection.schema_builder import (
    build_optimizers_config,
    build_vector_config,
    get_hnsw_config,
    get_per_vector_quantization_config,
)

from benchmarks.vector_service.toolkit.settings import load_environment

ENTITY_TYPE_FIELD = "entity_type"
AVERAGE_VECTOR = "image_main_average"


class ImageCollectionExistsError(RuntimeError):
    def __init__(self, collection: str) -> None:
        super().__init__(f"{collection} exists; pass --recreate to rebuild it")


@dataclass(frozen=True)
class EntityImages:
    entity_type: str
    points: int
    most_images: int


@dataclass(frozen=True)
class ImagePoint:
    point_id: int
    entity_type: str
    images: list[list[float]]


def image_points(
    kinds: Sequence[EntityImages], dimensions: int, seed: int, first_id: int = 1
) -> Iterator[ImagePoint]:
    """Points with 1 to ``most_images`` random unit vectors each, in ``kinds`` order."""
    generator = np.random.default_rng(seed)
    point_id = first_id
    for kind in kinds:
        for _ in range(kind.points):
            count = int(generator.integers(1, kind.most_images + 1))
            images = generator.standard_normal((count, dimensions))
            images /= np.linalg.norm(images, axis=1, keepdims=True)
            yield ImagePoint(
                point_id, kind.entity_type, images.astype(np.float32).tolist()
            )
            point_id += 1


def to_qdrant_point(
    point: ImagePoint, vector_name: str, average_vector: str | None = None
) -> models.PointStruct:
    """The point's images; with ``average_vector``, also their unit-length average."""
    vectors: dict[str, Any] = {vector_name: point.images}
    if average_vector:
        average = np.asarray(point.images, dtype=np.float32).mean(axis=0)
        vectors[average_vector] = (average / np.linalg.norm(average)).tolist()
    return models.PointStruct(
        id=point.point_id,
        vector=vectors,
        payload={ENTITY_TYPE_FIELD: point.entity_type},
    )


def vectors_config(
    config: QdrantConfig, image_vector: str, average_vector: str | None
) -> dict[str, models.VectorParams]:
    vectors = {image_vector: build_vector_config(config)[image_vector]}
    if average_vector:
        vectors[average_vector] = models.VectorParams(
            size=config.vector_names[image_vector],
            distance=models.Distance.COSINE,
            hnsw_config=get_hnsw_config(config, "high"),
            quantization_config=get_per_vector_quantization_config(config, "high"),
        )
    return vectors


def create_collection(
    client: QdrantClient, collection: str, recreate: bool, average_vector: str | None
) -> QdrantConfig:
    """Create the collection; ``average_vector`` adds an indexed single vector
    per point (the average of its images) for a two-stage image search."""
    config = QdrantConfig()
    if client.collection_exists(collection):
        if not recreate:
            raise ImageCollectionExistsError(collection)
        client.delete_collection(collection)
    image_vector = config.primary_image_vector_name
    client.create_collection(
        collection,
        vectors_config=vectors_config(config, image_vector, average_vector),
        optimizers_config=build_optimizers_config(config),
    )
    client.create_payload_index(
        collection, ENTITY_TYPE_FIELD, models.PayloadSchemaType.KEYWORD
    )
    return config


def wait_until_indexed(client: QdrantClient, collection: str) -> None:
    while client.get_collection(collection).status != models.CollectionStatus.GREEN:
        time.sleep(5)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--environment", default="laptop")
    parser.add_argument(
        "--set", action="append", default=[], help="override, e.g. qdrant.url=..."
    )
    parser.add_argument("--anime", type=int, default=40346)
    parser.add_argument("--characters", type=int, default=769693)
    parser.add_argument(
        "--anime-images", type=int, default=6, help="most images per anime"
    )
    parser.add_argument(
        "--character-images", type=int, default=2, help="most images per character"
    )
    parser.add_argument("--seed", type=int, default=54)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument(
        "--average-vector",
        action="store_true",
        help=f"also store each point's average image as {AVERAGE_VECTOR} (indexed)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    environment = load_environment(args.environment, args.set)
    collection = environment.collections.image_load
    client = QdrantClient(
        url=environment.qdrant.url,
        api_key=environment.qdrant.api_key(),
        prefer_grpc=True,
        timeout=300,
    )
    average_vector = AVERAGE_VECTOR if args.average_vector else None
    config = create_collection(client, collection, args.recreate, average_vector)
    image_vector = config.primary_image_vector_name
    kinds = [
        EntityImages("anime", args.anime, args.anime_images),
        EntityImages("character", args.characters, args.character_images),
    ]
    started = time.perf_counter()
    client.upload_points(
        collection,
        (
            to_qdrant_point(point, image_vector, average_vector)
            for point in image_points(
                kinds, config.vector_names[image_vector], args.seed
            )
        ),
        batch_size=256,
        parallel=4,
    )
    wait_until_indexed(client, collection)
    info = client.get_collection(collection)
    print(
        f"{collection}: {info.points_count} points, {info.segments_count} segments, "
        f"built in {time.perf_counter() - started:.0f} s"
    )


if __name__ == "__main__":
    main()
