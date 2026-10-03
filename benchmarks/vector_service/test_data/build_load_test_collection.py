#!/usr/bin/env python3
"""Build a realistic Qdrant collection for the vector search load test.

Two steps, run separately:

``records``
    Build real anime records from the provider data cached in Redis, without
    any network access: every non-local DNS lookup and every browser start is
    blocked, so a cache miss fails instead of fetching. Each anime is merged from
    all providers' cached data; characters and episodes come from the cached
    AniDB XML. Image URLs are removed, since embedding images would download
    them. Records are written to ``data/load_test/records.jsonl``.

``collection``
    Create a separate collection (default ``anime_load_test``) with the
    service's collection settings and fill it with the service's own embedding
    code: the cached records with their characters and episodes, plus every
    other anime in the offline database from its offline fields. Characters and
    episodes are then topped up to the requested counts with copies of the real
    ones: each copy takes a real point's vectors plus small random noise and is
    attached to a random anime. Copies are not re-embedded, which would take
    hours; their near-duplicate vectors may make search cost differ from
    production.
"""

import argparse
import asyncio
import json
import socket
import sys
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import ExitStack, suppress
from pathlib import Path
from typing import Any
from unittest.mock import patch

import redis

OFFLINE_DATABASE = Path("assets/seed_data/anime-offline-database.json")
RECORDS_FILE = Path("data/load_test/records.jsonl")
MAL_RESULT_CACHE_PATTERN = "result_cache:mal_anime_scraped:*"
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
RECORD_ID_NAMESPACE = uuid.UUID("5f0c5c8e-54a1-4c4b-9d0e-0ec0a54e5454")
VALID_SEASONS = {"SPRING", "SUMMER", "FALL", "WINTER"}
EMBED_BATCH_RECORDS = 128
UPSERT_BATCH_SIZE = 256
COPY_BATCH_SIZE = 2048
# Per-dimension noise on a 1024-d unit vector; a copy keeps a cosine
# similarity of about 0.93 with its source point.
DENSE_NOISE_SCALE = 0.012
SPARSE_WEIGHT_JITTER = 0.1


class NetworkBlockedError(OSError):
    """Raised when code tries to reach a non-local host during a cache-only run."""

    def __init__(self, target: str) -> None:
        super().__init__(f"cache-only run: {target} blocked")


def cache_only_patches(blocked_hosts: list[str]) -> ExitStack:
    """Patch the enrichment code so a cache-only run never reaches the network.

    Non-local DNS lookups and browser starts fail and are recorded in
    ``blocked_hosts``. Retries are turned off, since a blocked request cannot
    succeed on a retry, and AniDB uses its own cache-only request, which answers
    a miss as no data. AniDB characters are mapped from the cached XML alone.
    """
    import zendriver
    from common.utils.retry import retry_with_backoff
    from enrichment.sources.anidb import anidb_helper
    from enrichment.sources.anilist import anilist_helper
    from enrichment.sources.kitsu import kitsu_helper

    allowed_lookup = socket.getaddrinfo

    def local_only_lookup(host: Any, *args: Any, **kwargs: Any) -> Any:
        name = host.decode() if isinstance(host, bytes) else str(host)
        if name not in LOCAL_HOSTS:
            blocked_hosts.append(name)
            raise NetworkBlockedError(name)
        return allowed_lookup(host, *args, **kwargs)

    async def refuse_browser(*args: Any, **kwargs: Any) -> Any:
        blocked_hosts.append("browser")
        raise NetworkBlockedError("browser")

    async def try_once(*args: Any, **kwargs: Any) -> Any:
        return await retry_with_backoff(*args, **{**kwargs, "max_retries": 0})

    async def anidb_from_cache(helper: Any, params: dict[str, Any]) -> str | None:
        await helper._ensure_session_health()
        return await helper._make_single_request(params, 0, cache_only=True)

    patches = ExitStack()
    for target, attribute, replacement in (
        (socket, "getaddrinfo", local_only_lookup),
        (zendriver, "start", refuse_browser),
        (anilist_helper, "retry_with_backoff", try_once),
        (kitsu_helper, "retry_with_backoff", try_once),
        (anidb_helper.AniDBHelper, "_make_request_with_retry", anidb_from_cache),
        (anidb_helper, "fetch_anidb_characters", no_character_pages),
    ):
        patches.enter_context(patch.object(target, attribute, replacement))
    return patches


async def no_character_pages(
    character_ids: list[int],
) -> AsyncIterator[tuple[int, None]]:
    """Stand-in for the AniDB character page crawler: no page for any character.

    AniDB characters are then mapped from the cached anime XML alone.
    """
    for character_id in character_ids:
        yield character_id, None


def cached_offline_entries() -> list[dict[str, Any]]:
    """Return offline database entries whose MAL page is in the result cache."""
    client = redis.Redis(port=6379)
    cached_urls = {
        key.decode().split(":", 3)[3]
        for key in client.scan_iter(MAL_RESULT_CACHE_PATTERN, count=1000)
    }
    offline = json.loads(OFFLINE_DATABASE.read_text(encoding="utf-8"))["data"]
    return [
        entry
        for entry in offline
        if any(source in cached_urls for source in entry.get("sources", []))
    ]


def strip_images(item: dict[str, Any]) -> dict[str, Any]:
    """Drop every image URL embedding would download: images and trailer thumbnails."""
    stripped = {key: value for key, value in item.items() if key != "images"}
    if stripped.get("trailers"):
        stripped["trailers"] = [
            {key: value for key, value in trailer.items() if key != "thumbnail"}
            for trailer in stripped["trailers"]
        ]
    return stripped


async def build_record(
    entry: dict[str, Any], fetcher: Any, anidb_helper: Any
) -> tuple[dict[str, Any] | None, list[str]]:
    """Merge one anime from cached provider data; return it with the providers used."""
    from common.models.anime import AnimeRecord
    from enrichment.pipeline.id_extractor import PlatformIDExtractor
    from enrichment.pipeline.metadata_merger import merge_provider_records

    ids = {
        key: value
        for key, value in PlatformIDExtractor().extract_all_ids(entry).items()
        if value
    }
    results = await fetcher.fetch_all_data(
        ids,
        entry,
        skip_services=["anidb", "animeschedule"],
        fetch_characters=False,
        fetch_episodes=False,
    )
    anidb = await anidb_helper.fetch_all(ids, entry) if ids.get("anidb_url") else None
    provider_records = {
        provider: payload["anime"]
        for provider, payload in {**results, "anidb": anidb}.items()
        if payload and payload.get("anime")
    }
    if not provider_records:
        return None, []
    anime = strip_images(merge_provider_records(provider_records, offline_data=entry))
    anime["id"] = str(uuid.uuid5(RECORD_ID_NAMESPACE, entry["sources"][0]))
    record = AnimeRecord.model_validate(
        {
            "anime": anime,
            "characters": [
                strip_images(item) for item in (anidb or {}).get("characters", [])
            ],
            "episodes": (anidb or {}).get("episodes", []),
        }
    )
    return record.model_dump(mode="json", exclude_none=True), sorted(provider_records)


async def build_records(limit: int | None) -> None:
    """Build records for every cached anime and write them as JSON lines."""
    from enrichment.pipeline.api_fetcher import ApiFetcher
    from enrichment.sources.anidb.anidb_helper import AniDBHelper

    blocked_hosts: list[str] = []
    with cache_only_patches(blocked_hosts):
        await write_records(limit, ApiFetcher(), AniDBHelper(), blocked_hosts)


async def write_records(
    limit: int | None, fetcher: Any, anidb_helper: Any, blocked_hosts: list[str]
) -> None:
    """Build and write one record per cached anime."""
    entries = cached_offline_entries()[:limit]
    print(f"{len(entries)} anime with cached provider data")

    RECORDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    built = failed = 0
    with RECORDS_FILE.open("w", encoding="utf-8") as output:
        for position, entry in enumerate(entries, 1):
            started = time.perf_counter()
            try:
                record, providers = await build_record(entry, fetcher, anidb_helper)
            except Exception as exc:  # noqa: BLE001 - one bad record must not stop the build
                record, providers = None, [f"{type(exc).__name__}: {exc}"]
            if record is None:
                failed += 1
            else:
                built += 1
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"{position:4}/{len(entries)} {entry['title'][:40]:42} "
                f"{len(record['characters']) if record else 0:3} chars "
                f"{len(record['episodes']) if record else 0:4} eps "
                f"{time.perf_counter() - started:5.1f}s  {','.join(providers)}",
                flush=True,
            )
    await anidb_helper.close()
    print(f"built {built}, failed {failed}, blocked lookups {len(blocked_hosts)}")
    if blocked_hosts:
        print(f"blocked: {sorted(set(blocked_hosts))}")


def offline_anime_record(entry: dict[str, Any]) -> dict[str, Any]:
    """Map an offline database entry to a record with the fields it has."""
    anime_season = entry.get("animeSeason") or {}
    anime: dict[str, Any] = {
        "id": str(uuid.uuid5(RECORD_ID_NAMESPACE, entry["sources"][0])),
        "title": entry["title"],
        "type": entry["type"],
        "status": entry["status"],
        "sources": entry["sources"],
        "synonyms": entry.get("synonyms", []),
        "tags": entry.get("tags", []),
        "episode_count": entry.get("episodes"),
        "year": anime_season.get("year"),
    }
    if anime_season.get("season") in VALID_SEASONS:
        anime["season"] = anime_season["season"]
    return {"anime": anime}


def load_records(limit: int | None) -> list[Any]:
    """Load the cached records, then every other offline anime, as AnimeRecords."""
    from common.models.anime import AnimeRecord
    from pydantic import ValidationError

    enriched = [
        json.loads(line)
        for line in RECORDS_FILE.read_text(encoding="utf-8").splitlines()
    ]
    enriched_ids = {record["anime"]["id"] for record in enriched}
    offline = json.loads(OFFLINE_DATABASE.read_text(encoding="utf-8"))["data"]
    others = [offline_anime_record(entry) for entry in offline if entry.get("sources")]
    raw_records = enriched + [
        record for record in others if record["anime"]["id"] not in enriched_ids
    ]
    records, skipped = [], 0
    for raw_record in raw_records[:limit]:
        try:
            records.append(AnimeRecord.model_validate(raw_record))
        except ValidationError:
            skipped += 1
    print(
        f"{len(records)} anime records ({len(enriched)} from the cache), {skipped} skipped"
    )
    return records


def build_embedding_manager(settings: Any) -> Any:
    """Create the service's embedding manager, as indexing does."""
    from vector_processing.embedding_models.factory import EmbeddingModelFactory
    from vector_processing.processors.anime_field_mapper import AnimeFieldMapper
    from vector_processing.processors.embedding_manager import (
        MultiVectorEmbeddingManager,
    )
    from vector_processing.processors.text_processor import TextProcessor
    from vector_processing.processors.vision_processor import VisionProcessor
    from vector_processing.utils.image_downloader import ImageDownloader

    text_processor = TextProcessor(
        model=EmbeddingModelFactory.create_text_model(settings.embedding),
        config=settings.embedding,
    )
    vision_processor = VisionProcessor(
        model=EmbeddingModelFactory.create_vision_model(settings.embedding),
        downloader=ImageDownloader(
            cache_dir=settings.embedding.model_cache_dir or "cache"
        ),
        config=settings.embedding,
    )
    return MultiVectorEmbeddingManager(
        text_processor=text_processor,
        vision_processor=vision_processor,
        field_mapper=AnimeFieldMapper(),
    )


async def recreate_collection(settings: Any, collection_name: str) -> Any:
    """Drop and create the test collection with the service's collection settings."""
    from qdrant_client import AsyncQdrantClient
    from qdrant_db.client import QdrantClient

    if collection_name == settings.qdrant.qdrant_collection_name:
        sys.exit(f"refusing to replace the service collection {collection_name}")
    client = QdrantClient(
        config=settings.qdrant,
        async_qdrant_client=AsyncQdrantClient(
            url=settings.qdrant.qdrant_url, api_key=settings.qdrant.qdrant_api_key
        ),
        url=settings.qdrant.qdrant_url,
        collection_name=collection_name,
    )
    with suppress(Exception):  # the collection may not exist yet
        await client.delete_collection()
    await client.create_collection()
    return client


async def index_records(
    client: Any, manager: Any, records: list[Any]
) -> dict[str, list[Any]]:
    """Embed and upsert every record; return the real character and episode points."""
    points_by_kind: dict[str, list[Any]] = {"character": [], "episode": []}
    started = time.perf_counter()
    for start in range(0, len(records), EMBED_BATCH_RECORDS):
        documents = await manager.process_anime_batch(
            records[start : start + EMBED_BATCH_RECORDS]
        )
        await client.add_documents(documents, batch_size=UPSERT_BATCH_SIZE)
        for document in documents:
            kind = document.payload.get("entity_type")
            if kind in points_by_kind:
                points_by_kind[kind].append(document)
        done = min(start + EMBED_BATCH_RECORDS, len(records))
        print(
            f"embedded {done}/{len(records)} anime, "
            f"{len(points_by_kind['character'])} characters, "
            f"{len(points_by_kind['episode'])} episodes, "
            f"{time.perf_counter() - started:.0f}s",
            flush=True,
        )
    return points_by_kind


def noisy_copy(source: Any, anime_id: str, generator: Any) -> Any:
    """Copy a point: its vectors plus small random noise, attached to another anime."""
    import numpy as np
    from vector_db_interface import VectorDocument

    dense = np.asarray(source.vectors["text_vector"], dtype=np.float32)
    dense += generator.normal(0.0, DENSE_NOISE_SCALE, dense.shape).astype(np.float32)
    dense /= np.linalg.norm(dense)
    vectors: dict[str, Any] = {"text_vector": dense.tolist()}
    sparse = source.vectors.get("text_sparse_vector")
    if sparse:
        indices = sparse["indices"] if isinstance(sparse, dict) else sparse.indices
        values = np.asarray(
            sparse["values"] if isinstance(sparse, dict) else sparse.values
        )
        values *= generator.uniform(
            1 - SPARSE_WEIGHT_JITTER, 1 + SPARSE_WEIGHT_JITTER, len(values)
        )
        vectors["text_sparse_vector"] = {
            "indices": list(indices),
            "values": values.tolist(),
        }
    payload = dict(source.payload)
    if payload.get("entity_type") == "episode":
        payload["anime_id"] = anime_id
    else:
        payload["anime_ids"] = [anime_id]
    return VectorDocument(
        id=str(uuid.UUID(bytes=generator.bytes(16), version=4)),
        vectors=vectors,
        payload=payload,
    )


async def add_copies(
    client: Any, sources: list[Any], count: int, anime_ids: list[str], kind: str
) -> None:
    """Upsert ``count`` noisy copies of the given real points."""
    import numpy as np

    generator = np.random.default_rng(54)
    made = 0
    started = time.perf_counter()
    while made < count:
        size = min(COPY_BATCH_SIZE, count - made)
        batch = [
            noisy_copy(
                sources[generator.integers(len(sources))],
                anime_ids[generator.integers(len(anime_ids))],
                generator,
            )
            for _ in range(size)
        ]
        await client.add_documents(batch, batch_size=UPSERT_BATCH_SIZE)
        made += size
        if made % 50_000 < size or made == count:
            print(
                f"{kind} copies {made}/{count}, {time.perf_counter() - started:.0f}s",
                flush=True,
            )


async def build_collection(
    collection_name: str, characters: int, episodes: int, limit: int | None
) -> None:
    """Fill the test collection with real points, then copies up to the targets."""
    from common.config import get_settings

    settings = get_settings()
    records = load_records(limit)
    manager = build_embedding_manager(settings)
    client = await recreate_collection(settings, collection_name)
    real = await index_records(client, manager, records)
    anime_ids = [record.anime.id for record in records]
    for kind, target in (("character", characters), ("episode", episodes)):
        if real[kind]:
            await add_copies(
                client, real[kind], max(0, target - len(real[kind])), anime_ids, kind
            )
    stats = await client.get_stats()
    print(f"collection {collection_name}: {stats.get('total_documents')} points")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    steps = parser.add_subparsers(dest="step", required=True)
    records = steps.add_parser("records", help="build records from the Redis cache")
    records.add_argument("--limit", type=int, default=None)
    collection = steps.add_parser(
        "collection", help="embed records into the test collection"
    )
    collection.add_argument("--name", default="anime_load_test")
    collection.add_argument("--characters", type=int, default=770_000)
    collection.add_argument("--episodes", type=int, default=514_050)
    collection.add_argument(
        "--limit", type=int, default=None, help="anime records to embed"
    )
    return parser.parse_args()


def main() -> None:
    """Run the requested step."""
    args = parse_args()
    if args.step == "records":
        asyncio.run(build_records(args.limit))
    elif args.step == "collection":
        asyncio.run(
            build_collection(args.name, args.characters, args.episodes, args.limit)
        )
    else:
        sys.exit(f"unknown step {args.step}")


if __name__ == "__main__":
    main()
