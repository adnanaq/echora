#!/usr/bin/env python3
"""Download the images referenced in local data files into the image cache.

Collects every image URL (jpg, jpeg, png, webp) from the given files, skips
excluded hosts (AniDB by default; ``--only-host anidb.net`` fetches just
AniDB, which should run one at a time with a longer pause), and
downloads the rest with the service's own ``ImageDownloader``, so they land in
the same cache (``cache/images``, named by URL hash) that indexing and the
image test tools read. Already cached images are skipped, so a run can be
stopped and resumed. Downloads run a few at a time with a short pause.

Run from the repository root:

    ./pants run benchmarks/vector_service/test_data/download_images.py -- \\
      temp/*/*.jsonl assets/seed_data/anime_database.json
"""

import argparse
import asyncio
import re
import time
from collections.abc import Iterable
from pathlib import Path

import aiohttp
from vector_processing.utils.image_downloader import ImageDownloader

IMAGE_URL = re.compile(
    r"https?://[^\"\s\\]+?\.(?:jpg|jpeg|png|webp)(?:\?[^\"\s\\]*)?", re.IGNORECASE
)
DEFAULT_EXCLUDED_HOSTS = ("anidb.net",)


def image_urls(
    text: str, excluded_hosts: Iterable[str], only_hosts: Iterable[str] = ()
) -> set[str]:
    """Image URLs in ``text``: from ``only_hosts`` if given, else not from ``excluded_hosts``."""
    only = tuple(only_hosts)
    excluded = () if only else tuple(excluded_hosts)
    found = set()
    for url in IMAGE_URL.findall(text):
        host = url.split("/")[2]
        if only and not any(name in host for name in only):
            continue
        if any(name in host for name in excluded):
            continue
        found.add(url)
    return found


async def download_all(
    urls: list[str], concurrency: int, pause_seconds: float
) -> tuple[int, int]:
    """Download ``urls`` into the image cache; return (downloaded or cached, failed)."""
    downloader = ImageDownloader()
    slots = asyncio.Semaphore(concurrency)
    done = failed = 0
    started = time.perf_counter()

    async def fetch(url: str, session: aiohttp.ClientSession) -> None:
        nonlocal done, failed
        async with slots:
            path = await downloader.download_and_cache_image(url, session)
            await asyncio.sleep(pause_seconds)
        if path:
            done += 1
        else:
            failed += 1
        if (done + failed) % 250 == 0:
            print(
                f"{done + failed}/{len(urls)}: {failed} failed, {time.perf_counter() - started:.0f} s",
                flush=True,
            )

    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*(fetch(url, session) for url in urls))
    return done, failed


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument(
        "--exclude-host", action="append", default=list(DEFAULT_EXCLUDED_HOSTS)
    )
    parser.add_argument(
        "--only-host", action="append", default=[], help="download only these hosts"
    )
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--pause-seconds", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    urls: set[str] = set()
    for path in args.files:
        urls |= image_urls(
            path.read_text(errors="ignore"), args.exclude_host, args.only_host
        )
    ordered = sorted(urls)
    print(f"{len(ordered)} image URLs from {len(args.files)} files", flush=True)
    done, failed = asyncio.run(
        download_all(ordered, args.concurrency, args.pause_seconds)
    )
    print(f"finished: {done} in the cache, {failed} failed", flush=True)


if __name__ == "__main__":
    main()
