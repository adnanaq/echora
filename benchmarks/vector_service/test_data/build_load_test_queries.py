#!/usr/bin/env python3
"""Build the search queries the k6 load test sends to the vector service.

Queries come from real anime data in three kinds, since query length drives
embedding cost:

- ``short``: two or three tags taken from one anime, like "slice of life trains"
- ``title``: a title or synonym
- ``long``: the first sentence of a synopsis

Output is deterministic for a given seed, so every load test run sends the
same mix.
"""

import argparse
import json
import random
import re
from pathlib import Path

OFFLINE_DATABASE = Path("assets/seed_data/anime-offline-database.json")
ENRICHED_DATABASE = Path("assets/seed_data/anime_database.json")
OUTPUT_FILE = Path("benchmarks/vector_service/load/search_queries.json")
LONG_QUERY_MAX_CHARS = 300


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--short-count", type=int, default=1000)
    parser.add_argument("--title-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=54)
    return parser.parse_args()


def build_short_queries(
    offline_anime: list[dict], count: int, random_generator: random.Random
) -> list[str]:
    """Join two or three tags from the same anime, so tags co-occur as in real data."""
    tagged = [anime["tags"] for anime in offline_anime if len(anime["tags"]) >= 3]
    queries: set[str] = set()
    while len(queries) < count:
        tags = random_generator.choice(tagged)
        queries.add(
            " ".join(random_generator.sample(tags, random_generator.choice((2, 3))))
        )
    return sorted(queries)


def build_title_queries(
    offline_anime: list[dict], count: int, random_generator: random.Random
) -> list[str]:
    """Pick titles or synonyms, as users search either."""
    names = [
        name
        for anime in offline_anime
        for name in (anime["title"], *anime["synonyms"])
        if name.isascii() and 3 <= len(name) <= 80
    ]
    return sorted(set(random_generator.sample(names, count)))


def build_long_queries(enriched_records: list[dict]) -> list[str]:
    """Take the first sentence of each synopsis, capped in length."""
    queries: set[str] = set()
    for record in enriched_records:
        synopsis = (record["anime"].get("synopsis") or "").strip()
        if not synopsis:
            continue
        first_sentence = re.split(r"(?<=[.!?])\s", synopsis, maxsplit=1)[0]
        queries.add(first_sentence[:LONG_QUERY_MAX_CHARS])
    return sorted(queries)


def main() -> None:
    """Write the query file."""
    args = parse_args()
    random_generator = random.Random(args.seed)  # noqa: S311 - picks test queries, not secrets
    offline_anime = json.loads(OFFLINE_DATABASE.read_text(encoding="utf-8"))["data"]
    enriched_records = json.loads(ENRICHED_DATABASE.read_text(encoding="utf-8"))["data"]
    queries = {
        "short": build_short_queries(offline_anime, args.short_count, random_generator),
        "title": build_title_queries(offline_anime, args.title_count, random_generator),
        "long": build_long_queries(enriched_records),
    }
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(queries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    counts = ", ".join(f"{kind}: {len(texts)}" for kind, texts in queries.items())
    print(f"Wrote {OUTPUT_FILE} ({counts})")


if __name__ == "__main__":
    main()
