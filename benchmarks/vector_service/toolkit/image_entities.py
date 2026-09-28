"""Group image URLs from local enrichment data by the entity they show.

Enrichment agent folders (``temp/<Anime>_agent<N>/``) hold one file per
provider and kind: anime files (``mal_anime.jsonl``, ``anilist.jsonl``, ...)
with an ``images`` mapping of covers, posters and banners, and
``*_characters.jsonl`` / ``*_episodes.jsonl`` files whose records carry an
``images`` list. Providers are not merged yet, so the same character appears
once per provider; records are grouped by normalized name (characters) or
episode number (episodes) within an anime, which folds several providers'
pictures into one entity. Folders of the same anime (``One_agent1``,
``One_refresh``) are grouped together by the name before the first ``_``.
"""

import json
import random
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

IsCached = Callable[[str], bool]
KIND_SUFFIXES = {"_characters.jsonl": "character", "_episodes.jsonl": "episode"}


@dataclass(frozen=True)
class ImageEntity:
    key: str
    entity_type: str
    urls: tuple[str, ...]


def normalized_name(name: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def image_list(images: Any) -> list[str]:
    """URLs from an ``images`` field: a list, or a mapping of lists."""
    if isinstance(images, dict):
        return [
            url
            for urls in images.values()
            if isinstance(urls, list)
            for url in urls
            if isinstance(url, str)
        ]
    if isinstance(images, list):
        return [url for url in images if isinstance(url, str)]
    return []


def read_records(path: Path) -> Iterable[dict[str, Any]]:
    for line in path.read_text(errors="ignore").splitlines():
        if line.strip():
            yield json.loads(line)


def kind_of(path: Path) -> str:
    for suffix, kind in KIND_SUFFIXES.items():
        if path.name.endswith(suffix):
            return kind
    return "anime"


def entity_key(anime: str, kind: str, record: dict[str, Any]) -> str:
    if kind == "character":
        return f"{anime}:character:{normalized_name(record.get('name'))}"
    if kind == "episode":
        return f"{anime}:episode:{record.get('episode_number')}"
    return f"{anime}:anime"


def build_entities(
    groups: dict[tuple[str, str], list[str]], is_cached: IsCached
) -> list[ImageEntity]:
    entities = []
    for (key, kind), urls in groups.items():
        kept = tuple(dict.fromkeys(url for url in urls if is_cached(url)))
        if kept:
            entities.append(ImageEntity(key, kind, kept))
    return entities


def collect_agent_entities(
    folders: Iterable[Path], is_cached: IsCached
) -> list[ImageEntity]:
    """Entities with their cached image URLs, from enrichment agent folders."""
    groups: dict[tuple[str, str], list[str]] = {}
    for folder in folders:
        anime = folder.name.split("_")[0]
        for path in sorted(folder.glob("*.jsonl")):
            kind = kind_of(path)
            for record in read_records(path):
                urls = image_list(record.get("images"))
                if urls:
                    groups.setdefault(
                        (entity_key(anime, kind, record), kind), []
                    ).extend(urls)
    return build_entities(groups, is_cached)


def collect_database_entities(path: Path, is_cached: IsCached) -> list[ImageEntity]:
    """Anime and their characters from an enriched database file (``{"data": [...]}``)."""
    groups: dict[tuple[str, str], list[str]] = {}
    for record in json.loads(path.read_text())["data"]:
        anime = record.get("anime") or {}
        title = normalized_name(anime.get("title"))
        groups.setdefault((f"db:{title}:anime", "anime"), []).extend(
            image_list(anime.get("images"))
        )
        for character in record.get("characters") or []:
            key = f"db:{title}:character:{normalized_name(character.get('name'))}"
            groups.setdefault((key, "character"), []).extend(
                image_list(character.get("images"))
            )
    return build_entities(groups, is_cached)


def hold_out_queries(
    entities: Iterable[ImageEntity], seed: int
) -> tuple[dict[str, tuple[str, ...]], list[tuple[str, str]]]:
    """Hold out one random image of each entity with two or more as its query.

    Returns the images each entity keeps in the collection, and the queries
    as ``(entity key, held-out image URL)``. Entities with one image are kept
    whole, as other entities a search must rank below the right one.
    """
    random_generator = random.Random(seed)  # noqa: S311 - repeatable split, not security
    stored: dict[str, tuple[str, ...]] = {}
    queries: list[tuple[str, str]] = []
    for entity in entities:
        if len(entity.urls) < 2:
            stored[entity.key] = entity.urls
            continue
        held_out = random_generator.choice(entity.urls)
        stored[entity.key] = tuple(url for url in entity.urls if url != held_out)
        queries.append((entity.key, held_out))
    return stored, queries
