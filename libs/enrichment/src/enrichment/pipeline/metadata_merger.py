#!/usr/bin/env python3
"""Consolidate the seven per-provider anime records into one canonical `Anime`.

Each provider mapper already emits canonical, model-shaped field values. What is
left is arbitration: seven records describe one work, and every field needs a
rule for which value survives.

Three mechanisms cover the whole model:

  * **first signal by priority** - scalars every provider agrees on. Sentinels
    (`UNKNOWN`, `OTHER`) never beat a concrete value, so a real `type` from the
    lowest-ranked provider wins over `UNKNOWN` from the highest. Disagreement is
    logged, because for these fields it means a mapper bug rather than a genuine
    difference of opinion.
  * **field-specific hierarchy** - fields where the default order is wrong.
    `episode_count` wants MAL, `synopsis` wants whichever text is longest.
  * **union** - list fields, deduplicated on a per-field key.

Single-source fields (`background`, `titles`, `hiatus`, ...) need no special
case: first-signal over one candidate is a passthrough.

`id` is assigned here rather than by a provider. It is derived from the work's
own identity, so re-running enrichment over the same anime produces the same id
and the record updates in place instead of duplicating.

Usage
-----
From an agent directory, reading the per-provider files::

    merged = merge_agent_metadata(Path("temp/One_agent5"))

Directly from ApiFetcher output, with no temp directory involved::

    results = await fetcher.fetch_all_data(ids, offline_data)
    merged = merge_provider_records(
        {
            provider: payload["anime"]
            for provider, payload in results.items()
            if payload and payload.get("anime")
        }
    )

Provider keys are the same service names ``ApiFetcher._REGISTRY`` uses, so no
translation is needed at either entry point.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from common.utils.id_generation import generate_deterministic_id

from enrichment.pipeline.identity import canonical_url_key
from enrichment.pipeline.relationship_merger import (
    PROVIDER_PRIORITY,
    is_signal,
    load_agent_providers,
)

logger = logging.getLogger(__name__)

# Fields where every provider that supplies a value reports the same one, so the
# only question is who answers first. Disagreement here is worth a log line.
_FIRST_SIGNAL: tuple[str, ...] = (
    "country_of_origin",
    "duration",
    "entity_type",
    "month",
    "nsfw",
    "season",
    "source_material",
    "status",
    "type",
    "year",
)

# Single-source fields. Kept apart from _FIRST_SIGNAL only so that a second
# provider appearing later is noticed rather than silently absorbed.
_SINGLE_SOURCE: tuple[str, ...] = (
    "background",
    "hiatus",
    "titles",
)

# Plain unions over scalar list values, deduplicated on the value itself.
_UNION: tuple[str, ...] = ("content_warnings",)

# Identity is seeded from whichever of these the work's URLs mention first. The
# order is fixed and independent of which providers a given run fetched, so a
# failed MAL fetch does not change the id as long as any provider linked to MAL.
_ID_SEED_ORDER: tuple[str, ...] = (
    "mal",
    "anilist",
    "anidb",
    "kitsu",
    "animeplanet",
    "anisearch",
)


def _ranked(records: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """Order provider records by trust, most trusted first.

    Args:
        records: Mapping of provider name to that provider's canonical record.

    Returns:
        ``(provider, record)`` pairs sorted by ``PROVIDER_PRIORITY``; unknown
        providers sort last.
    """
    return sorted(
        records.items(),
        key=lambda item: (
            PROVIDER_PRIORITY.index(item[0])
            if item[0] in PROVIDER_PRIORITY
            else len(PROVIDER_PRIORITY)
        ),
    )


def _plain(value: Any) -> Any:
    """Unwrap an enum to the value that will be compared and logged.

    Args:
        value: Field value, either a raw scalar or an enum member.

    Returns:
        The underlying value.
    """
    return value.value if hasattr(value, "value") else value


def _provider_supplied(value: Any) -> bool:
    """Report whether a provider actually supplied a value for this field.

    Mappers emit a key for every model field, so a field a provider knows
    nothing about arrives as an empty container rather than as a missing key.
    ``is_signal`` alone accepts ``{}`` as real, which lets six providers' empty
    ``titles`` outrank AniDB's 27 - the one field only AniDB supplies.

    Args:
        value: Field value as the provider's mapper emitted it.

    Returns:
        ``True`` when the value carries information.
    """
    if isinstance(value, (dict, list, str)) and not value:
        return False
    return is_signal(value)


def _first_signal(
    ranked: list[tuple[str, dict[str, Any]]], field: str, *, warn: bool = False
) -> Any:
    """Resolve a scalar field to the most trusted provider's concrete value.

    A sentinel never beats a real value, so every provider is asked for a
    concrete value before any is allowed to answer with ``UNKNOWN``.

    Args:
        ranked: Provider records in priority order.
        field: Field name to resolve.
        warn: Log when providers supply differing concrete values. Only set for
            fields expected to be uniform, where divergence means a mapper bug.

    Returns:
        The winning value, or ``None`` when no provider supplied the field.
    """
    concrete = [
        (provider, record[field])
        for provider, record in ranked
        if _provider_supplied(record.get(field))
    ]
    if warn and len({str(_plain(value)) for _, value in concrete}) > 1:
        reported = ", ".join(f"{p}={_plain(v)!r}" for p, v in concrete)
        logger.warning(f"Providers disagree on {field}: {reported}")
    if concrete:
        return concrete[0][1]
    # Every provider answered with a sentinel. Report it rather than None, so
    # "all sources say UNKNOWN" stays distinguishable from "nobody was asked".
    return next(
        (
            record[field]
            for _, record in ranked
            if record.get(field) not in (None, "", {}, [])
        ),
        None,
    )


def _union_values(ranked: list[tuple[str, dict[str, Any]]], field: str) -> list[Any]:
    """Union a list field across providers, keeping first-seen order.

    Args:
        ranked: Provider records in priority order.
        field: List field to union.

    Returns:
        Deduplicated values, most trusted provider's ordering first.
    """
    seen: dict[str, Any] = {}
    for _, record in ranked:
        for value in record.get(field) or []:
            if value:
                seen.setdefault(str(value), value)
    return list(seen.values())


def _merge_sources(
    ranked: list[tuple[str, dict[str, Any]]], offline_data: dict[str, Any] | None
) -> list[str]:
    """Union work URLs, one per work rather than one per spelling.

    Providers decorate the same id differently - MAL publishes
    ``/anime/21/One_Piece`` while AniList's cross-link to it is ``/anime/21``,
    and AnimeSchedule links ``anime-planet.com`` where Anime-Planet itself says
    ``www.anime-planet.com``. A plain string union keeps all of them: on One
    Piece that is 13 URLs for 7 works.

    The offline seed is folded in because neither side is complete on its own.
    The seed carries livechart, simkl and animecountdown, which no provider
    returns; the providers carry animeschedule, which the seed does not list.

    The longest spelling of each work wins, which is the one carrying the title
    slug. These are work pages with no query strings, so length here tracks how
    much identity the URL states.

    Args:
        ranked: Provider records in priority order.
        offline_data: The work's offline database entry, when one is known.

    Returns:
        One URL per distinct work, providers first.
    """
    best: dict[str, str] = {}
    lists = [record.get("sources") or [] for _, record in ranked]
    lists.append((offline_data or {}).get("sources") or [])
    for urls in lists:
        for url in urls:
            if not url:
                continue
            key = canonical_url_key(url)
            if len(url) > len(best.get(key, "")):
                best[key] = url
    return [url for key, url in best.items() if not _superseded_by_slug(key, best)]


def _superseded_by_slug(key: str, keys: dict[str, str]) -> bool:
    """Report whether a numeric-id key duplicates a slug key for the same work.

    Kitsu addresses one anime two ways - the providers report the slug
    (``kitsu.io/anime/one-piece``) and the offline seed the numeric id
    (``kitsu.app/anime/12``). Both are valid and neither string reveals the
    other, so they survive canonicalisation as separate works.

    Only a provider that uses both forms is affected. MAL and AniList identify
    every work numerically, so their keys never meet a slug rival and are never
    dropped.

    Args:
        key: A ``provider:kind:id`` key to judge.
        keys: Every key in the merged set.

    Returns:
        ``True`` when this key is the numeric form and a slug form exists.
    """
    provider, _, identifier = key.partition(":")
    kind = identifier.partition(":")[0]
    if not identifier.rpartition(":")[2].isdigit():
        return False
    return any(
        other != key
        and other.startswith(f"{provider}:{kind}:")
        and not other.rpartition(":")[2].isdigit()
        for other in keys
    )


def _merge_statistics(
    ranked: list[tuple[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Collect every provider's own statistics block under its provider key.

    Providers report different metrics and none of them are comparable across
    platforms, so nothing is arbitrated: each block is kept as its author sent
    it. Scores are already on the canonical 0-10 scale by the time they reach
    here - the mappers convert, since only a mapper knows its provider's scale.

    Args:
        ranked: Provider records in priority order.

    Returns:
        Provider name to that provider's statistics.
    """
    merged: dict[str, dict[str, Any]] = {}
    for _, record in ranked:
        for provider, stats in (record.get("statistics") or {}).items():
            if stats:
                merged[provider] = stats
    return dict(sorted(merged.items()))


def merged_anime_id(sources: list[str]) -> str:
    """Derive the stable id for a merged work from its provider URLs.

    The seed is one provider's canonical key rather than the whole URL set: the
    set changes whenever a fetch fails or a provider adds a link, and an id that
    moves would re-insert the work instead of updating it.

    A run that reaches none of the seed providers falls back to the full sorted
    key set, which is stable for that run but not across runs with different
    coverage. That is the honest outcome - there is no shared identity to hang a
    stable id on.

    Args:
        sources: Provider URLs for this work, as merged into ``sources``.

    Returns:
        A UUIDv5 string in the Echora namespace.

    Raises:
        ValueError: When ``sources`` is empty; a work with no URL has no
            identity to derive.
    """
    keys = sorted({canonical_url_key(url) for url in sources if url})
    if not keys:
        raise ValueError(f"Cannot derive an anime id from sources: {sources!r}")  # noqa: TRY003
    for provider in _ID_SEED_ORDER:
        match = next((key for key in keys if key.startswith(f"{provider}:")), None)
        if match:
            return generate_deterministic_id(match)
    logger.warning(f"No known provider URL among {keys}; id is coverage-dependent")
    return generate_deterministic_id("|".join(keys))


def merge_provider_records(
    records: dict[str, dict[str, Any]],
    offline_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge per-provider anime records already in memory into one record.

    Args:
        records: Mapping of provider name to that provider's canonical record.
        offline_data: The work's offline database entry. Contributes its
            ``sources`` only; every other field is the providers' to report.

    Returns:
        A canonical ``Anime``-shaped dict.
    """
    ranked = _ranked(records)
    merged: dict[str, Any] = {}

    for field in (*_FIRST_SIGNAL, *_SINGLE_SOURCE):
        value = _first_signal(ranked, field, warn=field in _FIRST_SIGNAL)
        if value is not None:
            merged[field] = value

    for field in _UNION:
        values = _union_values(ranked, field)
        if values:
            merged[field] = values

    merged["sources"] = _merge_sources(ranked, offline_data)

    statistics = _merge_statistics(ranked)
    if statistics:
        merged["statistics"] = statistics

    merged["id"] = merged_anime_id(merged.get("sources") or [])
    return merged


def merge_agent_metadata(
    agent_dir: Path, offline_data: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Merge the per-provider files in one agent directory into one record.

    Args:
        agent_dir: Directory holding the per-provider ``*.jsonl`` files.
        offline_data: The work's offline database entry, when one is known.

    Returns:
        A canonical ``Anime``-shaped dict.
    """
    return merge_provider_records(load_agent_providers(agent_dir), offline_data)
