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
  * **field-specific rules** - the fields where that default is measurably
    wrong. They live in ``metadata_rules``, one function per field, each
    carrying the evidence for its own rule.
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

from enrichment.pipeline.link_rules import (
    merge_external_sources,
    merge_images,
    merge_sources,
    merge_streaming_sources,
    merge_trailers,
    merged_anime_id,
)
from enrichment.pipeline.metadata_rules import (
    Ranked,
    merge_categories,
    merge_episode_count,
    merge_month,
    merge_nsfw,
    merge_object,
    merge_statistics,
    merge_synonyms,
    merge_synopsis,
    merge_title,
    merge_title_japanese,
    provider_supplied,
)
from enrichment.pipeline.relationship_merger import (
    PROVIDER_PRIORITY,
    load_agent_providers,
)
from enrichment.pipeline.relationship_merger import (
    merge_provider_records as merge_relationships,
)
from enrichment.sources.base.external_links import normalize_link_url

logger = logging.getLogger(__name__)

# Fields where every provider that supplies a value reports the same one, so the
# only question is who answers first. Disagreement here is worth a log line.
_FIRST_SIGNAL: tuple[str, ...] = (
    "country_of_origin",
    "duration",
    "entity_type",
    "rating",
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
# content_warnings is not here: it is one of the category fields, so a
# warning cannot also sit in tags or themes.
_UNION: tuple[str, ...] = ()


def _ranked(records: dict[str, dict[str, Any]]) -> Ranked:
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


def _first_signal(ranked: Ranked, field: str, *, warn: bool = False) -> Any:
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
        if provider_supplied(record.get(field))
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


def _union_values(ranked: Ranked, field: str) -> list[Any]:
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
        A canonical ``Anime``-shaped dict, relations included.
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

    for field in ("title", "title_english"):
        title = merge_title(ranked, field)
        if title is not None:
            merged[field] = title

    for field, resolve in (
        ("nsfw", merge_nsfw),
        ("title_japanese", merge_title_japanese),
        ("synopsis", merge_synopsis),
        ("images", merge_images),
    ):
        value = resolve(ranked)
        if value is not None:
            merged[field] = value

    for field in ("aired_dates", "broadcast"):
        value = merge_object(ranked, field)
        if value is not None:
            merged[field] = value

    # After aired_dates: the premiere date is the fallback when AnimeSchedule,
    # the only provider that names the month, is missing.
    month = merge_month(ranked, merged.get("aired_dates"))
    if month is not None:
        merged["month"] = month

    merged["episode_count"] = merge_episode_count(ranked)
    merged["sources"] = merge_sources(ranked, offline_data)
    merged.update(merge_categories(ranked))

    for field, values in (
        ("opening_themes", _union_values(ranked, "opening_themes")),
        ("ending_themes", _union_values(ranked, "ending_themes")),
        ("streaming_sources", merge_streaming_sources(ranked)),
        ("trailers", merge_trailers(ranked)),
    ):
        if values:
            merged[field] = values

    # external_sources is the residual: whatever sources and streaming_sources
    # have not already claimed. AniDB files streaming links here.
    claimed = {normalize_link_url(url) for url in merged["sources"]}
    claimed |= {
        normalize_link_url(entry["source"])
        for entry in merged.get("streaming_sources", [])
        if entry.get("source")
    }
    external = merge_external_sources(ranked, claimed)
    if external:
        merged["external_sources"] = external

    # Last, because it filters against the titles resolved above.
    synonyms = merge_synonyms(
        ranked,
        (
            merged.get("title"),
            merged.get("title_english"),
            merged.get("title_japanese"),
        ),
    )
    if synonyms:
        merged["synonyms"] = synonyms

    statistics = merge_statistics(ranked)
    if statistics:
        merged["statistics"] = statistics

    merged.update(merge_relationships(records))
    merged["id"] = merged_anime_id(merged["sources"])
    return merged


def merge_agent_metadata(
    agent_dir: Path, offline_data: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Merge the per-provider files in one agent directory into one record.

    Args:
        agent_dir: Directory holding the per-provider ``*.jsonl`` files.
        offline_data: The work's offline database entry, when one is known.

    Returns:
        A canonical ``Anime``-shaped dict, relations included.
    """
    return merge_provider_records(load_agent_providers(agent_dir), offline_data)
