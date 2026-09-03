#!/usr/bin/env python3
"""Consolidate per-source relationship data into the canonical grouped shape.

Each provider helper already writes canonical, model-shaped payloads:

    related_anime:           dict[AnimeRelationType,         list[RelatedAnime]]
    related_source_material: dict[SourceMaterialRelationType, list[RelatedSourceMaterial]]

This module merges those N per-provider payloads into one, so that a single
real-world work appears exactly once, under exactly one relation key, carrying
every source URL that mentioned it.

Resolution rules:
  * Identity      - exact source-URL match, else normalized title.
  * UNKNOWN/OTHER - treated as "no signal". Any concrete value supersedes them
                    regardless of source rank; they survive only when every
                    source agrees there is nothing better.
  * Ties          - broken by PROVIDER_PRIORITY (concrete values only).
  * sources/images- union across all providers, order-preserving.
                    ("sources" is the model's field name for a work's URLs;
                    the seven data providers are named "providers" here.)
  * scalars       - first non-null by PROVIDER_PRIORITY.

Usage
-----
From an agent directory, reading the per-provider files (what stage 3 does)::

    merged = merge_agent_relationships(Path("temp/One_agent5"))
    # {"related_anime": {...}, "related_source_material": {...}}

Directly from ApiFetcher output, with no temp directory involved::

    results = await fetcher.fetch_all_data(ids, offline_data)
    merged = merge_provider_records(
        {
            provider: payload["anime"]
            for provider, payload in results.items()
            if payload and payload.get("anime")
        }
    )

PROVIDER_PRIORITY uses the same service names as ``ApiFetcher._REGISTRY``, so no
key translation is needed. The only unwrapping required is ``payload["anime"]``:
``fetch_all_data`` returns the normalised envelope
``{"anime", "episodes", "characters", "extras"}``, and the relation fields live
on the ``anime`` record.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import jellyfish
from common.models.anime import (
    AnimeRelationType,
    AnimeType,
    RelatedAnime,
    RelatedSourceMaterial,
    SourceMaterialRelationType,
    SourceMaterialType,
)
from rapidfuzz import fuzz

from enrichment.pipeline.identity import NullIdentityResolver, WorkIdentityResolver
from enrichment.utils.text_utils import normalize_japanese_text

# Highest trust first, per docs/anime_relationship_and_format_type_mappings.md.
# All seven sources participate; only AniDB's character/episode files are out of
# scope, its anime-level relations are not.
#
# The doc ranks AniDB above AniList for format specifically. That deviation is
# not reproduced here because AniDB reports UNKNOWN for every observed relation,
# so Rule 0 (sentinels never win) discards it regardless of rank. One order is
# therefore used for title, format and relation alike.
#
# Kitsu is placed last: it supplies no relationship data in any observed run, so
# there is no evidence on which to rank it higher.
PROVIDER_PRIORITY: tuple[str, ...] = (
    "mal",
    "anilist",
    "anidb",
    "anime_planet",
    "anisearch",
    "animeschedule",
    "kitsu",
)

# Helpers do not name their output files uniformly - four carry an "_anime"
# suffix. That is a disk concern only, so the mapping lives here rather than
# leaking into PROVIDER_PRIORITY or onto in-memory callers.
PROVIDER_FILES: dict[str, str] = {
    "mal": "mal_anime.jsonl",
    "anilist": "anilist.jsonl",
    "anidb": "anidb_anime.jsonl",
    "anime_planet": "anime_planet_anime.jsonl",
    "anisearch": "anisearch.jsonl",
    "animeschedule": "animeschedule.jsonl",
    "kitsu": "kitsu_anime.jsonl",
}

# Values that carry no information and must never beat a concrete value.
_SENTINELS: frozenset[str] = frozenset(
    {
        AnimeType.UNKNOWN.value,
        AnimeRelationType.OTHER.value,
        SourceMaterialType.UNKNOWN.value,
        SourceMaterialType.OTHER.value,
        "UNKNOWN",
        "OTHER",
    }
)

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_ZERO_PAD = re.compile(r"\b0+(\d)")
_LONG_VOWEL = re.compile(r"([aeiou])\1+")
_TRAILING_NUM = re.compile(r"(\d+)\s*$")

# Tokens that slugs add but real titles omit (or vice versa). They carry no
# distinguishing power on their own, so token-set comparison should ignore them.
_NOISE_TOKENS = frozenset({"tv", "special", "movie", "ver", "episode", "ep", "the"})

# Ensemble acceptance threshold, calibrated against the near-duplicate fixture.
MATCH_THRESHOLD = 0.90


def normalize_title(title: str) -> str:
    """Fold a title (or an AnimeSchedule slug) to a comparable key.

    Lowercases and collapses every run of non-alphanumeric characters to a
    single space, so punctuation and slug hyphens stop being significant.

    Args:
        title: Raw title as reported by a source.

    Returns:
        Space-separated lowercase token string.
    """
    return " ".join(_NON_ALNUM.sub(" ", title.lower()).split())


def canonical_title(title: str) -> str:
    """Aggressively fold romaji/slug variance so equivalent titles converge.

    Handles the observed failure classes: kana vs romaji, long-vowel spellings
    (``Kyuutai``/``Kyutai``), zero-padding (``Movie 01``/``Movie 1``) and slug
    punctuation (``mamore`` from ``Mamore!``).

    Args:
        title: Raw title as reported by a source.

    Returns:
        Canonicalised title suitable for equality and similarity comparison.
    """
    text = title or ""
    if any("぀" <= char <= "ヿ" or "一" <= char <= "龯" for char in text):
        text = normalize_japanese_text(text)
    text = normalize_title(text)
    text = _ZERO_PAD.sub(r"\1", text)
    text = text.replace("ou", "o").replace("oh", "o")
    return _LONG_VOWEL.sub(r"\1", text)


def significant_tokens(title: str) -> set[str]:
    """Canonical tokens with slug noise removed.

    Args:
        title: Raw title as reported by a source.

    Returns:
        Set of canonical tokens excluding ``_NOISE_TOKENS``.
    """
    return {
        token for token in canonical_title(title).split() if token not in _NOISE_TOKENS
    }


def trailing_index(title: str) -> str | None:
    """Trailing sequence number, which distinguishes sequels from their base work.

    ``Collabo Special`` vs ``Collabo Special 2`` are different works even at 97%
    string similarity, so a differing trailing number must veto a match.

    Args:
        title: Raw title as reported by a source.

    Returns:
        The trailing integer as a string, or ``None`` when the title does not
        end in a number.
    """
    match = _TRAILING_NUM.search(canonical_title(title))
    return match.group(1) if match else None


def _phonetic(text: str) -> str:
    """Metaphone encoding of each token, for long-vowel-insensitive comparison.

    Args:
        text: Canonicalised title.

    Returns:
        Space-separated metaphone codes.
    """
    return " ".join(jellyfish.metaphone(token) for token in text.split() if token)


def titles_match(left: str, right: str, threshold: float = MATCH_THRESHOLD) -> bool:
    """Ensemble title match mirroring EnsembleFuzzyMatcher's cheap signals.

    Uses edit-distance, token-sort and phonetic agreement over canonicalised
    titles. A differing trailing sequence number vetoes the match outright,
    regardless of score.

    Args:
        left: First title.
        right: Second title.
        threshold: Minimum ensemble score in ``[0, 1]`` required to accept.

    Returns:
        ``True`` when both titles denote the same work.
    """
    left_key, right_key = canonical_title(left), canonical_title(right)
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True
    if trailing_index(left) != trailing_index(right):
        return False

    # Deliberately no subset rule: in a franchise every title shares a common
    # prefix ("one piece"), so token-subset matching collapses the entire
    # franchise into one entry. token_set_ratio has the same bias, so it is
    # excluded here and only the order-insensitive token_sort_ratio is kept.
    score = max(
        fuzz.ratio(left_key, right_key),
        fuzz.token_sort_ratio(left_key, right_key),
        fuzz.ratio(_phonetic(left_key), _phonetic(right_key)),
    )
    return score / 100.0 >= threshold


def normalize_url(url: str) -> str:
    """Normalize a source URL for identity comparison.

    Args:
        url: Source URL as reported by a source.

    Returns:
        Lowercased URL without the ``www.`` prefix or a trailing slash.
    """
    return url.lower().replace("://www.", "://").rstrip("/")


def is_signal(value: Any) -> bool:
    """Return whether a value carries real information.

    ``UNKNOWN`` and ``OTHER`` are sentinels meaning "no data", so they must
    never outrank a concrete value from a lower-priority source.

    Args:
        value: Field value, either a raw string or an enum member.

    Returns:
        ``False`` for ``None``, empty strings and sentinel values.
    """
    if value is None or value == "":
        return False
    plain_value = value.value if hasattr(value, "value") else value
    return str(plain_value) not in _SENTINELS


class _Groups:
    """Union-find over identity keys, grouping entries that denote one work.

    Each source entry is registered under every identity key it answers to
    (its URLs plus its normalized title). Registering the same key twice unions
    the two groups, so entries reached by any shared key end up together.
    """

    def __init__(self) -> None:
        """Initialise an empty union-find with no registered keys."""
        self._parent: dict[str, str] = {}
        self._members: dict[str, list[tuple[int, str, dict[str, Any]]]] = {}

    def _find(self, key: str) -> str:
        """Resolve a key to its group root, path-compressing on the way.

        Args:
            key: Identity key to resolve.

        Returns:
            The root key representing this key's group.
        """
        self._parent.setdefault(key, key)
        while self._parent[key] != key:
            self._parent[key] = self._parent[self._parent[key]]
            key = self._parent[key]
        return key

    def add(
        self, keys: list[str], rank: int, relation: str, entry: dict[str, Any]
    ) -> None:
        """Attach one source entry under every identity key it answers to.

        Args:
            keys: Identity keys for this entry (URLs and normalized title).
            rank: Source priority index; lower means more trusted.
            relation: Relation group the source filed this entry under.
            entry: The raw canonical entry from that source.
        """
        roots = {self._find(identity_key) for identity_key in keys}
        root = sorted(roots)[0]
        for other_root in roots:
            self._parent[other_root] = root
        for identity_key in keys:
            self._parent[self._find(identity_key)] = root
        self._members.setdefault(root, []).append((rank, relation, entry))

    def groups(self) -> list[list[tuple[int, str, dict[str, Any]]]]:
        """Collapse to final member lists, re-resolving roots after all unions.

        Returns:
            One member list per distinct work, each holding
            ``(rank, relation, entry)`` triples from every contributing source.
        """
        merged: dict[str, list[tuple[int, str, dict[str, Any]]]] = {}
        for root, members in self._members.items():
            merged.setdefault(self._find(root), []).extend(members)
        return list(merged.values())


def _fuzzy_union(
    groups: list[list[tuple[int, str, dict[str, Any]]]],
    resolver: WorkIdentityResolver,
) -> list[list[tuple[int, str, dict[str, Any]]]]:
    """Second pass: union groups whose titles match under the ensemble matcher.

    Exact URL/title keys catch identical spellings only. This pass folds together
    the romaji, spacing, zero-padding and slug variants that exact keys miss.

    The resolver is authoritative in both directions: groups it places on
    different work ids are never fused, however similar their titles look.
    ``Kyutai Panic Adventure!`` and ``Kyuutai Panic Adventure`` differ only by a
    long vowel but are genuinely distinct works, and only the resolver knows it.

    Args:
        groups: Member lists produced by exact-key grouping.
        resolver: Identity resolver used to veto merges across known work ids.

    Returns:
        Member lists after fuzzy-matching groups have been combined.
    """
    representative_titles: list[str] = []
    resolved_ids: list[set[str]] = []
    for members in groups:
        highest_priority = min(members, key=lambda member: member[0])
        representative_titles.append(highest_priority[2].get("title") or "")
        found = {
            work_id
            for _, _, entry in members
            for url in entry.get("sources") or []
            if (work_id := resolver.resolve(url)) is not None
        }
        resolved_ids.append(found)

    parent = list(range(len(groups)))

    def find(index: int) -> int:
        """Resolve a group index to its root, path-compressing on the way.

        Args:
            index: Group index to resolve.

        Returns:
            Index of the root group this index now belongs to.
        """
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left_index in range(len(groups)):
        for right_index in range(left_index + 1, len(groups)):
            if find(left_index) == find(right_index):
                continue
            left_ids, right_ids = resolved_ids[left_index], resolved_ids[right_index]
            if left_ids and right_ids and not (left_ids & right_ids):
                continue  # resolver says these are different works
            if titles_match(
                representative_titles[left_index], representative_titles[right_index]
            ):
                parent[find(right_index)] = find(left_index)

    combined: dict[int, list[tuple[int, str, dict[str, Any]]]] = {}
    for group_index, members in enumerate(groups):
        combined.setdefault(find(group_index), []).extend(members)
    return list(combined.values())


def _identity_keys(entry: dict[str, Any], resolver: WorkIdentityResolver) -> list[str]:
    """Build the identity keys an entry should be registered under.

    Args:
        entry: Canonical entry from one source.

    Returns:
        Namespaced keys: ``url:<url>`` per source URL, ``work:<id>`` when the
        resolver knows the work, and ``title:<normalized title>``.
    """
    keys = [f"url:{normalize_url(url)}" for url in entry.get("sources") or []]
    # Tier 1: a resolved work id is authoritative and links entries that share
    # no URL and no recognisable title (romaji vs English, for instance).
    work_id = next(
        (
            resolved
            for url in entry.get("sources") or []
            if (resolved := resolver.resolve(url)) is not None
        ),
        None,
    )
    if work_id is not None:
        keys.append(f"work:{work_id}")
    title = entry.get("title") or ""
    if title:
        keys.append(f"title:{normalize_title(title)}")
    return keys


def _pick(members: list[tuple[int, str, dict[str, Any]]], field: str) -> Any:
    """Resolve one scalar field across a group's contributing providers.

    Concrete values always beat sentinels, so a real format from the lowest
    priority source wins over ``UNKNOWN`` from the highest.

    Args:
        members: Group members, pre-sorted by source priority.
        field: Field name to resolve.

    Returns:
        The winning value, or ``None`` when no source supplied the field.
    """
    for _, _, entry in members:
        if is_signal(entry.get(field)):
            return entry.get(field)
    for _, _, entry in members:
        if entry.get(field) is not None:
            return entry.get(field)
    return None


def _pick_relation(members: list[tuple[int, str, dict[str, Any]]]) -> str:
    """Choose the relation group a merged work belongs to.

    Args:
        members: Group members, pre-sorted by source priority.

    Returns:
        The winning relation key; a concrete relation always beats ``OTHER``.
    """
    for _, relation, _ in members:
        if is_signal(relation):
            return relation
    return members[0][1]


def _union(members: list[tuple[int, str, dict[str, Any]]], field: str) -> list[str]:
    """Union a list-valued field across a group, preserving first-seen order.

    Args:
        members: Group members, pre-sorted by source priority.
        field: List field to union, such as ``sources`` or ``images``.

    Returns:
        Deduplicated values in priority order.
    """
    seen: dict[str, None] = {}
    for _, _, entry in members:
        for value in entry.get(field) or []:
            if value:
                seen.setdefault(value, None)
    return list(seen)


def merge_relation_field(
    per_provider: dict[str, dict[str, list[dict[str, Any]]]],
    *,
    is_source_material: bool,
    resolver: WorkIdentityResolver | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Merge one relation field across providers into the grouped canonical shape.

    Args:
        per_provider: Mapping of provider name to that provider's grouped payload.
        is_source_material: When ``True`` resolve ``chapters``/``volumes``
            instead of ``year``/``episode_count``.

    Returns:
        Relation key to merged entries, each entry denoting one distinct work.
    """
    resolver = resolver or NullIdentityResolver()
    groups = _Groups()
    for provider, grouped in per_provider.items():
        rank = (
            PROVIDER_PRIORITY.index(provider) if provider in PROVIDER_PRIORITY else 99
        )
        for relation, entries in (grouped or {}).items():
            for entry in entries:
                keys = _identity_keys(entry, resolver)
                if keys:
                    groups.add(keys, rank, relation, entry)

    # "score" is left unset: it is a join away from the related entity itself.
    scalars = ("status",)
    scalars += (
        ("chapters", "volumes") if is_source_material else ("year", "episode_count")
    )

    out: dict[str, list[dict[str, Any]]] = {}
    for members in _fuzzy_union(groups.groups(), resolver):
        members.sort(key=lambda m: m[0])
        merged: dict[str, Any] = {
            "title": _pick(members, "title"),
            "type": _pick(members, "type"),
            "sources": _union(members, "sources"),
            "images": _union(members, "images"),
        }
        for field in scalars:
            value = _pick(members, field)
            if value is not None:
                merged[field] = value
        out.setdefault(_pick_relation(members), []).append(merged)

    for entries in out.values():
        entries.sort(key=lambda e: (e.get("title") or "").lower())
    return dict(sorted(out.items()))


def load_agent_providers(agent_dir: Path) -> dict[str, dict[str, Any]]:
    """Read the canonical record from each provider file present in an agent dir.

    Missing provider files are skipped rather than raising, so a partial
    enrichment run still consolidates whatever was fetched.

    Args:
        agent_dir: Directory holding the per-source ``*.jsonl`` files.

    Returns:
        Mapping of provider name to its parsed record, in priority order.
    """
    records: dict[str, dict[str, Any]] = {}
    for provider in PROVIDER_PRIORITY:
        path = agent_dir / PROVIDER_FILES[provider]
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            line = handle.readline()
        if line.strip():
            records[provider] = json.loads(line)
    return records


def merge_agent_relationships(
    agent_dir: Path, resolver: WorkIdentityResolver | None = None
) -> dict[str, Any]:
    """Merge both relation fields for one agent directory.

    Args:
        agent_dir: Directory holding the per-source ``*.jsonl`` files.

    Returns:
        Mapping with ``related_anime`` and ``related_source_material``, each
        grouped by relation type.
    """
    return merge_provider_records(load_agent_providers(agent_dir), resolver)


def merge_provider_records(
    records: dict[str, dict[str, Any]],
    resolver: WorkIdentityResolver | None = None,
) -> dict[str, Any]:
    """Merge both relation fields from per-provider records already in memory.

    Entry point for in-process callers such as the enrichment pipeline, which
    hold the provider payloads directly and need no temp directory. Keys are the
    same service names ApiFetcher uses.

    Args:
        records: Mapping of provider name to that provider's canonical record.

    Returns:
        Mapping with ``related_anime`` and ``related_source_material``, each
        grouped by relation type.
    """
    return {
        "related_anime": merge_relation_field(
            {
                provider: record.get("related_anime") or {}
                for provider, record in records.items()
            },
            is_source_material=False,
            resolver=resolver,
        ),
        "related_source_material": merge_relation_field(
            {
                provider: record.get("related_source_material") or {}
                for provider, record in records.items()
            },
            is_source_material=True,
            resolver=resolver,
        ),
    }


def validate(merged: dict[str, Any]) -> list[str]:
    """Validate merged output against the canonical models.

    Args:
        merged: Output of :func:`merge_agent_relationships`.

    Returns:
        Human-readable error strings; empty when every entry and relation key
        validates against the models in ``common.models.anime``.
    """
    errors: list[str] = []
    for relation, entries in merged["related_anime"].items():
        try:
            AnimeRelationType(relation)
        except ValueError:
            errors.append(f"bad AnimeRelationType: {relation!r}")
        for entry in entries:
            try:
                RelatedAnime(**entry)
            except Exception as exc:
                errors.append(f"{relation}/{entry.get('title')!r}: {exc}")
    for relation, entries in merged["related_source_material"].items():
        try:
            SourceMaterialRelationType(relation)
        except ValueError:
            errors.append(f"bad SourceMaterialRelationType: {relation!r}")
        for entry in entries:
            try:
                RelatedSourceMaterial(**entry)
            except Exception as exc:
                errors.append(f"{relation}/{entry.get('title')!r}: {exc}")
    return errors
