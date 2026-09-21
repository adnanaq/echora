"""Merge rules for the fields that hold links, images and media.

These share one problem: the same thing arrives under several addresses.
Providers link one Crunchyroll page four different ways, publish the same
official site as both ``http`` and ``https``, and disagree on whether a service
is called ``Crunchyroll`` or ``crunchyroll``. Every rule here therefore compares
a folded form of the url rather than the url itself.

They also have to agree on who owns what. ``sources``, ``streaming_sources``
and ``external_sources`` can each hold a link to the same place, so
``external_sources`` is defined as the residual: whatever is left once the
other two have taken theirs, decided by platform as well as by url.
"""

from __future__ import annotations

import logging
from typing import Any

from common.utils.id_generation import generate_deterministic_id

from enrichment.pipeline.identity import canonical_url_key
from enrichment.pipeline.metadata_rules import Ranked, providers_supplying
from enrichment.sources.base.external_links import (
    canonical_platform,
    normalize_link_url,
)

logger = logging.getLogger(__name__)

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

# Platforms a dedicated field owns, so a link to one is never a residual.
# `sources` holds the seven providers we fetch plus the three the offline seed
# adds; `streaming_sources` holds the services people actually watch on.
# Sites that are a work's own page on a database we already track. `sources`
# holds exactly one url per work, so another spelling of one carries nothing new.
_PROVIDER_PLATFORMS: frozenset[str] = frozenset(
    {
        "myanimelist",
        "anilist",
        "anidb",
        "kitsu",
        "anime_planet",
        "anisearch",
        "animeschedule",
        "livechart",
        "simkl",
        "animecountdown",
    }
)

# Services people watch on. A link to one belongs in `streaming_sources`
# whichever field the provider filed it under - AniDB files them as external
# links, and its urls differ from everyone else's, so these are moved rather
# than dropped or the link is lost outright.
_STREAMING_PLATFORMS: frozenset[str] = frozenset(
    {
        "crunchyroll",
        "funimation",
        "netflix",
        "hidive",
        "hulu",
        "amazon",
        "prime_video",
        "disney_plus",
        "max",
        "abema",
        "u_next",
        "bilibili",
        "iqiyi",
        "wetv",
        "youku",
        "qq_video",
        "bahamut",
        "tubi",
        "viki",
        "vrv",
    }
)

_CLAIMED_PLATFORMS: frozenset[str] = _PROVIDER_PLATFORMS | _STREAMING_PLATFORMS


def merge_images(ranked: Ranked) -> dict[str, list[str]] | None:
    """Union each image category across providers.

    Every provider contributes a different artwork, and MAL alone supplies 20
    covers, so nothing is arbitrated - only exact duplicate URLs collapse. The
    categories are merged separately because a provider that has covers but no
    banners must not suppress another's banners.

    Args:
        ranked: Provider records in priority order.

    Returns:
        Image URLs by category, or ``None`` when no provider has any.
    """
    merged: dict[str, list[str]] = {}
    for _, images in providers_supplying(ranked, "images"):
        for category, urls in images.items():
            seen = merged.setdefault(category, [])
            seen.extend(url for url in urls or [] if url and url not in seen)
    return {k: v for k, v in merged.items() if v} or None


def merge_streaming_sources(ranked: Ranked) -> list[dict[str, Any]]:
    """Union streaming entries, one per platform-and-destination.

    Providers disagree on both halves of an entry: the platform arrives as
    ``Crunchyroll`` from MAL and ``crunchyroll`` from AnimeSchedule, and the
    same service is linked as ``crunchyroll.com/series-257631`` by one provider
    and ``crunchyroll.com/one-piece`` by another. Platform name and URL are
    therefore both folded before comparison.

    AnimeSchedule also publishes affiliate shorteners (``amzn.to``,
    ``apple.co``) whose host names no service, which is why the provider's own
    platform label is kept rather than deriving everything from the URL.

    Streaming links filed under ``external_sources`` are collected here too.
    AniDB files Crunchyroll, Amazon and Funimation there, at urls no other
    provider reports, so leaving them to the residual field and excluding them
    from it - which is what ownership by platform does - loses them outright.

    Args:
        ranked: Provider records in priority order.

    Returns:
        Merged streaming entries in priority order.
    """
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for field in ("streaming_sources", "external_sources"):
        for _, entries in providers_supplying(ranked, field):
            for entry in entries:
                source = entry.get("source") or ""
                platform = _streaming_platform(entry.get("platform"), source)
                if field == "external_sources" and platform not in _STREAMING_PLATFORMS:
                    continue
                key = (platform, normalize_link_url(source) if source else "")
                merged.setdefault(key, {**entry, "platform": platform})
    return list(merged.values())


def _streaming_platform(label: str | None, source: str) -> str:
    """Name a streaming service the same way whichever provider linked it.

    Providers label the same service ``Crunchyroll``, ``crunchyroll`` and
    ``Bilibili TV``, which makes the field useless for filtering. The url's
    host settles it where it names the service; AnimeSchedule's affiliate
    shorteners (``amzn.to``, ``apple.co``) do not, so their own label is
    folded instead.

    Args:
        label: The provider's name for the service.
        source: The streaming url.

    Returns:
        A canonical lowercase platform name.
    """
    from_host = canonical_platform(source) if source else "unknown"
    if from_host not in {"unknown", "official_site"}:
        return from_host
    return (label or from_host).strip().lower().replace(" ", "_")


def merge_trailers(ranked: Ranked) -> list[dict[str, Any]]:
    """Union trailers by video, keeping the richest description of each.

    MAL and Kitsu supply genuinely different videos for the same anime, so both
    survive; only the same video from two providers collapses. MAL also carries
    a title and thumbnail that Kitsu omits, so the entry with more fields wins
    a collision rather than the higher-ranked one.

    Args:
        ranked: Provider records in priority order.

    Returns:
        Merged trailer entries in priority order.
    """
    merged: dict[str, dict[str, Any]] = {}
    for _, entries in providers_supplying(ranked, "trailers"):
        for entry in entries:
            source = entry.get("source")
            if not source:
                continue
            key = normalize_link_url(source)
            if len(entry) > len(merged.get(key, {})):
                merged[key] = entry
    return list(merged.values())


def merge_external_sources(ranked: Ranked, claimed: set[str]) -> list[dict[str, Any]]:
    """Union external links, dropping those another field already owns.

    ``external_sources`` is the residual: whatever is left once ``sources`` and
    ``streaming_sources`` have taken their own. AniDB files streaming platforms
    and rival provider pages here where everyone else uses the dedicated field.

    Ownership is decided by platform as well as by url. AniDB links Crunchyroll
    as ``/series/GRMG8ZQZR`` where MAL links ``/series-257631``; the urls differ,
    so url comparison alone leaves a streaming link sitting in this field. The
    platform does not differ, and that is what settles it.

    Links are compared on the normalised url, which folds the ``x.com`` and
    ``twitter.com`` spellings of one account together, along with the ``http``
    and ``https`` spellings of one page. The richest entry wins a collision -
    MAL states a label, AniDB a language, and neither should erase the other.

    Args:
        ranked: Provider records in priority order.
        claimed: Normalised urls already owned by another field.

    Returns:
        Merged external links in priority order.
    """
    merged: dict[str, dict[str, Any]] = {}
    for _, entries in providers_supplying(ranked, "external_sources"):
        for entry in entries:
            source = entry.get("source")
            if not source:
                continue
            key = normalize_link_url(source)
            if key in claimed or _owned_elsewhere(entry.get("platform")):
                continue
            existing = merged.get(key)
            merged[key] = _richer_link(existing, entry) if existing else entry
    return list(merged.values())


def _owned_elsewhere(platform: str | None) -> bool:
    """Report whether a platform belongs to `sources` or `streaming_sources`.

    Args:
        platform: The link's canonical platform name.

    Returns:
        ``True`` when a dedicated field already owns this platform.
    """
    return (platform or "").strip().lower() in _CLAIMED_PLATFORMS


def _richer_link(existing: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Combine two records of one link, preferring the higher-ranked values.

    Args:
        existing: The entry already held, from a higher-ranked provider.
        candidate: A later provider's entry for the same URL.

    Returns:
        The existing entry filled in with any field only the candidate states.
    """
    return {**{k: v for k, v in candidate.items() if v}, **existing}


def merge_sources(ranked: Ranked, offline_data: dict[str, Any] | None) -> list[str]:
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
