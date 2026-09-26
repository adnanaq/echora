"""Per-field arbitration rules for the anime metadata merge.

Most fields are resolved by the generic mechanisms in ``metadata_merger`` —
first concrete value by provider priority, or a union. The fields here are the
ones where that default is measurably wrong, each for its own reason:

  * ``episode_count`` — four providers report 0 for a long-running series, and
    0 is also the model default, so priority alone answers with a non-count.
  * ``synopsis`` — priority picks MAL's 1111 characters over AniSearch's 1826.
  * ``title`` / ``title_english`` — priority is blind to ``ONE PIECE`` versus
    ``One Piece``.
  * ``title_japanese`` — four providers file romaji under it and three file
    kana, so priority fills a Japanese-title field with romaji.
  * ``sources`` — a string union keeps one work under several spellings.

Each rule takes the ranked provider records and answers for one field, so the
merger stays a list of field assignments rather than a nest of special cases.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from statistics import mean, median
from typing import Any

from common.utils.datetime_utils import to_japan_time

from enrichment.pipeline.relationship_merger import is_signal
from enrichment.pipeline.same_company import company_keys
from enrichment.pipeline.same_word import word_key
from enrichment.sources.base.external_links import normalize_link_url
from enrichment.utils.text_utils import (
    fold_for_comparison,
    has_cjk,
    is_shouting,
    strip_synopsis_markup,
)

logger = logging.getLogger(__name__)

# episode_count ignores the default order: it is the one field where a provider
# that tracks the airing closely beats one that merely lists the work. AniList,
# AniSearch, AnimeSchedule and Kitsu all report 0 for a long-running series.
_EPISODE_COUNT_PRIORITY: tuple[str, ...] = ("mal", "anidb", "anime_planet")

# A synopsis within this fraction of the longest is treated as equally complete,
# so provider priority rather than a few characters decides between them.
_SYNOPSIS_TIE = 0.10

# Most specific field first. A word goes to the highest one any provider
# filed it under, so tags only holds what nobody else claimed.
_CATEGORY_ORDER: tuple[str, ...] = (
    "content_warnings",
    "genres",
    "demographics",
    "themes",
    "tags",
)


Ranked = list[tuple[str, dict[str, Any]]]


def provider_supplied(value: Any) -> bool:
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


def providers_supplying(ranked: Ranked, field: str) -> list[tuple[str, Any]]:
    """Collect the providers that supplied a value for a field, most trusted first.

    Args:
        ranked: Provider records in priority order.
        field: Field name to collect.

    Returns:
        ``(provider, value)`` pairs for providers with a real value.
    """
    return [
        (provider, record[field])
        for provider, record in ranked
        if provider_supplied(record.get(field))
    ]


def merge_episode_count(ranked: Ranked) -> int:
    """Resolve the episode count against its own provider order.

    ``episode_count`` defaults to 0 in the model, so a provider that does not
    track it is indistinguishable from one reporting zero episodes. Zero is
    therefore read as "not supplied" - no aired anime has none, and letting it
    count as a value puts AniList's 0 ahead of AniDB's 1184.

    Args:
        ranked: Provider records in priority order.

    Returns:
        The winning count, or 0 when no provider tracks it.
    """
    counts = {
        provider: record["episode_count"]
        for provider, record in ranked
        if record.get("episode_count")
    }
    for provider in _EPISODE_COUNT_PRIORITY:
        if provider in counts:
            return int(counts[provider])
    return int(next(iter(counts.values()), 0))


def merge_nsfw(ranked: Ranked) -> bool:
    """Resolve the adult-content flag, taking any provider's yes as a yes.

    Priority is the wrong rule here. Only AniList and Kitsu supply the flag,
    AniList outranks Kitsu, so a work Kitsu marks adult and AniList does not
    would come out marked safe. The two errors are not equally bad, and this is
    the same principle the streaming routing already follows: union what the
    providers know rather than letting the highest-ranked one speak alone.

    The field is never left empty. When neither provider says anything the
    answer is ``False``, so consumers always get a usable flag rather than
    having to decide for themselves what an absent one means.

    Args:
        ranked: Provider records in priority order.

    Returns:
        ``True`` when any provider flags the work, otherwise ``False``.
    """
    return any(bool(value) for _, value in providers_supplying(ranked, "nsfw"))


def merge_synopsis(ranked: Ranked) -> str | None:
    """Choose the most complete synopsis, then store it without its markup.

    Length is the only available proxy for completeness, and it is a fair one
    once markup and attribution are gone: AniDB's 774 characters really do say
    less than AniSearch's 1826. Priority decides only between texts of
    comparable length.

    Args:
        ranked: Provider records in priority order.

    Returns:
        The winning synopsis, stripped, or ``None`` when no provider wrote one.
    """
    cleaned = [
        stripped
        for _, text in providers_supplying(ranked, "synopsis")
        if (stripped := strip_synopsis_markup(text))
    ]
    if not cleaned:
        return None
    longest = max(len(text) for text in cleaned)
    return next(text for text in cleaned if len(text) >= longest * (1 - _SYNOPSIS_TIE))


def merge_title(ranked: Ranked, field: str) -> str | None:
    """Resolve a latin-script title, preferring the one that is not shouting.

    AniList styles One Piece as ``ONE PIECE`` where everyone else writes
    ``One Piece``. Both name the same work, so the difference is typography and
    the readable form should win even from a lower-ranked provider. Titles that
    differ by more than case are left to priority.

    Args:
        ranked: Provider records in priority order.
        field: ``title`` or ``title_english``.

    Returns:
        The winning title, or ``None`` when no provider supplied one.
    """
    candidates = [value for _, value in providers_supplying(ranked, field)]
    if not candidates:
        return None
    best = candidates[0]
    if not is_shouting(best):
        return best
    return next(
        (
            other
            for other in candidates
            if other.casefold() == best.casefold() and not is_shouting(other)
        ),
        best,
    )


def merge_title_japanese(ranked: Ranked) -> str | None:
    """Resolve the Japanese title, preferring an actual Japanese script.

    Four providers file the romaji ``ONE PIECE`` under this field and three file
    ``ワンピース``. Priority alone therefore fills a Japanese-title field with
    romaji; the romaji is a synonym, not the Japanese title.

    Args:
        ranked: Provider records in priority order.

    Returns:
        The winning title, or ``None`` when no provider supplied one.
    """
    candidates = [value for _, value in providers_supplying(ranked, "title_japanese")]
    if not candidates:
        return None
    return next((value for value in candidates if has_cjk(value)), candidates[0])


def merge_categories(ranked: Ranked) -> dict[str, list[Any]]:
    """Put every genre, theme, demographic and tag in exactly one field.

    The four fields overlap because providers disagree about where a word goes,
    not about the word: ``Shounen`` arrives as a demographic from MAL and
    AniList, a genre from AnimeSchedule, a theme from Kitsu and a tag from
    AniDB. Keeping all four would store one word four times.

    The most trusted provider that actually classified the word decides where
    it goes. A ``tags`` entry is not a classification - it means the provider
    had nothing to say, and AniDB has no genre, theme or demographic field at
    all, so everything it knows arrives as a tag. Letting those count would
    demote ``Swordplay`` out of themes on AniDB's say-so alone.

    ``content_warnings`` is the exception: any provider flagging a word as
    adult content wins outright, whoever else disagrees.

    ``genres > demographics > themes > tags`` then settles a single provider
    that used two of its own fields - Kitsu files ``Super Power`` under both
    its genres and its themes. A word nobody classified stays a tag.

    There is no vocabulary of our own. The providers decide.

    The surviving spelling is the one most providers used, since AniDB writes
    everything lowercase and would otherwise decide the storage form on its
    own. Provider priority breaks a tie.

    Themes carry a description, so they are objects while the other three hold
    plain strings; a word changing field changes shape with it.

    Args:
        ranked: Provider records in priority order.

    Returns:
        ``genres``, ``demographics``, ``themes`` and ``tags``, each word in
        exactly one of them.
    """
    # word -> field -> spelling -> [rank of each provider that wrote it that way]
    claims: dict[str, dict[str, dict[str, list[int]]]] = {}
    descriptions: dict[str, str] = {}

    for rank, (_, record) in enumerate(ranked):
        for field in _CATEGORY_ORDER:
            for raw in record.get(field) or []:
                name = raw.get("name") if isinstance(raw, dict) else raw
                if not name:
                    continue
                key = word_key(name)
                claims.setdefault(key, {}).setdefault(field, {}).setdefault(
                    name, []
                ).append(rank)
                if isinstance(raw, dict) and raw.get("description"):
                    descriptions.setdefault(key, raw["description"])

    merged: dict[str, list[Any]] = {field: [] for field in _CATEGORY_ORDER}
    for key, by_field in claims.items():
        field = _chosen_field(by_field)
        name = _agreed_spelling(by_field[field])
        if field == "themes":
            description = descriptions.get(key)
            merged[field].append(
                {"name": name, "description": description}
                if description
                else {"name": name}
            )
        else:
            merged[field].append(name)
    return merged


def _chosen_field(by_field: dict[str, dict[str, list[int]]]) -> str:
    """Decide which field a word belongs in.

    Args:
        by_field: Each field the word was filed under, holding the spellings
            used and the priority rank of every provider that used them.

    Returns:
        The field the most trusted provider that classified the word used.
        ``tags`` only when no provider classified it.
    """
    # A content warning outranks everyone, including MAL. Another provider
    # calling the word a theme is not a denial that it is adult content, and
    # the two mistakes are not equally bad - the same reasoning as `nsfw`.
    if "content_warnings" in by_field:
        return "content_warnings"

    classified = {f: v for f, v in by_field.items() if f != "tags"}
    if not classified:
        return "tags"
    best = min(
        min(ranks) for spellings in classified.values() for ranks in spellings.values()
    )
    contenders = {
        field
        for field, spellings in classified.items()
        if any(min(ranks) == best for ranks in spellings.values())
    }
    return next(f for f in _CATEGORY_ORDER if f in contenders)


def _agreed_spelling(by_spelling: dict[str, list[int]]) -> str:
    """Choose how a word is written, from how the providers wrote it.

    Most providers win. AniDB publishes everything in lowercase, so letting the
    most trusted provider decide alone would store ``action`` wherever AniDB
    happened to be the only one filing it in the winning field. Counting
    instead lets six providers outvote it.

    An even split prefers a spelling that is not entirely lowercase, because
    AniDB's lowercase is a house style rather than how the word is written -
    it alone turns ``Ocean`` into ``ocean``. A word every provider writes in
    lowercase is unaffected, and the most trusted provider settles what is
    left.

    Args:
        by_spelling: Each spelling seen, with the priority rank of every
            provider that used it.

    Returns:
        The spelling the providers most agree on.
    """
    return max(
        by_spelling,
        key=lambda s: (len(by_spelling[s]), not s.islower(), -min(by_spelling[s])),
    )


def merge_synonyms(ranked: Ranked, titles: Iterable[str | None]) -> list[str]:
    """Union alternative titles, then remove the ones already stated elsewhere.

    Two kinds of duplicate survive a plain union. The same title reappears in
    different scripts and punctuation - ``All'arrembaggio!`` arrives with three
    different apostrophes (U+0027, U+0060, U+2019) - which NFKC and apostrophe
    folding collapse without a model. And the work's own title bleeds in:
    AniDB lists ``One Piece`` as a synonym of One Piece.

    This runs after the titles are resolved, since it needs to know what they
    are. Nothing semantic happens here: measured against an embedding model,
    the deterministic ladder removed 88 duplicates with 0 mistakes where the
    semantic pass removed 125 with 14 wrong.

    Args:
        ranked: Provider records in priority order.
        titles: The already-resolved title fields to filter against.

    Returns:
        Deduplicated synonyms in priority order.
    """
    claimed = {fold_for_comparison(title) for title in titles if title}
    merged: dict[str, str] = {}
    for _, values in providers_supplying(ranked, "synonyms"):
        for value in values:
            key = fold_for_comparison(value)
            if key and key not in claimed:
                merged.setdefault(key, value)
    return list(merged.values())


def merge_object(ranked: Ranked, field: str) -> dict[str, Any] | None:
    """Merge an object field one sub-field at a time, never whole-object.

    Providers partition these rather than duplicating them: MAL and AniSearch
    carry ``day``/``time``/``timezone``, AniList and Kitsu ``next_episode_at``,
    AnimeSchedule the ``jp_time``/``sub_time``/``dub_time`` and premiere dates.
    Taking the highest-ranked object whole would keep MAL's three keys and
    discard the other six.

    Args:
        ranked: Provider records in priority order.
        field: ``aired_dates`` or ``broadcast``.

    Returns:
        The union of every provider's sub-fields, or ``None`` when none.
    """
    merged: dict[str, Any] = {}
    for _, value in providers_supplying(ranked, field):
        for key, sub_value in value.items():
            if provider_supplied(sub_value):
                merged.setdefault(key, sub_value)
    return merged or None


def merge_month(ranked: Ranked, aired_dates: dict[str, Any] | None) -> str | None:
    """Resolve the premiere month, falling back to the premiere date.

    AnimeSchedule is the only provider that names the month outright, but every
    provider that reports ``aired_from`` states the same fact in a different
    shape. Treating the field as AnimeSchedule's alone would leave it empty
    whenever that one fetch fails, so the date is used as the second source.

    The date is read in Japan time. A premiere on the 1st is stored as 15:00
    UTC on the last day of the month before, so reading it in UTC names the
    wrong month.

    Args:
        ranked: Provider records in priority order.
        aired_dates: The already-merged ``aired_dates``, used as the fallback.

    Returns:
        The English month name, or ``None`` when nothing states one.
    """
    stated = providers_supplying(ranked, "month")
    if stated:
        return stated[0][1]
    aired_from = (aired_dates or {}).get("aired_from")
    if not aired_from:
        return None
    try:
        premiere = datetime.fromisoformat(str(aired_from))
    except ValueError:
        logger.warning(f"Cannot read a month from aired_from: {aired_from!r}")
        return None
    return to_japan_time(premiere).strftime("%B")


def merge_score(
    statistics: dict[str, dict[str, Any]],
    *,
    baseline_score: float,
    baseline_votes: int,
) -> dict[str, float] | None:
    """Reduce the providers' scores to the three figures we publish.

    Computed from the scores this run actually collected, not taken from the
    offline seed. The seed's aggregate is frozen - its upstream was archived -
    and it cannot reflect a provider being added, removed, or corrected. On
    One Piece the seed says 8.74 where the seven live scores average 8.60,
    most of the gap being the AniSearch five-star scale the seed never fixed.

    Every score is already on 0-10 by the time it reaches here; the mappers
    convert, since only a mapper knows its own provider's scale.

    ``weighted`` treats every anime as carrying ``baseline_votes`` votes of
    ``baseline_score`` on top of its real ones, so a title a handful of people
    rated sits near the ordinary score while a title hundreds of thousands
    rated keeps its own. Providers count once each toward the mean; the vote
    totals only decide how far that mean is trusted. It is left out when no
    provider reports a count, because the formula would otherwise return the
    baseline and claim an unrated anime had been measured and found average.
    See docs/merge_rules.md for the evidence behind both constants.

    Args:
        statistics: The merged per-provider statistics.
        baseline_score: Score the baseline votes carry.
        baseline_votes: How many baseline votes each anime carries.

    Returns:
        ``mean``, ``median`` and, when any provider reported a vote count,
        ``weighted``. ``None`` when no provider scored the work.
    """
    scores = [
        float(block["score"])
        for block in statistics.values()
        if isinstance(block.get("score"), (int, float))
    ]
    if not scores:
        return None

    average = mean(scores)
    merged = {"mean": round(average, 2), "median": round(median(scores), 2)}

    votes = sum(
        int(block["scored_by"])
        for block in statistics.values()
        if isinstance(block.get("scored_by"), int)
    )
    if votes:
        weighted = (votes * average + baseline_votes * baseline_score) / (
            votes + baseline_votes
        )
        merged["weighted"] = round(weighted, 2)
    return merged


def merge_statistics(ranked: Ranked) -> dict[str, dict[str, Any]]:
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


# Providers that carry more than one company field, so filing a company as a
# studio is a choice rather than the only option available. Anime-Planet,
# AniSearch and AnimeSchedule have a single studio field each: measured against
# these four on the same anime, their studio label is right 82-97% of the time,
# and when it is wrong it is because a distributor or a broadcaster had nowhere
# else to go. See docs/merge_rules.md.
_CHOOSES_COMPANY_ROLE = frozenset({"mal", "anilist", "kitsu", "anidb"})


def merge_companies(ranked: Ranked) -> list[dict[str, Any]]:
    """Reduce every provider's companies to one entry per company.

    Three questions have to be answered for each company, and they are settled
    separately because the evidence differs.

    **Is it the same company.** ``company_keys`` folds case, punctuation, legal
    suffixes and paired plurals, so ``Toei Animation Co., Ltd.`` and
    ``Toei Animation`` are one. Over 684 anime that collapses 21% of all names.

    **Which spelling to store.** The one most providers used. AniSearch alone
    writes the registered form and AniDB writes lowercase, so letting the most
    trusted provider decide would store one site's house style. Priority breaks
    a tie, which is needed more often than it sounds - a third of groups with
    two spellings have one provider each.

    **Which roles it holds.** Only the providers that have somewhere else to put
    a producer get a say. A provider with one studio field is not claiming the
    company animates, it is reporting an association, and treating that as
    evidence marks distributors like GKIDS as studios. Their label is used only
    when no other provider covers the company at all.

    Sources are unioned, so a well-covered company carries a link from every
    provider that named it - stronger evidence of identity than the name.

    Args:
        ranked: Provider records in priority order.

    Returns:
        One entry per company, each with its roles and pooled sources.
    """
    names: list[str] = []
    seen: list[tuple[str, int, dict[str, Any]]] = []
    for rank, (provider, record) in enumerate(ranked):
        for company in record.get("companies") or []:
            name = company.get("name")
            if name:
                names.append(name)
                seen.append((provider, rank, company))
    if not seen:
        return []

    keys = company_keys(names)
    grouped: dict[str, list[tuple[str, int, dict[str, Any]]]] = defaultdict(list)
    for provider, rank, company in seen:
        grouped[keys[company["name"]]].append((provider, rank, company))

    merged: list[dict[str, Any]] = []
    for members in grouped.values():
        spellings: dict[str, list[int]] = defaultdict(list)
        for _, rank, company in members:
            spellings[company["name"]].append(rank)

        chosen: dict[str, Any] = {
            "name": max(
                spellings,
                key=lambda name: (len(spellings[name]), -min(spellings[name])),
            ),
            "roles": _company_roles(members),
        }

        sources: dict[str, str] = {}
        for _, _, company in members:
            for url in company.get("sources") or []:
                sources.setdefault(normalize_link_url(url), url)
        if sources:
            chosen["sources"] = list(sources.values())
        merged.append(chosen)

    return merged


def _company_roles(members: list[tuple[str, int, dict[str, Any]]]) -> list[str]:
    """Decide a company's roles from the providers that had a choice.

    Args:
        members: Every ``(provider, rank, company)`` naming one company.

    Returns:
        The roles, ordered by the priority of the provider that first gave each
        one, so output does not depend on dict iteration order.
    """
    deciding = [m for m in members if m[0] in _CHOOSES_COMPANY_ROLE]
    roles: dict[str, int] = {}
    for _, rank, company in sorted(deciding or members, key=lambda m: m[1]):
        for role in company.get("roles") or []:
            roles.setdefault(role, rank)
    return list(roles)
