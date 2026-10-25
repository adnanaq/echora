"""Find groups of seed entries that may be one work split over several entries.

The seed sometimes lists one work as two entries: a site adds a page before the
seed joins it to the work's main entry, so the page becomes an entry of its
own. Such entries usually share a title, so candidates are found by title, type,
year and season. Seasons and parts often carry titles of their own, so entries
tied by a link (``split_work_links``) are candidates too. Neither a title nor a
link proves one work, so every candidate group goes to review, with the titles
and links that made it as evidence.

Each item gives the rule, then the reason for it.

- Title, type and year: two entries match when the title lookup of the
  known work entries finds one for the other. Titles are equal once case,
  punctuation, accents and season wording are folded; types are the same;
  years are equal, or one is missing.
- Season: matched titles also need seasons equal, next to each other, or
  one missing. A small entry often has no season yet; seasons far apart are
  different works ("Gelpiyo", winter and fall); neighbouring ones go either
  way.
- Links: two entries tied by a link (``split_work_links``) are candidates
  whatever their titles. Seasons and parts often carry titles of their own.
- Several works: an entry holding a page that covers several works
  (``split_work_links``) forms a group with the MAL entries linking that
  page, when it holds no MAL page and no title or link groups it. Under
  MAL's division the page belongs to none of those works, so a person
  decides whether the entry stays or merges into one of them.
- Groups: entries joined through a shared title match, link or
  several-works page form one group, and the group keeps every match and
  link as its evidence.
- Providers: a group found by title alone whose entries share two or more
  providers, or share MAL, is dropped and logged. Two providers listing
  the entries apart points to different works, and entries holding
  different MAL pages are never one work. A group with a link is kept: the
  link is evidence enough to ask a person. Simkl and AnimeCountdown count
  as one provider.
"""

import logging
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

from networkx.utils import UnionFind
from pydantic import BaseModel, ConfigDict
from seed_cleaning.split_work_links import SplitWorkLinks
from work_matching import KnownWorkEntriesFromSeed, WorkEntry, get_decided_provider

logger = logging.getLogger(__name__)

# Seasons in calendar order, so neighbouring seasons differ by one.
_SEASON_ORDER = ("WINTER", "SPRING", "SUMMER", "FALL")
# Entries sharing this provider are never one work: a work is what MAL lists.
_MAL_PROVIDER = "mal"


class SplitWorkCandidates(BaseModel):
    """Seed entries that may be one work, for a person to review as one group."""

    model_config = ConfigDict(frozen=True)

    # In seed file order.
    entries: tuple[WorkEntry, ...]
    # The one provider with pages in two or more of the entries, if any. That
    # provider lists the entries as separate pages, which alone does not decide:
    # such groups are sometimes one work and sometimes not.
    shared_provider: str | None = None
    # Why the entries are candidates, shown to the reviewer: each shared title
    # (``same title: 'A' and 'B'``) and each link (``<page key> links <page
    # key>``), titles first.
    candidate_evidence: tuple[str, ...] = ()
    # Whether a page tying the entries is also linked from entries outside the
    # group, so its link is weaker evidence; the review case warns about it.
    has_shared_tying_link: bool = False


def find_split_work_candidates(
    known_work_entries: KnownWorkEntriesFromSeed, split_work_links: SplitWorkLinks
) -> list[SplitWorkCandidates]:
    """Return every group of entries that may be one work split in several.

    The rules are in the module docstring.

    Args:
        known_work_entries: The entries to group, by page and by title.
        split_work_links: The links tying the entries, by their places in
            ``known_work_entries``.

    Returns:
        The kept groups, ordered by their first entry's place.
    """
    work_entries = list(known_work_entries)
    title_pairs = _find_title_pairs(known_work_entries, work_entries)
    entry_groups = UnionFind()
    for pair in [*title_pairs, *split_work_links.links_by_entry_pair]:
        entry_groups.union(*pair)
    several_works_pages = _join_several_works_entries(
        known_work_entries, work_entries, split_work_links, entry_groups
    )
    evidence_by_group = _collect_group_evidence(
        entry_groups, work_entries, title_pairs, split_work_links, several_works_pages
    )
    candidates: list[SplitWorkCandidates] = []
    for positions in sorted(sorted(group) for group in entry_groups.to_sets()):
        if built_candidates := _build_candidates(
            work_entries,
            positions,
            evidence_by_group[entry_groups[positions[0]]],
            split_work_links.linking_entries_by_page_key,
        ):
            candidates.append(built_candidates)
    return candidates


class _SeveralWorksPage(NamedTuple):
    """A page covering several works, the entry holding it, and why it is grouped."""

    holder_position: int
    page_key: str
    evidence: str


def _join_several_works_entries(
    known_work_entries: KnownWorkEntriesFromSeed,
    work_entries: list[WorkEntry],
    split_work_links: SplitWorkLinks,
    entry_groups: UnionFind,
) -> list[_SeveralWorksPage]:
    """Join each entry holding a several-works page with the MAL entries linking it.

    Args:
        known_work_entries: The entries, to find each page's holder.
        work_entries: The entries, by place.
        split_work_links: The MAL entries linking each several-works page.
        entry_groups: The groups from titles and links, joined in place.

    Returns:
        Each several-works page joined, with its holder's place and the
        evidence shown to the reviewer. An entry holding a MAL page, or one
        already grouped by a title or link, is left alone.
    """
    seed_position_by_entry_id = {
        id(work_entry): position for position, work_entry in enumerate(work_entries)
    }
    grouped_positions = set(entry_groups.parents)
    several_works_pages: list[_SeveralWorksPage] = []
    for page_key, mal_positions in sorted(
        split_work_links.mal_entries_by_several_works_page_key.items()
    ):
        holder = known_work_entries.find_by_page_key(page_key)
        if holder is None or _MAL_PROVIDER in _list_entry_providers(holder):
            continue
        holder_position = seed_position_by_entry_id[id(holder)]
        if holder_position in grouped_positions:
            continue
        for mal_position in mal_positions:
            entry_groups.union(holder_position, mal_position)
        several_works_pages.append(
            _SeveralWorksPage(
                holder_position,
                page_key,
                f"{page_key} is linked from {len(mal_positions)} entries holding "
                "their own MAL page, so it covers several works",
            )
        )
    return several_works_pages


@dataclass
class _GroupEvidence:
    """Why one group's entries are candidates."""

    # Each shared title, then each link, as shown to the reviewer.
    evidence: list[str] = field(default_factory=list)
    # The pages the group's links point to.
    linked_page_keys: set[str] = field(default_factory=set)


def _collect_group_evidence(
    entry_groups: UnionFind,
    work_entries: list[WorkEntry],
    title_pairs: list[tuple[int, int]],
    split_work_links: SplitWorkLinks,
    several_works_pages: list[_SeveralWorksPage],
) -> defaultdict[int, _GroupEvidence]:
    """Collect every title match and link of each group, by the group's root place.

    Args:
        entry_groups: The groups, after every pair is joined.
        work_entries: The entries, by place.
        title_pairs: The pairs that match by title.
        split_work_links: The pairs tied by links, with their links.
        several_works_pages: The several-works pages joined to groups.

    Returns:
        Each group's evidence, titles first, then links, then several-works
        pages, by its root place.
    """
    evidence_by_group: defaultdict[int, _GroupEvidence] = defaultdict(_GroupEvidence)
    for first, second in title_pairs:
        evidence_by_group[entry_groups[first]].evidence.append(
            f"same title: {work_entries[first].title!r} and {work_entries[second].title!r}"
        )
    for (first, _), links in split_work_links.links_by_entry_pair.items():
        group_evidence = evidence_by_group[entry_groups[first]]
        for link in links:
            group_evidence.evidence.append(
                f"{link.linking_page_key} links {link.linked_page_key}"
            )
            group_evidence.linked_page_keys.add(link.linked_page_key)
    for holder_position, page_key, evidence in several_works_pages:
        group_evidence = evidence_by_group[entry_groups[holder_position]]
        group_evidence.evidence.append(evidence)
        group_evidence.linked_page_keys.add(page_key)
    return evidence_by_group


def _build_candidates(
    work_entries: list[WorkEntry],
    positions: list[int],
    group_evidence: _GroupEvidence,
    linking_entries_by_page_key: dict[str, frozenset[int]],
) -> SplitWorkCandidates | None:
    """Return a group as candidates, or None when the provider rule drops it.

    Args:
        work_entries: The entries, by place.
        positions: The group's places, in seed order.
        group_evidence: Why the entries are candidates.
        linking_entries_by_page_key: The places of every entry linking each
            page, to tell whether a page tying the group is linked from outside.

    Returns:
        The candidates, or None for a group found by title alone whose entries
        share two or more providers or share MAL; that drop is logged.
    """
    entries = tuple(work_entries[position] for position in positions)
    shared_providers = _find_shared_providers(entries)
    if not group_evidence.linked_page_keys and (
        len(shared_providers) > 1 or _MAL_PROVIDER in shared_providers
    ):
        logger.warning(
            f"Split-work group dropped, its entries share {', '.join(shared_providers)}: "
            f"{', '.join(entry.title for entry in entries)}"
        )
        return None
    group_positions = frozenset(positions)
    return SplitWorkCandidates(
        entries=entries,
        shared_provider=shared_providers[0] if len(shared_providers) == 1 else None,
        candidate_evidence=tuple(group_evidence.evidence),
        has_shared_tying_link=any(
            linking_entries_by_page_key.get(key, frozenset()) - group_positions
            for key in group_evidence.linked_page_keys
        ),
    )


def _find_title_pairs(
    known_work_entries: KnownWorkEntriesFromSeed, work_entries: list[WorkEntry]
) -> list[tuple[int, int]]:
    """Return the pairs of entries that match by title, type, year and season.

    Entries are told apart by object, not reference page: two entries can list
    the same page, and ``find_by_title`` returns the very entries the lookup
    holds.

    Args:
        known_work_entries: The lookup over the entries.
        work_entries: The same entries, in iteration order.

    Returns:
        Each matching pair once, by the entries' places, the lower first, in
        order.
    """
    position_by_entry_id = {
        id(work_entry): position for position, work_entry in enumerate(work_entries)
    }
    title_pairs: set[tuple[int, int]] = set()
    for position, work_entry in enumerate(work_entries):
        for matching_work_entry in known_work_entries.find_by_title(
            work_entry.title, work_entry.type, work_entry.year
        ):
            matching_position = position_by_entry_id[id(matching_work_entry)]
            if matching_position > position and _is_season_close(
                work_entry.season, matching_work_entry.season
            ):
                title_pairs.add((position, matching_position))
    return sorted(title_pairs)


def _is_season_close(first_season: str | None, second_season: str | None) -> bool:
    """Report whether two seasons are equal, next to each other, or one is missing.

    Examples:
        >>> _is_season_close("SUMMER", "FALL")
        True
        >>> _is_season_close("WINTER", "FALL")
        False
        >>> _is_season_close(None, "FALL")
        True
    """
    if first_season is None or second_season is None:
        return True
    return (
        abs(_SEASON_ORDER.index(first_season) - _SEASON_ORDER.index(second_season)) <= 1
    )


def _find_shared_providers(entries: Sequence[WorkEntry]) -> list[str]:
    """Return the providers with pages in two or more of the entries, sorted."""
    entry_count_by_provider = Counter(
        provider for entry in entries for provider in _list_entry_providers(entry)
    )
    return sorted(
        provider
        for provider, entry_count in entry_count_by_provider.items()
        if entry_count > 1
    )


def _list_entry_providers(entry: WorkEntry) -> set[str]:
    """Return the providers an entry has pages from, Simkl and AnimeCountdown as one."""
    return {
        provider
        for key in entry.addresses_by_page_key
        if (provider := get_decided_provider(key))
    }
