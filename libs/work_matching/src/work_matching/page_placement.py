"""Decide where each page of an entry with doubled providers goes.

A doubled provider's decision says which of that provider's pages belong to the
entry. Correcting the entry needs more: a place for every page. A place is
written in the same words a review answer uses, so places from the rules and
places from a person's answer are applied the same way:

- ``entry``: the page stays in the entry.
- ``other work``: the page belongs to a work outside the entry. It joins that
  work's seed entry when the page's links name exactly one, and otherwise
  starts a new entry.
- ``work 2``, ``work 3`` and so on: the entry is mixed (it holds several works),
  and the page belongs to its second, third or later work. One review case
  covers the whole entry, so a number means the same work on every page.
- ``, main`` after ``entry`` or ``work N``: when a work keeps two or more pages
  of one provider, this marks the page enrichment fetches. Enrichment takes the
  first address of each provider, so the cleaned seed lists the main page first.

When the rules place an entry without a person:

- Every decision must be one of two kinds. Either the provider's addresses are
  one renamed page and the entry lists its current address, or exactly one
  page is linked with the entry and each other page moves to another seed
  entry: its links name exactly one seed entry, that entry holds no page of
  the same provider, and its year and type agree with that entry's.
- Anything else needs a person, for these reasons. A work that keeps two pages
  of one provider needs its main page named. A mixed entry needs every page
  placed, and some of its pages are linked with neither work. A page naming two
  other works has no single place to go. A page whose links name no seed entry
  most often holds a stale mapping to a page the provider has deleted, so it is
  a duplicate of the entry's own work rather than another work. A page whose
  year or type disagrees with the entry its links name may hold a wrong
  mapping. Details only stop a move; they never place a page.

When a person's answer is valid:

- It places every doubled page. An answer that uses ``work N`` splits the entry,
  so it must place every page of the entry, since each page has to go to one of
  the works. Otherwise a page the answer leaves out stays in the entry.
- In each work, a provider with two or more pages has exactly one main page,
  and no other page is marked main.

How a suggested answer is made for the reviewer:

- Pages joined by the decisions' links form groups, one group per work.
- The group holding the most trusted page is ``entry``; the others are numbered
  ``work 2``, ``work 3`` and so on, in order of trust.
- A group holding two pages that a decision found to be separate works is left
  out, because a page in it is linked with both works.
"""

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from networkx.utils import UnionFind
from pydantic import JsonValue
from work_matching.doubled_provider_check import (
    DoubledProviderDecision,
    Outcome,
    UnrelatedPage,
    get_decided_provider,
    is_page_details_matching,
)
from work_matching.provider_page import ProviderPage, index_pages_by_current_key
from work_matching.provider_page_key import get_page_provider, rank_by_provider_trust
from work_matching.work_entry import WorkEntry

# These words are written in review answers and in the corrections file, so
# changing them would make kept answers invalid.
ENTRY_WORK = "entry"
OTHER_WORK = "other work"
_MAIN_PAGE_MARK = ", main"
# A place as an answer writes it: ``entry`` or ``work N`` (N from 2), either
# optionally marked ``, main``, or ``other work``.
_PLACEMENT_ANSWER_TEXT = re.compile(
    rf"(?P<work>{ENTRY_WORK}|work (?:[2-9]|[1-9][0-9]+))(?P<main>{_MAIN_PAGE_MARK})?"
    rf"|{OTHER_WORK}"
)


@dataclass(frozen=True, slots=True)
class PagePlacement:
    """Where one page goes: its work, and whether it is that work's main page."""

    # ``entry``, ``other work`` or ``work 2``, ``work 3``…
    work: str
    is_main_page: bool = False

    @property
    def is_numbered_work(self) -> bool:
        """Report whether the place is ``work 2`` or later: a mixed entry's other work."""
        return self.work not in (ENTRY_WORK, OTHER_WORK)


def place_pages_without_review(
    entry: WorkEntry,
    decisions: Sequence[DoubledProviderDecision],
    pages: Iterable[ProviderPage],
) -> dict[str, PagePlacement] | None:
    """Place the doubled pages of an entry the rules settle without a person.

    Args:
        entry: The work entry.
        decisions: The entry's doubled-provider decisions.
        pages: The fetched pages the decisions were made from, for the details
            of a page that moves to another entry.

    Returns:
        Each doubled page's place, by its current page key, or None when a
        person has to place the entry's pages.
    """
    pages_by_current_key = index_pages_by_current_key(pages)
    if not all(
        _is_decision_settled(entry, decision, pages_by_current_key)
        for decision in decisions
    ):
        return None
    placements: dict[str, PagePlacement] = {}
    for decision in decisions:
        if decision.outcome is Outcome.SAME_PAGE:
            # The provider's addresses resolve to one current key, so a same-page
            # decision holds exactly one page.
            placements[decision.pages[0]] = PagePlacement(ENTRY_WORK)
        placements |= dict.fromkeys(decision.belonging_pages, PagePlacement(ENTRY_WORK))
        placements |= dict.fromkeys(decision.unrelated_pages, PagePlacement(OTHER_WORK))
    return placements


def list_placeable_pages(
    entry: WorkEntry, decisions: Sequence[DoubledProviderDecision]
) -> tuple[str, ...]:
    """Return the pages of an entry that an answer can place.

    The doubled pages come first, named by their current key as the decisions
    name them; then the entry's other pages, named by the key the entry lists.
    """
    decided_providers = {decision.provider for decision in decisions}
    return (
        *(key for decision in decisions for key in decision.pages),
        *(
            key
            for key in entry.pages
            if get_decided_provider(key) not in decided_providers
        ),
    )


def read_answer_placements(
    answer: JsonValue,
    entry: WorkEntry,
    decisions: Sequence[DoubledProviderDecision],
) -> dict[str, PagePlacement] | None:
    """Read a review answer into each page's place, when the answer is valid.

    Args:
        answer: The answer as written: each page key mapped to its place.
        entry: The work entry the answer places pages of.
        decisions: The entry's doubled-provider decisions.

    Returns:
        Each placed page's place, or None when the answer is not valid.
    """
    if not isinstance(answer, dict):
        return None
    placements = {
        key: placement
        for key, placement_text in answer.items()
        if isinstance(placement_text, str)
        and (placement := _read_placement(placement_text))
    }
    pages = set(list_placeable_pages(entry, decisions))
    is_mixed_entry_answer = any(
        placement.is_numbered_work for placement in placements.values()
    )
    required_pages = (
        pages
        if is_mixed_entry_answer
        else {key for decision in decisions for key in decision.pages}
    )
    is_valid = (
        len(placements) == len(answer)
        and required_pages <= placements.keys() <= pages
        and _is_main_page_named_once(placements)
    )
    return placements if is_valid else None


def suggest_page_placements(
    entry: WorkEntry, decisions: Sequence[DoubledProviderDecision]
) -> dict[str, str]:
    """Suggest a place for each page the decisions' links tie to a work.

    Args:
        entry: The work entry.
        decisions: The entry's doubled-provider decisions.

    Returns:
        Each suggested page's place in answer words, in answer page order.
        Pages linked with nothing, or linked with two works, are left out.
    """
    pages = list_placeable_pages(entry, decisions)
    page_groups = UnionFind(pages)
    for decision in decisions:
        for key, linked_keys in decision.belonging_pages.items():
            page_groups.union(key, *linked_keys)
    # Each mixed-entry decision's pages are separate works, so no group may hold
    # two of them.
    separate_work_pages = [
        set(decision.belonging_pages)
        for decision in decisions
        if decision.outcome is Outcome.MIXED_ENTRY
    ]
    page_position = {key: position for position, key in enumerate(pages)}
    groups_by_trust = sorted(
        (group for group in page_groups.to_sets() if len(group) > 1),
        key=lambda group: min(
            (rank_by_provider_trust(key), page_position.get(key, len(pages)))
            for key in group
        ),
    )
    suggested_placements: dict[str, str] = {}
    for work_number, group in enumerate(groups_by_trust, start=1):
        if not any(
            len(group & decision_pages) > 1 for decision_pages in separate_work_pages
        ):
            suggested_placements |= dict.fromkeys(
                group, ENTRY_WORK if work_number == 1 else f"work {work_number}"
            )
    suggested_placements |= {
        key: OTHER_WORK for decision in decisions for key in decision.unrelated_pages
    }
    return {
        key: suggested_placements[key] for key in pages if key in suggested_placements
    }


def find_move_target(
    key: str, unrelated_page: UnrelatedPage, page: ProviderPage | None
) -> WorkEntry | None:
    """Return the seed entry a page linking another work moves to, if the move is sure.

    The page's links must name exactly one seed entry, that entry must hold no
    page of the same provider, and the page's year and type must agree with
    that entry's. A link to a page outside the seed most often points to a page
    the provider has since deleted, kept in a stale mapping of a duplicate page,
    so it does not show another work. A page of the same provider already there
    would make the move a new doubled provider, which needs a person. A year or
    type that disagrees with the entry the link names shows the mapping itself
    may be wrong. Details only stop a move here; they never place a page. The
    same check picks the target for a page a review answer sends to another
    work, so an answer cannot make a doubled provider either.

    Args:
        key: The page's current key.
        unrelated_page: What the page's links say about other works.
        page: The page's fetched details, if any.

    Returns:
        The seed entry to move the page to, or None when no move is sure.
    """
    if len(unrelated_page.other_works) != 1:
        return None
    [other_work] = unrelated_page.other_works
    if other_work.has_page_of_same_provider(key):
        return None
    # A page links another work only through its own fetched details, so it
    # has them; without them nothing confirms the move.
    is_confirmed = page is not None and is_page_details_matching(page, other_work)
    return other_work if is_confirmed else None


def _is_decision_settled(
    entry: WorkEntry,
    decision: DoubledProviderDecision,
    pages_by_current_key: Mapping[str, ProviderPage],
) -> bool:
    """Report whether the rules can place one decision's pages without a person."""
    if decision.outcome is Outcome.SAME_PAGE:
        return decision.pages[0] in entry.pages
    return (
        decision.outcome is Outcome.ONE_WORK
        and len(decision.belonging_pages) == 1
        and all(
            find_move_target(key, unrelated_page, pages_by_current_key.get(key))
            is not None
            for key, unrelated_page in decision.unrelated_pages.items()
        )
    )


def _read_placement(placement_text: str) -> PagePlacement | None:
    """Read one page's place as an answer writes it; None when the text is not a place."""
    placement_match = _PLACEMENT_ANSWER_TEXT.fullmatch(placement_text)
    if not placement_match:
        return None
    return PagePlacement(
        placement_match["work"] or OTHER_WORK,
        is_main_page=bool(placement_match["main"]),
    )


def _is_main_page_named_once(placements: dict[str, PagePlacement]) -> bool:
    """Report whether each work's provider with several pages has one main page.

    A provider with a single page in a work has no main page, since there is
    nothing to choose.
    """
    placements_by_work_provider: defaultdict[
        tuple[str, str | None], list[PagePlacement]
    ] = defaultdict(list)
    for key, placement in placements.items():
        if placement.work != OTHER_WORK:
            placements_by_work_provider[placement.work, get_page_provider(key)].append(
                placement
            )
    return all(
        sum(placement.is_main_page for placement in provider_placements)
        == (len(provider_placements) > 1)
        for provider_placements in placements_by_work_provider.values()
    )
