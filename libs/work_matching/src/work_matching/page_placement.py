"""Decide where each page of an entry with doubled providers goes.

A doubled provider's decision says what that provider's pages are. Correcting
the entry needs more: a place for every page. A place is written in the same
words a review answer uses, so places from the rules and places from a person's
answer are applied the same way:

Each item reads ``<place>: <meaning>``.

- ``work 1``: the page belongs to the case's first work. The page stays in the
  entry that survives as that work (``find_surviving_entries``).
- ``other work``: the page belongs to a work outside the entry. It joins that
  work's seed entry when the move is sure (``UnrelatedPage.move_target_entry``),
  and otherwise starts a new entry.
- ``work 2``, ``work 3`` and so on: the entry holds several works, and the page
  belongs to its second, third or later work. One review case covers the whole
  entry, so a number means the same work on every page.
- ``, main`` after ``work N``: the work keeps two or more whole pages of one
  provider, and this one is the page enrichment fetches. Enrichment takes the
  first address of each provider, so the cleaned seed lists the main page first.
- ``, part`` after ``work N``: the page covers only part of the work, such as
  one of two OVAs, so enrichment combines it with the work's other part pages.

The rules place an entry without a person only when no decision needs review
(``DoubledProviderDecision.needs_review``). Every other entry goes to a person.

A person's answer is valid when it follows these rules:

Each item gives the rule, then the reason for it.

- It places every doubled page. Each doubled page needs a place.
- An answer using ``work 2`` or later places every page of the entry. Splitting
  the entry sends each page to one of the works.
- In each work, a provider's two or more pages are either all ``, part``, or
  exactly one is ``, main``. Enrichment needs either one page to fetch or every
  part to combine.
- A provider's single page in a work is not ``, main``. There is nothing to
  choose between.
- No work holds two MAL pages. A work is what MAL lists, so two MAL pages are
  two works.

A suggested answer is made for the reviewer this way:

- Pages joined by the decisions' links form groups, one group per work.
- Groups are numbered ``work 1``, ``work 2`` and so on in order of trust, the
  group holding the most trusted page first.
- A group holding two pages of one ``SEPARATE_WORKS`` decision is left out,
  because a page in it is linked with both works.
- A group holding two MAL pages is left out, because no work holds two MAL
  pages.
- Pages of a ``PART_PAGES`` decision are suggested with ``, part``.
- Pages of a decision whose years or types differ (``DETAILS_DIFFER``) get no
  suggestion, because they may be different works.
"""

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

from networkx.utils import UnionFind
from pydantic import JsonValue
from work_matching.doubled_provider_decision import (
    DoubledProviderDecision,
    DoubledProviderOutcome,
    DoubledProviderReviewReason,
)
from work_matching.doubled_provider_pages import get_decided_provider
from work_matching.provider_page_key import get_page_provider, rank_by_provider_trust
from work_matching.work_entry import WorkEntry

# These words are written in review answers and in the corrections file, so
# changing them would make kept answers invalid.
FIRST_WORK = "work 1"
OTHER_WORK = "other work"
_MAIN_PAGE_MARK = ", main"
_PART_PAGE_MARK = ", part"
# No work holds two pages of this provider: a work is what MAL lists.
_MAL_PROVIDER = "mal"
# A place as an answer writes it: ``work N`` (N from 1), optionally marked
# ``, main`` or ``, part``, or ``other work``.
_PLACEMENT_ANSWER_TEXT = re.compile(
    rf"(?P<work>work [1-9][0-9]*)"
    rf"(?P<mark>{_MAIN_PAGE_MARK}|{_PART_PAGE_MARK})?"
    rf"|{OTHER_WORK}"
)


@dataclass(frozen=True, slots=True)
class PagePlacement:
    """Hold where one page goes: its work, and whether it is a main or part page."""

    # ``work 1``, ``work 2`` and so on, or ``other work``.
    placement_word: str
    is_main_page: bool = False
    is_part_page: bool = False

    @property
    def is_later_work(self) -> bool:
        """Report whether the place is ``work 2`` or later, a second or later work."""
        return self.placement_word not in (FIRST_WORK, OTHER_WORK)


class PlacedPage(NamedTuple):
    """A page's placement, and the key it was placed under."""

    placed_key: str
    placement: PagePlacement


def find_page_placement(
    page_key: str,
    placements: Mapping[str, PagePlacement],
    current_key_by_page_key: Mapping[str, str],
) -> PlacedPage | None:
    """Find a page's placement under its current key, else under the key listed.

    Decisions name a doubled page by its current key, and answers name any
    other page by the key the entry lists, so both are tried, the current key
    first.

    Args:
        page_key: The key an entry lists the page under.
        placements: Each placed page's placement, by the key it was placed under.
        current_key_by_page_key: Each renamed page's current key, by the key an
            entry lists it under.

    Returns:
        The placement and the key it was found under, or None when the page
        has no placement.
    """
    for placed_key in (current_key_by_page_key.get(page_key, page_key), page_key):
        if placed_key in placements:
            return PlacedPage(placed_key, placements[placed_key])
    return None


def place_pages_without_review(
    decisions: Sequence[DoubledProviderDecision],
) -> dict[str, PagePlacement] | None:
    """Place the doubled pages of an entry when no decision needs review.

    Args:
        decisions: The entry's doubled-provider decisions.

    Returns:
        Each doubled page's place, by its current page key, or None when a
        person has to place the entry's pages.
    """
    if any(decision.needs_review for decision in decisions):
        return None
    placements: dict[str, PagePlacement] = {}
    for decision in decisions:
        if decision.outcome is DoubledProviderOutcome.SAME_PAGE:
            # The provider's addresses resolve to one current key, so a same-page
            # decision holds exactly one page.
            placements[decision.page_keys[0]] = PagePlacement(FIRST_WORK)
        placements |= dict.fromkeys(
            decision.belonging_page_keys, PagePlacement(FIRST_WORK)
        )
        placements |= dict.fromkeys(decision.unrelated_pages, PagePlacement(OTHER_WORK))
    return placements


def list_placeable_page_keys(
    entry: WorkEntry, decisions: Sequence[DoubledProviderDecision]
) -> tuple[str, ...]:
    """Return the keys of the entry's pages that an answer can place.

    The doubled pages come first, named by their current key as the decisions
    name them; then the entry's other pages, named by the key the entry lists.
    """
    decided_providers = {decision.provider for decision in decisions}
    return (
        *(key for decision in decisions for key in decision.page_keys),
        *(
            key
            for key in entry.addresses_by_page_key
            if get_decided_provider(key) not in decided_providers
        ),
    )


def read_answer_placements(
    answer: JsonValue,
    entry: WorkEntry,
    decisions: Sequence[DoubledProviderDecision],
    *,
    is_every_page_required: bool = False,
) -> dict[str, PagePlacement] | None:
    """Read a review answer into each page's place, when the answer is valid.

    Args:
        answer: The answer as written: each page key mapped to its place.
        entry: The work entry the answer places pages of.
        decisions: The entry's doubled-provider decisions.
        is_every_page_required: Whether the answer must place every page, as
            for a group of entries, even without ``work 2`` or later.

    Returns:
        Each placed page's place, or None when the answer is not valid.
    """
    if not isinstance(answer, dict):
        return None
    placements = {
        key: placement
        for key, placement_text in answer.items()
        if isinstance(placement_text, str)
        and (placement := read_placement(placement_text))
    }
    placeable_page_keys = set(list_placeable_page_keys(entry, decisions))
    is_mixed_entry_answer = any(
        placement.is_later_work for placement in placements.values()
    )
    required_page_keys = (
        placeable_page_keys
        if is_mixed_entry_answer or is_every_page_required
        else {key for decision in decisions for key in decision.page_keys}
    )
    is_valid = (
        len(placements) == len(answer)
        and required_page_keys <= placements.keys() <= placeable_page_keys
        and _is_each_provider_marked_once(placements)
        and not _has_work_with_two_mal_pages(placements)
    )
    return placements if is_valid else None


def suggest_page_placements(
    entry: WorkEntry,
    decisions: Sequence[DoubledProviderDecision],
    linked_page_key_pairs: Iterable[tuple[str, str]] = (),
) -> dict[str, str]:
    """Suggest a place for each page the decisions' links tie to a work.

    Args:
        entry: The work entry.
        decisions: The entry's doubled-provider decisions.
        linked_page_key_pairs: Further pairs of the entry's pages linked with
            each other, which also join pages into one work. A group of
            entries passes the links between its entries' pages.

    Returns:
        Each suggested page's place in answer words, in answer page order.
        Pages linked with nothing, linked with two works, or of a decision
        whose years or types differ, are left out.
    """
    placeable_page_keys = list_placeable_page_keys(entry, decisions)
    suggested_placements = _suggest_work_per_page(
        placeable_page_keys, decisions, linked_page_key_pairs
    )
    for decision in decisions:
        if DoubledProviderReviewReason.DETAILS_DIFFER in decision.review_reasons:
            for key in decision.belonging_page_keys:
                suggested_placements.pop(key, None)
        elif decision.outcome is DoubledProviderOutcome.PART_PAGES:
            for key in decision.belonging_page_keys:
                if key in suggested_placements:
                    suggested_placements[key] += _PART_PAGE_MARK
    suggested_placements |= {
        key: OTHER_WORK for decision in decisions for key in decision.unrelated_pages
    }
    return {
        key: suggested_placements[key]
        for key in placeable_page_keys
        if key in suggested_placements
    }


def _suggest_work_per_page(
    placeable_page_keys: Sequence[str],
    decisions: Sequence[DoubledProviderDecision],
    linked_page_key_pairs: Iterable[tuple[str, str]],
) -> dict[str, str]:
    """Group the pages the links join into works, and name each work.

    Args:
        placeable_page_keys: The keys of the pages an answer can place, in
            answer order.
        decisions: The entry's doubled-provider decisions.
        linked_page_key_pairs: Further pairs of pages linked with each other.

    Returns:
        Each grouped page's work in answer words: ``work 1`` for the group
        holding the most trusted page, then ``work 2``, ``work 3`` and so on.
        A group holding two pages of one ``SEPARATE_WORKS`` decision, or two
        MAL pages, is left out.
    """
    page_groups = _join_linked_pages(
        placeable_page_keys, decisions, linked_page_key_pairs
    )
    # Each separate-works decision's pages are separate works, so no group may
    # hold two of them.
    separate_work_page_keys = [
        set(decision.belonging_page_keys)
        for decision in decisions
        if decision.outcome is DoubledProviderOutcome.SEPARATE_WORKS
    ]
    page_position = {key: position for position, key in enumerate(placeable_page_keys)}
    groups_by_trust = sorted(
        (group for group in page_groups.to_sets() if len(group) > 1),
        key=lambda group: min(
            (
                rank_by_provider_trust(key),
                page_position.get(key, len(placeable_page_keys)),
            )
            for key in group
        ),
    )
    suggested_placements: dict[str, str] = {}
    for work_number, group in enumerate(groups_by_trust, start=1):
        if not any(
            len(group & decision_page_keys) > 1
            for decision_page_keys in separate_work_page_keys
        ) and not _has_two_mal_pages(group):
            suggested_placements |= dict.fromkeys(group, f"work {work_number}")
    return suggested_placements


def _join_linked_pages(
    placeable_page_keys: Sequence[str],
    decisions: Sequence[DoubledProviderDecision],
    linked_page_key_pairs: Iterable[tuple[str, str]],
) -> UnionFind:
    """Join each belonging page with its linked pages, and each linked pair."""
    page_groups = UnionFind(placeable_page_keys)
    for decision in decisions:
        for key, linked_keys in decision.belonging_page_keys.items():
            page_groups.union(key, *linked_keys)
    for page_key_pair in linked_page_key_pairs:
        page_groups.union(*page_key_pair)
    return page_groups


def read_placement(placement_text: str) -> PagePlacement | None:
    """Read one page's place as an answer writes it, or None for other text."""
    placement_match = _PLACEMENT_ANSWER_TEXT.fullmatch(placement_text)
    if not placement_match:
        return None
    return PagePlacement(
        placement_match["work"] or OTHER_WORK,
        is_main_page=placement_match["mark"] == _MAIN_PAGE_MARK,
        is_part_page=placement_match["mark"] == _PART_PAGE_MARK,
    )


def _is_each_provider_marked_once(placements: dict[str, PagePlacement]) -> bool:
    """Report whether each work's provider pages carry valid main and part marks.

    Each item reads ``<pages of the provider in the work>: <valid marks>``.

    - One page of the provider in the work: ``, part`` or no mark; never
      ``, main``.
    - Two or more pages of the provider in the work: every page ``, part``,
      or exactly one ``, main`` and no ``, part``.

    Args:
        placements: Each placed page's place, by its page key.

    Returns:
        True when every work's providers follow the rules above.
    """
    placements_by_work_provider: defaultdict[
        tuple[str, str | None], list[PagePlacement]
    ] = defaultdict(list)
    for key, placement in placements.items():
        if placement.placement_word != OTHER_WORK:
            placements_by_work_provider[
                placement.placement_word, get_page_provider(key)
            ].append(placement)
    return all(
        _is_provider_marked_once(provider_placements)
        for provider_placements in placements_by_work_provider.values()
    )


def _has_work_with_two_mal_pages(placements: dict[str, PagePlacement]) -> bool:
    """Report whether any work, ``other work`` aside, holds two MAL pages."""
    page_keys_by_work: defaultdict[str, set[str]] = defaultdict(set)
    for key, placement in placements.items():
        if placement.placement_word != OTHER_WORK:
            page_keys_by_work[placement.placement_word].add(key)
    return any(map(_has_two_mal_pages, page_keys_by_work.values()))


def _has_two_mal_pages(page_keys: Iterable[str]) -> bool:
    """Report whether two or more of these page keys are MAL pages."""
    return sum(get_page_provider(key) == _MAL_PROVIDER for key in page_keys) > 1


def _is_provider_marked_once(provider_placements: list[PagePlacement]) -> bool:
    """Report whether one provider's pages in one work carry valid main and part marks.

    Args:
        provider_placements: The places of one provider's pages in one work.

    Returns:
        True when one page is unmarked or a part, or when several pages are
        all parts or have exactly one main page and no part.
    """
    main_page_count = sum(placement.is_main_page for placement in provider_placements)
    part_page_count = sum(placement.is_part_page for placement in provider_placements)
    if len(provider_placements) == 1:
        return main_page_count == 0
    return part_page_count == len(provider_placements) or (
        main_page_count == 1 and part_page_count == 0
    )
