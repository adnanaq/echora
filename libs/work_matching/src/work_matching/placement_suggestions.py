"""Suggest a place for each page of an entry group, for the reviewer to accept or change.

A suggested answer is made this way. Each item gives the rule, then the
reason for it.

- Pages joined by the decisions' links form groups, one group per work, since
  linked pages describe the same work.
- Groups are numbered ``work 1``, ``work 2`` and so on in order of trust, the
  group holding the most trusted page first, so the most trusted page's work
  is the first work.
- A group holding two pages of one ``SEPARATE_WORKS`` decision is left out,
  because a page in it is linked with both works.
- A group holding two MAL pages is left out, because no work holds two MAL
  pages.
- Pages of a ``PART_PAGES`` decision are suggested with ``, part``.
- Pages of a decision whose years or types differ (``DETAILS_DIFFER``) get no
  suggestion, because they may be different works.
"""

from collections.abc import Iterable, Sequence

from networkx.utils import UnionFind
from work_matching.model_doubled_provider_decision import (
    DoubledProviderDecision,
    DoubledProviderOutcome,
    DoubledProviderReviewReason,
)
from work_matching.page_placement import (
    OTHER_WORK,
    PART_PAGE_MARK,
    has_two_mal_pages,
    list_placeable_page_keys,
)
from work_matching.provider_page_key import rank_by_provider_trust
from work_matching.work_entry import WorkEntry


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
            for page_key in decision.belonging_page_keys:
                suggested_placements.pop(page_key, None)
        elif decision.outcome is DoubledProviderOutcome.PART_PAGES:
            for page_key in decision.belonging_page_keys:
                if page_key in suggested_placements:
                    suggested_placements[page_key] += PART_PAGE_MARK
    suggested_placements |= {
        page_key: OTHER_WORK
        for decision in decisions
        for page_key in decision.unrelated_pages
    }
    return {
        page_key: suggested_placements[page_key]
        for page_key in placeable_page_keys
        if page_key in suggested_placements
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
    linked_page_sets = _join_linked_pages(
        placeable_page_keys, decisions, linked_page_key_pairs
    )
    # Each separate-works decision's pages are separate works, so no group may
    # hold two of them.
    separate_work_page_keys = [
        set(decision.belonging_page_keys)
        for decision in decisions
        if decision.outcome is DoubledProviderOutcome.SEPARATE_WORKS
    ]
    page_position = {
        page_key: position for position, page_key in enumerate(placeable_page_keys)
    }
    work_page_keys_by_trust = sorted(
        (
            work_page_keys
            for work_page_keys in linked_page_sets.to_sets()
            if len(work_page_keys) > 1
        ),
        key=lambda work_page_keys: min(
            (
                rank_by_provider_trust(page_key),
                page_position.get(page_key, len(placeable_page_keys)),
            )
            for page_key in work_page_keys
        ),
    )
    suggested_placements: dict[str, str] = {}
    for work_number, work_page_keys in enumerate(work_page_keys_by_trust, start=1):
        if not any(
            len(work_page_keys & decision_page_keys) > 1
            for decision_page_keys in separate_work_page_keys
        ) and not has_two_mal_pages(work_page_keys):
            suggested_placements |= dict.fromkeys(work_page_keys, f"work {work_number}")
    return suggested_placements


def _join_linked_pages(
    placeable_page_keys: Sequence[str],
    decisions: Sequence[DoubledProviderDecision],
    linked_page_key_pairs: Iterable[tuple[str, str]],
) -> UnionFind:
    """Join each belonging page with its linked pages, and each linked pair.

    Args:
        placeable_page_keys: The page keys an answer can place.
        decisions: The entry's doubled-provider decisions.
        linked_page_key_pairs: Additional pairs of linked page keys.

    Returns:
        Union-find sets of page keys joined by links.
    """
    linked_page_sets = UnionFind(placeable_page_keys)
    for decision in decisions:
        for page_key, linked_page_keys in decision.belonging_page_keys.items():
            linked_page_sets.union(page_key, *linked_page_keys)
    for page_key_pair in linked_page_key_pairs:
        linked_page_sets.union(*page_key_pair)
    return linked_page_sets
