"""Decide where each page of an entry with doubled providers goes.

A doubled provider's decision says what that provider's pages are. Correcting
the entry needs more: a place for every page. A place is written in the same
words a review answer uses, so places from the rules and places from a person's
answer are applied the same way:

| Place | Meaning |
| -- | -- |
| ``entry`` | The page stays in the entry. |
| ``other work`` | The page belongs to a work outside the entry. It joins that work's seed entry when the move is sure (``UnrelatedPage.move_target_entry``), and otherwise starts a new entry. |
| ``work 2``, ``work 3`` and so on | The entry holds several works, and the page belongs to its second, third or later work. One review case covers the whole entry, so a number means the same work on every page. |
| ``, main`` after ``entry`` or ``work N`` | The work keeps two or more whole pages of one provider, and this one is the page enrichment fetches. Enrichment takes the first address of each provider, so the cleaned seed lists the main page first. |
| ``, part`` after ``entry`` or ``work N`` | The page covers only part of the work, such as one of two OVAs, so enrichment combines it with the work's other part pages. |

The rules place an entry without a person only when no decision needs review
(``DoubledProviderDecision.needs_review``). Every other entry goes to a person.

A person's answer is valid when it follows these rules:

| Rule | Reason for the rule |
| -- | -- |
| It places every doubled page. | Each doubled page needs a place. |
| An answer using ``work N`` places every page of the entry. | Splitting the entry sends each page to one of the works. |
| In each work, a provider's two or more pages are either all ``, part``, or exactly one is ``, main``. | Enrichment needs either one page to fetch or every part to combine. |
| A provider's single page in a work is not ``, main``. | There is nothing to choose between. |

A suggested answer is made for the reviewer this way:

- Pages joined by the decisions' links form groups, one group per work.
- The group holding the most trusted page is ``entry``; the others are numbered
  ``work 2``, ``work 3`` and so on, in order of trust.
- A group holding two pages of one ``SEPARATE_WORKS`` decision is left out,
  because a page in it is linked with both works.
- Pages of a ``PART_PAGES`` decision are suggested with ``, part``.
- Pages of a decision whose years or types differ (``DETAILS_DIFFER``) get no
  suggestion, because they may be different works.
"""

import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from networkx.utils import UnionFind
from pydantic import JsonValue
from work_matching.doubled_provider_decision import (
    DoubledProviderDecision,
    Outcome,
    ReviewReason,
)
from work_matching.doubled_provider_pages import get_decided_provider
from work_matching.provider_page_key import get_page_provider, rank_by_provider_trust
from work_matching.work_entry import WorkEntry

# These words are written in review answers and in the corrections file, so
# changing them would make kept answers invalid.
ENTRY_WORK = "entry"
OTHER_WORK = "other work"
_MAIN_PAGE_MARK = ", main"
_PART_PAGE_MARK = ", part"
# A place as an answer writes it: ``entry`` or ``work N`` (N from 2), either
# optionally marked ``, main`` or ``, part``, or ``other work``.
_PLACEMENT_ANSWER_TEXT = re.compile(
    rf"(?P<work>{ENTRY_WORK}|work (?:[2-9]|[1-9][0-9]+))"
    rf"(?P<mark>{_MAIN_PAGE_MARK}|{_PART_PAGE_MARK})?"
    rf"|{OTHER_WORK}"
)


@dataclass(frozen=True, slots=True)
class PagePlacement:
    """Hold where one page goes: its work, and whether it is a main or part page."""

    # ``entry``, ``other work`` or ``work 2``, ``work 3``…
    placement_word: str
    is_main_page: bool = False
    is_part_page: bool = False

    @property
    def is_numbered_work(self) -> bool:
        """Report whether the place is ``work 2`` or later, a second or later work."""
        return self.placement_word not in (ENTRY_WORK, OTHER_WORK)


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
        if decision.outcome is Outcome.SAME_PAGE:
            # The provider's addresses resolve to one current key, so a same-page
            # decision holds exactly one page.
            placements[decision.page_keys[0]] = PagePlacement(ENTRY_WORK)
        placements |= dict.fromkeys(
            decision.belonging_page_keys, PagePlacement(ENTRY_WORK)
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
    placeable_page_keys = set(list_placeable_page_keys(entry, decisions))
    is_mixed_entry_answer = any(
        placement.is_numbered_work for placement in placements.values()
    )
    required_page_keys = (
        placeable_page_keys
        if is_mixed_entry_answer
        else {key for decision in decisions for key in decision.page_keys}
    )
    is_valid = (
        len(placements) == len(answer)
        and required_page_keys <= placements.keys() <= placeable_page_keys
        and _is_each_provider_marked_once(placements)
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
        Pages linked with nothing, linked with two works, or of a decision
        whose years or types differ, are left out.
    """
    placeable_page_keys = list_placeable_page_keys(entry, decisions)
    suggested_placements = _suggest_work_per_page(placeable_page_keys, decisions)
    for decision in decisions:
        if ReviewReason.DETAILS_DIFFER in decision.review_reasons:
            for key in decision.belonging_page_keys:
                suggested_placements.pop(key, None)
        elif decision.outcome is Outcome.PART_PAGES:
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
    placeable_page_keys: Sequence[str], decisions: Sequence[DoubledProviderDecision]
) -> dict[str, str]:
    """Group the pages the decisions' links join into works, and name each work.

    Args:
        placeable_page_keys: The keys of the pages an answer can place, in
            answer order.
        decisions: The entry's doubled-provider decisions.

    Returns:
        Each grouped page's work in answer words: ``entry`` for the group
        holding the most trusted page, then ``work 2``, ``work 3`` and so on.
        A group holding two pages of one ``SEPARATE_WORKS`` decision is left out.
    """
    page_groups = UnionFind(placeable_page_keys)
    for decision in decisions:
        for key, linked_keys in decision.belonging_page_keys.items():
            page_groups.union(key, *linked_keys)
    # Each separate-works decision's pages are separate works, so no group may
    # hold two of them.
    separate_work_page_keys = [
        set(decision.belonging_page_keys)
        for decision in decisions
        if decision.outcome is Outcome.SEPARATE_WORKS
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
        ):
            suggested_placements |= dict.fromkeys(
                group, ENTRY_WORK if work_number == 1 else f"work {work_number}"
            )
    return suggested_placements


def _read_placement(placement_text: str) -> PagePlacement | None:
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

    | Pages of the provider in the work | Valid marks |
    | -- | -- |
    | one | no ``, main``; ``, part`` or nothing |
    | two or more | every page ``, part``, or exactly one ``, main`` and no ``, part`` |

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
