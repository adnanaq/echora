"""Decide where each page of an entry with doubled providers goes.

A doubled provider's decision says what that provider's pages are. Correcting
the entry needs more: a place for every page. A place is written in the same
words a review answer uses, so places from the rules and places from a person's
answer are applied the same way:

Each item reads ``<place>: <meaning>``.

- ``work 1``: the page belongs to the case's first work. The page stays in the
  entry that survives as that work (``find_surviving_entries``).
- ``other work``: the page belongs to a work outside the entry. The page joins
  that work's seed entry when the move is sure
  (``UnrelatedPage.move_target_entry``), and otherwise starts a new entry.
- ``other work <page key>``: the page belongs to the seed entry holding that
  page, such as ``other work mal:anime:62841``. A reviewer names the entry
  when no link ties the page to its work, as for pages of unfetched providers.
- ``work 2``, ``work 3`` and so on: the entry holds several works, and the page
  belongs to its second, third or later work. One review case covers the whole
  entry, so a number means the same work on every page.
- ``, main`` after ``work N``: the work keeps two or more whole pages of one
  provider, and this one is the page enrichment fetches. Enrichment takes the
  first address of each provider, so the cleaned seed lists the main page first.
- ``, part`` after ``work N``: the page covers only part of the work, such as
  one of two OVAs, so enrichment combines it with the work's other part pages.

The rules place an entry without a person only when no decision needs review
(``DoubledProviderDecision.is_review_needed``). Every other entry goes to a person.

A person's answer is valid when it follows these rules:

Each item gives the rule, then the reason for it.

- An answer places every doubled page. Each doubled page needs a place.
- An answer using ``work 2`` or later places every page of the entry. Splitting
  the entry sends each page to one of the works.
- In each work, a provider's two or more pages are either all ``, part``, or
  exactly one is ``, main``. Enrichment needs either one page to fetch or every
  part to combine.
- A provider's single page in a work is not ``, main``. There is nothing to
  choose between.
- No work holds two MAL pages. A work is what MAL lists, so two MAL pages are
  two works.
- A page named after ``other work`` belongs to a seed entry outside the case.
  A page of the case's own entries is placed with ``work N`` instead.
- A seed entry holds the page named after ``other work``. The caller looks
  the names up in the seed (``named_entry_reference_pages``), so an unknown
  page is refused rather than moved nowhere.

Suggested answers are made by ``placement_suggestions``.
"""

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

from pydantic import JsonValue
from work_matching.answer_rules import PlacementRule
from work_matching.doubled_provider_pages import resolve_decided_provider
from work_matching.model_doubled_provider_decision import (
    DoubledProviderDecision,
    DoubledProviderOutcome,
)
from work_matching.provider_page_key import (
    build_page_address,
    is_mal_page,
    read_page_provider,
    resolve_page_key,
)
from work_matching.work_entry import WorkEntry

# These words are written in review answers and in the corrections file, so
# changing them would make kept answers invalid.
FIRST_WORK = "work 1"
OTHER_WORK = "other work"
# The marks a page's place can carry after ``work N``; no mark is written as
# nothing. An answer writes a mark after a comma, as in ``work 2, part``.
NO_MARK = ""
MAIN_MARK = "main"
PART_MARK = "part"
_MAIN_PAGE_MARK = f", {MAIN_MARK}"
PART_PAGE_MARK = f", {PART_MARK}"
# A place as an answer writes it: ``work N`` (N from 1), optionally marked
# ``, main`` or ``, part``, or ``other work``.
_PLACEMENT_ANSWER_TEXT = re.compile(
    rf"(?P<work>work [1-9][0-9]*)"
    rf"(?P<mark>{_MAIN_PAGE_MARK}|{PART_PAGE_MARK})?"
    rf"|{OTHER_WORK}(?: (?P<named_entry_page_key>\S+))?"
)


@dataclass(frozen=True, slots=True)
class PagePlacement:
    """Hold where one page goes: its work, and whether it is a main or part page."""

    # ``work 1``, ``work 2`` and so on, or ``other work``.
    placement_word: str
    is_main_page: bool = False
    is_part_page: bool = False
    # For ``other work <page key>``: the page key naming the seed entry the
    # page moves to.
    named_entry_page_key: str | None = None

    @property
    def is_later_work(self) -> bool:
        """Report whether the place is ``work 2`` or later, a second or later work."""
        return self.placement_word not in (FIRST_WORK, OTHER_WORK)


class PlacedPage(NamedTuple):
    """Hold a page's placement and the page key it was placed under."""

    placed_page_key: str
    placement: PagePlacement


@dataclass(frozen=True, slots=True)
class PlacementCheck:
    """Hold an answer's readable places and every placement rule it breaks."""

    # Each page whose place could be read, by the key the answer names it by.
    placements: dict[str, PagePlacement]
    # In the order the rules are checked; empty when the answer is valid.
    broken_rules: tuple[PlacementRule, ...]

    @property
    def is_valid(self) -> bool:
        """Report whether the answer breaks no placement rule."""
        return not self.broken_rules


def find_page_placement(
    page_key: str,
    placements: Mapping[str, PagePlacement],
    current_page_key_by_page_key: Mapping[str, str],
) -> PlacedPage | None:
    """Find a page's placement under its current key, else under the key listed.

    Decisions name a doubled page by its current key, and answers name any
    other page by the key the entry lists, so both are tried, the current key
    first.

    Args:
        page_key: The key an entry lists the page under.
        placements: Each placed page's placement, by the key it was placed under.
        current_page_key_by_page_key: Each renamed page's current key, by the
            key an entry lists it under.

    Returns:
        The placement and the key it was found under, or None when the page
        has no placement.
    """
    for placed_page_key in (
        current_page_key_by_page_key.get(page_key, page_key),
        page_key,
    ):
        if placed_page_key in placements:
            return PlacedPage(placed_page_key, placements[placed_page_key])
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
    if any(decision.is_review_needed for decision in decisions):
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

    Args:
        entry: The work entry whose pages may be placed.
        decisions: The entry's doubled-provider decisions.

    Returns:
        The placeable page keys in order.
    """
    decided_providers = {decision.provider for decision in decisions}
    return (
        *(page_key for decision in decisions for page_key in decision.page_keys),
        *(
            page_key
            for page_key in entry.addresses_by_page_key
            if resolve_decided_provider(page_key) not in decided_providers
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
        Each placed page's place, or None when the answer breaks a rule
        ``check_answer_placements`` checks.
    """
    placement_check = check_answer_placements(
        answer, entry, decisions, is_every_page_required=is_every_page_required
    )
    return placement_check.placements if placement_check.is_valid else None


def check_answer_placements(
    answer: JsonValue,
    entry: WorkEntry,
    decisions: Sequence[DoubledProviderDecision],
    *,
    is_every_page_required: bool = False,
    named_entry_reference_pages: Mapping[str, str] | None = None,
) -> PlacementCheck:
    """Read a review answer into places and list every placement rule it breaks.

    The rules are the ones the module docstring lists. Each is checked on its
    own, so a reviewer sees every problem of an answer at once.

    Args:
        answer: The answer as written: each page key mapped to its place.
        entry: The work entry the answer places pages of.
        decisions: The entry's doubled-provider decisions.
        is_every_page_required: Whether the answer must place every page, as
            for a group of entries, even without ``work 2`` or later.
        named_entry_reference_pages: Each page key named after ``other
            work``, mapped to the reference page of the seed entry holding
            it. A named key missing here breaks ``UNKNOWN_NAMED_ENTRY``.

    Returns:
        The places that could be read, and the broken rules in check order.
    """
    if not isinstance(answer, dict):
        return PlacementCheck({}, (PlacementRule.INVALID_PLACEMENT_MAP,))
    placements = {
        page_key: placement
        for page_key, placement_text in answer.items()
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
        else {page_key for decision in decisions for page_key in decision.page_keys}
    )
    named_entry_page_keys = {
        placement.named_entry_page_key
        for placement in placements.values()
        if placement.named_entry_page_key
    }
    named_page_keys_in_case = named_entry_page_keys & entry.addresses_by_page_key.keys()
    broken_rules = [
        rule
        for rule, is_broken in (
            (PlacementRule.INVALID_PAGE_PLACEMENT, len(placements) != len(answer)),
            (
                PlacementRule.PAGE_NOT_IN_CASE,
                bool(placements.keys() - placeable_page_keys),
            ),
            (
                PlacementRule.MISSING_REQUIRED_PLACEMENT,
                bool(required_page_keys - placements.keys()),
            ),
            (
                PlacementRule.WORK_WITH_TWO_MAL_PAGES,
                _has_work_with_two_mal_pages(placements),
            ),
            (PlacementRule.NAMED_ENTRY_IN_CASE, bool(named_page_keys_in_case)),
            (
                PlacementRule.UNKNOWN_NAMED_ENTRY,
                bool(
                    named_entry_page_keys
                    - named_page_keys_in_case
                    - (named_entry_reference_pages or {}).keys()
                ),
            ),
        )
        if is_broken
    ]
    broken_rules.extend(_list_broken_mark_rules(placements))
    return PlacementCheck(placements, tuple(broken_rules))


def read_placement(placement_text: str) -> PagePlacement | None:
    """Read one page's place as an answer writes it, or None for other text.

    ``other work`` may name a seed entry by one of its page keys. A named key
    must be a provider page key that ``resolve_page_key`` gives back from its own
    address, so a key with a missing or malformed id makes the text
    unreadable.

    Args:
        placement_text: The place as written, such as ``work 2, part`` or
            ``other work mal:anime:62841``.

    Returns:
        The place, or None when the text is not a place.
    """
    placement_match = _PLACEMENT_ANSWER_TEXT.fullmatch(placement_text)
    if not placement_match:
        return None
    named_entry_page_key = placement_match["named_entry_page_key"]
    if named_entry_page_key and not _is_page_key(named_entry_page_key):
        return None
    return PagePlacement(
        placement_match["work"] or OTHER_WORK,
        is_main_page=placement_match["mark"] == _MAIN_PAGE_MARK,
        is_part_page=placement_match["mark"] == PART_PAGE_MARK,
        named_entry_page_key=named_entry_page_key,
    )


def write_placement(placement: PagePlacement) -> str:
    """Write one page's place as an answer writes it; ``read_placement`` reads it back.

    Args:
        placement: The place to write.

    Returns:
        The place's text, such as ``work 2, part`` or ``other work mal:anime:9``.
    """
    if placement.named_entry_page_key:
        return f"{OTHER_WORK} {placement.named_entry_page_key}"
    mark = _MAIN_PAGE_MARK if placement.is_main_page else ""
    mark = PART_PAGE_MARK if placement.is_part_page else mark
    return placement.placement_word + mark


def has_two_mal_pages(page_keys: Iterable[str]) -> bool:
    """Report whether two or more of these page keys are MAL pages.

    Args:
        page_keys: The page keys to check.

    Returns:
        True when at least two page keys belong to MAL.
    """
    return sum(is_mal_page(page_key) for page_key in page_keys) > 1


def build_page_keys_by_work_provider(
    placements: Mapping[str, PagePlacement],
) -> dict[tuple[str, str | None], list[str]]:
    """Build the page keys of each provider in each work, ``other work`` aside.

    The mark rules and the MAL rule apply to one provider's pages in one work,
    and ``other work`` is not one work, so its pages are left out.

    Args:
        placements: Each placed page's place, by its page key.

    Returns:
        The page keys, in the given order, by their work's placement word and
        their provider.
    """
    page_keys_by_work_provider: defaultdict[tuple[str, str | None], list[str]] = (
        defaultdict(list)
    )
    for page_key, placement in placements.items():
        if placement.placement_word != OTHER_WORK:
            page_keys_by_work_provider[
                placement.placement_word, read_page_provider(page_key)
            ].append(page_key)
    return dict(page_keys_by_work_provider)


def _is_page_key(text: str) -> bool:
    """Report whether text is a provider page key, by its round trip through an address.

    Args:
        text: The text an answer names after ``other work``.

    Returns:
        True when the text names a provider page and reads back unchanged.
    """
    address = build_page_address(text)
    return address is not None and resolve_page_key(address) == text


def _list_broken_mark_rules(
    placements: dict[str, PagePlacement],
) -> list[PlacementRule]:
    """List the main and part mark rules each work's provider pages break.

    Each item reads ``<pages of the provider in the work>: <valid marks>``.

    - One page of the provider in the work: ``, part`` or no mark; never
      ``, main``.
    - Two or more pages of the provider in the work: every page ``, part``,
      or exactly one ``, main`` and no ``, part``.

    Args:
        placements: Each placed page's place, by its page key.

    Returns:
        Each broken rule once, in the order first found; empty when every
        work's providers follow the rules above.
    """
    return list(
        dict.fromkeys(
            broken_rule
            for page_keys in build_page_keys_by_work_provider(placements).values()
            if (
                broken_rule := _find_broken_mark_rule(
                    [placements[page_key] for page_key in page_keys]
                )
            )
        )
    )


def _has_work_with_two_mal_pages(placements: dict[str, PagePlacement]) -> bool:
    """Report whether any work, ``other work`` aside, holds two MAL pages.

    Args:
        placements: Each placed page's placement, by its page key.

    Returns:
        True when any numbered work holds two or more MAL pages.
    """
    return any(
        map(has_two_mal_pages, build_page_keys_by_work_provider(placements).values())
    )


def _find_broken_mark_rule(
    provider_placements: list[PagePlacement],
) -> PlacementRule | None:
    """Return the mark rule one provider's pages in one work break, if any.

    Args:
        provider_placements: The places of one provider's pages in one work.

    Returns:
        ``SINGLE_PAGE_MARKED_MAIN`` for a lone page marked main;
        ``INVALID_PROVIDER_PAGE_MARKS`` for several pages that are neither
        all parts nor one main page with no part; None otherwise.
    """
    main_page_count = sum(placement.is_main_page for placement in provider_placements)
    part_page_count = sum(placement.is_part_page for placement in provider_placements)
    if len(provider_placements) == 1:
        return PlacementRule.SINGLE_PAGE_MARKED_MAIN if main_page_count else None
    if part_page_count == len(provider_placements) or (
        main_page_count == 1 and part_page_count == 0
    ):
        return None
    return PlacementRule.INVALID_PROVIDER_PAGE_MARKS
