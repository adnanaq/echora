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
- ``, partial in`` after ``work N``, followed by works such as ``work 2,
  other work mal:anime:9``: the page covers several works. The page is a
  source of its own work and a partial source of each listed work, which is a
  work of the case or the seed entry holding the named page. Enrichment
  fetches only sources, so a page covering several works never stands for
  the works it is only a partial source of.
- ``dropped``: the page is a duplicate page, and the provider's other page in
  the work stays. The page is left out of every work entry.

The rules place an entry without a person only when no decision needs review
(``DoubledProviderDecision.is_review_needed``). Every other entry goes to a
person, whose answer is checked by ``answer_placement_check``.

Suggested answers are made by ``placement_suggestions``.
"""

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

from work_matching.answer_words import DROPPED, FIRST_WORK, OTHER_WORK
from work_matching.doubled_provider_pages import resolve_decided_provider
from work_matching.models import (
    DoubledProviderDecision,
    DoubledProviderOutcome,
)
from work_matching.provider_page_key import (
    build_page_address,
    read_page_provider,
    resolve_page_key,
)
from work_matching.work_entry import WorkEntry

# Written after ``work N``, before the works a page is a partial source of.
PARTIAL_SOURCE_PREFIX = ", partial in "
# A work a page can be a partial source of: a work of the case, or the seed
# entry holding a named page.
_PARTIAL_WORK_TEXT = rf"(?:work [1-9][0-9]*|{OTHER_WORK} [^,\s]+)"
# A place as an answer writes it: ``work N`` (N from 1), optionally followed
# by the works the page is a partial source of; ``other work``, optionally
# naming a page; or ``dropped``.
_PLACEMENT_ANSWER_TEXT = re.compile(
    rf"(?P<work>work [1-9][0-9]*)"
    rf"(?:{PARTIAL_SOURCE_PREFIX}"
    rf"(?P<partial_works>{_PARTIAL_WORK_TEXT}(?:, {_PARTIAL_WORK_TEXT})*))?"
    rf"|{OTHER_WORK}(?: (?P<named_entry_page_key>\S+))?"
    rf"|(?P<dropped>{DROPPED})"
)


@dataclass(frozen=True, slots=True)
class PagePlacement:
    """Hold where one page goes: its work, and the works it is a partial source of."""

    # ``work 1``, ``work 2`` and so on, ``other work`` or ``dropped``.
    placement_word: str
    # For ``other work <page key>``: the page key naming the seed entry the
    # page moves to.
    named_entry_page_key: str | None = None
    # The works the page is also a partial source of, in answer words: ``work
    # N`` for a work of the case, ``other work <page key>`` for the seed entry
    # holding that page. Empty unless the place is ``work N``.
    partial_works: tuple[str, ...] = ()

    @property
    def is_case_work(self) -> bool:
        """Report whether the place is a work of the case, ``work 1`` or later.

        Returns:
            True unless the place is ``other work`` or ``dropped``.
        """
        return self.placement_word not in (OTHER_WORK, DROPPED)

    @property
    def is_later_work(self) -> bool:
        """Report whether the place is ``work 2`` or later, a second or later work."""
        return self.is_case_work and self.placement_word != FIRST_WORK

    @property
    def partial_named_entry_page_keys(self) -> tuple[str, ...]:
        """Return the page keys naming the outside seed entries the page is a partial source of.

        Returns:
            The page key of each ``other work <page key>`` partial work, in
            answer order.
        """
        return tuple(
            partial_work.removeprefix(f"{OTHER_WORK} ")
            for partial_work in self.partial_works
            if partial_work.startswith(f"{OTHER_WORK} ")
        )


class PlacedPage(NamedTuple):
    """Hold a page's placement and the page key it was placed under."""

    placed_page_key: str
    placement: PagePlacement


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


def read_placement(placement_text: str) -> PagePlacement | None:
    """Read one page's place as an answer writes it, or None for other text.

    A page key named after ``other work``, as a place or as a partial work,
    must be a provider page key that ``resolve_page_key`` gives back from its
    own address. A key with a missing or malformed id makes the text
    unreadable.

    Args:
        placement_text: The place as written, such as ``work 2``, ``work 1,
            partial in work 2, other work mal:anime:9`` or ``dropped``.

    Returns:
        The place, or None when the text is not a place.
    """
    placement_match = _PLACEMENT_ANSWER_TEXT.fullmatch(placement_text)
    if not placement_match:
        return None
    if placement_match["dropped"]:
        return PagePlacement(DROPPED)
    named_entry_page_key = placement_match["named_entry_page_key"]
    partial_works = tuple(
        partial_work
        for partial_work in dict.fromkeys(
            (placement_match["partial_works"] or "").split(", ")
        )
        if partial_work
    )
    placement = PagePlacement(
        placement_match["work"] or OTHER_WORK,
        named_entry_page_key=named_entry_page_key,
        partial_works=partial_works,
    )
    named_page_keys = (
        named_entry_page_key,
        *placement.partial_named_entry_page_keys,
    )
    if not all(map(_is_page_key, filter(None, named_page_keys))):
        return None
    return placement


def read_placements(answer: Mapping[str, object]) -> dict[str, PagePlacement]:
    """Read the answer's places that follow the answer words; others are left out.

    Args:
        answer: The raw answer mapping page keys to placement strings.

    Returns:
        The readable placements by page key.
    """
    return {
        page_key: placement
        for page_key, placement_text in answer.items()
        if isinstance(placement_text, str)
        and (placement := read_placement(placement_text)) is not None
    }


def write_placement(placement: PagePlacement) -> str:
    """Write one page's place as an answer writes it; ``read_placement`` reads it back.

    Args:
        placement: The place to write.

    Returns:
        The place's text, such as ``work 2``, ``work 1, partial in work 2`` or
        ``other work mal:anime:9``.
    """
    if placement.named_entry_page_key:
        return f"{OTHER_WORK} {placement.named_entry_page_key}"
    if placement.partial_works:
        return (
            placement.placement_word
            + PARTIAL_SOURCE_PREFIX
            + ", ".join(placement.partial_works)
        )
    return placement.placement_word


def has_two_pages_of_one_provider(page_keys: Iterable[str]) -> bool:
    """Report whether two or more of these page keys belong to one provider.

    Args:
        page_keys: The page keys to check.

    Returns:
        True when at least two page keys share a known provider.
    """
    providers = [
        provider
        for page_key in page_keys
        if (provider := read_page_provider(page_key)) is not None
    ]
    return len(providers) != len(set(providers))


def build_page_keys_by_work_provider(
    placements: Mapping[str, PagePlacement],
) -> dict[tuple[str, str | None], list[str]]:
    """Build the page keys of each provider in each work of the case.

    The provider rule applies to one provider's pages in one work. Neither
    ``other work`` nor ``dropped`` is one work of the case, so their pages are
    left out.

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
        if placement.is_case_work:
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
