"""Decide whether a linked page joins the entry linking it.

A linked page is a page an entry's pages link that no known work holds, such as
a MAL page named by an AniList page the seed lists alone. Most such links are
stale, because the provider has deleted the page. Some are wrong mappings,
such as a Kitsu page mapped to a recap OVA instead of its own work. A linked
page therefore joins an entry only after these checks, run in this order:

Each item reads ``<check>: <outcome>. <reason for the check>``.

- The provider says the page does not exist: not added. The link is stale.
- A seed entry, or a linking entry, holds the page's current key: not added. The
  link spells a held page another way, such as a Kitsu slug for a page the seed
  lists by number.
- Two or more entries link the page: review. The page has no single entry to
  join.
- The entry holds a page of the page's provider: review. Adding the page would
  make a doubled provider; the page may be a part page.
- The entry links two or more such pages of one provider: review. Adding each
  page would make a doubled provider.
- The year or type differs from the entry's: review. The link may be a wrong
  mapping; moves use the same check (``is_page_details_matching``).
- Both episode counts are known and differ: review. A page with fewer episodes
  is often one season of the entry's whole run.
- None of the above: added.

Other rules:

- Only anime pages are linked pages: a manga link names source material.
- Episode counts and air dates never decide a match; they only stop an addition.

A review answer maps the page to one of these:

Each item reads ``<answer>: <effect>``.

- ``add``: add the page to the one linking entry, which must hold no page of the
  page's provider.
- ``add as part``: add the page as a part page of that entry's work; the entry
  may already hold pages of the provider.
- ``add to <reference page>``, ``add as part to <reference page>``: do the same
  for the named linking entry; required when several entries link the page.
- ``skip``: leave the page out.
"""

import re
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from pydantic import JsonValue
from work_matching.known_work_entries import KnownWorkEntries
from work_matching.page_details_comparison import is_page_details_matching
from work_matching.provider_page import ProviderPage
from work_matching.provider_page_key import (
    get_page_kind,
    get_page_provider,
    rank_by_provider_trust,
)
from work_matching.work_entry import WorkEntry

# These words are written in review answers, so changing them would make kept
# answers invalid.
SKIP_ANSWER: Final = "skip"
_WHOLE_PAGE_ANSWER: Final = "add"
_PART_PAGE_ANSWER: Final = "add as part"
# An answer adding the page, optionally naming the linking entry it joins. The
# longer word comes first, so "add as part" is never read as "add".
_LINKED_PAGE_ANSWER_TEXT: Final = re.compile(
    rf"(?P<addition>{re.escape(_PART_PAGE_ANSWER)}|{re.escape(_WHOLE_PAGE_ANSWER)})"
    r"(?: to (?P<entry_reference_page>\S+))?"
)


class LinkedPageOutcome(StrEnum):
    """What happens to a linked page."""

    ADD = "add"
    REVIEW = "review"
    SKIP = "skip"


class LinkedPageReason(StrEnum):
    """The rule deciding what happens to a linked page."""

    CHECKS_PASSED = (
        "the page exists, one entry links it, the entry holds no page of its "
        "provider, and their year, type and episode count agree"
    )
    NOT_FOUND = "the provider says the page does not exist"
    HELD_UNDER_OTHER_ADDRESS = (
        "the page is one a seed entry already holds, under another address"
    )
    SEVERAL_ENTRIES = "two or more entries link the page"
    PROVIDER_IN_ENTRY = "the entry already holds a page of the page's provider"
    SEVERAL_PAGES_OF_PROVIDER = (
        "the entry's pages link two or more pages of the page's provider that no "
        "known work holds"
    )
    DETAILS_DIFFER = "the page's year or type differs from the entry's"
    EPISODE_COUNTS_DIFFER = "the page and the entry list different episode counts"

    @property
    def outcome(self) -> LinkedPageOutcome:
        """Return what a decision with this reason does with the linked page.

        Returns:
            ``ADD``, ``REVIEW`` or ``SKIP``.
        """
        return _OUTCOME_BY_REASON[self]


_OUTCOME_BY_REASON: dict[LinkedPageReason, LinkedPageOutcome] = {
    LinkedPageReason.CHECKS_PASSED: LinkedPageOutcome.ADD,
    LinkedPageReason.NOT_FOUND: LinkedPageOutcome.SKIP,
    LinkedPageReason.HELD_UNDER_OTHER_ADDRESS: LinkedPageOutcome.SKIP,
    LinkedPageReason.SEVERAL_ENTRIES: LinkedPageOutcome.REVIEW,
    LinkedPageReason.PROVIDER_IN_ENTRY: LinkedPageOutcome.REVIEW,
    LinkedPageReason.SEVERAL_PAGES_OF_PROVIDER: LinkedPageOutcome.REVIEW,
    LinkedPageReason.DETAILS_DIFFER: LinkedPageOutcome.REVIEW,
    LinkedPageReason.EPISODE_COUNTS_DIFFER: LinkedPageOutcome.REVIEW,
}


@dataclass(frozen=True, slots=True)
class LinkingEntryDetails:
    """Hold an entry linking a page no known work holds, with the details the checks read.

    A ``LinkedPageDecision`` keeps only the entries in ``linking_entries``. Deciding
    also needs each entry's episode count and its other linked pages, so those
    details travel with the entry here.
    """

    entry: WorkEntry
    # As ``get_entry_episode_count`` gives it.
    episode_count: int | None
    # Every page the entry's pages link that no known work holds, as
    # ``find_unheld_linked_page_keys`` gives them, so two of one provider are seen.
    linked_page_keys: frozenset[str]


@dataclass(frozen=True, slots=True)
class LinkedPageDecision:
    """What happens to one linked page, and why."""

    page_key: str
    reason: LinkedPageReason
    # Every entry linking the page, sorted by reference page.
    linking_entries: tuple[WorkEntry, ...]
    # The episode count of the one linking entry, for a reviewer and the
    # suggested answer; None when several entries link the page or none is known.
    entry_episode_count: int | None = None

    @property
    def outcome(self) -> LinkedPageOutcome:
        """Return what the decision does with the linked page.

        Returns:
            ``ADD``, ``REVIEW`` or ``SKIP``; a person answers a ``REVIEW``.
        """
        return self.reason.outcome


@dataclass(frozen=True, slots=True)
class LinkedPageAddition:
    """An accepted addition: the entry a linked page joins, and whether as a part."""

    entry_reference_page: str
    is_part_page: bool


class NoLinkingEntryError(ValueError):
    """Report a linked page given to decide without an entry linking it.

    A linked page is found through an entry's links. A caller passing no entry
    has lost track of where the linked page came from.
    """

    def __init__(self, page_key: str) -> None:
        """Name the page given without a linking entry.

        Args:
            page_key: The key of the linked page.
        """
        super().__init__(f"No entry links the page to decide: {page_key}")


def find_unheld_linked_page_keys(
    entry: WorkEntry,
    pages: Iterable[ProviderPage],
    known_work_entries: KnownWorkEntries,
    providers: Collection[str],
) -> frozenset[str]:
    """Return the keys of linked pages that neither a known work nor the entry holds.

    Args:
        entry: The work entry.
        pages: The fetched details of the entry's pages and of pages searched
            by them, whose links are read.
        known_work_entries: The work entries already known.
        providers: The providers whose linked pages can be added: those whose
            pages are fetched, so each one can be checked live.

    Returns:
        The page keys of the linked pages.
    """
    return frozenset(
        linked_page_key
        for page in pages
        for linked_page_key in page.linked_page_keys
        if get_page_provider(linked_page_key) in providers
        and get_page_kind(linked_page_key) == "anime"
        and linked_page_key not in entry.addresses_by_page_key
        and known_work_entries.find_by_page_key(linked_page_key) is None
    )


def get_entry_episode_count(
    entry: WorkEntry, pages: Iterable[ProviderPage]
) -> int | None:
    """Return the episode count of an entry's most trusted page that gives one.

    The fetched pages are newer than the seed's own count. The most trusted
    provider, MAL first, decides how a work is divided, so that provider's
    count stands for the entry.

    Args:
        entry: The work entry.
        pages: The fetched details of the entry's pages; other pages are ignored.

    Returns:
        The episode count, or None when no page of the entry gives one.
    """
    episode_count_by_page_key = {
        page.key: page.episode_count
        for page in pages
        if page.key in entry.addresses_by_page_key and page.episode_count is not None
    }
    # min keeps the first of equally trusted pages, in the entry's own order.
    most_trusted_page_key = min(
        episode_count_by_page_key, key=rank_by_provider_trust, default=None
    )
    return (
        episode_count_by_page_key[most_trusted_page_key]
        if most_trusted_page_key is not None
        else None
    )


def decide_linked_page(
    page: ProviderPage,
    linking_entry_details: Iterable[LinkingEntryDetails],
    known_work_entries: KnownWorkEntries,
) -> LinkedPageDecision:
    """Decide whether a fetched linked page is added, reviewed or skipped.

    The page is judged at its current key. A link can spell a held page another
    way, such as a Kitsu slug for a page the seed lists by number. Only the
    fetched page shows its current key.

    Args:
        page: The linked page's fetched details.
        linking_entry_details: Every entry whose pages link the page; at least one.
        known_work_entries: The work entries already known, to tell a page
            already held.

    Returns:
        The decision named by the page's current key, with the linking entries
        sorted by reference page.

    Raises:
        NoLinkingEntryError: No linking entry was given.
    """
    sorted_entry_details = sorted(
        linking_entry_details,
        key=lambda entry_details: entry_details.entry.reference_page,
    )
    if not sorted_entry_details:
        raise NoLinkingEntryError(page.key)
    linking_entries = tuple(
        entry_details.entry for entry_details in sorted_entry_details
    )
    page_key = page.current_key
    if page.is_dead:
        reason = LinkedPageReason.NOT_FOUND
    elif known_work_entries.find_by_page_key(page_key) or any(
        page_key in entry.addresses_by_page_key for entry in linking_entries
    ):
        reason = LinkedPageReason.HELD_UNDER_OTHER_ADDRESS
    elif len(linking_entries) > 1:
        reason = LinkedPageReason.SEVERAL_ENTRIES
    else:
        [entry_details] = sorted_entry_details
        return LinkedPageDecision(
            page_key,
            _choose_single_entry_reason(page, entry_details),
            linking_entries,
            entry_details.episode_count,
        )
    return LinkedPageDecision(page_key, reason, linking_entries)


def suggest_linked_page_answer(
    decision: LinkedPageDecision, page: ProviderPage
) -> dict[str, str]:
    """Suggest ``add as part`` for a page that looks like one season of its entry's work.

    The suggestion needs all of these:

    - one entry links the page;
    - the page's year and type agree with the entry's;
    - the page lists fewer episodes than the entry.

    Providers that divide a run into seasons list fewer episodes per page.
    Without a suggestion, a person starts from the evidence alone.

    Args:
        decision: The decision on the linked page.
        page: The linked page's fetched details.

    Returns:
        The page key mapped to the suggested answer, or an empty mapping.
    """
    if len(decision.linking_entries) != 1:
        return {}
    [entry] = decision.linking_entries
    is_fewer_episodes = (
        page.episode_count is not None
        and decision.entry_episode_count is not None
        and page.episode_count < decision.entry_episode_count
    )
    if is_fewer_episodes and is_page_details_matching(page, entry):
        return {decision.page_key: _PART_PAGE_ANSWER}
    return {}


def read_linked_page_answer(
    answer: JsonValue, decision: LinkedPageDecision
) -> LinkedPageAddition | str | None:
    """Read a review answer about a linked page, when the answer is valid.

    Args:
        answer: The answer as written: the page key mapped to the answer text.
        decision: The decision the answer is about.

    Returns:
        The accepted addition, ``SKIP_ANSWER`` for a page left out, or None
        when the answer is not valid.
    """
    if not isinstance(answer, dict) or answer.keys() != {decision.page_key}:
        return None
    answer_text = answer[decision.page_key]
    if answer_text == SKIP_ANSWER:
        return SKIP_ANSWER
    if not isinstance(answer_text, str) or not (
        answer_match := _LINKED_PAGE_ANSWER_TEXT.fullmatch(answer_text)
    ):
        return None
    target_entry = _find_target_entry(decision, answer_match["entry_reference_page"])
    is_part_page = answer_match["addition"] == _PART_PAGE_ANSWER
    if target_entry is None or (
        not is_part_page and target_entry.has_page_of_same_provider(decision.page_key)
    ):
        return None
    return LinkedPageAddition(target_entry.reference_page, is_part_page)


def _choose_single_entry_reason(
    page: ProviderPage, entry_details: LinkingEntryDetails
) -> LinkedPageReason:
    """Return the reason for a linked page that exists and that one entry links.

    The checks run in the order the module docstring lists them, and the first
    check that applies gives the reason.

    Args:
        page: The linked page's fetched details.
        entry_details: The one entry linking the page, with its episode count
            and its other linked pages.

    Returns:
        The reason; ``CHECKS_PASSED`` when no check stops the addition.
    """
    entry = entry_details.entry
    page_key = page.current_key
    provider = get_page_provider(page_key)
    if entry.has_page_of_same_provider(page_key):
        return LinkedPageReason.PROVIDER_IN_ENTRY
    if any(
        key != page_key and get_page_provider(key) == provider
        for key in entry_details.linked_page_keys
    ):
        return LinkedPageReason.SEVERAL_PAGES_OF_PROVIDER
    if not is_page_details_matching(page, entry):
        return LinkedPageReason.DETAILS_DIFFER
    if None not in (page.episode_count, entry_details.episode_count) and (
        page.episode_count != entry_details.episode_count
    ):
        return LinkedPageReason.EPISODE_COUNTS_DIFFER
    return LinkedPageReason.CHECKS_PASSED


def _find_target_entry(
    decision: LinkedPageDecision, entry_reference_page: str | None
) -> WorkEntry | None:
    """Return the linking entry an answer names, or the only linking entry.

    Args:
        decision: The decision the answer is about.
        entry_reference_page: The reference page the answer names after
            ``to``, or None when the answer names no entry.

    Returns:
        The named linking entry; without a name, the one linking entry. None
        when the name matches no linking entry, or no name is given while
        several entries link the page.
    """
    entries = decision.linking_entries
    if entry_reference_page is None:
        return entries[0] if len(entries) == 1 else None
    return next(
        (entry for entry in entries if entry.reference_page == entry_reference_page),
        None,
    )
