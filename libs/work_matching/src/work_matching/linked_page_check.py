"""Decide whether a page an entry's pages link, and no known work holds, joins that entry.

Providers link pages the seed does not list: an AniList page names a MAL page
no seed entry holds. Such a linked page is added to the entry linking it only
after checks, because most of these links are stale (the provider has since
deleted the page) and some are wrong mappings (a Kitsu page mapped to a recap
OVA instead of its own work). The rules:

- Not found: a page the provider says does not exist is never added and needs
  no person.
- Linked from several entries: the page has no single entry to join, so a
  person picks one.
- Provider already in the entry: adding the page would make a doubled
  provider, so a person decides; the page may be a part of the entry's work.
- Several linked pages of one provider: when the entry's pages link two or
  more pages of the page's provider that no known work holds, adding each would
  make a doubled provider, so a person decides.
- Year or type differ from the entry's: the link may be a wrong mapping, so a
  person decides. The check is the one moves use (``is_page_details_matching``).
- Episode counts differ, where both are known: providers divide long runs
  differently, and a page listing fewer episodes is often one season of the
  entry's whole run, so a person decides whether it is the work or a part.
- Otherwise the page is added. Episode counts and air dates never decide a
  match; they only stop an addition.

A review answer maps the page to one of these:

- ``add``: add it to the one entry linking it, which must hold no page of the
  page's provider, so an answer never makes a doubled provider either;
- ``add as part``: add it as a part of that entry's work, which may already
  hold pages of the provider, since a work can hold several part pages;
- either followed by ``to <reference page>``, naming one of the linking
  entries, which is required when several entries link the page;
- ``skip``: leave the page out.
"""

import re
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from pydantic import JsonValue
from work_matching.doubled_provider_check import is_page_details_matching
from work_matching.known_works import KnownWorks
from work_matching.provider_page import ProviderPage
from work_matching.provider_page_key import get_page_provider, rank_by_provider_trust
from work_matching.work_entry import WorkEntry

# These words are written in review answers, so changing them would make kept
# answers invalid.
SKIP_ANSWER: Final = "skip"
_PART_PAGE_ANSWER = "add as part"
_LINKED_PAGE_ANSWER_TEXT = re.compile(
    r"(?P<addition>add as part|add)(?: to (?P<entry_reference_page>\S+))?"
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
        """What a decision with this reason does with the page."""
        return _OUTCOME_BY_REASON[self]


_OUTCOME_BY_REASON: dict[LinkedPageReason, LinkedPageOutcome] = {
    LinkedPageReason.CHECKS_PASSED: LinkedPageOutcome.ADD,
    LinkedPageReason.NOT_FOUND: LinkedPageOutcome.SKIP,
    LinkedPageReason.SEVERAL_ENTRIES: LinkedPageOutcome.REVIEW,
    LinkedPageReason.PROVIDER_IN_ENTRY: LinkedPageOutcome.REVIEW,
    LinkedPageReason.SEVERAL_PAGES_OF_PROVIDER: LinkedPageOutcome.REVIEW,
    LinkedPageReason.DETAILS_DIFFER: LinkedPageOutcome.REVIEW,
    LinkedPageReason.EPISODE_COUNTS_DIFFER: LinkedPageOutcome.REVIEW,
}


@dataclass(frozen=True, slots=True)
class LinkingEntry:
    """An entry whose pages link a page no known work holds, with what the checks read."""

    entry: WorkEntry
    # As ``get_entry_episode_count`` gives it.
    episode_count: int | None
    # Every page the entry's pages link that no known work holds, as
    # ``find_unheld_linked_pages`` gives them, so two of one provider are seen.
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
        """What the decision does with the page; a person answers ``REVIEW``."""
        return self.reason.outcome


@dataclass(frozen=True, slots=True)
class LinkedPageAddition:
    """An accepted addition: the entry a linked page joins, and whether as a part."""

    entry_reference_page: str
    is_part_page: bool


class NoLinkingEntryError(ValueError):
    """A linked page was given to decide without an entry linking it.

    A linked page is found through an entry's links, so a caller passing none
    has lost track of where the page came from.
    """

    def __init__(self, page_key: str) -> None:
        """Name the page given without a linking entry."""
        super().__init__(f"No entry links the page to decide: {page_key}")


def find_unheld_linked_pages(
    entry: WorkEntry,
    pages: Iterable[ProviderPage],
    known_works: KnownWorks,
    providers: Collection[str],
) -> frozenset[str]:
    """Return the pages an entry's pages link that no known work and not the entry holds.

    Args:
        entry: The work entry.
        pages: The fetched details of the entry's pages and of pages found
            through them, whose links are read.
        known_works: The works already known.
        providers: The providers whose linked pages can be added: those whose
            pages are fetched, so each one can be checked live.

    Returns:
        The page keys of the linked pages.
    """
    return frozenset(
        link
        for page in pages
        for link in page.links
        if get_page_provider(link) in providers
        and link not in entry.pages
        and known_works.find_by_page_key(link) is None
    )


def get_entry_episode_count(
    entry: WorkEntry, pages: Iterable[ProviderPage]
) -> int | None:
    """Return the episode count of an entry's most trusted page that gives one.

    The fetched pages are newer than the seed's own count, and the most trusted
    provider (MAL first) decides how a work is divided, so its count stands
    for the entry.

    Args:
        entry: The work entry.
        pages: The fetched details of the entry's pages; other pages are ignored.

    Returns:
        The episode count, or None when no page of the entry gives one.
    """
    episode_count_by_page = {
        page.key: page.episode_count
        for page in pages
        if page.key in entry.pages and page.episode_count is not None
    }
    return next(
        (
            episode_count_by_page[key]
            for key in sorted(entry.pages, key=rank_by_provider_trust)
            if key in episode_count_by_page
        ),
        None,
    )


def decide_linked_page(
    page: ProviderPage, linking_entries: Iterable[LinkingEntry]
) -> LinkedPageDecision:
    """Decide whether a fetched linked page is added, reviewed or skipped.

    Args:
        page: The linked page's fetched details.
        linking_entries: Every entry whose pages link the page; at least one.

    Returns:
        The decision, with the linking entries sorted by reference page.

    Raises:
        NoLinkingEntryError: No linking entry was given.
    """
    sorted_linking_entries = sorted(
        linking_entries, key=lambda linking_entry: linking_entry.entry.reference_page
    )
    if not sorted_linking_entries:
        raise NoLinkingEntryError(page.key)
    entries = tuple(linking_entry.entry for linking_entry in sorted_linking_entries)
    if page.is_dead:
        return LinkedPageDecision(page.key, LinkedPageReason.NOT_FOUND, entries)
    if len(entries) > 1:
        return LinkedPageDecision(page.key, LinkedPageReason.SEVERAL_ENTRIES, entries)
    [linking_entry] = sorted_linking_entries
    return LinkedPageDecision(
        page.key,
        _choose_single_entry_reason(page, linking_entry),
        entries,
        linking_entry.episode_count,
    )


def suggest_linked_page_answer(
    decision: LinkedPageDecision, page: ProviderPage
) -> dict[str, str]:
    """Suggest ``add as part`` for a page that looks like one season of its entry's work.

    The suggestion is made when one entry links the page, their year and type
    agree, and the page lists fewer episodes than the entry: providers that
    divide a run into seasons list fewer episodes per page. Otherwise nothing
    is suggested, and a person starts from the evidence alone.

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
        return {page.key: _PART_PAGE_ANSWER}
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
    page: ProviderPage, linking_entry: LinkingEntry
) -> LinkedPageReason:
    """Return the reason for a page that exists and one entry links."""
    entry = linking_entry.entry
    provider = get_page_provider(page.key)
    if entry.has_page_of_same_provider(page.key):
        return LinkedPageReason.PROVIDER_IN_ENTRY
    if any(
        key != page.key and get_page_provider(key) == provider
        for key in linking_entry.linked_page_keys
    ):
        return LinkedPageReason.SEVERAL_PAGES_OF_PROVIDER
    if not is_page_details_matching(page, entry):
        return LinkedPageReason.DETAILS_DIFFER
    if None not in (page.episode_count, linking_entry.episode_count) and (
        page.episode_count != linking_entry.episode_count
    ):
        return LinkedPageReason.EPISODE_COUNTS_DIFFER
    return LinkedPageReason.CHECKS_PASSED


def _find_target_entry(
    decision: LinkedPageDecision, entry_reference_page: str | None
) -> WorkEntry | None:
    """Return the linking entry an answer names, or the only one when it names none."""
    entries = decision.linking_entries
    if entry_reference_page is None:
        return entries[0] if len(entries) == 1 else None
    return next(
        (entry for entry in entries if entry.reference_page == entry_reference_page),
        None,
    )
