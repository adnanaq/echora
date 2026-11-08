"""Hold a linked page's decision, the details deciding it reads, and its checked answer.

A linked page is a page an entry's pages link that no known work holds;
``linked_page_check`` lists the checks deciding what happens to it. Each model
reads ``<model>: <what it holds>``.

- ``LinkedPageReason``: the rule that decided, which also sets the outcome.
- ``LinkedPageOutcome``: whether the page is added, reviewed or skipped.
- ``LinkingEntryDetails``: an entry linking the page, with the episode count
  and other linked pages the checks read.
- ``LinkedPageDecision``: the page, its reason and every entry linking it.
- ``LinkedPageAddition``: an accepted answer adding the page to an entry.
- ``LinkedPageAnswerCheck``: an answer once read, with any rule it breaks.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated

from pydantic import BeforeValidator, PlainSerializer
from work_matching.answer_rules import LinkedPageAnswerRule
from work_matching.enum_names import read_member_by_name, write_member_name
from work_matching.work_entry import WorkEntry


class LinkedPageOutcome(StrEnum):
    """Define what happens to a linked page."""

    ADD = "add"
    REVIEW = "review"
    SKIP = "skip"


class LinkedPageReason(StrEnum):
    """Define the rule deciding what happens to a linked page."""

    CHECKS_PASSED = (
        "the page exists, one entry links it, the entry holds no page of its "
        "provider, and their year, type and episode count agree"
    )
    NOT_FOUND = "the provider says the page does not exist"
    HELD_UNDER_OTHER_ADDRESS = (
        "a seed entry already holds this page under a different address"
    )
    SEVERAL_ENTRIES = "two or more entries link the page"
    PROVIDER_IN_ENTRY = "the entry already holds a page of the page's provider"
    SEVERAL_PAGES_OF_PROVIDER = (
        "the entry links to multiple unclaimed pages from this provider"
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
    # As ``find_entry_episode_count`` gives it.
    episode_count: int | None
    # Every page the entry's pages link that no known work holds, as
    # ``find_unheld_linked_page_keys`` gives them, so two of one provider are seen.
    linked_page_keys: frozenset[str]


@dataclass(frozen=True, slots=True)
class LinkedPageDecision:
    """Hold what happens to one linked page and why."""

    page_key: str
    # Saved by member name (``enum_names``), so rewording a sentence keeps
    # saved review cases readable.
    reason: Annotated[
        LinkedPageReason,
        BeforeValidator(read_member_by_name(LinkedPageReason)),
        PlainSerializer(write_member_name, return_type=str),
    ]
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
    """Hold an accepted addition, which names the entry a linked page joins and whether as a partial source."""

    entry_reference_page: str
    # Whether the page joins the entry's partial sources rather than its sources.
    is_partial_source: bool


@dataclass(frozen=True, slots=True)
class LinkedPageAnswerCheck:
    """Hold a linked-page answer once read, with the accepted answer or the rule it breaks."""

    # The addition, ``SKIP_ANSWER``, or None when a rule is broken.
    accepted_answer: LinkedPageAddition | str | None
    # Empty when the answer is valid; otherwise the one rule broken first.
    broken_rules: tuple[LinkedPageAnswerRule, ...]

    @property
    def is_valid(self) -> bool:
        """Report whether the answer breaks no rule."""
        return not self.broken_rules
