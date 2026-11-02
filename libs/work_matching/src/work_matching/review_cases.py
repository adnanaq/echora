"""Review cases: decisions the matching rules leave to a person, with their answers.

Every entry group the rules cannot place (``entry_group``) and every linked
page they cannot add on their own becomes a review case: one JSON line in the
review file. A person, or later the review agent, writes the answer into the
line, and the next run hands it back to the caller, who applies it. The rules:

- One case per entry group: all the group's pages are placed together, because
  telling which works the group holds needs all its pages at once, and because
  an answer's ``work 2`` then names the same work on every page. Every group of
  two or more entries is a case, since a shared title never proves one work.
- Case name: a case is named by its kind and its sorted page keys, never by an
  entry's position, so the same case in a later seed release or discovery run
  meets its kept answer. A single entry's case (``doubled provider``) names its
  doubled pages and its reference page. A group's case (``split work``) names
  every page of its entries, since its answer places every page.
- Changed pages: a case whose pages changed, or whose doubled pages moved to
  another entry, gets a new name, so it is asked again.
- Answers: an entry group's answer places pages with the words in
  ``page_placement`` (``work 1``, ``work 2`` and so on, ``other work``, ``, main`` and
  ``, part``); a linked page's answer uses the words in ``linked_page_check``
  (``add``, ``add as part``, either with ``to <reference page>``, and
  ``skip``). Any other answer leaves the case open.
- Linked-page case name: the linked page and the sorted reference pages of
  the entries linking it, so a page linked from a new entry is asked again.
- Suggested answer: an entry group's case carries the places its links point
  to, and a linked-page case may carry ``add as part``, so a person starts from
  them rather than from nothing.
- Kept answers: an answer whose case no longer comes up is kept as written,
  never applied, and reported.
- Rewrite: the file is rewritten whole on each run, because people edit lines in
  place; only each line's case name and answer are read back.

Real lines of the review file are shown in the README, under "Lines of the
review file".
"""

import logging
from collections.abc import Iterable, Sequence
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, JsonValue
from work_matching.answer_rules import LinkedPageAnswerRule, PlacementRule
from work_matching.doubled_provider_decision import DoubledProviderDecision
from work_matching.entry_group import (
    check_group_answer_placements,
    list_mal_page_review_reasons,
    read_group_answer_placements,
    suggest_group_page_placements,
)
from work_matching.linked_page_check import (
    LinkedPageAddition,
    LinkedPageDecision,
    check_linked_page_answer,
    read_linked_page_answer,
    suggest_linked_page_answer,
)
from work_matching.page_details_comparison import MalPageComparison
from work_matching.page_placement import PagePlacement
from work_matching.provider_page import ProviderPage, index_pages_by_current_key
from work_matching.split_work_check import SplitWorkReviewReason
from work_matching.work_entry import WorkEntry

logger = logging.getLogger(__name__)


class ReviewCaseKind(StrEnum):
    """What a review case decides, named as its case name begins."""

    DOUBLED_PROVIDER = "doubled provider"
    SPLIT_WORK = "split work"
    LINKED_PAGE = "linked page"

    @property
    def description(self) -> str:
        """Return what a case of this kind asks the reviewer to decide."""
        return _DESCRIPTION_BY_CASE_KIND[self]


_DESCRIPTION_BY_CASE_KIND: dict[ReviewCaseKind, str] = {
    ReviewCaseKind.DOUBLED_PROVIDER: (
        "An entry holds two or more pages of one provider. The answer places "
        "each page in a work."
    ),
    ReviewCaseKind.SPLIT_WORK: (
        "Two or more seed entries may be one work split up. The answer says "
        "which pages form each work."
    ),
    ReviewCaseKind.LINKED_PAGE: (
        "An entry links a page no seed entry holds. The answer says whether the "
        "page joins the entry."
    ),
}


class ReviewCase(BaseModel):
    """One decision left to a person: what the rules found, and the answer given.

    Fields are written in this order, so ``case_name``, ``answer`` and
    ``suggested_answer`` open each line of the review file and a person finds
    them first.
    """

    model_config = ConfigDict(frozen=True)

    # Its kind and its sorted page keys.
    case_name: str
    # As written in the file; ``accepted_answer`` says whether it is valid.
    answer: JsonValue = None
    # For an entry group's case, the place the links point to for each page
    # they tie to a work; for a linked-page case, ``add as part`` when the page
    # looks like one season of its entry's work.
    suggested_answer: dict[str, str] = {}
    # Names, which stay when the wording changes: for an entry group's case,
    # each decision's outcome followed by its review reasons, then the
    # split-work review reasons, then why the MAL check left pages without a
    # suggestion; for a linked page, the decision's reason.
    reasons: tuple[str, ...]
    # For a split-work case, why its entries are candidates: each shared title
    # and each link that tied them; empty for other cases.
    candidate_evidence: tuple[str, ...] = ()
    # Each entry involved, by its reference page.
    entries: dict[str, WorkEntry]
    # Every doubled-provider decision of the group's combined entry (none when
    # no provider is doubled), or the one decision on a linked page.
    decisions: tuple[DoubledProviderDecision, ...] | tuple[LinkedPageDecision]
    # The entries' fetched pages and the pages searched by them, which hold
    # the links behind the decisions.
    pages: tuple[ProviderPage, ...] = ()

    @property
    def kind(self) -> ReviewCaseKind:
        """Return what the case decides, read from its decisions and entries.

        Returns:
            ``LINKED_PAGE`` for a linked page's decision, ``SPLIT_WORK`` for a
            group of two or more entries, ``DOUBLED_PROVIDER`` otherwise.
        """
        if any(isinstance(decision, LinkedPageDecision) for decision in self.decisions):
            return ReviewCaseKind.LINKED_PAGE
        if len(self.entries) > 1:
            return ReviewCaseKind.SPLIT_WORK
        return ReviewCaseKind.DOUBLED_PROVIDER

    @property
    def accepted_answer(
        self,
    ) -> dict[str, PagePlacement] | LinkedPageAddition | str | None:
        """Return the case's answer read for the caller, when the answer is valid.

        Returns:
            The answer, read by the case's kind:

            Each item reads ``<case kind>: <what a valid answer comes back as>``.

            - Entry group: each placed page's ``PagePlacement``.
            - Linked page: the ``LinkedPageAddition``, or ``skip``.

            None when the case is open or the answer is not valid.
        """
        doubled_provider_decisions: list[DoubledProviderDecision] = []
        for decision in self.decisions:
            if isinstance(decision, LinkedPageDecision):
                return read_linked_page_answer(self.answer, decision)
            doubled_provider_decisions.append(decision)
        return read_group_answer_placements(
            self.answer,
            list(self.entries.values()),
            doubled_provider_decisions,
            {page.key: page.renamed_to for page in self.pages if page.renamed_to},
        )

    def list_broken_answer_rules(
        self, answer: JsonValue
    ) -> tuple[PlacementRule, ...] | tuple[LinkedPageAnswerRule, ...]:
        """List the rules an answer to this case breaks, with the run's own checks.

        The review page checks an answer with this before saving it, so it
        never saves an answer the run would refuse.

        Args:
            answer: The answer as it would be written in the file.

        Returns:
            Each broken rule; empty when the run would accept the answer.
        """
        doubled_provider_decisions: list[DoubledProviderDecision] = []
        for decision in self.decisions:
            if isinstance(decision, LinkedPageDecision):
                return check_linked_page_answer(answer, decision).broken_rules
            doubled_provider_decisions.append(decision)
        return check_group_answer_placements(
            answer,
            list(self.entries.values()),
            doubled_provider_decisions,
            {page.key: page.renamed_to for page in self.pages if page.renamed_to},
        ).broken_rules

    @classmethod
    def from_entry_group(
        cls,
        entries: Sequence[WorkEntry],
        decisions: Sequence[DoubledProviderDecision],
        split_work_review_reasons: Sequence[SplitWorkReviewReason],
        pages: Iterable[ProviderPage],
        *,
        candidate_evidence: Sequence[str] = (),
    ) -> Self:
        """Make the review case for an entry group whose pages the rules could not place.

        Args:
            entries: The group's entries: one entry with doubled providers, or
                two or more entries that may be one split work.
            decisions: Every decision ``decide_doubled_providers`` made for the
                combined entry (``combine_group_entries``).
            split_work_review_reasons: What speaks against the entries being
                one work; empty for a single entry.
            pages: The fetched pages; the details of the group's pages are kept.
            candidate_evidence: Why the entries are candidates, shown to the
                reviewer; empty for a single entry.

        Returns:
            The case. A single entry's case is named by its doubled pages and
            its reference page; a group's case by every page of its entries.
        """
        pages = tuple(pages)
        doubled_page_keys = [
            key for decision in decisions for key in decision.page_keys
        ]
        group_page_keys = [
            key for entry in entries for key in entry.addresses_by_page_key
        ]
        return cls(
            case_name=_build_entry_group_case_name(
                entries, doubled_page_keys, group_page_keys
            ),
            suggested_answer=suggest_group_page_placements(entries, decisions, pages),
            reasons=_list_entry_group_reason_names(
                decisions,
                split_work_review_reasons,
                list_mal_page_review_reasons(entries, decisions, pages),
            ),
            candidate_evidence=tuple(candidate_evidence),
            entries={entry.reference_page: entry for entry in entries},
            decisions=tuple(decisions),
            pages=_select_page_details(
                pages, frozenset(group_page_keys) | set(doubled_page_keys)
            ),
        )

    @classmethod
    def from_linked_page(
        cls, decision: LinkedPageDecision, pages: Iterable[ProviderPage]
    ) -> Self:
        """Make the review case for a linked page the rules could not add on their own.

        Args:
            decision: The decision ``decide_linked_page`` made.
            pages: The fetched pages, including the linked page itself; the
                details of the linked page and of the linking entries' pages
                are kept.

        Returns:
            The case named by the linked page's current key and the linking
            entries' reference pages, with a suggested answer when one applies.
        """
        pages = tuple(pages)
        entries = {entry.reference_page: entry for entry in decision.linking_entries}
        linked_page = next(
            page for page in pages if page.current_key == decision.page_key
        )
        return cls(
            case_name=(
                f"{_build_case_name('linked page', [decision.page_key])}"
                f" from {' '.join(entries)}"
            ),
            suggested_answer=suggest_linked_page_answer(decision, linked_page),
            reasons=(decision.reason.name,),
            entries=entries,
            decisions=(decision,),
            pages=_select_page_details(
                pages,
                frozenset(
                    {
                        decision.page_key,
                        *(
                            key
                            for entry in decision.linking_entries
                            for key in entry.addresses_by_page_key
                        ),
                    }
                ),
            ),
        )


def _build_entry_group_case_name(
    entries: Sequence[WorkEntry],
    doubled_page_keys: Iterable[str],
    group_page_keys: Iterable[str],
) -> str:
    """Name an entry group's case.

    A single entry's case names its doubled pages and its reference page, since
    its answer places pages relative to that entry. A group's case names every
    page of its entries, since its answer places every page.
    """
    if len(entries) == 1:
        return (
            f"{_build_case_name('doubled provider', doubled_page_keys)}"
            f" in {entries[0].reference_page}"
        )
    return _build_case_name("split work", set(group_page_keys))


def _list_entry_group_reason_names(
    decisions: Iterable[DoubledProviderDecision],
    split_work_review_reasons: Iterable[SplitWorkReviewReason],
    mal_page_review_reasons: Iterable[MalPageComparison],
) -> tuple[str, ...]:
    """List each decision's outcome and review reasons, then the other reasons.

    The split-work reasons come next, then why the MAL check left pages out.
    """
    return (
        *(
            name
            for decision in decisions
            for name in (
                decision.outcome.name,
                *(reason.name for reason in decision.review_reasons),
            )
        ),
        *(reason.name for reason in split_work_review_reasons),
        *(comparison.name for comparison in mal_page_review_reasons),
    )


def _build_case_name(case_kind: str, page_keys: Iterable[str]) -> str:
    """Name a case by its kind and its sorted page keys."""
    return f"{case_kind}: {' '.join(sorted(page_keys))}"


def _select_page_details(
    pages: Iterable[ProviderPage], page_keys: frozenset[str]
) -> tuple[ProviderPage, ...]:
    """Return the details of these pages and of pages searched by them.

    A page is kept when its key or its current key is one of ``page_keys``, or
    when it was searched by one of them (an AnimeSchedule page searched by MAL
    id), so a reviewer sees every link the decision read. Sorted by current key.
    """
    return tuple(
        page
        for key, page in sorted(index_pages_by_current_key(pages).items())
        if {key, page.key, *page.searched_by} & page_keys
    )
