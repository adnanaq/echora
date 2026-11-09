"""Resolve which review answer places an entry group's pages, and the seed entries it names.

A review case is placed by its own accepted answer, or, when a seed release
renamed the case, by the previous answer it carries (``attach_previous_answers``).
This module finds the previous cases a pass's cases can replace, attaches the
ones left after both passes, and reads the placements and named seed entries of
whichever answer applies.
"""

from collections.abc import Mapping, Sequence
from typing import NamedTuple

from work_matching import (
    KnownWorkEntries,
    PagePlacement,
    ReviewCase,
    ReviewFile,
    attach_previous_answers,
    resolve_named_entries,
)


class AnswerPlacements(NamedTuple):
    """Hold the placements an answer gives, with the entries it was given for."""

    placements: dict[str, PagePlacement]
    named_entry_reference_pages: dict[str, str]
    # The reference pages of the entries the answer was given for; only these
    # can survive as a work, so an entry a previous answer never saw stays as
    # the seed lists it.
    answered_reference_pages: frozenset[str]


def find_previous_cases(
    review_file: ReviewFile,
    review_cases: Sequence[ReviewCase],
    known_work_entries: KnownWorkEntries,
) -> dict[str, ReviewCase]:
    """Find the answered cases a pass's cases may replace.

    Each item gives the rule, then the reason for it.

    - Only answered cases of a kind among the pass's cases are found. A pass
      sees one kind of entry group case, and a previous case of the other kind
      may still come up in the other pass under its own name.
    - Their named seed entries are looked up again in this release, so a
      previous answer moving a page to a named entry follows where that entry
      is now.

    Args:
        review_file: The review file, for its answered cases.
        review_cases: The review cases of this pass.
        known_work_entries: The seed's entries, for the named entries.

    Returns:
        Each answered case of the pass's kinds not among the pass's cases, by
        case name.
    """
    case_names = {review_case.case_name for review_case in review_cases}
    case_kinds = {review_case.kind for review_case in review_cases}
    return {
        kept_case.case_name: resolve_named_entries(kept_case, known_work_entries)
        for kept_case in review_file.list_kept_cases()
        if kept_case.case_name not in case_names and kept_case.kind in case_kinds
    }


def attach_remaining_previous_answers(
    review_cases: Sequence[ReviewCase], review_file: ReviewFile
) -> tuple[ReviewCase, ...]:
    """Attach, after both passes, the previous answers no case of their kind took.

    A split-work group the release merged into one entry with a doubled
    provider comes up as a doubled-provider case, which neither pass offers
    the split-work answer. The answer is then shown on that case and never
    applied, since the case's kind changed.

    Args:
        review_cases: Every review case of the run, with the previous answers
            the passes attached.
        review_file: The review file, for its answered cases.

    Returns:
        The cases in their order, with the remaining previous answers attached
        to cases that carry none yet.
    """
    claimed_case_names = {review_case.case_name for review_case in review_cases} | {
        review_case.previous_answer.case_name
        for review_case in review_cases
        if review_case.previous_answer is not None
    }
    attached_cases_by_name = {
        review_case.case_name: review_case
        for review_case in attach_previous_answers(
            [
                review_case
                for review_case in review_cases
                if review_case.previous_answer is None
            ],
            [
                kept_case
                for kept_case in review_file.list_kept_cases()
                if kept_case.case_name not in claimed_case_names
            ],
        )
    }
    return tuple(
        attached_cases_by_name.get(review_case.case_name, review_case)
        for review_case in review_cases
    )


def resolve_answer_placements(
    review_case: ReviewCase | None, previous_cases_by_name: Mapping[str, ReviewCase]
) -> AnswerPlacements | None:
    """Resolve which answer places a review case's pages, if any.

    Each item gives the rule, then the reason for it.

    - The case's own accepted answer places its pages.
    - Without one, an applied previous answer places the pages the previous
      case held, since the person answered those. Pages the release added get
      no placement and stay in their seed entry.

    Args:
        review_case: The group's review case.
        previous_cases_by_name: The answered cases that did not come up, by
            case name.

    Returns:
        The placements and the seed entries the answer names, or None when
        no answer places the case.
    """
    if review_case is None:
        return None
    answering_case = review_case
    previous_answer = review_case.previous_answer
    if (
        review_case.accepted_answer is None
        and previous_answer is not None
        and previous_answer.is_applied
    ):
        answering_case = previous_cases_by_name[previous_answer.case_name]
    accepted_placements = answering_case.accepted_answer
    if not isinstance(accepted_placements, dict):
        return None
    return AnswerPlacements(
        accepted_placements,
        answering_case.named_entry_reference_pages,
        frozenset(answering_case.entries),
    )


def find_named_target_reference_pages(
    placements: Mapping[str, PagePlacement],
    named_entry_reference_pages: Mapping[str, str],
    current_page_key_by_page_key: Mapping[str, str],
) -> dict[str, str]:
    """Return the reference page each page an answer sends to a named seed entry moves to.

    An answer's named entry replaces any target the links found, since a
    person chose it.

    Args:
        placements: Each placed page's place, by the key the answer names it by.
        named_entry_reference_pages: Each named page key, mapped to the reference
            page of the seed entry holding it.
        current_page_key_by_page_key: Each renamed page's current key, by the key
            an entry lists it under.

    Returns:
        Each named page's current key, mapped to the reference page it moves to.
    """
    return {
        current_page_key_by_page_key.get(page_key, page_key): reference_page
        for page_key, placement in placements.items()
        if placement.named_entry_page_key
        and (
            reference_page := named_entry_reference_pages.get(
                placement.named_entry_page_key
            )
        )
    }
