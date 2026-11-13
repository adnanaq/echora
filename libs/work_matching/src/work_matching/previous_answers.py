"""Carry an answer to the case that replaces its case after a seed release changed its pages.

A case is named by its pages, so a seed release adding a page to one of its
entries renames the case, and the answer kept under the old name no longer meets
it. This module finds, for each case of a run, the answered case it replaces.

Each item gives the rule, then the reason for it.

- A previous case is an answered case that did not come up this run. An
  entry group case sharing a page with it replaces it, when each case is the
  only match of the other. Two matches cannot tell which case the answer
  was about.
- Linked-page cases are left out. A linked-page case name holds the linked page and the
  linking entries, so pages added to those entries never rename them.
- The caller chooses which cases meet. The passes offer a case only previous
  cases of its own kind, and the run then offers the cases left without a
  previous answer every previous case still unclaimed. A doubled-provider case
  can then show the answer of a split-work group the release merged into one
  entry.
- The previous answer applies when the case is of the same kind, still holds
  every entry the previous case held, and the answer was valid there. Pages
  and entries the release added get no placement, so they stay as the seed
  lists them.
- When the release merged entries the answer saw, or changed the case's kind,
  the answer is shown but stops applying. The release then disagrees with the
  answer about which works there are, and the release wins until a person
  answers again.
- The case keeps no answer of its own, so it stays open until a person answers
  it again with the added pages in view. A previous answer that applies fills
  the case's suggested answer for the pages it placed, so the reviewer starts
  from it.
"""

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence

from work_matching.models import PreviousAnswer
from work_matching.review_cases import ReviewCase, ReviewCaseKind


def attach_previous_answers(
    cases: Sequence[ReviewCase], previous_cases: Iterable[ReviewCase]
) -> tuple[ReviewCase, ...]:
    """Return the cases, each with the answer of the previous case it replaces.

    Args:
        cases: The review cases of this run.
        previous_cases: The answered cases whose case did not come up this run
            and that the cases may replace.

    Returns:
        The cases in their order; a case replacing exactly one previous case,
        which no other case replaces, carries that case's answer.
    """
    previous_cases_by_name = {
        previous_case.case_name: previous_case
        for previous_case in previous_cases
        if previous_case.answer is not None
        and previous_case.kind is not ReviewCaseKind.LINKED_PAGE
    }
    replaced_case_name_by_case_name = _match_previous_case_names(
        cases, previous_cases_by_name
    )
    return tuple(
        _attach_previous_answer(
            case,
            previous_cases_by_name[replaced_case_name_by_case_name[case.case_name]],
        )
        if case.case_name in replaced_case_name_by_case_name
        else case
        for case in cases
    )


def build_previous_answer(
    case: ReviewCase, previous_case: ReviewCase
) -> PreviousAnswer:
    """Build the answer a case takes over from the previous case it replaces.

    Args:
        case: The case of this run.
        previous_case: The answered case it replaces.

    Returns:
        The previous case's name and answer, the pages added since, and whether
        the answer still places the case's pages.
    """
    return PreviousAnswer(
        case_name=previous_case.case_name,
        answer=previous_case.answer,
        added_page_keys=tuple(
            sorted(case.entry_page_keys - previous_case.entry_page_keys)
        ),
        is_applied=case.kind is previous_case.kind
        and previous_case.entries.keys() <= case.entries.keys()
        and isinstance(previous_case.accepted_answer, dict),
    )


def _attach_previous_answer(case: ReviewCase, previous_case: ReviewCase) -> ReviewCase:
    """Return the case carrying the previous case's answer.

    A previous answer that still applies also fills the case's suggested
    answer for the pages it placed, over the links' suggestions, so the
    reviewer starts from it and only places the added pages. A previous
    answer that no longer applies is only shown, since the release disagrees
    with it.

    Args:
        case: The case of this run.
        previous_case: The answered case it replaces.

    Returns:
        The case with its previous answer, and its suggested answer filled
        from that answer when the answer applies.
    """
    previous_answer = build_previous_answer(case, previous_case)
    suggested_answer = case.suggested_answer
    if previous_answer.is_applied and isinstance(previous_case.answer, dict):
        case_page_keys = case.entry_page_keys
        suggested_answer = suggested_answer | {
            page_key: placement_word
            for page_key, placement_word in previous_case.answer.items()
            if page_key in case_page_keys and isinstance(placement_word, str)
        }
    return case.model_copy(
        update={
            "previous_answer": previous_answer,
            "suggested_answer": suggested_answer,
        }
    )


def _match_previous_case_names(
    cases: Sequence[ReviewCase], previous_cases_by_name: dict[str, ReviewCase]
) -> dict[str, str]:
    """Match each case to the one previous case sharing a page with it.

    Args:
        cases: The review cases of this run.
        previous_cases_by_name: The answered entry group cases that did not
            come up this run, by case name.

    Returns:
        The previous case name each case replaces, by case name, for the cases
        sharing pages with exactly one previous case that shares pages with no
        other case.
    """
    previous_case_names_by_page_key: defaultdict[str, set[str]] = defaultdict(set)
    for previous_case in previous_cases_by_name.values():
        for page_key in previous_case.entry_page_keys:
            previous_case_names_by_page_key[page_key].add(previous_case.case_name)
    matched_names_by_case_name = {
        case.case_name: {
            previous_case_name
            for page_key in case.entry_page_keys
            for previous_case_name in previous_case_names_by_page_key.get(page_key, ())
        }
        for case in cases
        if case.kind is not ReviewCaseKind.LINKED_PAGE
    }
    match_counts = Counter(
        previous_case_name
        for previous_case_names in matched_names_by_case_name.values()
        for previous_case_name in previous_case_names
    )
    return {
        case_name: previous_case_name
        for case_name, previous_case_names in matched_names_by_case_name.items()
        if len(previous_case_names) == 1
        and match_counts[previous_case_name := next(iter(previous_case_names))] == 1
    }
