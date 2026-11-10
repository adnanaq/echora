"""Offer a reviewer, page by page, only the choices the answer rules allow.

The review page asks these functions what to offer, rather than copying the
rules into the browser, so the page and a run never disagree. An entry
group's answer gives each page a work and a mark; a linked page's answer is
one of a few words. The rules:

Each item gives the rule, then the reason for it.

- Work choices are ``work 1`` up to one more than the highest work in use
  (at most one work per page), and ``other work``. Works are numbered in
  use, so the next new work is always offered, and a case of 110 pages does
  not offer 110 works in every dropdown.
- A ``work N`` already holding another page of the page's provider is
  disabled. A work is the finest split any provider lists, so it holds one
  page of each provider.
- ``dropped`` is disabled while no other page of the page's provider has a
  work of the case, since dropping a duplicate page keeps the other page.
- A page keeps the works it is a partial source of when its place changes,
  except its new work. A partial work left without placed pages is taken off
  every page, so a choice never leaves a partial source the rules refuse.
- A work's partial sources are chosen per work. Any page that is a source of
  another work of the case can be one. A work of the case is listed only when
  the case has two or more, since a lone work has no other work's page to
  take. A seed entry outside the case is listed when a page of the case
  covers it (``covered_outside_entries``) or the answer names it.
- Rules about the whole answer, such as every page needing a place, are not
  used to disable choices. Whole-answer rules are checked when the answer is
  saved, since a reviewer fills the pages in one at a time.
- A linked page's choices are ``add`` and ``add as partial`` when one entry
  links the page, and ``add to`` and ``add as partial to`` each linking entry when
  several do, then ``skip``. Each is disabled with the rule
  ``check_linked_page_answer`` reports for it.
- Two linked-page answers adding whole pages of one provider to one entry
  conflict, because the run adds neither (``find_linked_page_conflicts``).
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace

from pydantic import JsonValue
from work_matching.answer_rules import PlacementRule
from work_matching.entry_group import combine_group_entries
from work_matching.linked_page_check import (
    PARTIAL_SOURCE_ANSWER,
    SKIP_ANSWER,
    WHOLE_PAGE_ANSWER,
    check_linked_page_answer,
)
from work_matching.model_linked_page_decision import LinkedPageAddition
from work_matching.model_review_choices import (
    ChoiceOption,
    GroupPageChoices,
    LinkedPageConflict,
    WorkPartialSources,
)
from work_matching.page_placement import (
    DROPPED,
    OTHER_WORK,
    PagePlacement,
    list_placeable_page_keys,
    read_placement,
    read_placements,
    write_placement,
)
from work_matching.provider_page_key import read_page_provider
from work_matching.review_cases import ReviewCase, ReviewCaseKind


def list_placeable_case_page_keys(case: ReviewCase) -> tuple[str, ...]:
    """Return the keys of an entry group case's pages an answer can place.

    Args:
        case: A doubled-provider or split-work case.

    Returns:
        The doubled pages first, then the entries' other pages, as
        ``list_placeable_page_keys`` orders them.
    """
    return list_placeable_page_keys(
        combine_group_entries(list(case.entries.values())),
        case.doubled_provider_decisions,
    )


def list_required_page_keys(
    case: ReviewCase, answer: Mapping[str, JsonValue]
) -> frozenset[str]:
    """Return the keys of the pages this answer must place to be valid.

    Args:
        case: A doubled-provider or split-work case.
        answer: The answer so far: page keys mapped to places.

    Returns:
        Every placeable page for a split-work case or an answer using
        ``work 2`` or later; otherwise only the doubled pages.
    """
    is_later_work_used = any(
        placement.is_later_work for placement in read_placements(answer).values()
    )
    if case.kind is ReviewCaseKind.SPLIT_WORK or is_later_work_used:
        return frozenset(list_placeable_case_page_keys(case))
    return frozenset(
        page_key
        for decision in case.doubled_provider_decisions
        for page_key in decision.page_keys
    )


def list_group_page_choices(
    case: ReviewCase, answer: Mapping[str, JsonValue]
) -> tuple[GroupPageChoices, ...]:
    """List each page's current place and the choices left, for an entry group case.

    Args:
        case: A doubled-provider or split-work case.
        answer: The answer so far: page keys mapped to places.

    Returns:
        One entry per placeable page, in ``list_placeable_case_page_keys``
        order, with each disabled choice's rule.
    """
    placements = read_placements(answer)
    placeable_page_keys = list_placeable_case_page_keys(case)
    required_page_keys = list_required_page_keys(case, answer)
    highest_work_number = max(
        (
            int(placement.placement_word.removeprefix("work "))
            for placement in placements.values()
            if placement.is_case_work
        ),
        default=0,
    )
    offered_work_count = min(highest_work_number + 1, len(placeable_page_keys))
    work_words = [
        *(f"work {number}" for number in range(1, offered_work_count + 1)),
        OTHER_WORK,
        DROPPED,
    ]
    return tuple(
        GroupPageChoices(
            page_key=page_key,
            work=placement.placement_word if placement else None,
            named_entry_page_key=placement.named_entry_page_key if placement else None,
            partial_works=placement.partial_works if placement else (),
            is_required=page_key in required_page_keys,
            work_options=tuple(
                ChoiceOption(
                    work_word,
                    _find_work_disabling_rule(page_key, work_word, placements),
                )
                for work_word in work_words
            ),
        )
        for page_key in placeable_page_keys
        for placement in [placements.get(page_key)]
    )


def choose_group_page_place(
    answer: Mapping[str, JsonValue], page_key: str, work: str | None
) -> dict[str, JsonValue]:
    """Return the answer after one page's place is chosen, with partial works kept valid.

    Args:
        answer: The answer so far: page keys mapped to places.
        page_key: The page whose place is chosen.
        work: ``work N``, ``other work``, ``other work <page key>`` or
            ``dropped``; None removes the page's place. Text that is not a
            place is kept as written, so the answer's check reports it.

    Returns:
        The new answer: the chosen place, keeping the partial works the module
        docstring lists. Places that could not be read are kept as written.
    """
    chosen_placement = read_placement(work) if work is not None else None
    if work is not None and chosen_placement is None:
        return {**answer, page_key: work}
    placements = read_placements(answer)
    previous_placement = placements.pop(page_key, None)
    if chosen_placement is not None:
        kept_partial_works = (
            tuple(
                partial_work
                for partial_work in previous_placement.partial_works
                if partial_work != chosen_placement.placement_word
            )
            if previous_placement is not None and chosen_placement.is_case_work
            else ()
        )
        placements[page_key] = replace(
            chosen_placement, partial_works=kept_partial_works
        )
    _remove_empty_partial_works(placements)
    return _write_answer(answer, page_key, placements)


def list_work_partial_sources(
    case: ReviewCase, answer: Mapping[str, JsonValue]
) -> tuple[WorkPartialSources, ...]:
    """List each work an entry group's pages can be partial sources of, with its choices.

    Args:
        case: A doubled-provider or split-work case.
        answer: The answer so far: page keys mapped to places.

    Returns:
        The case's works in use, in number order, when there are two or more;
        then the seed entries outside the case that a page of the case covers,
        and any other entry the answer names as a partial work.
    """
    placements = read_placements(answer)
    case_work_words = sorted(
        {
            placement.placement_word
            for placement in placements.values()
            if placement.is_case_work
        },
        key=lambda work_word: int(work_word.removeprefix("work ")),
    )
    if len(case_work_words) < 2:
        case_work_words = []
    outside_work_words = dict.fromkeys(
        [
            *(
                f"{OTHER_WORK} {covered_entry.entry_reference_page}"
                for covered_entry in case.covered_outside_entries
            ),
            *(
                partial_work
                for placement in placements.values()
                for partial_work in placement.partial_works
                if partial_work.startswith(f"{OTHER_WORK} ")
            ),
        ]
    )
    return tuple(
        WorkPartialSources(
            work=work_word,
            partial_source_page_keys=tuple(
                page_key
                for page_key, placement in placements.items()
                if work_word in placement.partial_works
            ),
            page_key_options=tuple(
                page_key
                for page_key, placement in placements.items()
                if placement.is_case_work
                and placement.placement_word != work_word
                and work_word not in placement.partial_works
            ),
        )
        for work_word in (*case_work_words, *outside_work_words)
    )


def choose_partial_sources(
    answer: Mapping[str, JsonValue],
    page_keys: Sequence[str],
    work: str,
    *,
    is_partial_source: bool,
) -> dict[str, JsonValue]:
    """Return the answer after pages are made, or stop being, partial sources of a work.

    Args:
        answer: The answer so far: page keys mapped to places.
        page_keys: The pages whose partial works change, as chosen together.
        work: ``work N`` or ``other work <page key>``.
        is_partial_source: Whether the pages become partial sources of the
            work, or stop being ones.

    Returns:
        The new answer. A page without a work of the case, or the work's own
        page, is left as it is, since it cannot be a partial source there.
    """
    new_answer = {**answer}
    for page_key in page_keys:
        new_answer = _choose_partial_source(
            new_answer, page_key, work, is_partial_source=is_partial_source
        )
    return new_answer


def list_linked_page_options(case: ReviewCase) -> tuple[ChoiceOption, ...]:
    """List a linked-page case's answers, each with the rule that disables it.

    Args:
        case: The review case; only a linked-page case has answers to list.

    Returns:
        The additions the module docstring lists, then ``skip``; empty for an
        entry group case.
    """
    decision = case.linked_page_decision
    if decision is None:
        return ()
    if len(decision.linking_entries) == 1:
        values = [WHOLE_PAGE_ANSWER, PARTIAL_SOURCE_ANSWER]
    else:
        values = [
            value
            for entry in decision.linking_entries
            for value in (
                f"{WHOLE_PAGE_ANSWER} to {entry.reference_page_key}",
                f"{PARTIAL_SOURCE_ANSWER} to {entry.reference_page_key}",
            )
        ]
    values.append(SKIP_ANSWER)
    return tuple(
        ChoiceOption(
            value,
            next(
                iter(
                    check_linked_page_answer(
                        {decision.page_key: value}, decision
                    ).broken_rules
                ),
                None,
            ),
        )
        for value in values
    )


def find_linked_page_conflicts(
    cases: Iterable[ReviewCase],
) -> tuple[LinkedPageConflict, ...]:
    """Find linked-page answers adding whole pages of one provider to one entry.

    Args:
        cases: Review cases with their answers; other kinds are ignored.

    Returns:
        One conflict per entry and provider that two or more accepted
        answers add a whole page of, with their case names in the given order.
    """
    case_names_by_target: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for case in cases:
        decision = case.linked_page_decision
        if decision is None:
            continue
        accepted_answer = case.accepted_answer
        provider = read_page_provider(decision.page_key)
        if (
            isinstance(accepted_answer, LinkedPageAddition)
            and not accepted_answer.is_partial_source
            and provider is not None
        ):
            case_names_by_target[accepted_answer.entry_reference_page, provider].append(
                case.case_name
            )
    return tuple(
        LinkedPageConflict(entry_reference_page, provider, tuple(case_names))
        for (entry_reference_page, provider), case_names in case_names_by_target.items()
        if len(case_names) > 1
    )


def _choose_partial_source(
    answer: Mapping[str, JsonValue],
    page_key: str,
    work: str,
    *,
    is_partial_source: bool,
) -> dict[str, JsonValue]:
    """Return the answer after a page is made, or stops being, a partial source of a work.

    Args:
        answer: The answer so far: page keys mapped to places.
        page_key: The page whose partial works change.
        work: ``work N`` or ``other work <page key>``.
        is_partial_source: Whether the page becomes a partial source of the
            work, or stops being one.

    Returns:
        The new answer. A page without a work of the case, or the work's own
        page, is left as it is, since it cannot be a partial source there.
    """
    placements = read_placements(answer)
    placement = placements.get(page_key)
    if (
        placement is None
        or not placement.is_case_work
        or placement.placement_word == work
    ):
        return {**answer}
    kept_partial_works = tuple(
        partial_work for partial_work in placement.partial_works if partial_work != work
    )
    placements[page_key] = replace(
        placement,
        partial_works=(*kept_partial_works, work)
        if is_partial_source
        else kept_partial_works,
    )
    return _write_answer(answer, page_key, placements)


def _find_work_disabling_rule(
    page_key: str, work_word: str, placements: Mapping[str, PagePlacement]
) -> PlacementRule | None:
    """Return the rule disabling a work word for a page, as the module docstring lists.

    Args:
        page_key: The page key being evaluated.
        work_word: The work option word being evaluated.
        placements: The current placements by page key.

    Returns:
        ``WORK_WITH_TWO_PAGES_OF_ONE_PROVIDER`` for a work already holding
        another page of the page's provider, ``DROPPED_PAGE_WITHOUT_KEPT_PAGE``
        for ``dropped`` while no other page of that provider has a work, or
        None when the choice is allowed.
    """
    if work_word == OTHER_WORK:
        return None
    provider = read_page_provider(page_key)
    other_provider_placements = [
        placement
        for other_page_key, placement in placements.items()
        if other_page_key != page_key
        and provider is not None
        and read_page_provider(other_page_key) == provider
    ]
    if work_word == DROPPED:
        return (
            None
            if any(placement.is_case_work for placement in other_provider_placements)
            else PlacementRule.DROPPED_PAGE_WITHOUT_KEPT_PAGE
        )
    return (
        PlacementRule.WORK_WITH_TWO_PAGES_OF_ONE_PROVIDER
        if any(
            placement.placement_word == work_word
            for placement in other_provider_placements
        )
        else None
    )


def _remove_empty_partial_works(placements: dict[str, PagePlacement]) -> None:
    """Take works of the case that hold no placed page off every page's partial works.

    Args:
        placements: The current placements by page key, updated in place.
    """
    used_work_words = {
        placement.placement_word
        for placement in placements.values()
        if placement.is_case_work
    }
    for page_key, placement in placements.items():
        kept_partial_works = tuple(
            partial_work
            for partial_work in placement.partial_works
            if partial_work in used_work_words
            or partial_work.startswith(f"{OTHER_WORK} ")
        )
        if kept_partial_works != placement.partial_works:
            placements[page_key] = replace(placement, partial_works=kept_partial_works)


def _write_answer(
    answer: Mapping[str, JsonValue],
    page_key: str,
    placements: Mapping[str, PagePlacement],
) -> dict[str, JsonValue]:
    """Write the placements back as an answer, keeping places that could not be read.

    Args:
        answer: The answer before the choice.
        page_key: The page whose place was chosen, left out when it lost its place.
        placements: The placements after the choice, by page key.

    Returns:
        The new answer: unreadable places as written, then every placement.
    """
    return {
        **{
            existing_page_key: placement_text
            for existing_page_key, placement_text in answer.items()
            if existing_page_key != page_key and existing_page_key not in placements
        },
        **{
            placed_page_key: write_placement(placement)
            for placed_page_key, placement in placements.items()
        },
    }
