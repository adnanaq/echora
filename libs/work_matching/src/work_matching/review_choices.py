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
- A ``work N`` already holding another MAL page is disabled for a MAL page.
  A work is what MAL lists, so it holds one MAL page.
- ``main`` is disabled for a provider's only page in its work: a main page
  stands beside the provider's other pages there, so a lone page has none.
- ``main`` and ``part`` stay offered while the provider's other pages in the
  work carry the other mark. Choosing either changes those pages too (see
  below), so the answer still keeps one main page or only part pages.
- ``main`` and ``part`` are disabled for a page with no work yet, since a
  mark says how the page stands within its work (``ChoiceRule``).
- ``main`` and ``part`` are disabled for a page in ``other work``, since the
  mark would belong to a work outside the case (``ChoiceRule``).
- Choosing ``part`` makes the provider's other pages in that work ``part``
  too; choosing ``main`` leaves them unmarked; taking ``part`` off one takes
  it off all. A work's pages of one provider are all parts or none, so one
  choice keeps them consistent.
- After each choice, a provider's only page in a work loses ``main``, and a
  page joining a work whose pages of its provider are parts becomes a part,
  so a choice never leaves a mark the rules refuse.
- Rules about the whole answer, such as every page needing a place, are not
  used to disable choices. Whole-answer rules are checked when the answer is
  saved, since a reviewer fills the pages in one at a time.
- A linked page's choices are ``add`` and ``add as part`` when one entry links
  the page, and ``add to`` and ``add as part to`` each linking entry when
  several do, then ``skip``. Each is disabled with the rule
  ``check_linked_page_answer`` reports for it.
- Two linked-page answers adding whole pages of one provider to one entry
  conflict, because the run adds neither (``find_linked_page_conflicts``).
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import replace

from pydantic import JsonValue
from work_matching.answer_rules import PlacementRule
from work_matching.entry_group import combine_group_entries
from work_matching.linked_page_check import (
    PART_PAGE_ANSWER,
    SKIP_ANSWER,
    WHOLE_PAGE_ANSWER,
    check_linked_page_answer,
)
from work_matching.model_linked_page_decision import LinkedPageAddition
from work_matching.model_review_choices import (
    ChoiceOption,
    ChoiceRule,
    GroupPageChoices,
    LinkedPageConflict,
)
from work_matching.page_placement import (
    MAIN_MARK,
    NO_MARK,
    OTHER_WORK,
    PART_MARK,
    PagePlacement,
    build_page_keys_by_work_provider,
    has_two_mal_pages,
    list_placeable_page_keys,
    read_placement,
    write_placement,
)
from work_matching.provider_page_key import is_mal_page, read_page_provider
from work_matching.review_cases import ReviewCase, ReviewCaseKind

_MARKS = (NO_MARK, MAIN_MARK, PART_MARK)


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
        (placement := _read_placement(placement_text)) is not None
        and placement.is_later_work
        for placement_text in answer.values()
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
    placements = _read_placements(answer)
    placeable_page_keys = list_placeable_case_page_keys(case)
    required_page_keys = list_required_page_keys(case, answer)
    highest_work_number = max(
        (
            int(placement.placement_word.removeprefix("work "))
            for placement in placements.values()
            if placement.placement_word != OTHER_WORK
        ),
        default=0,
    )
    offered_work_count = min(highest_work_number + 1, len(placeable_page_keys))
    work_words = [
        *(f"work {number}" for number in range(1, offered_work_count + 1)),
        OTHER_WORK,
    ]
    return tuple(
        GroupPageChoices(
            page_key=page_key,
            work=placement.placement_word if placement else None,
            named_entry_page_key=placement.named_entry_page_key if placement else None,
            mark=_resolve_mark(placement),
            is_required=page_key in required_page_keys,
            work_options=tuple(
                ChoiceOption(
                    work_word,
                    _find_work_disabling_rule(page_key, work_word, placements),
                )
                for work_word in work_words
            ),
            mark_options=tuple(
                ChoiceOption(
                    mark, _find_mark_disabling_rule(page_key, mark, placements)
                )
                for mark in _MARKS
            ),
        )
        for page_key in placeable_page_keys
        for placement in [placements.get(page_key)]
    )


def choose_group_page_place(
    answer: Mapping[str, JsonValue],
    page_key: str,
    work: str | None,
    mark: str,
) -> dict[str, JsonValue]:
    """Return the answer after one page's place is chosen, with marks kept consistent.

    Args:
        answer: The answer so far: page keys mapped to places.
        page_key: The page whose place is chosen.
        work: ``work N``, ``other work`` or ``other work <page key>``; None
            removes the page's place. Text that is not a place is kept as
            written, so the answer's check reports it.
        mark: ``NO_MARK``, ``MAIN_MARK`` or ``PART_MARK``.

    Returns:
        The new answer: the chosen place, then the mark changes the module
        docstring lists. Places that could not be read are kept as written.
    """
    chosen_placement = read_placement(work) if work is not None else None
    if work is not None and chosen_placement is None:
        return {**answer, page_key: work}
    placements = _read_placements(answer)
    previous_placement = placements.pop(page_key, None)
    if chosen_placement is not None:
        is_marked_work = chosen_placement.placement_word != OTHER_WORK
        placements[page_key] = replace(
            chosen_placement,
            is_main_page=is_marked_work and mark == MAIN_MARK,
            is_part_page=is_marked_work and mark == PART_MARK,
        )
        _spread_mark(placements, page_key, mark, previous_placement)
    _normalize_marks(placements)
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
        values = [WHOLE_PAGE_ANSWER, PART_PAGE_ANSWER]
    else:
        values = [
            value
            for entry in decision.linking_entries
            for value in (
                f"{WHOLE_PAGE_ANSWER} to {entry.reference_page_key}",
                f"{PART_PAGE_ANSWER} to {entry.reference_page_key}",
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
            and not accepted_answer.is_part_page
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


def _find_work_disabling_rule(
    page_key: str, work_word: str, placements: Mapping[str, PagePlacement]
) -> PlacementRule | None:
    """Return the rule disabling a work for a page: a second MAL page in it.

    Args:
        page_key: The page key being evaluated.
        work_word: The work option word being evaluated.
        placements: The current placements by page key.

    Returns:
        ``WORK_WITH_TWO_MAL_PAGES`` when the work already holds another MAL
        page, or None otherwise.
    """
    if work_word == OTHER_WORK:
        return None
    work_page_keys = [
        page_key,
        *(
            other_page_key
            for other_page_key, placement in placements.items()
            if other_page_key != page_key and placement.placement_word == work_word
        ),
    ]
    return (
        PlacementRule.WORK_WITH_TWO_MAL_PAGES
        if is_mal_page(page_key) and has_two_mal_pages(work_page_keys)
        else None
    )


def _find_mark_disabling_rule(
    page_key: str, mark: str, placements: Mapping[str, PagePlacement]
) -> PlacementRule | ChoiceRule | None:
    """Return the rule disabling a mark for a page, as the module docstring lists.

    Args:
        page_key: The page key being evaluated.
        mark: The mark option being evaluated.
        placements: The current placements by page key.

    Returns:
        The rule that disables the mark, or None when the mark is allowed.
    """
    if mark == NO_MARK:
        return None
    placement = placements.get(page_key)
    if placement is None:
        return ChoiceRule.MARK_WITHOUT_WORK
    if placement.placement_word == OTHER_WORK:
        return ChoiceRule.MARK_IN_OTHER_WORK
    if mark == MAIN_MARK and not _list_provider_page_keys_in_work(placements, page_key):
        return PlacementRule.SINGLE_PAGE_MARKED_MAIN
    return None


def _spread_mark(
    placements: dict[str, PagePlacement],
    page_key: str,
    mark: str,
    previous_placement: PagePlacement | None,
) -> None:
    """Change the marks of the page's provider pages in its work after its mark is chosen.

    Args:
        placements: The current placements by page key, updated in place.
        page_key: The page key whose mark was just chosen.
        mark: The chosen mark.
        previous_placement: The page's previous placement, if any.
    """
    placement = placements[page_key]
    if placement.placement_word == OTHER_WORK:
        return
    was_part_in_same_work = (
        previous_placement is not None
        and previous_placement.is_part_page
        and previous_placement.placement_word == placement.placement_word
    )
    if mark == PART_MARK:
        other_mark = PART_MARK
    elif mark == MAIN_MARK or was_part_in_same_work:
        other_mark = NO_MARK
    else:
        return
    for provider_page_key in _list_provider_page_keys_in_work(placements, page_key):
        placements[provider_page_key] = PagePlacement(
            placements[provider_page_key].placement_word,
            is_part_page=other_mark == PART_MARK,
        )


def _normalize_marks(placements: dict[str, PagePlacement]) -> None:
    """Make each work's pages of one provider all parts when any is, and drop a lone main.

    Args:
        placements: The current placements by page key, updated in place.
    """
    for page_keys in build_page_keys_by_work_provider(placements).values():
        if any(placements[page_key].is_part_page for page_key in page_keys):
            for page_key in page_keys:
                placements[page_key] = PagePlacement(
                    placements[page_key].placement_word, is_part_page=True
                )
        elif len(page_keys) == 1 and placements[page_keys[0]].is_main_page:
            placements[page_keys[0]] = PagePlacement(
                placements[page_keys[0]].placement_word
            )


def _list_provider_page_keys_in_work(
    placements: Mapping[str, PagePlacement], page_key: str
) -> list[str]:
    """Return the other pages of the page's provider placed in the page's work.

    Args:
        placements: The current placements by page key.
        page_key: The reference page key.

    Returns:
        The other page keys of the same provider in the same work.
    """
    work_provider = (placements[page_key].placement_word, read_page_provider(page_key))
    return [
        other_page_key
        for other_page_key in build_page_keys_by_work_provider(placements).get(
            work_provider, []
        )
        if other_page_key != page_key
    ]


def _read_placements(answer: Mapping[str, JsonValue]) -> dict[str, PagePlacement]:
    """Read the answer's places that follow the answer words; others are left out.

    Args:
        answer: The raw answer mapping page keys to placement strings.

    Returns:
        The readable placements by page key.
    """
    return {
        page_key: placement
        for page_key, placement_text in answer.items()
        if (placement := _read_placement(placement_text)) is not None
    }


def _read_placement(placement_text: object) -> PagePlacement | None:
    """Read one place as an answer writes it, or None for anything else.

    Args:
        placement_text: The raw placement value.

    Returns:
        The parsed ``PagePlacement``, or None if not a valid placement string.
    """
    return read_placement(placement_text) if isinstance(placement_text, str) else None


def _resolve_mark(placement: PagePlacement | None) -> str:
    """Resolve a place's mark as a choice value.

    Args:
        placement: The page's placement, or None.

    Returns:
        ``MAIN_MARK``, ``PART_MARK``, or ``NO_MARK``.
    """
    if placement is None:
        return NO_MARK
    if placement.is_main_page:
        return MAIN_MARK
    return PART_MARK if placement.is_part_page else NO_MARK
