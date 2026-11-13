"""Place the pages of each entry group, from the rules or a review answer.

An entry group is one entry with doubled providers, or two or more entries that
may be one split work (``entry_groups``). The rules:

- Checked as one: the group's pages are checked as one combined entry
  (``combine_group_entries``), so a provider listed in two of its entries is a
  doubled provider like any other.
- Rules or review: the rules place a single entry when no doubled-provider
  decision needs review. A group of two or more entries always goes to review,
  because a shared title never proves one work.
- Works: each work keeps the entry that survives as it
  (``find_surviving_entries``). The other entries' pages in that work move
  into it, and an entry left without pages is removed. Only an entry the
  answer was given for can survive, so an entry a previous answer never saw
  stays as the seed lists it.
- Unplaced pages stay. The rules and a previous answer can leave pages
  unplaced, and such pages keep the entry as its first work.
- Previous answers. A case a seed release renamed is placed by the answer of
  the case it replaces while it has no answer of its own
  (``answer_placements``). The pages that answer did not see stay unplaced, so
  the case is asked again with them.
- Destinations: the entry each placed page ends up in, as its correction
  names it, follows the rules of ``page_destinations``.
- Second pass: groups of the second pass hold entries as the first pass left
  them (``corrected_seed``), but corrections name the downloaded seed. So each
  page's correction names the seed entry listing it, its address, and whether
  it was added, and each entry it joins is named by the seed entry it stands
  in for or its new entry key. A page moved in by the first pass that now
  stays where it is records that place, so the second pass's correction can
  stand in for the first's.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import NamedTuple

from seed_cleaning.answer_placements import (
    find_named_target_reference_pages,
    find_previous_cases,
    resolve_answer_placements,
)
from seed_cleaning.models import (
    AnswerPlacements,
    CorrectedSeed,
    CorrectionNames,
    EntryGroup,
    GroupPlacement,
    PlacementCorrections,
    SeedCorrection,
)
from seed_cleaning.page_destinations import (
    find_page_destination,
    find_partial_work_destination,
)
from seed_cleaning.seed_corrections import (
    REVIEW_ANSWER_REASON,
)
from work_matching import (
    DROPPED,
    DoubledProviderDecision,
    KnownWorkEntries,
    PagePlacement,
    ProviderPage,
    ProviderPageStore,
    ReviewCase,
    ReviewFile,
    WorkEntry,
    attach_previous_answers,
    combine_group_entries,
    decide_doubled_providers,
    find_covered_outside_entries,
    find_page_placement,
    find_surviving_entries,
    list_split_work_review_reasons,
    place_pages_without_review,
    resolve_named_entries,
    suggest_target_entries,
)


class _DecidedEntryGroup(NamedTuple):
    """Hold an entry group as the rules leave it, before any answer places it."""

    entry_group: EntryGroup
    pages: list[ProviderPage]
    combined_entry: WorkEntry
    decisions: list[DoubledProviderDecision]
    # The rules' placements, or None when the group goes to review.
    rule_placements: dict[str, PagePlacement] | None
    # The group's review case with its kept answer, or None when the rules
    # placed the group.
    review_case: ReviewCase | None


def correct_entry_groups(
    entry_groups: Iterable[EntryGroup],
    store: ProviderPageStore,
    known_work_entries: KnownWorkEntries,
    review_file: ReviewFile,
    corrected_seed: CorrectedSeed | None = None,
) -> PlacementCorrections:
    """Place the pages of each entry group, or ask a person.

    Every group is decided before any answer is applied, because a case the
    seed release renamed finds its previous answer only among all the pass's
    cases (``attach_previous_answers``).

    Args:
        entry_groups: The entry groups, with why their entries are grouped.
        store: The fetched pages, including pages searched by the entries' pages.
        known_work_entries: Every entry the groups' entries come from, to find
            the entries pages move to.
        review_file: The review file, for the answers kept in it.
        corrected_seed: In the second pass, the corrected seed the groups'
            entries come from, so corrections name the downloaded seed; None
            in the first pass.

    Returns:
        The corrections from groups needing no review, from accepted answers
        and from applied previous answers, and every review case of this run.
    """
    decided_groups = [
        _decide_entry_group(entry_group, store, known_work_entries, review_file)
        for entry_group in entry_groups
    ]
    review_cases = [
        decided_group.review_case
        for decided_group in decided_groups
        if decided_group.review_case is not None
    ]
    previous_cases_by_name = find_previous_cases(
        review_file, review_cases, known_work_entries
    )
    review_cases_by_name = {
        review_case.case_name: review_case
        for review_case in attach_previous_answers(
            review_cases, previous_cases_by_name.values()
        )
    }
    corrections: list[SeedCorrection] = []
    correction_names = CorrectionNames(corrected_seed)
    for decided_group in decided_groups:
        review_case = (
            review_cases_by_name[decided_group.review_case.case_name]
            if decided_group.review_case is not None
            else None
        )
        group_placement = _place_entry_group(
            decided_group._replace(review_case=review_case),
            previous_cases_by_name,
            correction_names,
        )
        if group_placement is not None:
            for entry in decided_group.entry_group.entries:
                corrections.extend(_build_entry_corrections(entry, group_placement))
    return PlacementCorrections(
        tuple(corrections), tuple(review_cases_by_name.values())
    )


def _decide_entry_group(
    entry_group: EntryGroup,
    store: ProviderPageStore,
    known_work_entries: KnownWorkEntries,
    review_file: ReviewFile,
) -> _DecidedEntryGroup:
    """Decide one entry group by the rules, or build its review case.

    Args:
        entry_group: The group's entries and why they are grouped.
        store: The fetched pages, including pages searched by the entries' pages.
        known_work_entries: The seed's entries, to find the entries pages move to.
        review_file: The review file, for the answer kept for the group's case.

    Returns:
        The group with its pages and decisions, and either the rules'
        placements or its review case.
    """
    entries = entry_group.entries
    pages = _read_group_pages(entries, store)
    combined_entry = combine_group_entries(entries)
    decisions = decide_doubled_providers(combined_entry, pages, known_work_entries)
    rule_placements = (
        place_pages_without_review(decisions) if len(entries) == 1 else None
    )
    review_case = (
        _build_review_case(
            entry_group, decisions, pages, known_work_entries, store, review_file
        )
        if rule_placements is None
        else None
    )
    return _DecidedEntryGroup(
        entry_group, pages, combined_entry, decisions, rule_placements, review_case
    )


def _place_entry_group(
    decided_group: _DecidedEntryGroup,
    previous_cases_by_name: Mapping[str, ReviewCase],
    correction_names: CorrectionNames,
) -> GroupPlacement | None:
    """Place one decided entry group's pages by the rules or by an answer.

    Args:
        decided_group: The group as the rules leave it, with its review case
            and the case's previous answer.
        previous_cases_by_name: The answered cases that did not come up, by
            case name, for the previous answer a case carries.
        correction_names: How the group's corrections name pages and entries.

    Returns:
        The group's placement, or None while its review case is open with no
        previous answer applied, or its answer is not valid.
    """
    resolved_placements = _resolve_group_placements(
        decided_group, previous_cases_by_name
    )
    if resolved_placements is None:
        return None
    answer_placements, reason_by_page_key = resolved_placements
    placements = answer_placements.placements
    current_page_key_by_page_key = {
        page.key: page.renamed_to for page in decided_group.pages if page.renamed_to
    }
    return GroupPlacement(
        decided_group.combined_entry,
        placements,
        reason_by_page_key,
        find_surviving_entries(
            [
                entry
                for entry in decided_group.entry_group.entries
                if entry.reference_page_key
                in answer_placements.answered_reference_pages
            ],
            placements,
            current_page_key_by_page_key,
        ),
        current_page_key_by_page_key,
        _find_target_reference_pages(decided_group.decisions)
        | find_named_target_reference_pages(
            placements,
            answer_placements.named_entry_reference_pages,
            current_page_key_by_page_key,
        ),
        answer_placements.named_entry_reference_pages,
        correction_names,
    )


def _resolve_group_placements(
    decided_group: _DecidedEntryGroup,
    previous_cases_by_name: Mapping[str, ReviewCase],
) -> tuple[AnswerPlacements, dict[str, str]] | None:
    """Resolve the placements of one group, with why each page is placed there.

    Args:
        decided_group: The group as the rules leave it, with its review case.
        previous_cases_by_name: The answered cases that did not come up, by
            case name, for the previous answer a case carries.

    Returns:
        The rules' placements with each decision's outcome as the reason, or
        an answer's placements with the review answer reason; None when no
        answer places the group.
    """
    if decided_group.rule_placements is not None:
        return AnswerPlacements(
            decided_group.rule_placements,
            {},
            frozenset(
                entry.reference_page_key for entry in decided_group.entry_group.entries
            ),
        ), {
            page_key: decision.outcome.name
            for decision in decided_group.decisions
            for page_key in decision.page_keys
        }
    answer_placements = resolve_answer_placements(
        decided_group.review_case, previous_cases_by_name
    )
    if answer_placements is None:
        return None
    return answer_placements, dict.fromkeys(
        answer_placements.placements, REVIEW_ANSWER_REASON
    )


def _build_review_case(
    entry_group: EntryGroup,
    decisions: Sequence[DoubledProviderDecision],
    pages: Sequence[ProviderPage],
    known_work_entries: KnownWorkEntries,
    store: ProviderPageStore,
    review_file: ReviewFile,
) -> ReviewCase:
    """Make an entry group's review case, with why its entries are grouped and its kept answer.

    Args:
        entry_group: The group's entries and why they are grouped.
        decisions: The doubled-provider decisions of the combined entry.
        pages: The entries' fetched pages and the pages searched by them.
        known_work_entries: The seed's entries, to suggest the entries a page
            in ``other work`` could move to.
        store: The fetched pages, for the suggested entries' details.
        review_file: The review file, for the answer kept for the case.

    Returns:
        The case, with its seed entries of the same series as suggestions,
        those of them a page of the case covers too, its kept answer, and the
        seed entries that answer names looked up.
    """
    review_case = ReviewCase.build_entry_group_case(
        entry_group.entries,
        decisions,
        list_split_work_review_reasons(
            entry_group.entries,
            pages,
            has_shared_tying_link=entry_group.has_shared_tying_link,
        ),
        pages,
        candidate_evidence=entry_group.candidate_evidence,
    )
    review_case = review_case.model_copy(
        update={
            "target_entry_suggestions": suggest_target_entries(
                review_case, known_work_entries
            )
        }
    )
    entry_pages_by_reference_page = {
        suggestion.entry_reference_page: store.get_entry_pages(work_entry)
        for suggestion in review_case.target_entry_suggestions
        if (
            work_entry := known_work_entries.find_by_page_key(
                suggestion.entry_reference_page
            )
        )
    }
    review_case = review_case.model_copy(
        update={
            "covered_outside_entries": find_covered_outside_entries(
                review_case, entry_pages_by_reference_page
            )
        }
    )
    return resolve_named_entries(
        review_file.attach_kept_answer(review_case), known_work_entries
    )


def _read_group_pages(
    entries: Sequence[WorkEntry], store: ProviderPageStore
) -> list[ProviderPage]:
    """Read the stored pages of a group's entries, each page once.

    Args:
        entries: The group's entries.
        store: The fetched pages.

    Returns:
        The stored pages of every entry, each page once.
    """
    pages_by_key: dict[str, ProviderPage] = {}
    for entry in entries:
        pages_by_key |= {page.key: page for page in store.get_entry_pages(entry)}
    return list(pages_by_key.values())


def _build_entry_corrections(
    entry: WorkEntry, group_placement: GroupPlacement
) -> list[SeedCorrection]:
    """Turn one entry's page places into corrections, in the entry's page order.

    A doubled page listed under an old address takes the place of its current
    page, since the decisions name doubled pages by their current key. The old
    address is dropped only when the group also lists the current one;
    otherwise dropping it would lose the page, so it keeps its address. Any
    other page is placed by the key the entry lists, as an answer names it.
    Pages without a place stay as they are and get no correction.

    Args:
        entry: The seed entry whose pages are corrected.
        group_placement: Where the entry's group places its pages.

    Returns:
        One correction per placed page of the entry.
    """
    corrections: list[SeedCorrection] = []
    for page_key, address in entry.addresses_by_page_key.items():
        placed_page = find_page_placement(
            page_key,
            group_placement.placements,
            group_placement.current_page_key_by_page_key,
        )
        if placed_page is None:
            continue
        corrections.append(
            _build_seed_correction(
                entry,
                page_key,
                address,
                placed_page.placed_page_key,
                placed_page.placement,
                group_placement,
            )
        )
    return corrections


def _build_seed_correction(
    entry: WorkEntry,
    page_key: str,
    address: str,
    current_page_key: str,
    placement: PagePlacement,
    group_placement: GroupPlacement,
) -> SeedCorrection:
    """Build the correction placing one page an entry lists.

    The page records its current key as renamed only when the group also lists
    that key, so dropping the old address loses nothing.

    Args:
        entry: The entry the page is listed in.
        page_key: The key the entry lists the page under.
        address: The page's address in the entry.
        current_page_key: The key the page was placed by.
        placement: The page's placement.
        group_placement: Where the entry's group places its pages.

    Returns:
        The correction naming where the page goes and why.
    """
    origin = group_placement.correction_names.find_page_origin(entry, page_key, address)
    destination = find_page_destination(
        entry, page_key, placement, current_page_key, origin, group_placement
    )
    is_renamed_page_listed = (
        current_page_key != page_key
        and current_page_key in group_placement.combined_entry.addresses_by_page_key
    )
    partial_destinations = [
        find_partial_work_destination(partial_work, group_placement)
        for partial_work in placement.partial_works
    ]
    return SeedCorrection(
        entry_reference_page=origin.entry_reference_page,
        page_key=page_key,
        address=origin.address,
        is_added_page=origin.is_added_page,
        placement_word=placement.placement_word,
        is_dropped_page=placement.placement_word == DROPPED,
        partial_entry_reference_pages=tuple(
            partial_destination.target_entry_reference_page
            for partial_destination in partial_destinations
            if partial_destination.target_entry_reference_page
        ),
        partial_new_entry_keys=tuple(
            partial_destination.new_entry_key
            for partial_destination in partial_destinations
            if partial_destination.new_entry_key
        ),
        renamed_to=current_page_key if is_renamed_page_listed else None,
        target_entry_reference_page=destination.target_entry_reference_page,
        new_entry_key=destination.new_entry_key,
        named_entry_page_key=placement.named_entry_page_key,
        reason=group_placement.reason_by_page_key[current_page_key],
    )


def _find_target_reference_pages(
    decisions: Sequence[DoubledProviderDecision],
) -> dict[str, str]:
    """Return the reference page each unrelated page moves to, when the move is sure.

    The rules and review answers share one target
    (``UnrelatedPage.move_target_entry``), so a page an answer sends to another
    work never makes a doubled provider in the entry it joins. Without a sure
    target the page starts a new entry.

    Args:
        decisions: The group's doubled-provider decisions.

    Returns:
        Each unrelated page's current key, mapped to the reference page of the
        seed entry it moves to; pages without a sure target are left out.
    """
    return {
        page_key: unrelated_page.move_target_entry.reference_page_key
        for decision in decisions
        for page_key, unrelated_page in decision.unrelated_pages.items()
        if unrelated_page.move_target_entry is not None
    }
