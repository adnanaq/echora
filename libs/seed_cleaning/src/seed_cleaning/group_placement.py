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
  into it, and an entry left without pages is removed.
- Unplaced pages stay: only the rules leave pages unplaced, and they keep the
  entry as its first work.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import NamedTuple

from seed_cleaning.seed_corrections import (
    REVIEW_ANSWER_REASON,
    PlacementCorrections,
    SeedCorrection,
)
from work_matching import (
    OTHER_WORK,
    DoubledProviderDecision,
    KnownWorkEntries,
    PagePlacement,
    ProviderPage,
    ProviderPageStore,
    ReviewCase,
    ReviewFile,
    WorkEntry,
    combine_group_entries,
    decide_doubled_providers,
    find_page_placement,
    find_surviving_entries,
    list_split_work_review_reasons,
    place_pages_without_review,
)


class _PageDestination(NamedTuple):
    """The entry a placed page ends up in; both fields None when the page stays."""

    target_entry_reference_page: str | None
    new_entry_key: str | None


class _GroupPlacement(NamedTuple):
    """Where an entry group's pages go, and what the group's corrections read."""

    combined_entry: WorkEntry
    # Each placed page's placement, and why it is placed, by its key.
    placements: Mapping[str, PagePlacement]
    reason_by_page_key: Mapping[str, str]
    # Each work's surviving entry, as ``find_surviving_entries`` gives them.
    surviving_reference_page_by_work: Mapping[str, str]
    # Each renamed page's current key, by the key an entry lists it under.
    current_key_by_page_key: Mapping[str, str]
    # The seed entry each unrelated page surely moves to, by its current key.
    target_reference_page_by_page_key: Mapping[str, str]


def correct_entry_groups(
    entry_groups: Iterable[Sequence[WorkEntry]],
    store: ProviderPageStore,
    known_work_entries: KnownWorkEntries,
    review_file: ReviewFile,
) -> PlacementCorrections:
    """Place the pages of each entry group, or ask a person.

    Args:
        entry_groups: The entry groups, as ``find_entry_groups`` gives them.
        store: The fetched pages, including pages searched by the entries' pages.
        known_work_entries: Every seed entry, to find the entries pages move to.
        review_file: The review file, for the answers kept in it.

    Returns:
        The corrections from groups needing no review and from accepted
        answers, and every review case of this run.
    """
    corrections: list[SeedCorrection] = []
    review_cases: list[ReviewCase] = []
    for entries in entry_groups:
        group_placement, review_case = _place_entry_group(
            entries, store, known_work_entries, review_file
        )
        if review_case is not None:
            review_cases.append(review_case)
        if group_placement is not None:
            for entry in entries:
                corrections.extend(_build_entry_corrections(entry, group_placement))
    return PlacementCorrections(tuple(corrections), tuple(review_cases))


def _place_entry_group(
    entries: Sequence[WorkEntry],
    store: ProviderPageStore,
    known_work_entries: KnownWorkEntries,
    review_file: ReviewFile,
) -> tuple[_GroupPlacement | None, ReviewCase | None]:
    """Place one entry group's pages by the rules, or by its accepted answer.

    Returns:
        The group's placement, or None while its review case is open or its
        answer is not valid; and its review case, or None when the rules
        placed the group.
    """
    pages = _get_group_pages(entries, store)
    combined_entry = combine_group_entries(entries)
    decisions = decide_doubled_providers(combined_entry, pages, known_work_entries)
    placements = place_pages_without_review(decisions) if len(entries) == 1 else None
    reason_by_page_key = {
        key: decision.outcome.name
        for decision in decisions
        for key in decision.page_keys
    }
    review_case = None
    if placements is None:
        review_case = review_file.attach_kept_answer(
            ReviewCase.from_entry_group(
                entries,
                decisions,
                list_split_work_review_reasons(entries, pages),
                pages,
            )
        )
        accepted_placements = review_case.accepted_answer
        if not isinstance(accepted_placements, dict):
            return None, review_case
        placements = accepted_placements
        reason_by_page_key = dict.fromkeys(placements, REVIEW_ANSWER_REASON)
    current_key_by_page_key = {
        page.key: page.renamed_to for page in pages if page.renamed_to
    }
    group_placement = _GroupPlacement(
        combined_entry,
        placements,
        reason_by_page_key,
        find_surviving_entries(entries, placements, current_key_by_page_key),
        current_key_by_page_key,
        _find_target_reference_pages(decisions),
    )
    return group_placement, review_case


def _get_group_pages(
    entries: Sequence[WorkEntry], store: ProviderPageStore
) -> list[ProviderPage]:
    """Return the stored pages of a group's entries, each page once."""
    pages_by_key: dict[str, ProviderPage] = {}
    for entry in entries:
        pages_by_key |= {page.key: page for page in store.get_entry_pages(entry)}
    return list(pages_by_key.values())


def _build_entry_corrections(
    entry: WorkEntry, group_placement: _GroupPlacement
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
    for key, address in entry.addresses_by_page_key.items():
        placed_page = find_page_placement(
            key, group_placement.placements, group_placement.current_key_by_page_key
        )
        if placed_page is None:
            continue
        corrections.append(
            _build_seed_correction(
                entry,
                key,
                address,
                placed_page.placed_key,
                placed_page.placement,
                group_placement,
            )
        )
    return corrections


def _build_seed_correction(
    entry: WorkEntry,
    page_key: str,
    address: str,
    current_key: str,
    placement: PagePlacement,
    group_placement: _GroupPlacement,
) -> SeedCorrection:
    """Build the correction placing one page an entry lists.

    The page records its current key as renamed only when the group also lists
    that key, so dropping the old address loses nothing.
    """
    destination = _find_page_destination(
        entry, page_key, placement, current_key, group_placement
    )
    is_renamed_page_listed = (
        current_key != page_key
        and current_key in group_placement.combined_entry.addresses_by_page_key
    )
    return SeedCorrection(
        entry_reference_page=entry.reference_page,
        page_key=page_key,
        address=address,
        placement_word=placement.placement_word,
        is_main_page=placement.is_main_page,
        is_part_page=placement.is_part_page,
        renamed_to=current_key if is_renamed_page_listed else None,
        target_entry_reference_page=destination.target_entry_reference_page,
        new_entry_key=destination.new_entry_key,
        reason=group_placement.reason_by_page_key[current_key],
    )


def _find_page_destination(
    entry: WorkEntry,
    page_key: str,
    placement: PagePlacement,
    current_key: str,
    group_placement: _GroupPlacement,
) -> _PageDestination:
    """Find the entry a placed page ends up in.

    Pages with the same new entry key become one new entry.

    Each item reads ``<placement>: <where the page goes>``.

    - ``other work`` with a sure move: the page joins that seed entry.
    - ``other work`` without one: the key is ``<entry reference page> <page
      key>``, so the page alone forms a new entry.
    - A work the page's own entry survives as: the page stays.
    - A work another seed entry survives as: the page joins that entry.
    - A work no entry survives as: the key is ``<group reference page> work
      N``, so the group's pages in that work form one new entry together.

    Args:
        entry: The entry the page is listed in.
        page_key: The key the entry lists the page under.
        placement: The page's placement.
        current_key: The key the page was placed by.
        group_placement: Where the entry's group places its pages.

    Returns:
        The seed entry the page joins, or the new entry key it shares.
    """
    if placement.placement_word == OTHER_WORK:
        move_target_reference_page = (
            group_placement.target_reference_page_by_page_key.get(current_key)
        )
        if move_target_reference_page:
            return _PageDestination(move_target_reference_page, None)
        return _PageDestination(None, f"{entry.reference_page} {page_key}")
    surviving_reference_page = group_placement.surviving_reference_page_by_work.get(
        placement.placement_word
    )
    if surviving_reference_page == entry.reference_page:
        return _PageDestination(None, None)
    if surviving_reference_page:
        return _PageDestination(surviving_reference_page, None)
    return _PageDestination(
        None,
        f"{group_placement.combined_entry.reference_page} {placement.placement_word}",
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
        key: unrelated_page.move_target_entry.reference_page
        for decision in decisions
        for key, unrelated_page in decision.unrelated_pages.items()
        if unrelated_page.move_target_entry is not None
    }
