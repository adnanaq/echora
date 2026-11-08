"""Find the entry each placed page of an entry group ends up in, as corrections name it.

A correction names the entry a page joins by its seed entry's reference page,
or by a new entry key when the page starts an entry the seed lacks. Pages with
the same new entry key become one new entry. Two different keys never read the
same, because reference pages and page keys hold no spaces, and a page key
never reads ``work N``.

Each item reads ``<placement>: <where the page goes>``.

- ``other work``: the page joins the seed entry of its sure move. Without one,
  the page alone forms a new entry, keyed ``<entry reference page> <page key>``.
- ``dropped``: the page goes nowhere, since the correction's
  ``is_dropped_page`` takes it out of its entry.
- A work the page's own entry survives as: the page stays.
- A work another seed entry survives as: the page joins that entry.
- A work no entry survives as: the key is ``<group reference page> work N``,
  so the group's pages in that work form one new entry together.
- A partial work: the entry that work becomes, or the seed entry holding the
  page an ``other work <page key>`` partial work names.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import NamedTuple

from seed_cleaning.corrected_seed import CorrectedSeed, PageOrigin
from work_matching import DROPPED, OTHER_WORK, PagePlacement, WorkEntry


class PageDestination(NamedTuple):
    """The entry a placed page ends up in; both fields None when the page stays."""

    target_entry_reference_page: str | None
    new_entry_key: str | None


@dataclass(frozen=True)
class CorrectionNames:
    """How corrections name pages and entries: by the seed, or by a corrected seed."""

    # None in the first pass, whose entries are the seed's own.
    corrected_seed: CorrectedSeed | None

    def find_page_origin(
        self, entry: WorkEntry, page_key: str, address: str
    ) -> PageOrigin:
        """Return the seed entry listing a page of an entry, its address and whether added.

        Args:
            entry: The entry listing the page in this pass.
            page_key: The key the entry lists the page under.
            address: The page's address in the entry.

        Returns:
            The page's origin in the downloaded seed.
        """
        if self.corrected_seed is None:
            return PageOrigin(entry.reference_page_key, address)
        return self.corrected_seed.find_page_origin(entry.reference_page_key, page_key)

    def find_entry_destination(self, reference_page: str) -> PageDestination:
        """Return how a correction names the entry with this reference page.

        Args:
            reference_page: The reference page of an entry of this pass.

        Returns:
            The seed entry the entry stands for, or its new entry key.
        """
        if self.corrected_seed is None:
            return PageDestination(reference_page, None)
        identity = self.corrected_seed.identity_by_reference_page[reference_page]
        return PageDestination(
            identity.seed_entry_reference_page, identity.new_entry_key
        )


class GroupPlacement(NamedTuple):
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
    # The reference page of the seed entry holding each page key an answer
    # names after ``other work``.
    named_entry_reference_pages: Mapping[str, str]
    correction_names: CorrectionNames


def find_page_destination(
    entry: WorkEntry,
    page_key: str,
    placement: PagePlacement,
    current_key: str,
    origin: PageOrigin,
    group_placement: GroupPlacement,
) -> PageDestination:
    """Find the entry a placed page ends up in, as its correction names it.

    A dropped page goes nowhere, since its correction takes it out of every
    entry. A page staying in its entry still moves when that entry is not the
    seed entry listing the page, such as a page the first pass moved in.

    Args:
        entry: The entry the page is listed in.
        page_key: The key the entry lists the page under.
        placement: The page's placement.
        current_key: The key the page was placed by.
        origin: The seed entry listing the page, and its address there.
        group_placement: Where the entry's group places its pages.

    Returns:
        The seed entry the page joins, or the new entry key it shares; both
        fields None when the page is dropped or stays in the seed entry
        listing it.
    """
    if placement.placement_word == DROPPED:
        return PageDestination(None, None)
    destination = _find_placed_page_destination(
        entry, page_key, placement, current_key, group_placement
    )
    if destination is not None:
        return destination
    entry_destination = group_placement.correction_names.find_entry_destination(
        entry.reference_page_key
    )
    if entry_destination == (origin.entry_reference_page, None):
        return PageDestination(None, None)
    return entry_destination


def find_partial_work_destination(
    partial_work: str, group_placement: GroupPlacement
) -> PageDestination:
    """Find the entry a page joins as a partial source of one partial work.

    Each item reads ``<partial work>: <entry>``.

    - ``work N``: the entry that work becomes (``_find_work_destination``).
    - ``other work <page key>``: the seed entry holding the named page. A run
      reads answers only after their named keys are looked up, so the key is
      always found.

    Args:
        partial_work: The partial work in answer words.
        group_placement: Where the group places its pages.

    Returns:
        The seed entry or the new entry the page is a partial source of.
    """
    named_page_key = partial_work.removeprefix(f"{OTHER_WORK} ")
    if named_page_key != partial_work:
        return group_placement.correction_names.find_entry_destination(
            group_placement.named_entry_reference_pages[named_page_key]
        )
    return _find_work_destination(partial_work, group_placement)


def _find_placed_page_destination(
    entry: WorkEntry,
    page_key: str,
    placement: PagePlacement,
    current_key: str,
    group_placement: GroupPlacement,
) -> PageDestination | None:
    """Find the entry a placement sends a page to, or None when the page stays in its entry.

    Args:
        entry: The entry the page is listed in.
        page_key: The key the entry lists the page under.
        placement: The page's placement.
        current_key: The key the page was placed by.
        group_placement: Where the entry's group places its pages.

    Returns:
        The seed entry the page joins, or the new entry key it shares; None
        when the page stays in its entry.
    """
    if placement.placement_word == OTHER_WORK:
        return _find_other_work_destination(
            entry, page_key, current_key, group_placement
        )
    surviving_reference_page = group_placement.surviving_reference_page_by_work.get(
        placement.placement_word
    )
    if surviving_reference_page == entry.reference_page_key:
        return None
    return _find_work_destination(placement.placement_word, group_placement)


def _find_work_destination(
    work_word: str, group_placement: GroupPlacement
) -> PageDestination:
    """Find the entry a work of the group becomes: its surviving entry, or a new entry.

    Args:
        work_word: The work's word, ``work 1`` and so on.
        group_placement: Where the group places its pages.

    Returns:
        The seed entry the work survives as, as corrections name it; without
        one, the new entry keyed ``<group reference page> work N``, so the
        group's pages in that work form one new entry together.
    """
    surviving_reference_page = group_placement.surviving_reference_page_by_work.get(
        work_word
    )
    if surviving_reference_page:
        return group_placement.correction_names.find_entry_destination(
            surviving_reference_page
        )
    return PageDestination(
        None, f"{group_placement.combined_entry.reference_page_key} {work_word}"
    )


def _find_other_work_destination(
    entry: WorkEntry, page_key: str, current_key: str, group_placement: GroupPlacement
) -> PageDestination:
    """Find where a page placed in ``other work`` goes: its sure move, or a new entry.

    Args:
        entry: The entry the page is listed in.
        page_key: The key the entry lists the page under.
        current_key: The key the page was placed by.
        group_placement: Where the entry's group places its pages.

    Returns:
        The seed entry of the page's sure move, as corrections name it; without
        one, the new entry keyed ``<entry reference page> <page key>``.
    """
    move_target_reference_page = group_placement.target_reference_page_by_page_key.get(
        current_key
    )
    if move_target_reference_page:
        return group_placement.correction_names.find_entry_destination(
            move_target_reference_page
        )
    return PageDestination(None, f"{entry.reference_page_key} {page_key}")
