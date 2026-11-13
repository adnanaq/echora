"""Check a person's answer placing an entry group's pages against the placement rules.

The places an answer writes are read with ``page_placement``. Every rule the
answer breaks is reported, not only the first, so a reviewer sees every problem
of an answer at once.

A person's answer is valid when it follows these rules:

Each item gives the rule, then the reason for it.

- An answer places every doubled page. Each doubled page needs a place.
- An answer using ``work 2`` or later places every page of the entry. Splitting
  the entry sends each page to one of the works.
- No work holds two pages of one provider. A work is the finest split any
  provider lists, so two pages of one provider are two works or a duplicate
  page.
- A dropped page has another page of its provider placed in a work of the
  case. Dropping a duplicate page keeps the other page, so the provider is not
  lost.
- A page is not a partial source of its own work, since the page is already
  a source there.
- A page is a partial source only of a work that holds placed pages, since an
  empty work becomes no work entry.
- A page named after ``other work`` belongs to a seed entry outside the case,
  both as a place and among the partial works. A page of the case's own
  entries is placed with ``work N`` instead.
- A seed entry holds each page named after ``other work``. The caller looks
  the names up in the seed (``named_entry_reference_pages``), so an unknown
  page is refused rather than moved nowhere.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pydantic import JsonValue
from work_matching.models import DoubledProviderDecision, PlacementRule
from work_matching.page_placement import (
    DROPPED,
    OTHER_WORK,
    PagePlacement,
    build_page_keys_by_work_provider,
    has_two_pages_of_one_provider,
    list_placeable_page_keys,
    read_placements,
)
from work_matching.provider_page_key import read_page_provider
from work_matching.work_entry import WorkEntry


@dataclass(frozen=True, slots=True)
class PlacementCheck:
    """Hold an answer's readable places and every placement rule it breaks."""

    # Each page whose place could be read, by the key the answer names it by.
    placements: dict[str, PagePlacement]
    # In the order the rules are checked; empty when the answer is valid.
    broken_rules: tuple[PlacementRule, ...]

    @property
    def is_valid(self) -> bool:
        """Report whether the answer breaks no placement rule."""
        return not self.broken_rules


def read_answer_placements(
    answer: JsonValue,
    entry: WorkEntry,
    decisions: Sequence[DoubledProviderDecision],
    *,
    is_every_page_required: bool = False,
) -> dict[str, PagePlacement] | None:
    """Read a review answer into each page's place, when the answer is valid.

    Args:
        answer: The answer as written: each page key mapped to its place.
        entry: The work entry the answer places pages of.
        decisions: The entry's doubled-provider decisions.
        is_every_page_required: Whether the answer must place every page, as
            for a group of entries, even without ``work 2`` or later.

    Returns:
        Each placed page's place, or None when the answer breaks a rule
        ``check_answer_placements`` checks.
    """
    placement_check = check_answer_placements(
        answer, entry, decisions, is_every_page_required=is_every_page_required
    )
    return placement_check.placements if placement_check.is_valid else None


def check_answer_placements(
    answer: JsonValue,
    entry: WorkEntry,
    decisions: Sequence[DoubledProviderDecision],
    *,
    is_every_page_required: bool = False,
    named_entry_reference_pages: Mapping[str, str] | None = None,
) -> PlacementCheck:
    """Read a review answer into places and list every placement rule it breaks.

    The rules are the ones the module docstring lists. Each is checked on its
    own, so a reviewer sees every problem of an answer at once.

    Args:
        answer: The answer as written: each page key mapped to its place.
        entry: The work entry the answer places pages of.
        decisions: The entry's doubled-provider decisions.
        is_every_page_required: Whether the answer must place every page, as
            for a group of entries, even without ``work 2`` or later.
        named_entry_reference_pages: Each page key named after ``other
            work``, mapped to the reference page of the seed entry holding
            it. A named key missing here breaks ``UNKNOWN_NAMED_ENTRY``.

    Returns:
        The places that could be read, and the broken rules in check order.
    """
    if not isinstance(answer, dict):
        return PlacementCheck({}, (PlacementRule.INVALID_PLACEMENT_MAP,))
    placements = read_placements(answer)
    placeable_page_keys = set(list_placeable_page_keys(entry, decisions))
    required_page_keys = _find_required_page_keys(
        placements, placeable_page_keys, decisions, is_every_page_required
    )
    broken_rules = [
        rule
        for rule, is_broken in (
            (PlacementRule.INVALID_PAGE_PLACEMENT, len(placements) != len(answer)),
            (
                PlacementRule.PAGE_NOT_IN_CASE,
                bool(placements.keys() - placeable_page_keys),
            ),
            (
                PlacementRule.MISSING_REQUIRED_PLACEMENT,
                bool(required_page_keys - placements.keys()),
            ),
            *_check_work_rules(placements),
            *_check_named_entry_rules(
                placements, entry, named_entry_reference_pages or {}
            ),
        )
        if is_broken
    ]
    return PlacementCheck(placements, tuple(broken_rules))


def _find_required_page_keys(
    placements: Mapping[str, PagePlacement],
    placeable_page_keys: set[str],
    decisions: Sequence[DoubledProviderDecision],
    is_every_page_required: bool,
) -> set[str]:
    """Find the pages an answer must place.

    An answer placing a page in ``work 2`` or later splits the entry, so every
    page needs a place. Otherwise only the doubled pages need one.

    Args:
        placements: Each placed page's place, by its key.
        placeable_page_keys: The keys of the pages an answer can place.
        decisions: The entry's doubled-provider decisions.
        is_every_page_required: Whether the answer must place every page, as
            for a group of entries.

    Returns:
        The keys of the pages the answer must place.
    """
    is_mixed_entry_answer = any(
        placement.is_later_work for placement in placements.values()
    )
    if is_mixed_entry_answer or is_every_page_required:
        return placeable_page_keys
    return {page_key for decision in decisions for page_key in decision.page_keys}


def _check_work_rules(
    placements: Mapping[str, PagePlacement],
) -> tuple[tuple[PlacementRule, bool], ...]:
    """Check the rules on what each work and partial work of an answer holds.

    Args:
        placements: Each placed page's place, by its key.

    Returns:
        Each work rule, in check order, with whether the answer breaks it.
    """
    return (
        (
            PlacementRule.WORK_WITH_TWO_PAGES_OF_ONE_PROVIDER,
            any(
                map(
                    has_two_pages_of_one_provider,
                    build_page_keys_by_work_provider(placements).values(),
                )
            ),
        ),
        (
            PlacementRule.DROPPED_PAGE_WITHOUT_KEPT_PAGE,
            _has_dropped_page_without_kept_page(placements),
        ),
        (
            PlacementRule.PARTIAL_SOURCE_OF_OWN_WORK,
            any(
                placement.placement_word in placement.partial_works
                for placement in placements.values()
            ),
        ),
        (
            PlacementRule.PARTIAL_SOURCE_OF_EMPTY_WORK,
            _has_partial_source_of_empty_work(placements),
        ),
    )


def _check_named_entry_rules(
    placements: Mapping[str, PagePlacement],
    entry: WorkEntry,
    named_entry_reference_pages: Mapping[str, str],
) -> tuple[tuple[PlacementRule, bool], ...]:
    """Check the rules on the seed entries an answer names after ``other work``.

    A key is named both as a page's place and among the works a page is a
    partial source of.

    Args:
        placements: Each placed page's place, by its key.
        entry: The work entry the answer places pages of.
        named_entry_reference_pages: Each named page key a seed entry holds,
            mapped to that entry's reference page.

    Returns:
        Each named-entry rule, in check order, with whether the answer breaks it.
    """
    named_entry_page_keys = {
        named_entry_page_key
        for placement in placements.values()
        for named_entry_page_key in (
            placement.named_entry_page_key,
            *placement.partial_named_entry_page_keys,
        )
        if named_entry_page_key
    }
    named_page_keys_in_case = named_entry_page_keys & entry.addresses_by_page_key.keys()
    return (
        (PlacementRule.NAMED_ENTRY_IN_CASE, bool(named_page_keys_in_case)),
        (
            PlacementRule.UNKNOWN_NAMED_ENTRY,
            bool(
                named_entry_page_keys
                - named_page_keys_in_case
                - named_entry_reference_pages.keys()
            ),
        ),
    )


def _has_dropped_page_without_kept_page(
    placements: Mapping[str, PagePlacement],
) -> bool:
    """Report whether a dropped page's provider has no page placed in a work of the case.

    Args:
        placements: Each placed page's place, by its page key.

    Returns:
        True when some dropped page has no other page of its provider kept.
    """
    kept_providers = {
        read_page_provider(page_key)
        for page_key, placement in placements.items()
        if placement.is_case_work
    }
    return any(
        read_page_provider(page_key) not in kept_providers
        for page_key, placement in placements.items()
        if placement.placement_word == DROPPED
    )


def _has_partial_source_of_empty_work(
    placements: Mapping[str, PagePlacement],
) -> bool:
    """Report whether a page is a partial source of a work of the case no page is placed in.

    Args:
        placements: Each placed page's place, by its page key.

    Returns:
        True when some partial work of the case holds no placed page.
    """
    used_work_words = {
        placement.placement_word
        for placement in placements.values()
        if placement.is_case_work
    }
    return any(
        partial_work not in used_work_words
        for placement in placements.values()
        for partial_work in placement.partial_works
        if not partial_work.startswith(OTHER_WORK)
    )
