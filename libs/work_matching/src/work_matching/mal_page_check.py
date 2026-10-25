"""Check the suggested places of an entry group's pages against MAL pages.

A link or a shared seed entry alone has joined separate works, such as a page
covering three OVA series suggested as one of them. A work is what MAL lists,
so each suggested page is compared with the MAL page of its suggested work
(``compare_page_with_mal_page``) before the suggestion reaches a reviewer.

Each item reads ``<comparison>: <suggestion>``.

- ``SAME_WORK``: kept.
- ``PART_OF_WORK``: kept, marked ``, part``.
- ``DETAILS_UNKNOWN``: kept for a page the seed lists in the MAL page's own
  entry, since keeping it there merges nothing; left out otherwise.
- ``SEVERAL_WORKS`` or ``DETAILS_DIFFER``: left out.
- ``NO_MAL_PAGE``, a work holding no MAL page: every page left out. Nothing
  sets where such a work starts and ends, and other providers divide works
  differently, so links alone decide too little.
- ``other work``: kept unchecked. The page leaves the group, so it joins no
  work here.

A doubled provider's pages in one work are marked ``, part`` only when all of
them are parts, since an answer may not mix a provider's part and whole pages
in one work (``page_placement``). Each comparison that left a page out is
returned, so the review case can tell the reviewer why.
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple, Self

from work_matching.doubled_provider_decision import DoubledProviderDecision
from work_matching.page_details_comparison import (
    MalPageComparison,
    compare_page_with_mal_page,
)
from work_matching.page_placement import OTHER_WORK, PART_PAGE_MARK, read_placement
from work_matching.provider_page import ProviderPage, index_pages_by_current_key
from work_matching.provider_page_key import get_page_provider
from work_matching.provider_page_links import ProviderPageLinks
from work_matching.work_entry import WorkEntry

# A work is what MAL lists, so the pages of a work are measured against its
# MAL page.
_MAL_PROVIDER = "mal"


@dataclass(frozen=True, slots=True)
class _GroupMalPageDetails:
    """What every page of a group shares when checked against MAL pages."""

    links: ProviderPageLinks
    pages_by_current_key: dict[str, ProviderPage]
    # Each work's MAL page key, by the work's ``work N`` word.
    mal_page_key_by_work: dict[str, str]
    # The reference page of the entry listing each page.
    reference_page_by_page_key: dict[str, str]
    doubled_page_keys: frozenset[str]

    @classmethod
    def from_group(
        cls,
        entries: Sequence[WorkEntry],
        decisions: Sequence[DoubledProviderDecision],
        suggested_placements: dict[str, str],
        pages: Sequence[ProviderPage],
    ) -> Self:
        """Collect what a group's pages share for the MAL check."""
        return cls(
            links=ProviderPageLinks(
                pages,
                [key for entry in entries for key in entry.addresses_by_page_key],
            ),
            pages_by_current_key=index_pages_by_current_key(pages),
            mal_page_key_by_work={
                placement.placement_word: key
                for key, placement_text in suggested_placements.items()
                if get_page_provider(key) == _MAL_PROVIDER
                and (placement := read_placement(placement_text))
            },
            reference_page_by_page_key={
                key: entry.reference_page
                for entry in entries
                for key in entry.addresses_by_page_key
            },
            doubled_page_keys=frozenset(
                key for decision in decisions for key in decision.page_keys
            ),
        )


class _CheckedPage(NamedTuple):
    """One page's place after the MAL check, and the comparison that matters."""

    # None when the check left the page's suggestion out.
    placement_text: str | None
    # Why the page was left out, or ``PART_OF_WORK`` for a doubled page that
    # is a part; None otherwise.
    comparison: MalPageComparison | None


class MalPageCheck(NamedTuple):
    """The suggestions left after the MAL check, and why any page was left out."""

    placements: dict[str, str]
    left_out_comparisons: tuple[MalPageComparison, ...]


def check_with_mal_pages(
    entries: Sequence[WorkEntry],
    decisions: Sequence[DoubledProviderDecision],
    suggested_placements: dict[str, str],
    pages: Sequence[ProviderPage],
) -> MalPageCheck:
    """Keep, mark or leave out each suggested place by the rules in the module docstring.

    Args:
        entries: The group's entries.
        decisions: The doubled-provider decisions of the combined entry.
        suggested_placements: The places suggested from links and entries.
        pages: Details of the group's fetched pages.

    Returns:
        The suggestions kept, with ``, part`` added where the page is a part,
        and each comparison that left a page out.
    """
    group_details = _GroupMalPageDetails.from_group(
        entries, decisions, suggested_placements, pages
    )
    checked_placements: dict[str, str] = {}
    left_out_comparisons: set[MalPageComparison] = set()
    doubled_part_page_keys: set[str] = set()
    for key, suggested_placement_text in suggested_placements.items():
        placement_text, comparison = _check_page_placement(
            key, suggested_placement_text, group_details
        )
        if placement_text is not None:
            checked_placements[key] = placement_text
            if comparison is MalPageComparison.PART_OF_WORK:
                doubled_part_page_keys.add(key)
        elif comparison is not None:
            left_out_comparisons.add(comparison)
    return MalPageCheck(
        _mark_doubled_part_pages(checked_placements, doubled_part_page_keys),
        tuple(
            comparison
            for comparison in MalPageComparison
            if comparison in left_out_comparisons
        ),
    )


def _check_page_placement(
    page_key: str, placement_text: str, group_details: _GroupMalPageDetails
) -> _CheckedPage:
    """Check one page's suggested place against the MAL page of its work.

    Args:
        page_key: The key an entry lists the page under.
        placement_text: The page's suggested place in answer words.
        group_details: What the group's pages share for the check.

    Returns:
        The place unchanged, the place marked ``, part``, or no place when the
        suggestion is left out, with the comparison. A page placed in ``other work``,
        and the MAL page itself, keep their place uncompared; a page whose
        work holds no MAL page is left out. A
        doubled page keeps the mark its decision gave, since its provider's
        other pages in the work decide whether it is a part
        (``_mark_doubled_part_pages``).
    """
    placement = read_placement(placement_text)
    if placement is None:
        return _CheckedPage(placement_text, None)
    if placement.placement_word == OTHER_WORK:
        return _CheckedPage(placement_text, None)
    mal_page_key = group_details.mal_page_key_by_work.get(placement.placement_word)
    if mal_page_key is None:
        return _CheckedPage(None, MalPageComparison.NO_MAL_PAGE)
    if page_key == mal_page_key:
        return _CheckedPage(placement_text, None)
    links = group_details.links
    comparison = _compare_with_mal_page(
        group_details.pages_by_current_key.get(links.get_current_key(page_key)),
        group_details.pages_by_current_key.get(links.get_current_key(mal_page_key)),
    )
    reference_page_by_page_key = group_details.reference_page_by_page_key
    if not _is_suggestion_kept(
        comparison,
        reference_page_by_page_key[page_key]
        == reference_page_by_page_key[mal_page_key],
    ):
        return _CheckedPage(None, comparison)
    if links.get_current_key(page_key) in group_details.doubled_page_keys:
        return _CheckedPage(
            placement_text,
            comparison if comparison is MalPageComparison.PART_OF_WORK else None,
        )
    if comparison is MalPageComparison.PART_OF_WORK:
        return _CheckedPage(placement.placement_word + PART_PAGE_MARK, None)
    return _CheckedPage(placement_text, None)


def _mark_doubled_part_pages(
    placements: dict[str, str], doubled_part_page_keys: set[str]
) -> dict[str, str]:
    """Mark a doubled provider's pages ``, part`` when every one in a work is a part.

    An answer may not mix a provider's part and whole pages in one work, so
    the pages are marked only together.

    Args:
        placements: Each kept page's place.
        doubled_part_page_keys: The doubled pages the MAL check found to be
            parts of their work.

    Returns:
        The places, with every page of a doubled provider in a work marked
        ``, part`` when all of them are parts and none carries a mark yet.
    """
    page_keys_by_work_provider: defaultdict[tuple[str, str | None], list[str]] = (
        defaultdict(list)
    )
    for key, placement_text in placements.items():
        if placement := read_placement(placement_text):
            page_keys_by_work_provider[
                placement.placement_word, get_page_provider(key)
            ].append(key)
    marked_placements = dict(placements)
    for (placement_word, _), page_keys in page_keys_by_work_provider.items():
        if len(page_keys) > 1 and all(
            key in doubled_part_page_keys and placements[key] == placement_word
            for key in page_keys
        ):
            marked_placements |= dict.fromkeys(
                page_keys, placement_word + PART_PAGE_MARK
            )
    return marked_placements


def _is_suggestion_kept(
    comparison: MalPageComparison, is_listed_with_mal_page: bool
) -> bool:
    """Report whether a page keeps its suggestion after the MAL check.

    A page the seed lists in the MAL page's own entry keeps it when its details
    are only unknown, since keeping it there merges nothing.
    """
    return comparison in (
        MalPageComparison.SAME_WORK,
        MalPageComparison.PART_OF_WORK,
    ) or (is_listed_with_mal_page and comparison is MalPageComparison.DETAILS_UNKNOWN)


def _compare_with_mal_page(
    page: ProviderPage | None, mal_page: ProviderPage | None
) -> MalPageComparison:
    """Compare a page with a MAL page; a page not fetched or dead has unknown details."""
    if page is None or mal_page is None or page.is_dead or mal_page.is_dead:
        return MalPageComparison.DETAILS_UNKNOWN
    return compare_page_with_mal_page(page, mal_page)
