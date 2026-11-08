"""Check the suggested places of an entry group's pages against MAL pages.

A link or a shared seed entry alone has joined separate works, such as a page
covering three OVA series suggested as one of them. A work is what MAL lists,
so each suggested page is compared with the MAL page of its suggested work
(``compare_page_with_mal_page``) before the suggestion reaches a reviewer.

Each item reads ``<comparison>: <suggestion>``.

- ``SAME_WORK``: kept.
- ``PART_OF_WORK``: left out. The page lists a finer split than the MAL page,
  so the page is a work of its own.
- ``DETAILS_UNKNOWN``: kept for a page the seed lists in the MAL page's own
  entry, since keeping it there merges nothing; left out otherwise.
- ``SEVERAL_WORKS`` or ``DETAILS_DIFFER``: left out.
- ``NO_MAL_PAGE``, a work holding no MAL page: every page left out. Nothing
  sets where such a work starts and ends, and other providers divide works
  differently, so links alone decide too little.
- ``other work``: kept unchecked. The page leaves the group, so it joins no
  work here.

Each comparison that left a page out is returned, so the review case can tell
the reviewer why.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple, Self

from work_matching.model_doubled_provider_decision import DoubledProviderDecision
from work_matching.page_details_comparison import (
    MalPageComparison,
    compare_page_with_mal_page,
)
from work_matching.page_placement import (
    OTHER_WORK,
    read_placement,
)
from work_matching.provider_page import ProviderPage, index_pages_by_current_page_key
from work_matching.provider_page_key import is_mal_page
from work_matching.provider_page_links import ProviderPageLinks
from work_matching.work_entry import WorkEntry


@dataclass(frozen=True, slots=True)
class _GroupMalPageDetails:
    """Hold what every page of a group shares when checked against MAL pages."""

    links: ProviderPageLinks
    pages_by_current_page_key: dict[str, ProviderPage]
    # Each work's MAL page key, by the work's ``work N`` word.
    mal_page_key_by_work: dict[str, str]
    # The reference page of the entry listing each page.
    reference_page_by_page_key: dict[str, str]
    doubled_page_keys: frozenset[str]

    @classmethod
    def build_group_mal_details(
        cls,
        entries: Sequence[WorkEntry],
        decisions: Sequence[DoubledProviderDecision],
        suggested_placements: dict[str, str],
        pages: Sequence[ProviderPage],
    ) -> Self:
        """Build what a group's pages share for the MAL check.

        Args:
            entries: The group's work entries.
            decisions: The doubled-provider decisions of the combined entry.
            suggested_placements: The suggested placements by page key.
            pages: Details of the group's fetched pages.

        Returns:
            The shared details for comparing the group's pages against MAL pages.
        """
        return cls(
            links=ProviderPageLinks(
                pages,
                [
                    page_key
                    for entry in entries
                    for page_key in entry.addresses_by_page_key
                ],
            ),
            pages_by_current_page_key=index_pages_by_current_page_key(pages),
            mal_page_key_by_work={
                placement.placement_word: page_key
                for page_key, placement_text in suggested_placements.items()
                if is_mal_page(page_key)
                and (placement := read_placement(placement_text))
            },
            reference_page_by_page_key={
                page_key: entry.reference_page_key
                for entry in entries
                for page_key in entry.addresses_by_page_key
            },
            doubled_page_keys=frozenset(
                page_key for decision in decisions for page_key in decision.page_keys
            ),
        )


class _CheckedPage(NamedTuple):
    """Hold one page's place after the MAL check."""

    # None when the check left the page's suggestion out.
    placement_text: str | None
    # Why the page's suggestion was left out; None when it was kept.
    left_out_comparison: MalPageComparison | None = None


class MalPageCheck(NamedTuple):
    """Hold the suggestions left after the MAL check and why any page was left out."""

    placements: dict[str, str]
    left_out_comparisons: tuple[MalPageComparison, ...]


def check_with_mal_pages(
    entries: Sequence[WorkEntry],
    decisions: Sequence[DoubledProviderDecision],
    suggested_placements: dict[str, str],
    pages: Sequence[ProviderPage],
) -> MalPageCheck:
    """Keep or leave out each suggested place by the rules in the module docstring.

    Args:
        entries: The group's entries.
        decisions: The doubled-provider decisions of the combined entry.
        suggested_placements: The places suggested from links and entries, by
            the page keys the entries list, so each page's entry is known.
        pages: Details of the group's fetched pages.

    Returns:
        The suggestions kept, and each comparison that left a page out.
    """
    group_details = _GroupMalPageDetails.build_group_mal_details(
        entries, decisions, suggested_placements, pages
    )
    checked_placements: dict[str, str] = {}
    left_out_comparisons: set[MalPageComparison] = set()
    for page_key, suggested_placement_text in suggested_placements.items():
        checked_page = _check_page_placement(
            page_key, suggested_placement_text, group_details
        )
        if checked_page.left_out_comparison is not None:
            left_out_comparisons.add(checked_page.left_out_comparison)
        if checked_page.placement_text is None:
            continue
        checked_placements[page_key] = checked_page.placement_text
    return MalPageCheck(
        checked_placements,
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
        The checked page. Each item reads ``<page>: <result>``.

        - A page in ``other work``, or the work's MAL page: its place, uncompared.
        - A page whose work holds no MAL page: left out, with ``NO_MAL_PAGE``.
        - A page the comparison leaves out: left out, with the comparison.
        - Any other page: its place unchanged.
    """
    placement = read_placement(placement_text)
    if placement is None or placement.placement_word == OTHER_WORK:
        return _CheckedPage(placement_text)
    mal_page_key = group_details.mal_page_key_by_work.get(placement.placement_word)
    if mal_page_key is None:
        return _CheckedPage(None, MalPageComparison.NO_MAL_PAGE)
    if page_key == mal_page_key:
        return _CheckedPage(placement_text)
    links = group_details.links
    comparison = _compare_with_mal_page(
        group_details.pages_by_current_page_key.get(
            links.get_current_page_key(page_key)
        ),
        group_details.pages_by_current_page_key.get(
            links.get_current_page_key(mal_page_key)
        ),
    )
    reference_page_by_page_key = group_details.reference_page_by_page_key
    if not _is_suggestion_kept(
        comparison,
        reference_page_by_page_key[page_key]
        == reference_page_by_page_key[mal_page_key],
    ):
        return _CheckedPage(None, comparison)
    return _CheckedPage(placement_text)


def _is_suggestion_kept(
    comparison: MalPageComparison, is_listed_with_mal_page: bool
) -> bool:
    """Report whether a page keeps its suggestion after the MAL check.

    A page the seed lists in the MAL page's own entry keeps it when its details
    are only unknown, since keeping it there merges nothing.

    Args:
        comparison: The page's comparison with its work's MAL page.
        is_listed_with_mal_page: Whether the seed lists the page in the entry
            holding that MAL page.

    Returns:
        True for the same work, and for unknown details of a page listed with
        the MAL page.
    """
    return comparison is MalPageComparison.SAME_WORK or (
        is_listed_with_mal_page and comparison is MalPageComparison.DETAILS_UNKNOWN
    )


def _compare_with_mal_page(
    page: ProviderPage | None, mal_page: ProviderPage | None
) -> MalPageComparison:
    """Compare a page with the MAL page of its work.

    Args:
        page: The page's details, or None when it is not fetched.
        mal_page: The MAL page's details, or None when it is not fetched.

    Returns:
        ``DETAILS_UNKNOWN`` when either page is not fetched or is dead, since
        neither has details to compare; otherwise ``compare_page_with_mal_page``.
    """
    if page is None or mal_page is None or page.is_dead or mal_page.is_dead:
        return MalPageComparison.DETAILS_UNKNOWN
    return compare_page_with_mal_page(page, mal_page)
