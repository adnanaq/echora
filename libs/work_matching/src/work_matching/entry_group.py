"""Place, suggest, and validate pages for groups of seed entries.

An entry group is one entry with doubled providers, or two or more entries that
may be one split work, or both when they overlap. The group's pages are placed
together, because which works the group holds can only be told from all its
pages at once. The rules:

- Combined entry: the group's pages are checked as one entry holding them all,
  so a provider listing a page in two of the entries is a doubled provider like
  any other. The combined entry takes its title, type, year and season from the
  entry with the most trusted reference page, the entry most likely to survive.
- Suggestion: pages linked with each other are suggested as one work, numbered
  by trust (``placement_suggestions``). A page no link reaches follows the
  other pages of its own entry when all of them are suggested in one work,
  because the seed listing them together is the only thing known about such a
  page. Such a page gets no suggestion when its entry's suggested pages are in
  different works, or when it is a doubled page, which the reviewer must
  judge. These rules hold for a group of one entry too, so the reviewer only
  answers the pages actually in question.
- MAL check: in a group of two or more entries, each suggestion is then kept
  or left out by comparing its page with the MAL page of its suggested work
  (``mal_page_check``), because a link or a shared entry alone has joined
  separate works.
- Split works: when a provider lists the group's work as part pages, the
  suggestion is one work per part instead (``part_page_splits``), keeping only
  the link suggestions that send a page to ``other work``.
- Duplicate pages get no suggestion, because only a review answer drops one.
- Answers follow the placement rules of ``page_placement``. A group of two or
  more entries adds two rules:

Each item gives the rule, then the reason for it.

- The answer places every page. Merging entries moves pages between them, so
  each page needs a place.
- A work holding pages of two or more entries has the reference page of one
  of them placed in it, so one entry survives as the work
  (``find_surviving_entries``). Otherwise the work would become a new entry,
  and every merged entry would lose its anime id.
"""

from collections.abc import Iterable, Mapping, Sequence

from pydantic import JsonValue
from work_matching.answer_placement_check import (
    PlacementCheck,
    check_answer_placements,
)
from work_matching.answer_rules import PlacementRule
from work_matching.mal_page_check import check_with_mal_pages
from work_matching.model_doubled_provider_decision import DoubledProviderDecision
from work_matching.page_details_comparison import MalPageComparison
from work_matching.page_placement import (
    OTHER_WORK,
    PagePlacement,
    find_page_placement,
    list_placeable_page_keys,
    read_placement,
)
from work_matching.part_page_splits import (
    find_splitting_decision,
    list_duplicate_page_keys,
    suggest_part_page_splits,
)
from work_matching.placement_suggestions import suggest_page_placements
from work_matching.provider_page import ProviderPage
from work_matching.provider_page_key import rank_by_provider_trust
from work_matching.provider_page_links import ProviderPageLinks
from work_matching.surviving_entries import find_surviving_entries
from work_matching.work_entry import WorkEntry


def combine_group_entries(entries: Sequence[WorkEntry]) -> WorkEntry:
    """Combine a group's entries into one entry holding all their pages.

    Args:
        entries: The group's entries, in seed order.

    Returns:
        The single entry itself for a group of one. Otherwise a new entry with
        every entry's pages in the given order, and the title, type, year and
        season of the entry whose reference page is most trusted.
    """
    if len(entries) == 1:
        return entries[0]
    most_trusted_entry = min(
        entries,
        key=lambda entry: (
            rank_by_provider_trust(entry.reference_page_key),
            entry.reference_page_key,
        ),
    )
    addresses_by_page_key: dict[str, str] = {}
    for entry in entries:
        addresses_by_page_key |= entry.addresses_by_page_key
    return most_trusted_entry.model_copy(
        update={"addresses_by_page_key": addresses_by_page_key}
    )


def suggest_group_page_placements(
    entries: Sequence[WorkEntry],
    decisions: Sequence[DoubledProviderDecision],
    pages: Iterable[ProviderPage],
) -> dict[str, str]:
    """Suggest a place for each page of a group that links tie to a work.

    Args:
        entries: The group's entries.
        decisions: The doubled-provider decisions of the combined entry.
        pages: Details of the group's fetched pages.

    Returns:
        Each suggested page's place in answer words. A split work gets one work
        per part, as ``suggest_part_page_splits`` gives them. Otherwise the
        places are those ``suggest_page_placements`` gives for pages linked
        with each other, and a page no link reaches gets the work of its own
        entry's suggested pages. A group of two or more entries then keeps
        only the suggestions the MAL check keeps.
        Duplicate pages are never suggested.
    """
    pages = tuple(pages)
    suggested_placements = _suggest_linked_group_placements(entries, decisions, pages)
    splitting_decision = find_splitting_decision(decisions)
    if splitting_decision is not None:
        suggested_placements = {
            page_key: placement_text
            for page_key, placement_text in suggested_placements.items()
            if placement_text == OTHER_WORK
        } | suggest_part_page_splits(
            list_placeable_page_keys(combine_group_entries(entries), decisions),
            splitting_decision,
            pages,
        )
    elif len(entries) > 1:
        suggested_placements = check_with_mal_pages(
            entries, decisions, suggested_placements, pages
        ).placements
    duplicate_page_keys = list_duplicate_page_keys(decisions)
    return {
        page_key: placement_text
        for page_key, placement_text in suggested_placements.items()
        if page_key not in duplicate_page_keys
    }


def list_mal_page_review_reasons(
    entries: Sequence[WorkEntry],
    decisions: Sequence[DoubledProviderDecision],
    pages: Iterable[ProviderPage],
) -> tuple[MalPageComparison, ...]:
    """List why the MAL check left pages of a group without a suggestion.

    Args:
        entries: The group's entries.
        decisions: The doubled-provider decisions of the combined entry.
        pages: Details of the group's fetched pages.

    Returns:
        Each comparison that left a page out, once, in ``MalPageComparison``
        order; empty for a single entry, which the check does not apply to.
    """
    if len(entries) == 1:
        return ()
    pages = tuple(pages)
    suggested_placements = _suggest_linked_group_placements(entries, decisions, pages)
    return check_with_mal_pages(
        entries, decisions, suggested_placements, pages
    ).left_out_comparisons


def read_group_answer_placements(
    answer: JsonValue,
    entries: Sequence[WorkEntry],
    decisions: Sequence[DoubledProviderDecision],
    current_page_key_by_page_key: Mapping[str, str],
) -> dict[str, PagePlacement] | None:
    """Read a review answer for a group into each page's place, when it is valid.

    Args:
        answer: The answer as written: each page key mapped to its place.
        entries: The group's entries.
        decisions: The doubled-provider decisions of the combined entry.
        current_page_key_by_page_key: Each renamed page's current key, by the
            key an entry lists it under, since a doubled page is placed under
            it.

    Returns:
        Each placed page's place, or None when the answer breaks a placement
        rule or a group rule in the module docstring.
    """
    placement_check = check_group_answer_placements(
        answer, entries, decisions, current_page_key_by_page_key
    )
    return placement_check.placements if placement_check.is_valid else None


def check_group_answer_placements(
    answer: JsonValue,
    entries: Sequence[WorkEntry],
    decisions: Sequence[DoubledProviderDecision],
    current_page_key_by_page_key: Mapping[str, str],
    *,
    named_entry_reference_pages: Mapping[str, str] | None = None,
) -> PlacementCheck:
    """Read a group's review answer into places and list every rule it breaks.

    The group rules in the module docstring are checked only on an answer
    that follows the placement rules, because which entry survives as a work
    can only be told from valid places.

    Args:
        answer: The answer as written: each page key mapped to its place.
        entries: The group's entries.
        decisions: The doubled-provider decisions of the combined entry.
        current_page_key_by_page_key: Each renamed page's current key, by the
            key an entry lists it under, since a doubled page is placed under
            it.
        named_entry_reference_pages: Each page key named after ``other work``,
            mapped to the reference page of the seed entry holding it.

    Returns:
        The places that could be read, and the broken rules in check order.
    """
    is_entry_group = len(entries) > 1
    placement_check = check_answer_placements(
        answer,
        combine_group_entries(entries),
        decisions,
        is_every_page_required=is_entry_group,
        named_entry_reference_pages=named_entry_reference_pages,
    )
    if not placement_check.is_valid or not is_entry_group:
        return placement_check
    placements = placement_check.placements
    surviving_reference_page_by_work = find_surviving_entries(
        entries, placements, current_page_key_by_page_key
    )
    if any(
        len(listing_reference_pages) > 1
        and placement_word not in surviving_reference_page_by_work
        for placement_word, listing_reference_pages in _list_entries_per_work(
            entries, placements, current_page_key_by_page_key
        ).items()
    ):
        return PlacementCheck(placements, (PlacementRule.WORK_WITHOUT_REFERENCE_PAGE,))
    return placement_check


def _suggest_linked_group_placements(
    entries: Sequence[WorkEntry],
    decisions: Sequence[DoubledProviderDecision],
    pages: Sequence[ProviderPage],
) -> dict[str, str]:
    """Suggest places from links and entries alone, before the MAL check.

    Args:
        entries: The group's entries.
        decisions: The doubled-provider decisions of the combined entry.
        pages: Details of the group's fetched pages.

    Returns:
        The suggested placement text for each page key tied by links or by its
        own entry's agreeing suggestions.
    """
    combined_entry = combine_group_entries(entries)
    links = ProviderPageLinks(pages, combined_entry.addresses_by_page_key)
    group_page_keys = frozenset(
        map(links.get_current_page_key, combined_entry.addresses_by_page_key)
    )
    suggested_placements = suggest_page_placements(
        combined_entry,
        decisions,
        [
            (page_key, linked_page_key)
            for page_key in sorted(group_page_keys)
            for linked_page_key in sorted(
                links.get_linked_page_keys(page_key) & group_page_keys
            )
        ],
    )
    doubled_page_keys = {
        page_key for decision in decisions for page_key in decision.page_keys
    }
    for entry in entries:
        suggested_placements |= _suggest_unlinked_entry_pages(
            entry, suggested_placements, doubled_page_keys
        )
    return {
        page_key: suggested_placements[page_key]
        for page_key in combined_entry.addresses_by_page_key
        if page_key in suggested_placements
    }


def _suggest_unlinked_entry_pages(
    entry: WorkEntry,
    suggested_placements: dict[str, str],
    doubled_page_keys: set[str],
) -> dict[str, str]:
    """Suggest the work of an entry's suggested pages for its other pages.

    Args:
        entry: One entry of the group.
        suggested_placements: The places suggested from links so far.
        doubled_page_keys: The keys of the combined entry's doubled pages.

    Returns:
        Each of the entry's pages without a suggestion, doubled pages aside,
        mapped to the one ``work N`` all its suggested pages share; empty when
        the suggested pages name different places, ``other work`` among them,
        or when none is suggested.
    """
    suggested_placement_words = {
        placement.placement_word
        for page_key in entry.addresses_by_page_key
        if page_key in suggested_placements
        and (placement := read_placement(suggested_placements[page_key]))
    }
    if len(suggested_placement_words) != 1 or OTHER_WORK in suggested_placement_words:
        return {}
    [placement_word] = suggested_placement_words
    return {
        page_key: placement_word
        for page_key in entry.addresses_by_page_key
        if page_key not in suggested_placements and page_key not in doubled_page_keys
    }


def _list_entries_per_work(
    entries: Sequence[WorkEntry],
    placements: dict[str, PagePlacement],
    current_page_key_by_page_key: Mapping[str, str],
) -> dict[str, set[str]]:
    """Return the reference pages of the entries listing pages in each work.

    A renamed page counts for the entry listing it under its old key.

    Args:
        entries: The group's work entries.
        placements: Each placed page's placement, by its page key.
        current_page_key_by_page_key: Each renamed page's current key, by the
            key an entry lists it under.

    Returns:
        Each numbered work's placement word mapped to the reference pages of
        the entries that contribute pages to that work.
    """
    listing_reference_pages_by_work: dict[str, set[str]] = {}
    for entry in entries:
        for page_key in entry.addresses_by_page_key:
            placed_page = find_page_placement(
                page_key, placements, current_page_key_by_page_key
            )
            if placed_page and placed_page.placement.placement_word != OTHER_WORK:
                listing_reference_pages_by_work.setdefault(
                    placed_page.placement.placement_word, set()
                ).add(entry.reference_page_key)
    return listing_reference_pages_by_work
