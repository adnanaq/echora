"""Suggest a work for each part of a split work, and the partial sources covering several.

A provider that lists a work as several part pages splits it, and every split
any provider makes is kept as its own work. A case holding part pages is
therefore suggested as one work per part, and every other page as the part or
parts it covers (``find_covered_part_pages``). The rules:

Each item gives the rule, then the reason for it.

- The splitting provider is the provider of the ``PART_PAGES`` decision with
  the most pages, the first one on a tie. That provider makes the finest split
  the decisions show.
- Its part pages are numbered ``work 1``, ``work 2`` and so on in air order,
  so the works follow the run of the work.
- A page covering one part is suggested in that part's work. A page covering
  several parts in a row is suggested as a source of the first part's work
  and a partial source of the others, because enrichment fetches only sources
  and the first part is where the page's run starts. The reviewer still
  chooses the source work.
- A page covering no part, or whose parts cannot be told, gets no suggestion,
  since only a person can place it.
- Two pages of one provider suggested in the same work are both left out, for
  the reviewer to tell a duplicate page from a finer split.
- The pages of a ``DUPLICATE_PAGES`` decision get no suggestion, because only
  a review answer drops a duplicate page.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

from work_matching.covered_part_pages import find_covered_part_pages
from work_matching.models import (
    DoubledProviderDecision,
    DoubledProviderOutcome,
)
from work_matching.page_placement import PARTIAL_SOURCE_PREFIX
from work_matching.provider_page import ProviderPage, index_pages_by_any_key
from work_matching.provider_page_key import read_page_provider


def find_splitting_decision(
    decisions: Sequence[DoubledProviderDecision],
) -> DoubledProviderDecision | None:
    """Find the part-pages decision whose provider splits the work finest.

    Args:
        decisions: The doubled-provider decisions of the case's combined entry.

    Returns:
        The ``PART_PAGES`` decision with the most pages, the first one on a
        tie, or None when no decision found part pages.
    """
    part_page_decisions = [
        decision
        for decision in decisions
        if decision.outcome is DoubledProviderOutcome.PART_PAGES
    ]
    if not part_page_decisions:
        return None
    return max(part_page_decisions, key=lambda decision: len(decision.page_keys))


def suggest_part_page_splits(
    placeable_page_keys: Sequence[str],
    splitting_decision: DoubledProviderDecision,
    pages: Iterable[ProviderPage],
) -> dict[str, str]:
    """Suggest a place for each page of a case whose work a provider splits.

    Args:
        placeable_page_keys: The case's placeable page keys, in answer order.
        splitting_decision: The decision ``find_splitting_decision`` chose.
        pages: The fetched pages, looked up by their current key and by the
            key an entry lists them under.

    Returns:
        Each suggested page's place in answer words, by the rules in the
        module docstring, in answer page order.
    """
    pages_by_key = index_pages_by_any_key(pages)
    part_pages = sorted(
        (
            pages_by_key[page_key]
            for page_key in splitting_decision.page_keys
            if page_key in pages_by_key
        ),
        key=lambda part_page: (part_page.aired_from is None, part_page.aired_from),
    )
    work_word_by_part_key = {
        part_page.current_key: f"work {work_number}"
        for work_number, part_page in enumerate(part_pages, start=1)
    }
    suggested_placements: dict[str, str] = {}
    for page_key in placeable_page_keys:
        page = pages_by_key.get(page_key)
        if page is None:
            continue
        if page.current_key in work_word_by_part_key:
            suggested_placements[page_key] = work_word_by_part_key[page.current_key]
            continue
        covered_work_words = [
            work_word_by_part_key[part_page.current_key]
            for part_page in find_covered_part_pages(page, part_pages)
        ]
        if covered_work_words:
            suggested_placements[page_key] = _write_covered_works(covered_work_words)
    return _leave_out_shared_works(suggested_placements)


def list_duplicate_page_keys(
    decisions: Sequence[DoubledProviderDecision],
) -> frozenset[str]:
    """Return the page keys of the decisions that found duplicate pages.

    Args:
        decisions: The doubled-provider decisions of the case's combined entry.

    Returns:
        Every page key of a ``DUPLICATE_PAGES`` decision.
    """
    return frozenset(
        page_key
        for decision in decisions
        if decision.outcome is DoubledProviderOutcome.DUPLICATE_PAGES
        for page_key in decision.page_keys
    )


def _write_covered_works(covered_work_words: Sequence[str]) -> str:
    """Write the place of a page covering one or more works in answer words.

    Args:
        covered_work_words: The works the page covers, in air order.

    Returns:
        ``work N`` for one work, or the first work followed by the works the
        page is a partial source of.
    """
    source_work_word, *partial_work_words = covered_work_words
    if not partial_work_words:
        return source_work_word
    return source_work_word + PARTIAL_SOURCE_PREFIX + ", ".join(partial_work_words)


def _leave_out_shared_works(suggested_placements: Mapping[str, str]) -> dict[str, str]:
    """Leave out pages whose provider has another page suggested in the same work.

    Args:
        suggested_placements: Each suggested page's place in answer words.

    Returns:
        The suggestions without the pages of a provider suggested twice in one
        work, in the given order.
    """
    page_keys_by_work_provider: defaultdict[tuple[str, str | None], list[str]] = (
        defaultdict(list)
    )
    for page_key, placement_text in suggested_placements.items():
        work_word = placement_text.split(",")[0]
        page_keys_by_work_provider[work_word, read_page_provider(page_key)].append(
            page_key
        )
    shared_work_page_keys = {
        page_key
        for page_keys in page_keys_by_work_provider.values()
        if len(page_keys) > 1
        for page_key in page_keys
    }
    return {
        page_key: placement_text
        for page_key, placement_text in suggested_placements.items()
        if page_key not in shared_work_page_keys
    }
