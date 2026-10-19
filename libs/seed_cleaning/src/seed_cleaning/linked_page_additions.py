"""Add the pages entries' pages link that no seed entry holds, as seed corrections.

Providers link pages the seed lacks, most often a MAL page linked from an
entry's AniList, Kitsu or AniDB page. Each such page is fetched, checked by the
work matching rules (``linked_page_check``), and added to the entry linking it
when the checks pass or a person's answer accepts it. The rules:

- Which entries: each entry to check and each entry without a MAL page, as
  the doubled-provider corrections leave it, so a page moved to another entry
  is read under that entry. An entry whose doubled providers wait for a review
  answer is left out until it is answered: its pages are not placed yet, so a
  link on one of them may belong to another work. New entries built from
  answers are left out too, since a correction must name a seed entry.
- Which links: links to pages of providers whose pages are fetched, so every
  linked page can be checked live; other providers' pages are filled in from
  seed releases instead.
- Waiting: a linked page not fetched yet, or whose fetch failed, gets no
  decision; the next run fetches it again.
- One step: corrections are rebuilt from the downloaded seed on every run, so
  an added page's own links are never followed.
- Two answers adding two pages of one provider to one entry would make a
  doubled provider, so neither is added and a warning names the entry.
"""

import logging
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

from seed_cleaning.provider_page_fetcher import FETCHED_PROVIDERS
from seed_cleaning.seed_corrections import (
    REVIEW_ANSWER_REASON,
    CorrectedEntry,
    DoubledProviderCorrections,
    SeedCorrection,
    build_corrected_entries,
)
from work_matching import (
    ENTRY_WORK,
    KnownWorks,
    LinkedPageAddition,
    LinkedPageDecision,
    LinkedPageOutcome,
    LinkingEntry,
    ProviderPage,
    ProviderPageStore,
    ReviewCase,
    ReviewFile,
    WorkEntry,
    decide_linked_page,
    find_unheld_linked_pages,
    get_entry_episode_count,
    get_page_provider,
)

logger = logging.getLogger(__name__)

# The providers whose pages carry links to other providers' pages, so fetching
# them for their links alone is worth it. Anime-Planet and AniSearch pages
# carry none.
LINK_CARRYING_PROVIDERS: frozenset[str] = frozenset({"anilist", "kitsu", "anidb"})


@dataclass(frozen=True, slots=True)
class LinkingSeedEntry:
    """An entry linking a page no seed entry holds, and the seed entry it stands for."""

    # The reference page of the seed entry, which a correction names; the
    # corrected entry's own reference page can differ after a page moved out.
    seed_entry_reference_page: str
    linking_entry: LinkingEntry


class _PageAddition(NamedTuple):
    """A linked page to add, the entry it joins, whether as a part, and why."""

    page: ProviderPage
    # The reference page of the work entry the page joins, as decided.
    entry_reference_page: str
    is_part_page: bool
    # The decision's reason by name, or ``review answer``.
    reason: str


@dataclass(frozen=True)
class LinkedPageCorrections:
    """The added pages, the review cases, and the linked pages without a decision."""

    corrections: tuple[SeedCorrection, ...]
    # Every linked-page case of this run, with its kept answer, for the review file.
    review_cases: tuple[ReviewCase, ...]
    # Linked pages not fetched yet, or whose fetch failed, sorted.
    waiting_page_keys: tuple[str, ...]
    # Linked pages the provider says do not exist, sorted.
    not_found_page_keys: tuple[str, ...]
    # Linked pages answers add as whole pages next to another page of their
    # provider in one entry, so none of them is added, sorted.
    conflicting_page_keys: tuple[str, ...] = ()


def find_entries_without_mal_page(entries: Iterable[WorkEntry]) -> list[WorkEntry]:
    """Return the entries holding no MAL page, in the order given.

    A work is what MAL lists, and an entry's anime id comes from its MAL page
    first, so an entry without one is where a missing MAL page matters most.
    """
    return [
        entry
        for entry in entries
        if not any(get_page_provider(key) == "mal" for key in entry.pages)
    ]


def list_entries_with_placed_pages(
    entries: Iterable[WorkEntry],
    doubled_provider_corrections: DoubledProviderCorrections,
    known_works: KnownWorks,
    store: ProviderPageStore,
) -> list[CorrectedEntry]:
    """Return the entries whose links can be read, as the corrections leave them.

    Args:
        entries: The entries whose links to read.
        doubled_provider_corrections: The corrections and review cases of the
            entries with doubled providers.
        known_works: Every seed entry.
        store: The fetched pages.

    Returns:
        Each seed entry once, corrected where the corrections change it: the
        given entries in their order, then the entries pages moved into. An
        entry with an unanswered doubled-provider case is left out, and so are
        new entries.
    """
    unplaced_reference_pages = {
        reference_page
        for review_case in doubled_provider_corrections.review_cases
        if review_case.accepted_answer is None
        for reference_page in review_case.entries
    }
    corrected_entries = {
        corrected.seed_entry_reference_page: corrected
        for corrected in build_corrected_entries(
            doubled_provider_corrections.corrections, known_works, store
        )
        if corrected.seed_entry_reference_page is not None
    }
    placed_entries = {
        entry.reference_page: corrected_entries.get(
            entry.reference_page, CorrectedEntry(entry.reference_page, entry)
        )
        for entry in entries
    } | corrected_entries
    return [
        placed_entry
        for reference_page, placed_entry in placed_entries.items()
        if reference_page not in unplaced_reference_pages
    ]


def find_linked_pages(
    entries: Iterable[CorrectedEntry],
    store: ProviderPageStore,
    known_works: KnownWorks,
) -> dict[str, tuple[LinkingSeedEntry, ...]]:
    """Return each page the entries' pages link that no seed entry holds.

    Args:
        entries: The entries whose links to read, as
            ``list_entries_with_placed_pages`` gives them.
        store: The fetched pages, whose links are read.
        known_works: Every seed entry.

    Returns:
        Each linked page's key, sorted, mapped to the entries linking it. A
        new entry, which no seed entry stands for, links nothing.
    """
    linking_entries_by_page: defaultdict[str, list[LinkingSeedEntry]] = defaultdict(
        list
    )
    for corrected in entries:
        if corrected.seed_entry_reference_page is None:
            continue
        entry_pages = store.get_entry_pages(corrected.entry)
        linked_page_keys = find_unheld_linked_pages(
            corrected.entry, entry_pages, known_works, FETCHED_PROVIDERS
        )
        linking_entry = LinkingEntry(
            corrected.entry,
            get_entry_episode_count(corrected.entry, entry_pages),
            linked_page_keys,
        )
        for key in linked_page_keys:
            linking_entries_by_page[key].append(
                LinkingSeedEntry(corrected.seed_entry_reference_page, linking_entry)
            )
    return {
        key: tuple(linking_entries_by_page[key])
        for key in sorted(linking_entries_by_page)
    }


def correct_linked_pages(
    linked_pages: Mapping[str, Sequence[LinkingSeedEntry]],
    store: ProviderPageStore,
    review_file: ReviewFile,
) -> LinkedPageCorrections:
    """Decide each fetched linked page, and turn additions into corrections.

    Args:
        linked_pages: Each linked page's key mapped to the entries linking it,
            as ``find_linked_pages`` gives them.
        store: The fetched pages, holding the linked pages fetched so far.
        review_file: The review file, for the answers kept in it.

    Returns:
        The corrections adding pages, the review cases, and the pages not added.
    """
    page_additions: list[_PageAddition] = []
    review_cases: list[ReviewCase] = []
    waiting_page_keys: list[str] = []
    not_found_page_keys: list[str] = []
    for key, linking_seed_entries in linked_pages.items():
        page = store.get_page(key)
        if page is None:
            waiting_page_keys.append(key)
            continue
        decision = decide_linked_page(
            page, (linking.linking_entry for linking in linking_seed_entries)
        )
        if decision.outcome is LinkedPageOutcome.SKIP:
            not_found_page_keys.append(key)
            continue
        review_case, page_addition = _place_linked_page(
            decision, page, store, review_file
        )
        if review_case is not None:
            review_cases.append(review_case)
        if page_addition is not None:
            page_additions.append(page_addition)
    kept_additions, conflicting_page_keys = _drop_conflicting_additions(page_additions)
    return LinkedPageCorrections(
        corrections=tuple(_build_addition_corrections(kept_additions, linked_pages)),
        review_cases=tuple(review_cases),
        waiting_page_keys=tuple(waiting_page_keys),
        not_found_page_keys=tuple(not_found_page_keys),
        conflicting_page_keys=conflicting_page_keys,
    )


def _place_linked_page(
    decision: LinkedPageDecision,
    page: ProviderPage,
    store: ProviderPageStore,
    review_file: ReviewFile,
) -> tuple[ReviewCase | None, _PageAddition | None]:
    """Turn a decision to add or review a linked page into its addition or review case.

    Returns:
        The review case with its kept answer, None for a page the rules add;
        and the addition, when the rules or an accepted answer add the page.
    """
    if decision.outcome is LinkedPageOutcome.ADD:
        [entry] = decision.linking_entries
        reason = decision.reason.name
        return None, _PageAddition(
            page, entry.reference_page, is_part_page=False, reason=reason
        )
    entry_pages = (
        entry_page
        for entry in decision.linking_entries
        for entry_page in store.get_entry_pages(entry)
    )
    review_case = review_file.attach_kept_answer(
        ReviewCase.from_linked_page(decision, [page, *entry_pages])
    )
    addition = review_case.accepted_answer
    if not isinstance(addition, LinkedPageAddition):
        return review_case, None
    return review_case, _PageAddition(
        page, addition.entry_reference_page, addition.is_part_page, REVIEW_ANSWER_REASON
    )


def _drop_conflicting_additions(
    page_additions: Sequence[_PageAddition],
) -> tuple[list[_PageAddition], tuple[str, ...]]:
    """Leave out whole pages added two or more per provider to one entry.

    A work holds one whole page of each provider but can hold several part
    pages, so two whole pages of one provider joining one entry would make a
    doubled provider. The rules never add two (an entry linking two such pages
    sends both to review), so only answers can; neither page is added, and a
    warning names the entry.

    Returns:
        The additions kept, and the sorted keys of the pages left out.
    """
    whole_page_counts = Counter(
        (page_addition.entry_reference_page, get_page_provider(page_addition.page.key))
        for page_addition in page_additions
        if not page_addition.is_part_page
    )
    kept_additions: list[_PageAddition] = []
    conflicting_page_keys: list[str] = []
    for page_addition in page_additions:
        entry_provider = (
            page_addition.entry_reference_page,
            get_page_provider(page_addition.page.key),
        )
        if not page_addition.is_part_page and whole_page_counts[entry_provider] > 1:
            logger.warning(
                f"Not adding {page_addition.page.key}: answers add two or more "
                f"pages of its provider to {page_addition.entry_reference_page}"
            )
            conflicting_page_keys.append(page_addition.page.key)
        else:
            kept_additions.append(page_addition)
    return kept_additions, tuple(sorted(conflicting_page_keys))


def _build_addition_corrections(
    page_additions: Iterable[_PageAddition],
    linked_pages: Mapping[str, Sequence[LinkingSeedEntry]],
) -> list[SeedCorrection]:
    """Turn additions into corrections, each naming the seed entry the page joins.

    The seed entry is named by its own reference page, so the cleaned seed
    finds it whatever the corrections did to the entry's reference page.
    """
    seed_reference_page_by_entry = {
        linking.linking_entry.entry.reference_page: linking.seed_entry_reference_page
        for linking_seed_entries in linked_pages.values()
        for linking in linking_seed_entries
    }
    return [
        SeedCorrection(
            entry_reference_page=seed_reference_page_by_entry[
                page_addition.entry_reference_page
            ],
            page_key=page_addition.page.key,
            address=page_addition.page.address,
            work=ENTRY_WORK,
            is_added_page=True,
            is_part_page=page_addition.is_part_page,
            reason=page_addition.reason,
        )
        for page_addition in page_additions
    ]
