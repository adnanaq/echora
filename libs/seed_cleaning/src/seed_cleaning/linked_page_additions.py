"""Add the pages entries link that no seed entry holds, as seed corrections.

Providers link pages the seed lacks, most often a MAL page linked from an
entry's AniList, Kitsu or AniDB page. Each such linked page is fetched and
checked by the work matching rules (``linked_page_check``). The page joins the
entry linking it when the checks pass or a person's answer accepts it.

| Rule | Reason for the rule |
| -- | -- |
| Read the links of each entry to check and each entry without a MAL page, as the doubled-provider corrections leave the entry. | A page moved to another entry is read under that entry. |
| Leave out an entry whose doubled providers wait for a review answer. | The entry's pages are not placed yet, so a link on one of them may belong to another work. |
| Leave out new entries built from answers. | A correction must name a seed entry. |
| Keep only links to anime pages of providers whose pages are fetched. | Each linked page can then be checked live; other providers' pages come from seed releases. |
| Group the linked pages by the current key their fetch reports. | One page can be linked under several spellings, such as a Kitsu slug and the Kitsu number. |
| Give a linked page not fetched yet, or whose fetch failed, no decision. | The next run fetches the page again. |
| Never follow an added page's own links. | Corrections are rebuilt from the downloaded seed on every run, so additions never chain. |
| Add neither page when answers add two whole pages of one provider to one entry. | Both pages would make a doubled provider; a warning names the entry. |
| Mark an added page the entry's anime id now comes from (`is_anime_id_page`). | Enrichment derives the id from the entry's pages, MAL first, so the search entry under the old id must be replaced. |
"""

import logging
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

from enrichment.pipeline.link_rules import find_anime_id_page_key
from seed_cleaning.linked_page_groups import (
    LinkingSeedEntry,
    group_linked_pages_by_current_key,
)
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
    KnownWorkEntries,
    LinkedPageAddition,
    LinkedPageDecision,
    LinkedPageOutcome,
    LinkedPageReason,
    LinkingEntryDetails,
    ProviderPage,
    ProviderPageStore,
    ReviewCase,
    ReviewFile,
    WorkEntry,
    build_page_address,
    decide_linked_page,
    find_unheld_linked_page_keys,
    get_entry_episode_count,
    get_page_provider,
)

logger = logging.getLogger(__name__)

# The providers whose pages carry links to other providers' pages, so fetching
# them for their links alone is worth it. Anime-Planet and AniSearch pages
# carry none.
LINK_CARRYING_PROVIDERS: frozenset[str] = frozenset({"anilist", "kitsu", "anidb"})


class _PageAddition(NamedTuple):
    """Hold a linked page to add, the entry it joins, whether as a part, and why."""

    # The page's current key and the address the seed spells it with, which
    # can differ from the link's own spelling, such as a Kitsu slug.
    page_key: str
    address: str
    # The reference page of the work entry the page joins, as decided.
    entry_reference_page: str
    is_part_page: bool
    # The decision's reason by name, or ``review answer``.
    reason: str


@dataclass(frozen=True)
class LinkedPageCorrections:
    """Hold the added pages, the review cases, and the linked pages not added."""

    corrections: tuple[SeedCorrection, ...]
    # Every linked-page case of this run, with its kept answer, for the review file.
    review_cases: tuple[ReviewCase, ...]
    # Linked pages not fetched yet, or whose fetch failed, sorted.
    waiting_page_keys: tuple[str, ...]
    # Linked pages the provider says do not exist, sorted.
    not_found_page_keys: tuple[str, ...]
    # Linked pages a seed entry already holds under another address, by their
    # current key, sorted.
    held_page_keys: tuple[str, ...] = ()
    # Linked pages answers add as whole pages next to another page of their
    # provider in one entry, so none of them is added, sorted.
    conflicting_page_keys: tuple[str, ...] = ()


def find_entries_without_mal_page(entries: Iterable[WorkEntry]) -> list[WorkEntry]:
    """Return the entries holding no MAL page, in the order given.

    A work is what MAL lists. An entry's anime id comes from the entry's MAL
    page first. A missing MAL page therefore matters most for these entries.

    Args:
        entries: The entries to filter, such as every seed entry.

    Returns:
        The entries without a MAL page.
    """
    return [
        entry
        for entry in entries
        if not any(
            get_page_provider(key) == "mal" for key in entry.addresses_by_page_key
        )
    ]


def list_entries_with_placed_pages(
    entries: Iterable[WorkEntry],
    doubled_provider_corrections: DoubledProviderCorrections,
    known_work_entries: KnownWorkEntries,
    store: ProviderPageStore,
) -> list[CorrectedEntry]:
    """Return the entries whose links can be read, as the corrections leave them.

    Args:
        entries: The entries whose links to read.
        doubled_provider_corrections: The corrections and review cases of the
            entries with doubled providers.
        known_work_entries: Every seed entry.
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
            doubled_provider_corrections.corrections, known_work_entries, store
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


def find_linking_seed_entries(
    entries: Iterable[CorrectedEntry],
    store: ProviderPageStore,
    known_work_entries: KnownWorkEntries,
) -> dict[str, tuple[LinkingSeedEntry, ...]]:
    """Return the entries linking each page that no seed entry holds, by page key.

    Args:
        entries: The entries whose links to read, as
            ``list_entries_with_placed_pages`` gives them.
        store: The fetched pages, whose links are read.
        known_work_entries: Every seed entry.

    Returns:
        Each linked page's key, sorted, mapped to the entries linking it. A
        new entry, which no seed entry stands for, links nothing.
    """
    linking_seed_entries_by_page_key: defaultdict[str, list[LinkingSeedEntry]] = (
        defaultdict(list)
    )
    for corrected in entries:
        if corrected.seed_entry_reference_page is None:
            continue
        entry_pages = store.get_entry_pages(corrected.entry)
        linked_page_keys = find_unheld_linked_page_keys(
            corrected.entry, entry_pages, known_work_entries, FETCHED_PROVIDERS
        )
        entry_details = LinkingEntryDetails(
            corrected.entry,
            get_entry_episode_count(corrected.entry, entry_pages),
            linked_page_keys,
        )
        for key in linked_page_keys:
            linking_seed_entries_by_page_key[key].append(
                LinkingSeedEntry(corrected.seed_entry_reference_page, entry_details)
            )
    return {
        key: tuple(linking_seed_entries_by_page_key[key])
        for key in sorted(linking_seed_entries_by_page_key)
    }


def correct_linked_pages(
    linking_seed_entries_by_page_key: Mapping[str, Sequence[LinkingSeedEntry]],
    store: ProviderPageStore,
    known_work_entries: KnownWorkEntries,
    review_file: ReviewFile,
) -> LinkedPageCorrections:
    """Decide each fetched linked page, and turn additions into corrections.

    Args:
        linking_seed_entries_by_page_key: Each linked page's key mapped to the
            entries linking it, as ``find_linking_seed_entries`` gives them.
        store: The fetched pages, holding the linked pages fetched so far.
        known_work_entries: Every seed entry, to tell a page held under another address.
        review_file: The review file, for the answers kept in it.

    Returns:
        The corrections adding pages, the review cases, and the pages not added.
    """
    linked_page_groups_by_current_key, waiting_page_keys = (
        group_linked_pages_by_current_key(linking_seed_entries_by_page_key, store)
    )
    page_additions: list[_PageAddition] = []
    review_cases: list[ReviewCase] = []
    skipped_page_keys: defaultdict[LinkedPageReason, list[str]] = defaultdict(list)
    for page, linking_seed_entries in linked_page_groups_by_current_key.values():
        decision = decide_linked_page(
            page,
            (
                linking_seed_entry.entry_details
                for linking_seed_entry in linking_seed_entries
            ),
            known_work_entries,
        )
        if decision.outcome is LinkedPageOutcome.SKIP:
            skipped_page_keys[decision.reason].append(decision.page_key)
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
        corrections=tuple(
            _build_addition_corrections(
                kept_additions, linking_seed_entries_by_page_key
            )
        ),
        review_cases=tuple(review_cases),
        waiting_page_keys=waiting_page_keys,
        not_found_page_keys=tuple(skipped_page_keys[LinkedPageReason.NOT_FOUND]),
        held_page_keys=tuple(
            skipped_page_keys[LinkedPageReason.HELD_UNDER_OTHER_ADDRESS]
        ),
        conflicting_page_keys=conflicting_page_keys,
    )


def _place_linked_page(
    decision: LinkedPageDecision,
    page: ProviderPage,
    store: ProviderPageStore,
    review_file: ReviewFile,
) -> tuple[ReviewCase | None, _PageAddition | None]:
    """Turn a decision to add or review a linked page into its addition or review case.

    Args:
        decision: A decision with the outcome ``ADD`` or ``REVIEW``.
        page: The linked page's fetched details.
        store: The fetched pages, for the details the review case shows.
        review_file: The review file, for the answer kept for the case.

    Returns:
        The review case with its kept answer, or None for a page the rules add.
        Then the addition, when the rules or an accepted answer add the page.
    """
    page_key = decision.page_key
    # The seed's spelling of the page's current key, not the link's.
    address = build_page_address(page_key) or page.address
    if decision.outcome is LinkedPageOutcome.ADD:
        # Only a page one entry links passes every check, so an added page has
        # exactly one linking entry (``decide_linked_page``).
        [entry] = decision.linking_entries
        reason = decision.reason.name
        return None, _PageAddition(
            page_key, address, entry.reference_page, is_part_page=False, reason=reason
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
        page_key,
        address,
        addition.entry_reference_page,
        addition.is_part_page,
        REVIEW_ANSWER_REASON,
    )


def _drop_conflicting_additions(
    page_additions: Sequence[_PageAddition],
) -> tuple[list[_PageAddition], tuple[str, ...]]:
    """Leave out whole pages added two or more per provider to one entry.

    A work holds one whole page of each provider but can hold several part
    pages. Two whole pages of one provider joining one entry would make a
    doubled provider. The rules never add two such pages, because an entry
    linking two of them sends both to review. Only answers can add two, so
    neither page is added and a warning names the entry.

    Args:
        page_additions: The additions from the rules and the accepted answers.

    Returns:
        The additions kept, and the sorted keys of the pages left out.
    """
    whole_page_counts = Counter(
        (page_addition.entry_reference_page, get_page_provider(page_addition.page_key))
        for page_addition in page_additions
        if not page_addition.is_part_page
    )
    kept_additions: list[_PageAddition] = []
    conflicting_page_keys: list[str] = []
    for page_addition in page_additions:
        entry_provider = (
            page_addition.entry_reference_page,
            get_page_provider(page_addition.page_key),
        )
        if not page_addition.is_part_page and whole_page_counts[entry_provider] > 1:
            logger.warning(
                f"Not adding {page_addition.page_key}: answers add two or more "
                f"pages of its provider to {page_addition.entry_reference_page}"
            )
            conflicting_page_keys.append(page_addition.page_key)
        else:
            kept_additions.append(page_addition)
    return kept_additions, tuple(sorted(conflicting_page_keys))


def _build_addition_corrections(
    page_additions: Sequence[_PageAddition],
    linking_seed_entries_by_page_key: Mapping[str, Sequence[LinkingSeedEntry]],
) -> list[SeedCorrection]:
    """Turn additions into corrections, each naming the seed entry the page joins.

    - Each correction names the seed entry by the seed entry's own reference
      page. The cleaned seed then finds the entry whatever the corrections did
      to the entry's current reference page.
    - A correction marks the added page the entry's anime id now comes from
      (``is_anime_id_page``). Enrichment derives the id from the entry's
      pages, MAL first (``find_anime_id_page_key``), so the search entry stored
      under the old id must be replaced.

    Args:
        page_additions: The additions kept.
        linking_seed_entries_by_page_key: Each linked page's key mapped to the
            entries linking it, for each entry's seed reference page and pages.

    Returns:
        One correction per addition, in the order given.
    """
    # Keyed by the corrected entry's reference page, which is unique: each page
    # belongs to one seed entry, and the corrections move a page, never copy it.
    linking_seed_entry_by_entry_reference_page = {
        linking_seed_entry.entry_details.entry.reference_page: linking_seed_entry
        for linking_seed_entries in linking_seed_entries_by_page_key.values()
        for linking_seed_entry in linking_seed_entries
    }
    added_page_keys_by_entry_reference_page: defaultdict[str, list[str]] = defaultdict(
        list
    )
    for page_addition in page_additions:
        added_page_keys_by_entry_reference_page[
            page_addition.entry_reference_page
        ].append(page_addition.page_key)
    anime_id_page_key_by_entry_reference_page = {
        reference_page: find_anime_id_page_key(
            [
                *linking_seed_entry_by_entry_reference_page[
                    reference_page
                ].entry_details.entry.addresses_by_page_key,
                *added_page_keys,
            ]
        )
        for reference_page, added_page_keys in (
            added_page_keys_by_entry_reference_page.items()
        )
    }
    return [
        SeedCorrection(
            entry_reference_page=linking_seed_entry_by_entry_reference_page[
                page_addition.entry_reference_page
            ].seed_entry_reference_page,
            page_key=page_addition.page_key,
            address=page_addition.address,
            placement_word=ENTRY_WORK,
            is_added_page=True,
            is_part_page=page_addition.is_part_page,
            is_anime_id_page=page_addition.page_key
            == anime_id_page_key_by_entry_reference_page[
                page_addition.entry_reference_page
            ],
            reason=page_addition.reason,
        )
        for page_addition in page_additions
    ]
