"""Find, fetch and correct the seed entries that need checking before enrichment.

Seed cleaning reads the downloaded seed release, never changes it, and lists
the entries to check in two passes: entries with a doubled provider first, then
groups of split-work candidates found on the entries as the first pass leaves
them. The library fetches their pages into the provider page store, where the
work matching rules decide from their links, and records where each page of an
entry group ends up as seed corrections. Seed cleaning also adds the pages
entries link that no seed entry holds, once checked. The library is deleted once
discovery replaces the seed.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from seed_cleaning.cleaning_report import (
    ANSWER_REASON_SENTENCE,
    CORRECTION_COUNT_NAMES,
    build_entries_to_check_document,
    count_corrected_entries,
    count_corrections,
    describe_correction_reason,
    describe_linked_page_outcomes,
    find_correction_change,
)
from seed_cleaning.group_placement import correct_entry_groups
from seed_cleaning.models import (
    CORRECTIONS_FILE_NAME,
    ENTRIES_TO_CHECK_FILE_NAME,
    REVIEW_FILE_NAME,
    RUN_RESULT_FILE_NAME,
    CorrectedEntry,
    CorrectedSeed,
    EntryGroup,
    EntryIdentity,
    LinkedPageCorrections,
    LinkedPageGroup,
    LinkingSeedEntry,
    PageOrigin,
    PlacementCorrections,
    SeedCleaningRunResult,
    SeedCorrection,
    SeedFetchSummary,
    SplitWorkCandidates,
    SplitWorkLink,
    SplitWorkLinks,
    SplitWorkPass,
    read_seed_cleaning_run_result,
)
from seed_cleaning.not_listed_searches import NotListedSearchFile
from seed_cleaning.seed_corrections import (
    REVIEW_ANSWER_REASON,
    follow_named_entry_pages,
    merge_corrections,
    write_corrections,
)
from seed_cleaning.split_work_candidates import find_split_work_candidates
from seed_cleaning.split_work_links import find_split_work_links

if TYPE_CHECKING:
    from seed_cleaning.corrected_entries import (
        UnknownSeedEntryError,
        build_corrected_entries,
        mark_anime_id_pages,
    )
    from seed_cleaning.corrected_seed import (
        UnknownPageOriginError,
        build_corrected_seed,
        translate_corrected_seed_corrections,
    )
    from seed_cleaning.linked_page_additions import (
        correct_linked_pages,
        find_entries_without_mal_page,
        find_linking_seed_entries,
        group_linked_pages_by_current_key,
        list_entries_with_placed_pages,
        rename_linked_pages_to_current_keys,
    )
    from seed_cleaning.seed_cleaning_run import (
        find_doubled_provider_entries,
        run_seed_cleaning,
    )
    from seed_cleaning.seed_fetch import (
        fetch_seed_and_linked_pages,
        list_unheld_linked_page_titles,
        read_seed_release_name,
    )
    from seed_cleaning.split_work_pass import prepare_split_work_pass

_LAZY_ENRICHMENT_EXPORTS: dict[str, str] = {
    "find_doubled_provider_entries": "seed_cleaning.seed_cleaning_run",
    "run_seed_cleaning": "seed_cleaning.seed_cleaning_run",
    "prepare_split_work_pass": "seed_cleaning.split_work_pass",
    "UnknownPageOriginError": "seed_cleaning.corrected_seed",
    "build_corrected_seed": "seed_cleaning.corrected_seed",
    "translate_corrected_seed_corrections": "seed_cleaning.corrected_seed",
    "UnknownSeedEntryError": "seed_cleaning.corrected_entries",
    "build_corrected_entries": "seed_cleaning.corrected_entries",
    "mark_anime_id_pages": "seed_cleaning.corrected_entries",
    "correct_linked_pages": "seed_cleaning.linked_page_additions",
    "find_entries_without_mal_page": "seed_cleaning.linked_page_additions",
    "find_linking_seed_entries": "seed_cleaning.linked_page_additions",
    "group_linked_pages_by_current_key": "seed_cleaning.linked_page_additions",
    "list_entries_with_placed_pages": "seed_cleaning.linked_page_additions",
    "rename_linked_pages_to_current_keys": "seed_cleaning.linked_page_additions",
    "fetch_seed_and_linked_pages": "seed_cleaning.seed_fetch",
    "list_unheld_linked_page_titles": "seed_cleaning.seed_fetch",
    "read_seed_release_name": "seed_cleaning.seed_fetch",
}


def __getattr__(name: str) -> Any:
    """Resolve enrichment-backed seed-cleaning exports lazily on access.

    Args:
        name: Attribute name requested from ``seed_cleaning``.

    Returns:
        The exported symbol from the corresponding submodule.

    Raises:
        AttributeError: If ``name`` is not a public export of ``seed_cleaning``.
    """
    module_name = _LAZY_ENRICHMENT_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    return getattr(import_module(module_name), name)


__all__ = [
    "REVIEW_FILE_NAME",
    "RUN_RESULT_FILE_NAME",
    "SeedCleaningRunResult",
    "read_seed_cleaning_run_result",
    "run_seed_cleaning",
    "ANSWER_REASON_SENTENCE",
    "CORRECTION_COUNT_NAMES",
    "CORRECTIONS_FILE_NAME",
    "ENTRIES_TO_CHECK_FILE_NAME",
    "build_entries_to_check_document",
    "count_corrected_entries",
    "count_corrections",
    "describe_correction_reason",
    "find_correction_change",
    "describe_linked_page_outcomes",
    "REVIEW_ANSWER_REASON",
    "CorrectedEntry",
    "CorrectedSeed",
    "EntryGroup",
    "EntryIdentity",
    "PageOrigin",
    "SplitWorkLink",
    "SplitWorkLinks",
    "SplitWorkPass",
    "UnknownPageOriginError",
    "PlacementCorrections",
    "LinkedPageCorrections",
    "LinkedPageGroup",
    "LinkingSeedEntry",
    "group_linked_pages_by_current_key",
    "rename_linked_pages_to_current_keys",
    "NotListedSearchFile",
    "SeedFetchSummary",
    "fetch_seed_and_linked_pages",
    "list_unheld_linked_page_titles",
    "read_seed_release_name",
    "SeedCorrection",
    "SplitWorkCandidates",
    "UnknownSeedEntryError",
    "correct_entry_groups",
    "build_corrected_seed",
    "translate_corrected_seed_corrections",
    "find_split_work_links",
    "correct_linked_pages",
    "find_entries_without_mal_page",
    "find_linking_seed_entries",
    "list_entries_with_placed_pages",
    "build_corrected_entries",
    "mark_anime_id_pages",
    "find_doubled_provider_entries",
    "find_split_work_candidates",
    "follow_named_entry_pages",
    "merge_corrections",
    "prepare_split_work_pass",
    "write_corrections",
]
