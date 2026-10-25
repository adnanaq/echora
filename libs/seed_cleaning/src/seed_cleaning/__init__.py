"""Find, fetch and correct the seed entries that need checking before enrichment.

Seed cleaning reads the downloaded seed release, never changes it, and lists
the entries to check in two passes: entries with a doubled provider first, then
groups of split-work candidates found on the entries as the first pass leaves
them. It fetches their pages into the provider page store, where the work
matching rules decide from their links, and records where each page of an entry
group ends up as seed corrections. It also adds the pages entries link that no
seed entry holds, once checked. The library is deleted once discovery replaces
the seed.
"""

from seed_cleaning.cleaning_report import (
    CORRECTION_COUNT_NAMES,
    build_entries_to_check_document,
    count_corrections,
)
from seed_cleaning.corrected_entries import (
    CorrectedEntry,
    UnknownSeedEntryError,
    build_corrected_entries,
)
from seed_cleaning.corrected_seed import (
    CorrectedSeed,
    EntryIdentity,
    PageOrigin,
    UnknownPageOriginError,
    build_corrected_seed,
)
from seed_cleaning.doubled_provider_entries import find_doubled_provider_entries
from seed_cleaning.entry_groups import EntryGroup
from seed_cleaning.group_placement import correct_entry_groups
from seed_cleaning.linked_page_additions import (
    LINK_CARRYING_PROVIDERS,
    LinkedPageCorrections,
    correct_linked_pages,
    find_entries_without_mal_page,
    find_linking_seed_entries,
    list_entries_with_placed_pages,
)
from seed_cleaning.linked_page_groups import (
    LinkedPageGroup,
    LinkingSeedEntry,
    group_linked_pages_by_current_key,
    rename_linked_pages_to_current_keys,
)
from seed_cleaning.provider_page_fetcher import (
    FETCHED_PROVIDERS,
    FetchOutcome,
    FetchSummary,
    ProviderPageFetcher,
)
from seed_cleaning.seed_corrections import (
    REVIEW_ANSWER_REASON,
    PlacementCorrections,
    SeedCorrection,
    merge_corrections,
    write_corrections,
)
from seed_cleaning.split_work_candidates import (
    SplitWorkCandidates,
    find_split_work_candidates,
)
from seed_cleaning.split_work_links import (
    SplitWorkLink,
    SplitWorkLinks,
    find_split_work_links,
)
from seed_cleaning.split_work_pass import SplitWorkPass, prepare_split_work_pass

__all__ = [
    "CORRECTION_COUNT_NAMES",
    "build_entries_to_check_document",
    "count_corrections",
    "FETCHED_PROVIDERS",
    "LINK_CARRYING_PROVIDERS",
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
    "FetchOutcome",
    "FetchSummary",
    "LinkedPageCorrections",
    "LinkedPageGroup",
    "LinkingSeedEntry",
    "group_linked_pages_by_current_key",
    "rename_linked_pages_to_current_keys",
    "ProviderPageFetcher",
    "SeedCorrection",
    "SplitWorkCandidates",
    "UnknownSeedEntryError",
    "correct_entry_groups",
    "build_corrected_seed",
    "find_split_work_links",
    "correct_linked_pages",
    "find_entries_without_mal_page",
    "find_linking_seed_entries",
    "list_entries_with_placed_pages",
    "build_corrected_entries",
    "find_doubled_provider_entries",
    "find_split_work_candidates",
    "merge_corrections",
    "prepare_split_work_pass",
    "write_corrections",
]
