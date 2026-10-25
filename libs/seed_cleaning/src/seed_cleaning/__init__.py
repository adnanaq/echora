"""Find, fetch and correct the seed entries that need checking before enrichment.

Seed cleaning reads the downloaded seed release, never changes it, and lists
the entries to check: entries with a doubled provider and groups of split-work
candidates, merged into entry groups. It fetches every page of them into the
provider page store, where the work matching rules decide from their links, and
records where each page of an entry group ends up as seed corrections. It also adds
the pages entries link that no seed entry holds, once checked. The library is
deleted once discovery replaces the seed.
"""

from seed_cleaning.corrected_entries import (
    CorrectedEntry,
    UnknownSeedEntryError,
    build_corrected_entries,
)
from seed_cleaning.doubled_provider_entries import find_doubled_provider_entries
from seed_cleaning.entry_groups import find_entry_groups
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
    write_corrections,
)
from seed_cleaning.split_work_candidates import (
    SplitWorkCandidates,
    find_split_work_candidates,
)

__all__ = [
    "FETCHED_PROVIDERS",
    "LINK_CARRYING_PROVIDERS",
    "REVIEW_ANSWER_REASON",
    "CorrectedEntry",
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
    "find_entry_groups",
    "correct_linked_pages",
    "find_entries_without_mal_page",
    "find_linking_seed_entries",
    "list_entries_with_placed_pages",
    "build_corrected_entries",
    "find_doubled_provider_entries",
    "find_split_work_candidates",
    "write_corrections",
]
