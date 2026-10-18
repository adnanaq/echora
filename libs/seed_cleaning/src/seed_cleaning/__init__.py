"""Find, fetch and correct the seed entries that need checking before enrichment.

Seed cleaning reads the downloaded seed release, never changes it, and lists
the entries to check: entries with a doubled provider and groups of split-work
candidates. It fetches every page of them into the provider page store, where
the work matching rules decide from their links, and records where each page
of an entry with doubled providers ends up as seed corrections. The library is
deleted once discovery replaces the seed.
"""

from seed_cleaning.doubled_provider_entries import find_doubled_provider_entries
from seed_cleaning.provider_page_fetcher import (
    FetchOutcome,
    FetchSummary,
    ProviderPageFetcher,
)
from seed_cleaning.seed_corrections import (
    REVIEW_ANSWER_REASON,
    CorrectedEntry,
    DoubledProviderCorrections,
    SeedCorrection,
    UnknownSeedEntryError,
    build_corrected_entries,
    correct_doubled_provider_entries,
    write_corrections,
)
from seed_cleaning.split_work_candidates import (
    SplitWorkCandidates,
    find_split_work_candidates,
)

__all__ = [
    "REVIEW_ANSWER_REASON",
    "CorrectedEntry",
    "DoubledProviderCorrections",
    "FetchOutcome",
    "FetchSummary",
    "ProviderPageFetcher",
    "SeedCorrection",
    "SplitWorkCandidates",
    "UnknownSeedEntryError",
    "correct_doubled_provider_entries",
    "build_corrected_entries",
    "find_doubled_provider_entries",
    "find_split_work_candidates",
    "write_corrections",
]
