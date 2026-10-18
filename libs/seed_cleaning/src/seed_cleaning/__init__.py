"""Find and fetch the seed entries that need checking before enrichment.

Seed cleaning reads the downloaded seed release, never changes it, and lists
the entries to check: entries with a doubled provider and groups of split-work
candidates. It fetches every page of them into the provider page store, where
the work matching rules decide from their links. The library is deleted once
discovery replaces the seed.
"""

from seed_cleaning.doubled_provider_entries import doubled_provider_entries
from seed_cleaning.provider_page_fetcher import (
    FetchOutcome,
    FetchSummary,
    ProviderPageFetcher,
)
from seed_cleaning.split_work_candidates import (
    SplitWorkCandidates,
    split_work_candidates,
)

__all__ = [
    "FetchOutcome",
    "FetchSummary",
    "ProviderPageFetcher",
    "SplitWorkCandidates",
    "doubled_provider_entries",
    "split_work_candidates",
]
