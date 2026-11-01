"""Decide which provider pages belong to the same work.

Seed cleaning uses this now; seed link filling and discovery will use it later.
Terms are as in ``GLOSSARY.md``. This library imports nothing from enrichment.
"""

from work_matching.comparable_title import fold_title_for_comparison
from work_matching.doubled_provider_check import decide_doubled_providers
from work_matching.doubled_provider_decision import (
    DoubledProviderDecision,
    DoubledProviderOutcome,
    DoubledProviderReviewReason,
    UnrelatedPage,
)
from work_matching.doubled_provider_pages import (
    find_doubled_provider_page_keys,
    get_decided_provider,
)
from work_matching.entry_group import (
    combine_group_entries,
    list_mal_page_review_reasons,
    read_group_answer_placements,
    suggest_group_page_placements,
)
from work_matching.file_replacement import replace_file_lines
from work_matching.known_work_entries import KnownWorkEntries
from work_matching.known_work_entries_from_seed import KnownWorkEntriesFromSeed
from work_matching.linked_page_check import (
    SKIP_ANSWER,
    LinkedPageAddition,
    LinkedPageDecision,
    LinkedPageOutcome,
    LinkedPageReason,
    LinkingEntryDetails,
    NoLinkingEntryError,
    decide_linked_page,
    find_unheld_linked_page_keys,
    get_entry_episode_count,
    read_linked_page_answer,
    suggest_linked_page_answer,
)
from work_matching.page_details_comparison import (
    MalPageComparison,
    compare_page_with_mal_page,
    get_seed_type,
    is_page_details_matching,
)
from work_matching.page_placement import (
    FIRST_WORK,
    OTHER_WORK,
    PagePlacement,
    PlacedPage,
    find_page_placement,
    list_placeable_page_keys,
    place_pages_without_review,
    read_answer_placements,
    suggest_page_placements,
)
from work_matching.provider_page import ProviderPage, index_pages_by_current_key
from work_matching.provider_page_key import (
    build_page_address,
    get_page_key,
    get_page_provider,
    rank_by_provider_trust,
)
from work_matching.provider_page_links import ProviderPageLinks
from work_matching.provider_page_store import ProviderPageStore
from work_matching.review_cases import DuplicateCaseError, ReviewCase, ReviewFile
from work_matching.split_work_check import (
    SplitWorkReviewReason,
    list_split_work_review_reasons,
)
from work_matching.surviving_entries import find_surviving_entries
from work_matching.work_entry import WorkEntry

__all__ = [
    "FIRST_WORK",
    "OTHER_WORK",
    "SKIP_ANSWER",
    "LinkedPageAddition",
    "LinkedPageDecision",
    "LinkedPageOutcome",
    "LinkedPageReason",
    "LinkingEntryDetails",
    "NoLinkingEntryError",
    "PagePlacement",
    "PlacedPage",
    "find_page_placement",
    "DoubledProviderDecision",
    "DuplicateCaseError",
    "KnownWorkEntries",
    "KnownWorkEntriesFromSeed",
    "DoubledProviderOutcome",
    "ProviderPage",
    "ProviderPageLinks",
    "ProviderPageStore",
    "DoubledProviderReviewReason",
    "ReviewCase",
    "ReviewFile",
    "SplitWorkReviewReason",
    "UnrelatedPage",
    "WorkEntry",
    "read_answer_placements",
    "list_placeable_page_keys",
    "fold_title_for_comparison",
    "decide_doubled_providers",
    "get_decided_provider",
    "find_doubled_provider_page_keys",
    "find_surviving_entries",
    "combine_group_entries",
    "list_split_work_review_reasons",
    "read_group_answer_placements",
    "suggest_group_page_placements",
    "list_mal_page_review_reasons",
    "MalPageComparison",
    "compare_page_with_mal_page",
    "index_pages_by_current_key",
    "get_page_provider",
    "get_page_key",
    "rank_by_provider_trust",
    "replace_file_lines",
    "get_seed_type",
    "place_pages_without_review",
    "suggest_page_placements",
    "build_page_address",
    "decide_linked_page",
    "find_unheld_linked_page_keys",
    "get_entry_episode_count",
    "is_page_details_matching",
    "read_linked_page_answer",
    "suggest_linked_page_answer",
]
