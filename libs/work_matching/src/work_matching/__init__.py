"""Decide which provider pages belong to the same work.

Seed cleaning uses this now; seed link filling and discovery will use it later.
Terms are as in ``GLOSSARY.md``. This library imports nothing from enrichment.
"""

from work_matching.comparable_title import fold_title_for_comparison
from work_matching.doubled_provider_check import (
    DoubledProviderDecision,
    Outcome,
    Reason,
    UnrelatedPage,
    decide_doubled_providers,
    find_doubled_provider_page_keys,
    get_decided_provider,
    get_seed_type,
    is_page_details_matching,
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
from work_matching.page_placement import (
    ENTRY_WORK,
    OTHER_WORK,
    PagePlacement,
    find_move_target,
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
    SplitWorkDecision,
    SplitWorkReason,
    decide_split_work,
)
from work_matching.work_entry import WorkEntry

__all__ = [
    "ENTRY_WORK",
    "OTHER_WORK",
    "SKIP_ANSWER",
    "LinkedPageAddition",
    "LinkedPageDecision",
    "LinkedPageOutcome",
    "LinkedPageReason",
    "LinkingEntryDetails",
    "NoLinkingEntryError",
    "PagePlacement",
    "DoubledProviderDecision",
    "DuplicateCaseError",
    "KnownWorkEntries",
    "KnownWorkEntriesFromSeed",
    "Outcome",
    "ProviderPage",
    "ProviderPageLinks",
    "ProviderPageStore",
    "Reason",
    "ReviewCase",
    "ReviewFile",
    "SplitWorkDecision",
    "SplitWorkReason",
    "UnrelatedPage",
    "WorkEntry",
    "read_answer_placements",
    "list_placeable_page_keys",
    "fold_title_for_comparison",
    "decide_doubled_providers",
    "get_decided_provider",
    "decide_split_work",
    "find_doubled_provider_page_keys",
    "find_move_target",
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
