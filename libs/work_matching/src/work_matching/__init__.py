"""Decide which provider pages belong to the same work.

Seed cleaning uses this now; seed link filling and discovery will use it later.
Terms are as in ``GLOSSARY.md``. This library imports nothing from enrichment.
"""

from work_matching.answer_rules import LinkedPageAnswerRule, PlacementRule
from work_matching.answer_words import (
    MAIN_MARK,
    NO_MARK,
    PART_MARK,
    describe_answer_word,
)
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
    check_group_answer_placements,
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
    LinkedPageAnswerCheck,
    LinkedPageDecision,
    LinkedPageOutcome,
    LinkedPageReason,
    LinkingEntryDetails,
    NoLinkingEntryError,
    check_linked_page_answer,
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
    PlacementCheck,
    check_answer_placements,
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
from work_matching.review_case_reasons import describe_case_reasons
from work_matching.review_cases import ReviewCase, ReviewCaseKind
from work_matching.review_choices import (
    ChoiceOption,
    ChoiceRule,
    GroupPageChoices,
    LinkedPageConflict,
    choose_group_page_place,
    find_linked_page_conflicts,
    list_group_page_choices,
    list_linked_page_options,
    list_placeable_case_page_keys,
    list_required_page_keys,
)
from work_matching.review_file import (
    DuplicateCaseError,
    ReviewFile,
    SavedAnswerChangedError,
    UnknownCaseError,
)
from work_matching.review_file_lock import (
    ReviewFileLockedError,
    hold_review_file_lock,
    is_review_file_locked,
)
from work_matching.split_work_check import (
    SplitWorkReviewReason,
    list_split_work_review_reasons,
)
from work_matching.surviving_entries import find_surviving_entries
from work_matching.work_entry import WorkEntry

__all__ = [
    "MAIN_MARK",
    "NO_MARK",
    "PART_MARK",
    "ChoiceOption",
    "ChoiceRule",
    "GroupPageChoices",
    "LinkedPageConflict",
    "ReviewFileLockedError",
    "choose_group_page_place",
    "describe_answer_word",
    "describe_case_reasons",
    "find_linked_page_conflicts",
    "hold_review_file_lock",
    "is_review_file_locked",
    "list_group_page_choices",
    "list_linked_page_options",
    "list_placeable_case_page_keys",
    "list_required_page_keys",
    "LinkedPageAnswerCheck",
    "LinkedPageAnswerRule",
    "PlacementCheck",
    "PlacementRule",
    "check_answer_placements",
    "check_group_answer_placements",
    "check_linked_page_answer",
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
    "ReviewCaseKind",
    "SavedAnswerChangedError",
    "UnknownCaseError",
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
