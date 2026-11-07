"""Decide which provider pages belong to the same work.

Seed cleaning uses this package now; seed link filling and discovery will use
this package later. Terms are as in ``GLOSSARY.md``. This library imports
nothing from enrichment.
"""

from work_matching.answer_rules import LinkedPageAnswerRule, PlacementRule
from work_matching.answer_words import describe_answer_word
from work_matching.comparable_title import fold_title_for_comparison
from work_matching.doubled_provider_check import decide_doubled_providers
from work_matching.doubled_provider_pages import (
    find_doubled_provider_page_keys,
    resolve_decided_provider,
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
    NoLinkingEntryError,
    check_linked_page_answer,
    decide_linked_page,
    find_entry_episode_count,
    find_unheld_linked_page_keys,
    read_linked_page_answer,
    suggest_linked_page_answer,
)
from work_matching.model_doubled_provider_decision import (
    DoubledProviderDecision,
    DoubledProviderOutcome,
    DoubledProviderReviewReason,
    UnrelatedPage,
)
from work_matching.model_linked_page_decision import (
    LinkedPageAddition,
    LinkedPageAnswerCheck,
    LinkedPageDecision,
    LinkedPageOutcome,
    LinkedPageReason,
    LinkingEntryDetails,
)
from work_matching.model_review_choices import (
    ChoiceOption,
    ChoiceRule,
    GroupPageChoices,
    LinkedPageConflict,
)
from work_matching.named_entries import (
    MAX_TARGET_ENTRY_SUGGESTIONS,
    list_named_entry_page_keys,
    resolve_named_entries,
    suggest_target_entries,
)
from work_matching.page_details_comparison import (
    MalPageComparison,
    compare_page_with_mal_page,
    is_page_details_matching,
    resolve_seed_type,
)
from work_matching.page_placement import (
    FIRST_WORK,
    MAIN_MARK,
    NO_MARK,
    OTHER_WORK,
    PART_MARK,
    PagePlacement,
    PlacedPage,
    PlacementCheck,
    check_answer_placements,
    find_page_placement,
    list_placeable_page_keys,
    place_pages_without_review,
    read_answer_placements,
    read_placement,
    write_placement,
)
from work_matching.placement_suggestions import suggest_page_placements
from work_matching.provider_page import ProviderPage, index_pages_by_current_page_key
from work_matching.provider_page_key import (
    build_page_address,
    is_mal_page,
    rank_by_provider_trust,
    read_page_provider,
    resolve_page_key,
)
from work_matching.provider_page_links import ProviderPageLinks
from work_matching.provider_page_store import ProviderPageStore
from work_matching.review_case_reasons import describe_case_reasons
from work_matching.review_cases import ReviewCase, ReviewCaseKind, TargetEntrySuggestion
from work_matching.review_choices import (
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
    "FIRST_WORK",
    "MAIN_MARK",
    "MAX_TARGET_ENTRY_SUGGESTIONS",
    "NO_MARK",
    "OTHER_WORK",
    "PART_MARK",
    "SKIP_ANSWER",
    "ChoiceOption",
    "ChoiceRule",
    "DoubledProviderDecision",
    "DoubledProviderOutcome",
    "DoubledProviderReviewReason",
    "DuplicateCaseError",
    "GroupPageChoices",
    "KnownWorkEntries",
    "KnownWorkEntriesFromSeed",
    "LinkedPageAddition",
    "LinkedPageAnswerCheck",
    "LinkedPageAnswerRule",
    "LinkedPageConflict",
    "LinkedPageDecision",
    "LinkedPageOutcome",
    "LinkedPageReason",
    "LinkingEntryDetails",
    "MalPageComparison",
    "NoLinkingEntryError",
    "PagePlacement",
    "PlacedPage",
    "PlacementCheck",
    "PlacementRule",
    "ProviderPage",
    "ProviderPageLinks",
    "ProviderPageStore",
    "ReviewCase",
    "ReviewCaseKind",
    "ReviewFile",
    "ReviewFileLockedError",
    "SavedAnswerChangedError",
    "SplitWorkReviewReason",
    "TargetEntrySuggestion",
    "UnknownCaseError",
    "UnrelatedPage",
    "WorkEntry",
    "build_page_address",
    "check_answer_placements",
    "check_group_answer_placements",
    "check_linked_page_answer",
    "choose_group_page_place",
    "combine_group_entries",
    "compare_page_with_mal_page",
    "decide_doubled_providers",
    "decide_linked_page",
    "describe_answer_word",
    "describe_case_reasons",
    "find_doubled_provider_page_keys",
    "find_entry_episode_count",
    "find_linked_page_conflicts",
    "find_page_placement",
    "find_surviving_entries",
    "find_unheld_linked_page_keys",
    "fold_title_for_comparison",
    "hold_review_file_lock",
    "index_pages_by_current_page_key",
    "is_mal_page",
    "is_page_details_matching",
    "is_review_file_locked",
    "list_group_page_choices",
    "list_linked_page_options",
    "list_mal_page_review_reasons",
    "list_named_entry_page_keys",
    "list_placeable_case_page_keys",
    "list_placeable_page_keys",
    "list_required_page_keys",
    "list_split_work_review_reasons",
    "place_pages_without_review",
    "rank_by_provider_trust",
    "read_answer_placements",
    "read_group_answer_placements",
    "read_linked_page_answer",
    "read_page_provider",
    "read_placement",
    "replace_file_lines",
    "resolve_decided_provider",
    "resolve_named_entries",
    "resolve_page_key",
    "resolve_seed_type",
    "suggest_group_page_placements",
    "suggest_linked_page_answer",
    "suggest_page_placements",
    "suggest_target_entries",
    "write_placement",
]
