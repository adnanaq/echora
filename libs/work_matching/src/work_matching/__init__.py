"""Decide which provider pages belong to the same work.

Seed cleaning uses this package now; seed link filling and discovery will use
this package later. Terms are as in ``GLOSSARY.md``. This library imports
nothing from enrichment.
"""

from work_matching.answer_placement_check import (
    PlacementCheck,
    check_answer_placements,
    read_answer_placements,
)
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
from work_matching.errors import (
    DuplicateCaseError,
    NoLinkingEntryError,
    ReviewCaseNamesError,
    ReviewFileLockedError,
    SavedAnswerChangedError,
    UnknownCaseError,
    UnknownMemberNameError,
    WorkMatchingError,
)
from work_matching.file_replacement import replace_file_lines
from work_matching.known_work_entries import KnownWorkEntries
from work_matching.known_work_entries_from_seed import KnownWorkEntriesFromSeed
from work_matching.linked_page_check import (
    SKIP_ANSWER,
    check_linked_page_answer,
    decide_linked_page,
    find_entry_episode_count,
    find_unheld_linked_page_keys,
    read_linked_page_answer,
    suggest_linked_page_answer,
)
from work_matching.models import (
    ChoiceOption,
    DoubledProviderDecision,
    DoubledProviderOutcome,
    DoubledProviderReviewReason,
    GroupPageChoices,
    LinkedPageAddition,
    LinkedPageAnswerCheck,
    LinkedPageAnswerRule,
    LinkedPageConflict,
    LinkedPageDecision,
    LinkedPageOutcome,
    LinkedPageReason,
    LinkingEntryDetails,
    PlacementRule,
    PreviousAnswer,
    UnrelatedPage,
    WorkPartialSources,
)
from work_matching.named_entries import (
    MAX_TARGET_ENTRY_SUGGESTIONS,
    build_target_entry_suggestion,
    find_covered_outside_entries,
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
    DROPPED,
    FIRST_WORK,
    OTHER_WORK,
    PagePlacement,
    PlacedPage,
    find_page_placement,
    list_placeable_page_keys,
    place_pages_without_review,
    read_placement,
    read_placements,
    write_placement,
)
from work_matching.placement_suggestions import suggest_page_placements
from work_matching.previous_answers import (
    attach_previous_answers,
    build_previous_answer,
)
from work_matching.provider_page import (
    ProviderPage,
    index_pages_by_any_key,
    index_pages_by_current_page_key,
)
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
    choose_partial_sources,
    find_linked_page_conflicts,
    list_group_page_choices,
    list_linked_page_options,
    list_placeable_case_page_keys,
    list_required_page_keys,
    list_work_partial_sources,
)
from work_matching.review_file import ReviewFile
from work_matching.review_file_lock import (
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
    "MAX_TARGET_ENTRY_SUGGESTIONS",
    "DROPPED",
    "OTHER_WORK",
    "SKIP_ANSWER",
    "ChoiceOption",
    "DoubledProviderDecision",
    "DoubledProviderOutcome",
    "DoubledProviderReviewReason",
    "DuplicateCaseError",
    "PreviousAnswer",
    "GroupPageChoices",
    "KnownWorkEntries",
    "KnownWorkEntriesFromSeed",
    "LinkedPageAddition",
    "LinkedPageAnswerCheck",
    "LinkedPageAnswerRule",
    "LinkedPageConflict",
    "WorkPartialSources",
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
    "ReviewCaseNamesError",
    "ReviewFile",
    "ReviewFileLockedError",
    "SavedAnswerChangedError",
    "SplitWorkReviewReason",
    "TargetEntrySuggestion",
    "UnknownCaseError",
    "UnknownMemberNameError",
    "UnrelatedPage",
    "WorkEntry",
    "WorkMatchingError",
    "build_page_address",
    "check_answer_placements",
    "check_group_answer_placements",
    "check_linked_page_answer",
    "choose_group_page_place",
    "choose_partial_sources",
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
    "index_pages_by_any_key",
    "index_pages_by_current_page_key",
    "is_mal_page",
    "is_page_details_matching",
    "is_review_file_locked",
    "list_group_page_choices",
    "list_work_partial_sources",
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
    "read_placements",
    "replace_file_lines",
    "resolve_decided_provider",
    "resolve_named_entries",
    "attach_previous_answers",
    "build_previous_answer",
    "resolve_page_key",
    "resolve_seed_type",
    "suggest_group_page_placements",
    "suggest_linked_page_answer",
    "suggest_page_placements",
    "build_target_entry_suggestion",
    "find_covered_outside_entries",
    "suggest_target_entries",
    "write_placement",
]
