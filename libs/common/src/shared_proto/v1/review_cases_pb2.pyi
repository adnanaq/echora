import datetime

from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class PageAddress(_message.Message):
    __slots__ = ("page_key", "address")
    PAGE_KEY_FIELD_NUMBER: _ClassVar[int]
    ADDRESS_FIELD_NUMBER: _ClassVar[int]
    page_key: str
    address: str
    def __init__(self, page_key: _Optional[str] = ..., address: _Optional[str] = ...) -> None: ...

class WorkEntry(_message.Message):
    __slots__ = ("title", "type", "year", "season", "addresses_by_page_key")
    TITLE_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    YEAR_FIELD_NUMBER: _ClassVar[int]
    SEASON_FIELD_NUMBER: _ClassVar[int]
    ADDRESSES_BY_PAGE_KEY_FIELD_NUMBER: _ClassVar[int]
    title: str
    type: str
    year: int
    season: str
    addresses_by_page_key: _containers.RepeatedCompositeFieldContainer[PageAddress]
    def __init__(self, title: _Optional[str] = ..., type: _Optional[str] = ..., year: _Optional[int] = ..., season: _Optional[str] = ..., addresses_by_page_key: _Optional[_Iterable[_Union[PageAddress, _Mapping]]] = ...) -> None: ...

class CaseEntry(_message.Message):
    __slots__ = ("reference_page", "entry")
    REFERENCE_PAGE_FIELD_NUMBER: _ClassVar[int]
    ENTRY_FIELD_NUMBER: _ClassVar[int]
    reference_page: str
    entry: WorkEntry
    def __init__(self, reference_page: _Optional[str] = ..., entry: _Optional[_Union[WorkEntry, _Mapping]] = ...) -> None: ...

class ProviderPage(_message.Message):
    __slots__ = ("key", "address", "titles", "type", "year", "season", "episode_count", "aired_from", "aired_to", "linked_page_keys", "renamed_to", "searched_by", "is_dead")
    KEY_FIELD_NUMBER: _ClassVar[int]
    ADDRESS_FIELD_NUMBER: _ClassVar[int]
    TITLES_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    YEAR_FIELD_NUMBER: _ClassVar[int]
    SEASON_FIELD_NUMBER: _ClassVar[int]
    EPISODE_COUNT_FIELD_NUMBER: _ClassVar[int]
    AIRED_FROM_FIELD_NUMBER: _ClassVar[int]
    AIRED_TO_FIELD_NUMBER: _ClassVar[int]
    LINKED_PAGE_KEYS_FIELD_NUMBER: _ClassVar[int]
    RENAMED_TO_FIELD_NUMBER: _ClassVar[int]
    SEARCHED_BY_FIELD_NUMBER: _ClassVar[int]
    IS_DEAD_FIELD_NUMBER: _ClassVar[int]
    key: str
    address: str
    titles: _containers.RepeatedScalarFieldContainer[str]
    type: str
    year: int
    season: str
    episode_count: int
    aired_from: _timestamp_pb2.Timestamp
    aired_to: _timestamp_pb2.Timestamp
    linked_page_keys: _containers.RepeatedScalarFieldContainer[str]
    renamed_to: str
    searched_by: _containers.RepeatedScalarFieldContainer[str]
    is_dead: bool
    def __init__(self, key: _Optional[str] = ..., address: _Optional[str] = ..., titles: _Optional[_Iterable[str]] = ..., type: _Optional[str] = ..., year: _Optional[int] = ..., season: _Optional[str] = ..., episode_count: _Optional[int] = ..., aired_from: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., aired_to: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., linked_page_keys: _Optional[_Iterable[str]] = ..., renamed_to: _Optional[str] = ..., searched_by: _Optional[_Iterable[str]] = ..., is_dead: _Optional[bool] = ...) -> None: ...

class UnrelatedPage(_message.Message):
    __slots__ = ("linked_page_keys", "other_work_entries", "move_target_entry")
    LINKED_PAGE_KEYS_FIELD_NUMBER: _ClassVar[int]
    OTHER_WORK_ENTRIES_FIELD_NUMBER: _ClassVar[int]
    MOVE_TARGET_ENTRY_FIELD_NUMBER: _ClassVar[int]
    linked_page_keys: _containers.RepeatedScalarFieldContainer[str]
    other_work_entries: _containers.RepeatedCompositeFieldContainer[WorkEntry]
    move_target_entry: WorkEntry
    def __init__(self, linked_page_keys: _Optional[_Iterable[str]] = ..., other_work_entries: _Optional[_Iterable[_Union[WorkEntry, _Mapping]]] = ..., move_target_entry: _Optional[_Union[WorkEntry, _Mapping]] = ...) -> None: ...

class BelongingPage(_message.Message):
    __slots__ = ("page_key", "linked_page_keys")
    PAGE_KEY_FIELD_NUMBER: _ClassVar[int]
    LINKED_PAGE_KEYS_FIELD_NUMBER: _ClassVar[int]
    page_key: str
    linked_page_keys: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, page_key: _Optional[str] = ..., linked_page_keys: _Optional[_Iterable[str]] = ...) -> None: ...

class UnrelatedPageEntry(_message.Message):
    __slots__ = ("page_key", "unrelated_page")
    PAGE_KEY_FIELD_NUMBER: _ClassVar[int]
    UNRELATED_PAGE_FIELD_NUMBER: _ClassVar[int]
    page_key: str
    unrelated_page: UnrelatedPage
    def __init__(self, page_key: _Optional[str] = ..., unrelated_page: _Optional[_Union[UnrelatedPage, _Mapping]] = ...) -> None: ...

class PagePlace(_message.Message):
    __slots__ = ("page_key", "place")
    PAGE_KEY_FIELD_NUMBER: _ClassVar[int]
    PLACE_FIELD_NUMBER: _ClassVar[int]
    page_key: str
    place: str
    def __init__(self, page_key: _Optional[str] = ..., place: _Optional[str] = ...) -> None: ...

class DoubledProviderDecision(_message.Message):
    __slots__ = ("provider", "page_keys", "outcome", "review_reasons", "belonging_page_keys", "unrelated_pages", "contradicting_page_keys", "unlinked_page_keys", "title_matches", "year_type_matches", "season_matches")
    PROVIDER_FIELD_NUMBER: _ClassVar[int]
    PAGE_KEYS_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_FIELD_NUMBER: _ClassVar[int]
    REVIEW_REASONS_FIELD_NUMBER: _ClassVar[int]
    BELONGING_PAGE_KEYS_FIELD_NUMBER: _ClassVar[int]
    UNRELATED_PAGES_FIELD_NUMBER: _ClassVar[int]
    CONTRADICTING_PAGE_KEYS_FIELD_NUMBER: _ClassVar[int]
    UNLINKED_PAGE_KEYS_FIELD_NUMBER: _ClassVar[int]
    TITLE_MATCHES_FIELD_NUMBER: _ClassVar[int]
    YEAR_TYPE_MATCHES_FIELD_NUMBER: _ClassVar[int]
    SEASON_MATCHES_FIELD_NUMBER: _ClassVar[int]
    provider: str
    page_keys: _containers.RepeatedScalarFieldContainer[str]
    outcome: str
    review_reasons: _containers.RepeatedScalarFieldContainer[str]
    belonging_page_keys: _containers.RepeatedCompositeFieldContainer[BelongingPage]
    unrelated_pages: _containers.RepeatedCompositeFieldContainer[UnrelatedPageEntry]
    contradicting_page_keys: _containers.RepeatedScalarFieldContainer[str]
    unlinked_page_keys: _containers.RepeatedScalarFieldContainer[str]
    title_matches: _containers.RepeatedScalarFieldContainer[str]
    year_type_matches: _containers.RepeatedScalarFieldContainer[str]
    season_matches: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, provider: _Optional[str] = ..., page_keys: _Optional[_Iterable[str]] = ..., outcome: _Optional[str] = ..., review_reasons: _Optional[_Iterable[str]] = ..., belonging_page_keys: _Optional[_Iterable[_Union[BelongingPage, _Mapping]]] = ..., unrelated_pages: _Optional[_Iterable[_Union[UnrelatedPageEntry, _Mapping]]] = ..., contradicting_page_keys: _Optional[_Iterable[str]] = ..., unlinked_page_keys: _Optional[_Iterable[str]] = ..., title_matches: _Optional[_Iterable[str]] = ..., year_type_matches: _Optional[_Iterable[str]] = ..., season_matches: _Optional[_Iterable[str]] = ...) -> None: ...

class LinkedPageDecision(_message.Message):
    __slots__ = ("page_key", "reason", "linking_entries", "entry_episode_count")
    PAGE_KEY_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    LINKING_ENTRIES_FIELD_NUMBER: _ClassVar[int]
    ENTRY_EPISODE_COUNT_FIELD_NUMBER: _ClassVar[int]
    page_key: str
    reason: str
    linking_entries: _containers.RepeatedCompositeFieldContainer[WorkEntry]
    entry_episode_count: int
    def __init__(self, page_key: _Optional[str] = ..., reason: _Optional[str] = ..., linking_entries: _Optional[_Iterable[_Union[WorkEntry, _Mapping]]] = ..., entry_episode_count: _Optional[int] = ...) -> None: ...

class ReviewCase(_message.Message):
    __slots__ = ("case_name", "answer", "suggested_answer", "reasons", "candidate_evidence", "entries", "doubled_provider_decisions", "linked_page_decisions", "pages", "named_entry_reference_pages", "target_entry_suggestions")
    class NamedEntryReferencePagesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    CASE_NAME_FIELD_NUMBER: _ClassVar[int]
    ANSWER_FIELD_NUMBER: _ClassVar[int]
    SUGGESTED_ANSWER_FIELD_NUMBER: _ClassVar[int]
    REASONS_FIELD_NUMBER: _ClassVar[int]
    CANDIDATE_EVIDENCE_FIELD_NUMBER: _ClassVar[int]
    ENTRIES_FIELD_NUMBER: _ClassVar[int]
    DOUBLED_PROVIDER_DECISIONS_FIELD_NUMBER: _ClassVar[int]
    LINKED_PAGE_DECISIONS_FIELD_NUMBER: _ClassVar[int]
    PAGES_FIELD_NUMBER: _ClassVar[int]
    NAMED_ENTRY_REFERENCE_PAGES_FIELD_NUMBER: _ClassVar[int]
    TARGET_ENTRY_SUGGESTIONS_FIELD_NUMBER: _ClassVar[int]
    case_name: str
    answer: _struct_pb2.Value
    suggested_answer: _containers.RepeatedCompositeFieldContainer[PagePlace]
    reasons: _containers.RepeatedScalarFieldContainer[str]
    candidate_evidence: _containers.RepeatedScalarFieldContainer[str]
    entries: _containers.RepeatedCompositeFieldContainer[CaseEntry]
    doubled_provider_decisions: _containers.RepeatedCompositeFieldContainer[DoubledProviderDecision]
    linked_page_decisions: _containers.RepeatedCompositeFieldContainer[LinkedPageDecision]
    pages: _containers.RepeatedCompositeFieldContainer[ProviderPage]
    named_entry_reference_pages: _containers.ScalarMap[str, str]
    target_entry_suggestions: _containers.RepeatedCompositeFieldContainer[TargetEntrySuggestion]
    def __init__(self, case_name: _Optional[str] = ..., answer: _Optional[_Union[_struct_pb2.Value, _Mapping]] = ..., suggested_answer: _Optional[_Iterable[_Union[PagePlace, _Mapping]]] = ..., reasons: _Optional[_Iterable[str]] = ..., candidate_evidence: _Optional[_Iterable[str]] = ..., entries: _Optional[_Iterable[_Union[CaseEntry, _Mapping]]] = ..., doubled_provider_decisions: _Optional[_Iterable[_Union[DoubledProviderDecision, _Mapping]]] = ..., linked_page_decisions: _Optional[_Iterable[_Union[LinkedPageDecision, _Mapping]]] = ..., pages: _Optional[_Iterable[_Union[ProviderPage, _Mapping]]] = ..., named_entry_reference_pages: _Optional[_Mapping[str, str]] = ..., target_entry_suggestions: _Optional[_Iterable[_Union[TargetEntrySuggestion, _Mapping]]] = ...) -> None: ...

class TargetEntrySuggestion(_message.Message):
    __slots__ = ("entry_reference_page", "title", "providers")
    ENTRY_REFERENCE_PAGE_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    PROVIDERS_FIELD_NUMBER: _ClassVar[int]
    entry_reference_page: str
    title: str
    providers: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, entry_reference_page: _Optional[str] = ..., title: _Optional[str] = ..., providers: _Optional[_Iterable[str]] = ...) -> None: ...

class CaseAnswer(_message.Message):
    __slots__ = ("case_name", "answer", "seen_saved_answer")
    CASE_NAME_FIELD_NUMBER: _ClassVar[int]
    ANSWER_FIELD_NUMBER: _ClassVar[int]
    SEEN_SAVED_ANSWER_FIELD_NUMBER: _ClassVar[int]
    case_name: str
    answer: _struct_pb2.Value
    seen_saved_answer: _struct_pb2.Value
    def __init__(self, case_name: _Optional[str] = ..., answer: _Optional[_Union[_struct_pb2.Value, _Mapping]] = ..., seen_saved_answer: _Optional[_Union[_struct_pb2.Value, _Mapping]] = ...) -> None: ...

class RuleCheckResult(_message.Message):
    __slots__ = ("case_name", "is_saved", "broken_rule_names", "broken_rule_sentences", "warnings")
    CASE_NAME_FIELD_NUMBER: _ClassVar[int]
    IS_SAVED_FIELD_NUMBER: _ClassVar[int]
    BROKEN_RULE_NAMES_FIELD_NUMBER: _ClassVar[int]
    BROKEN_RULE_SENTENCES_FIELD_NUMBER: _ClassVar[int]
    WARNINGS_FIELD_NUMBER: _ClassVar[int]
    case_name: str
    is_saved: bool
    broken_rule_names: _containers.RepeatedScalarFieldContainer[str]
    broken_rule_sentences: _containers.RepeatedScalarFieldContainer[str]
    warnings: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, case_name: _Optional[str] = ..., is_saved: _Optional[bool] = ..., broken_rule_names: _Optional[_Iterable[str]] = ..., broken_rule_sentences: _Optional[_Iterable[str]] = ..., warnings: _Optional[_Iterable[str]] = ...) -> None: ...

class NamedCount(_message.Message):
    __slots__ = ("name", "count")
    NAME_FIELD_NUMBER: _ClassVar[int]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    name: str
    count: int
    def __init__(self, name: _Optional[str] = ..., count: _Optional[int] = ...) -> None: ...

class SeedCleaningRunResult(_message.Message):
    __slots__ = ("seed_file", "started_at", "finished_at", "correction_counts", "corrected_entry_counts", "linked_page_counts", "rejected_answer_case_names", "conflicting_linked_page_keys", "unused_answer_case_names", "case_names", "new_case_names", "original_answers")
    class OriginalAnswersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _struct_pb2.Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_struct_pb2.Value, _Mapping]] = ...) -> None: ...
    SEED_FILE_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_FIELD_NUMBER: _ClassVar[int]
    FINISHED_AT_FIELD_NUMBER: _ClassVar[int]
    CORRECTION_COUNTS_FIELD_NUMBER: _ClassVar[int]
    CORRECTED_ENTRY_COUNTS_FIELD_NUMBER: _ClassVar[int]
    LINKED_PAGE_COUNTS_FIELD_NUMBER: _ClassVar[int]
    REJECTED_ANSWER_CASE_NAMES_FIELD_NUMBER: _ClassVar[int]
    CONFLICTING_LINKED_PAGE_KEYS_FIELD_NUMBER: _ClassVar[int]
    UNUSED_ANSWER_CASE_NAMES_FIELD_NUMBER: _ClassVar[int]
    CASE_NAMES_FIELD_NUMBER: _ClassVar[int]
    NEW_CASE_NAMES_FIELD_NUMBER: _ClassVar[int]
    ORIGINAL_ANSWERS_FIELD_NUMBER: _ClassVar[int]
    seed_file: str
    started_at: _timestamp_pb2.Timestamp
    finished_at: _timestamp_pb2.Timestamp
    correction_counts: _containers.RepeatedCompositeFieldContainer[NamedCount]
    corrected_entry_counts: _containers.RepeatedCompositeFieldContainer[NamedCount]
    linked_page_counts: _containers.RepeatedCompositeFieldContainer[NamedCount]
    rejected_answer_case_names: _containers.RepeatedScalarFieldContainer[str]
    conflicting_linked_page_keys: _containers.RepeatedScalarFieldContainer[str]
    unused_answer_case_names: _containers.RepeatedScalarFieldContainer[str]
    case_names: _containers.RepeatedScalarFieldContainer[str]
    new_case_names: _containers.RepeatedScalarFieldContainer[str]
    original_answers: _containers.MessageMap[str, _struct_pb2.Value]
    def __init__(self, seed_file: _Optional[str] = ..., started_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., finished_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., correction_counts: _Optional[_Iterable[_Union[NamedCount, _Mapping]]] = ..., corrected_entry_counts: _Optional[_Iterable[_Union[NamedCount, _Mapping]]] = ..., linked_page_counts: _Optional[_Iterable[_Union[NamedCount, _Mapping]]] = ..., rejected_answer_case_names: _Optional[_Iterable[str]] = ..., conflicting_linked_page_keys: _Optional[_Iterable[str]] = ..., unused_answer_case_names: _Optional[_Iterable[str]] = ..., case_names: _Optional[_Iterable[str]] = ..., new_case_names: _Optional[_Iterable[str]] = ..., original_answers: _Optional[_Mapping[str, _struct_pb2.Value]] = ...) -> None: ...

class SeedCorrection(_message.Message):
    __slots__ = ("entry_reference_page", "page_key", "address", "placement_word", "is_main_page", "renamed_to", "target_entry_reference_page", "new_entry_key", "is_added_page", "is_part_page", "is_anime_id_page", "reason", "change", "reason_sentence", "is_from_answer", "named_entry_page_key")
    ENTRY_REFERENCE_PAGE_FIELD_NUMBER: _ClassVar[int]
    PAGE_KEY_FIELD_NUMBER: _ClassVar[int]
    ADDRESS_FIELD_NUMBER: _ClassVar[int]
    PLACEMENT_WORD_FIELD_NUMBER: _ClassVar[int]
    IS_MAIN_PAGE_FIELD_NUMBER: _ClassVar[int]
    RENAMED_TO_FIELD_NUMBER: _ClassVar[int]
    TARGET_ENTRY_REFERENCE_PAGE_FIELD_NUMBER: _ClassVar[int]
    NEW_ENTRY_KEY_FIELD_NUMBER: _ClassVar[int]
    IS_ADDED_PAGE_FIELD_NUMBER: _ClassVar[int]
    IS_PART_PAGE_FIELD_NUMBER: _ClassVar[int]
    IS_ANIME_ID_PAGE_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    CHANGE_FIELD_NUMBER: _ClassVar[int]
    REASON_SENTENCE_FIELD_NUMBER: _ClassVar[int]
    IS_FROM_ANSWER_FIELD_NUMBER: _ClassVar[int]
    NAMED_ENTRY_PAGE_KEY_FIELD_NUMBER: _ClassVar[int]
    entry_reference_page: str
    page_key: str
    address: str
    placement_word: str
    is_main_page: bool
    renamed_to: str
    target_entry_reference_page: str
    new_entry_key: str
    is_added_page: bool
    is_part_page: bool
    is_anime_id_page: bool
    reason: str
    change: str
    reason_sentence: str
    is_from_answer: bool
    named_entry_page_key: str
    def __init__(self, entry_reference_page: _Optional[str] = ..., page_key: _Optional[str] = ..., address: _Optional[str] = ..., placement_word: _Optional[str] = ..., is_main_page: _Optional[bool] = ..., renamed_to: _Optional[str] = ..., target_entry_reference_page: _Optional[str] = ..., new_entry_key: _Optional[str] = ..., is_added_page: _Optional[bool] = ..., is_part_page: _Optional[bool] = ..., is_anime_id_page: _Optional[bool] = ..., reason: _Optional[str] = ..., change: _Optional[str] = ..., reason_sentence: _Optional[str] = ..., is_from_answer: _Optional[bool] = ..., named_entry_page_key: _Optional[str] = ...) -> None: ...

class CheckedEntry(_message.Message):
    __slots__ = ("entry_reference_page", "title", "type", "year")
    ENTRY_REFERENCE_PAGE_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    YEAR_FIELD_NUMBER: _ClassVar[int]
    entry_reference_page: str
    title: str
    type: str
    year: int
    def __init__(self, entry_reference_page: _Optional[str] = ..., title: _Optional[str] = ..., type: _Optional[str] = ..., year: _Optional[int] = ...) -> None: ...

class RunState(_message.Message):
    __slots__ = ("is_running", "last_exit_code")
    IS_RUNNING_FIELD_NUMBER: _ClassVar[int]
    LAST_EXIT_CODE_FIELD_NUMBER: _ClassVar[int]
    is_running: bool
    last_exit_code: int
    def __init__(self, is_running: _Optional[bool] = ..., last_exit_code: _Optional[int] = ...) -> None: ...
