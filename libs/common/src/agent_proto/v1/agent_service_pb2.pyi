from shared_proto.v1 import review_cases_pb2 as _review_cases_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SuggestPageAnswersRequest(_message.Message):
    __slots__ = ("review_case",)
    REVIEW_CASE_FIELD_NUMBER: _ClassVar[int]
    review_case: _review_cases_pb2.ReviewCase
    def __init__(self, review_case: _Optional[_Union[_review_cases_pb2.ReviewCase, _Mapping]] = ...) -> None: ...

class PageAnswerSuggestion(_message.Message):
    __slots__ = ("page_key", "answer_word", "score")
    PAGE_KEY_FIELD_NUMBER: _ClassVar[int]
    ANSWER_WORD_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    page_key: str
    answer_word: str
    score: float
    def __init__(self, page_key: _Optional[str] = ..., answer_word: _Optional[str] = ..., score: _Optional[float] = ...) -> None: ...

class SuggestPageAnswersResponse(_message.Message):
    __slots__ = ("page_suggestions",)
    PAGE_SUGGESTIONS_FIELD_NUMBER: _ClassVar[int]
    page_suggestions: _containers.RepeatedCompositeFieldContainer[PageAnswerSuggestion]
    def __init__(self, page_suggestions: _Optional[_Iterable[_Union[PageAnswerSuggestion, _Mapping]]] = ...) -> None: ...

class JudgeOpenCasesRequest(_message.Message):
    __slots__ = ("review_cases", "synopses_by_page_key")
    class SynopsesByPageKeyEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    REVIEW_CASES_FIELD_NUMBER: _ClassVar[int]
    SYNOPSES_BY_PAGE_KEY_FIELD_NUMBER: _ClassVar[int]
    review_cases: _containers.RepeatedCompositeFieldContainer[_review_cases_pb2.ReviewCase]
    synopses_by_page_key: _containers.ScalarMap[str, str]
    def __init__(self, review_cases: _Optional[_Iterable[_Union[_review_cases_pb2.ReviewCase, _Mapping]]] = ..., synopses_by_page_key: _Optional[_Mapping[str, str]] = ...) -> None: ...

class JudgeOpenCasesResponse(_message.Message):
    __slots__ = ("judged_pair_count", "failed_pair_count", "input_token_count")
    JUDGED_PAIR_COUNT_FIELD_NUMBER: _ClassVar[int]
    FAILED_PAIR_COUNT_FIELD_NUMBER: _ClassVar[int]
    INPUT_TOKEN_COUNT_FIELD_NUMBER: _ClassVar[int]
    judged_pair_count: int
    failed_pair_count: int
    input_token_count: int
    def __init__(self, judged_pair_count: _Optional[int] = ..., failed_pair_count: _Optional[int] = ..., input_token_count: _Optional[int] = ...) -> None: ...
