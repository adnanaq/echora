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
