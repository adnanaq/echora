from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class SearchRequest(_message.Message):
    __slots__ = ("query", "image_data")
    QUERY_FIELD_NUMBER: _ClassVar[int]
    IMAGE_DATA_FIELD_NUMBER: _ClassVar[int]
    query: str
    image_data: str
    def __init__(self, query: _Optional[str] = ..., image_data: _Optional[str] = ...) -> None: ...

class SearchResponse(_message.Message):
    __slots__ = ("anime_ids", "reasoning")
    ANIME_IDS_FIELD_NUMBER: _ClassVar[int]
    REASONING_FIELD_NUMBER: _ClassVar[int]
    anime_ids: _containers.RepeatedScalarFieldContainer[str]
    reasoning: str
    def __init__(self, anime_ids: _Optional[_Iterable[str]] = ..., reasoning: _Optional[str] = ...) -> None: ...
