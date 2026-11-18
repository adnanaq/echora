import datetime

from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class PageKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PAGE_KIND_UNSPECIFIED: _ClassVar[PageKind]
    PAGE_KIND_ANIME: _ClassVar[PageKind]
    PAGE_KIND_EPISODE: _ClassVar[PageKind]
    PAGE_KIND_CHARACTER: _ClassVar[PageKind]
PAGE_KIND_UNSPECIFIED: PageKind
PAGE_KIND_ANIME: PageKind
PAGE_KIND_EPISODE: PageKind
PAGE_KIND_CHARACTER: PageKind

class PageKey(_message.Message):
    __slots__ = ("provider", "page_kind", "address")
    PROVIDER_FIELD_NUMBER: _ClassVar[int]
    PAGE_KIND_FIELD_NUMBER: _ClassVar[int]
    ADDRESS_FIELD_NUMBER: _ClassVar[int]
    provider: str
    page_kind: PageKind
    address: str
    def __init__(self, provider: _Optional[str] = ..., page_kind: _Optional[_Union[PageKind, str]] = ..., address: _Optional[str] = ...) -> None: ...

class PageResponse(_message.Message):
    __slots__ = ("request_name", "content_type", "body")
    REQUEST_NAME_FIELD_NUMBER: _ClassVar[int]
    CONTENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    BODY_FIELD_NUMBER: _ClassVar[int]
    request_name: str
    content_type: str
    body: bytes
    def __init__(self, request_name: _Optional[str] = ..., content_type: _Optional[str] = ..., body: _Optional[bytes] = ...) -> None: ...

class StoredPage(_message.Message):
    __slots__ = ("page_key", "fetched_at", "is_dead", "responses", "is_not_found")
    PAGE_KEY_FIELD_NUMBER: _ClassVar[int]
    FETCHED_AT_FIELD_NUMBER: _ClassVar[int]
    IS_DEAD_FIELD_NUMBER: _ClassVar[int]
    RESPONSES_FIELD_NUMBER: _ClassVar[int]
    IS_NOT_FOUND_FIELD_NUMBER: _ClassVar[int]
    page_key: PageKey
    fetched_at: _timestamp_pb2.Timestamp
    is_dead: bool
    responses: _containers.RepeatedCompositeFieldContainer[PageResponse]
    is_not_found: bool
    def __init__(self, page_key: _Optional[_Union[PageKey, _Mapping]] = ..., fetched_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., is_dead: _Optional[bool] = ..., responses: _Optional[_Iterable[_Union[PageResponse, _Mapping]]] = ..., is_not_found: _Optional[bool] = ...) -> None: ...
