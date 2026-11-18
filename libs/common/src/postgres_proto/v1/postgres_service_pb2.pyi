from shared_proto.v1 import anime_pb2 as _anime_pb2
from shared_proto.v1 import merge_pb2 as _merge_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class StorePageRequest(_message.Message):
    __slots__ = ("page",)
    PAGE_FIELD_NUMBER: _ClassVar[int]
    page: _merge_pb2.StoredPage
    def __init__(self, page: _Optional[_Union[_merge_pb2.StoredPage, _Mapping]] = ...) -> None: ...

class StorePageResponse(_message.Message):
    __slots__ = ("has_changed",)
    HAS_CHANGED_FIELD_NUMBER: _ClassVar[int]
    has_changed: bool
    def __init__(self, has_changed: _Optional[bool] = ...) -> None: ...

class LoadAnimeRequest(_message.Message):
    __slots__ = ("anime_id",)
    ANIME_ID_FIELD_NUMBER: _ClassVar[int]
    anime_id: str
    def __init__(self, anime_id: _Optional[str] = ...) -> None: ...

class LoadAnimeResponse(_message.Message):
    __slots__ = ("version", "source_pages", "partial_source_pages", "corrections")
    VERSION_FIELD_NUMBER: _ClassVar[int]
    SOURCE_PAGES_FIELD_NUMBER: _ClassVar[int]
    PARTIAL_SOURCE_PAGES_FIELD_NUMBER: _ClassVar[int]
    CORRECTIONS_FIELD_NUMBER: _ClassVar[int]
    version: int
    source_pages: _containers.RepeatedCompositeFieldContainer[_merge_pb2.StoredPage]
    partial_source_pages: _containers.RepeatedCompositeFieldContainer[_merge_pb2.PageKey]
    corrections: _containers.RepeatedCompositeFieldContainer[_merge_pb2.FieldCorrection]
    def __init__(self, version: _Optional[int] = ..., source_pages: _Optional[_Iterable[_Union[_merge_pb2.StoredPage, _Mapping]]] = ..., partial_source_pages: _Optional[_Iterable[_Union[_merge_pb2.PageKey, _Mapping]]] = ..., corrections: _Optional[_Iterable[_Union[_merge_pb2.FieldCorrection, _Mapping]]] = ...) -> None: ...

class SaveAnimeRequest(_message.Message):
    __slots__ = ("anime_id", "expected_version", "anime", "merge_version")
    ANIME_ID_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_VERSION_FIELD_NUMBER: _ClassVar[int]
    ANIME_FIELD_NUMBER: _ClassVar[int]
    MERGE_VERSION_FIELD_NUMBER: _ClassVar[int]
    anime_id: str
    expected_version: int
    anime: _anime_pb2.Anime
    merge_version: int
    def __init__(self, anime_id: _Optional[str] = ..., expected_version: _Optional[int] = ..., anime: _Optional[_Union[_anime_pb2.Anime, _Mapping]] = ..., merge_version: _Optional[int] = ...) -> None: ...

class SaveAnimeResponse(_message.Message):
    __slots__ = ("has_changed", "version")
    HAS_CHANGED_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    has_changed: bool
    version: int
    def __init__(self, has_changed: _Optional[bool] = ..., version: _Optional[int] = ...) -> None: ...
