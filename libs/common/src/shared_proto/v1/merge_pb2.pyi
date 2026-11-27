import datetime

from google.protobuf import any_pb2 as _any_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from shared_proto.v1 import anime_pb2 as _anime_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class PageKey(_message.Message):
    __slots__ = ("provider", "entity_type", "address")
    PROVIDER_FIELD_NUMBER: _ClassVar[int]
    ENTITY_TYPE_FIELD_NUMBER: _ClassVar[int]
    ADDRESS_FIELD_NUMBER: _ClassVar[int]
    provider: str
    entity_type: _anime_pb2.EntityType
    address: str
    def __init__(self, provider: _Optional[str] = ..., entity_type: _Optional[_Union[_anime_pb2.EntityType, str]] = ..., address: _Optional[str] = ...) -> None: ...

class PageSource(_message.Message):
    __slots__ = ("page_key", "source")
    PAGE_KEY_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    page_key: PageKey
    source: str
    def __init__(self, page_key: _Optional[_Union[PageKey, _Mapping]] = ..., source: _Optional[str] = ...) -> None: ...

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

class FieldCorrection(_message.Message):
    __slots__ = ("field_path", "value", "approved_by", "approved_at")
    FIELD_PATH_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    APPROVED_BY_FIELD_NUMBER: _ClassVar[int]
    APPROVED_AT_FIELD_NUMBER: _ClassVar[int]
    field_path: str
    value: _any_pb2.Any
    approved_by: str
    approved_at: _timestamp_pb2.Timestamp
    def __init__(self, field_path: _Optional[str] = ..., value: _Optional[_Union[_any_pb2.Any, _Mapping]] = ..., approved_by: _Optional[str] = ..., approved_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class GeneratedText(_message.Message):
    __slots__ = ("field_path", "text", "written_at")
    FIELD_PATH_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    WRITTEN_AT_FIELD_NUMBER: _ClassVar[int]
    field_path: str
    text: str
    written_at: _timestamp_pb2.Timestamp
    def __init__(self, field_path: _Optional[str] = ..., text: _Optional[str] = ..., written_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class RelationClaim(_message.Message):
    __slots__ = ("provider", "relation_type", "target_page")
    PROVIDER_FIELD_NUMBER: _ClassVar[int]
    RELATION_TYPE_FIELD_NUMBER: _ClassVar[int]
    TARGET_PAGE_FIELD_NUMBER: _ClassVar[int]
    provider: str
    relation_type: _anime_pb2.AnimeRelationType
    target_page: PageKey
    def __init__(self, provider: _Optional[str] = ..., relation_type: _Optional[_Union[_anime_pb2.AnimeRelationType, str]] = ..., target_page: _Optional[_Union[PageKey, _Mapping]] = ...) -> None: ...

class EpisodeListEntry(_message.Message):
    __slots__ = ("provider", "title", "aired_at", "duration_seconds", "is_filler", "is_recap")
    PROVIDER_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    AIRED_AT_FIELD_NUMBER: _ClassVar[int]
    DURATION_SECONDS_FIELD_NUMBER: _ClassVar[int]
    IS_FILLER_FIELD_NUMBER: _ClassVar[int]
    IS_RECAP_FIELD_NUMBER: _ClassVar[int]
    provider: str
    title: str
    aired_at: _timestamp_pb2.Timestamp
    duration_seconds: int
    is_filler: bool
    is_recap: bool
    def __init__(self, provider: _Optional[str] = ..., title: _Optional[str] = ..., aired_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., duration_seconds: _Optional[int] = ..., is_filler: _Optional[bool] = ..., is_recap: _Optional[bool] = ...) -> None: ...

class EpisodeReference(_message.Message):
    __slots__ = ("episode_id", "episode_number", "title", "page_keys", "list_entries")
    EPISODE_ID_FIELD_NUMBER: _ClassVar[int]
    EPISODE_NUMBER_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    PAGE_KEYS_FIELD_NUMBER: _ClassVar[int]
    LIST_ENTRIES_FIELD_NUMBER: _ClassVar[int]
    episode_id: str
    episode_number: int
    title: str
    page_keys: _containers.RepeatedCompositeFieldContainer[PageKey]
    list_entries: _containers.RepeatedCompositeFieldContainer[EpisodeListEntry]
    def __init__(self, episode_id: _Optional[str] = ..., episode_number: _Optional[int] = ..., title: _Optional[str] = ..., page_keys: _Optional[_Iterable[_Union[PageKey, _Mapping]]] = ..., list_entries: _Optional[_Iterable[_Union[EpisodeListEntry, _Mapping]]] = ...) -> None: ...
