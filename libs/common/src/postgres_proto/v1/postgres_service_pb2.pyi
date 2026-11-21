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
    __slots__ = ("version", "source_pages", "partial_source_pages", "corrections", "known_episodes")
    VERSION_FIELD_NUMBER: _ClassVar[int]
    SOURCE_PAGES_FIELD_NUMBER: _ClassVar[int]
    PARTIAL_SOURCE_PAGES_FIELD_NUMBER: _ClassVar[int]
    CORRECTIONS_FIELD_NUMBER: _ClassVar[int]
    KNOWN_EPISODES_FIELD_NUMBER: _ClassVar[int]
    version: int
    source_pages: _containers.RepeatedCompositeFieldContainer[_merge_pb2.StoredPage]
    partial_source_pages: _containers.RepeatedCompositeFieldContainer[_merge_pb2.PageKey]
    corrections: _containers.RepeatedCompositeFieldContainer[_merge_pb2.FieldCorrection]
    known_episodes: _containers.RepeatedCompositeFieldContainer[_merge_pb2.EpisodeReference]
    def __init__(self, version: _Optional[int] = ..., source_pages: _Optional[_Iterable[_Union[_merge_pb2.StoredPage, _Mapping]]] = ..., partial_source_pages: _Optional[_Iterable[_Union[_merge_pb2.PageKey, _Mapping]]] = ..., corrections: _Optional[_Iterable[_Union[_merge_pb2.FieldCorrection, _Mapping]]] = ..., known_episodes: _Optional[_Iterable[_Union[_merge_pb2.EpisodeReference, _Mapping]]] = ...) -> None: ...

class SaveAnimeRequest(_message.Message):
    __slots__ = ("anime_id", "expected_version", "anime", "merge_version", "relation_claims", "episodes")
    ANIME_ID_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_VERSION_FIELD_NUMBER: _ClassVar[int]
    ANIME_FIELD_NUMBER: _ClassVar[int]
    MERGE_VERSION_FIELD_NUMBER: _ClassVar[int]
    RELATION_CLAIMS_FIELD_NUMBER: _ClassVar[int]
    EPISODES_FIELD_NUMBER: _ClassVar[int]
    anime_id: str
    expected_version: int
    anime: _anime_pb2.Anime
    merge_version: int
    relation_claims: _containers.RepeatedCompositeFieldContainer[_merge_pb2.RelationClaim]
    episodes: _containers.RepeatedCompositeFieldContainer[_merge_pb2.EpisodeReference]
    def __init__(self, anime_id: _Optional[str] = ..., expected_version: _Optional[int] = ..., anime: _Optional[_Union[_anime_pb2.Anime, _Mapping]] = ..., merge_version: _Optional[int] = ..., relation_claims: _Optional[_Iterable[_Union[_merge_pb2.RelationClaim, _Mapping]]] = ..., episodes: _Optional[_Iterable[_Union[_merge_pb2.EpisodeReference, _Mapping]]] = ...) -> None: ...

class SaveAnimeResponse(_message.Message):
    __slots__ = ("has_changed", "version", "created_episode_ids", "updated_episode_ids")
    HAS_CHANGED_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    CREATED_EPISODE_IDS_FIELD_NUMBER: _ClassVar[int]
    UPDATED_EPISODE_IDS_FIELD_NUMBER: _ClassVar[int]
    has_changed: bool
    version: int
    created_episode_ids: _containers.RepeatedScalarFieldContainer[str]
    updated_episode_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, has_changed: _Optional[bool] = ..., version: _Optional[int] = ..., created_episode_ids: _Optional[_Iterable[str]] = ..., updated_episode_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class LoadEpisodeRequest(_message.Message):
    __slots__ = ("episode_id",)
    EPISODE_ID_FIELD_NUMBER: _ClassVar[int]
    episode_id: str
    def __init__(self, episode_id: _Optional[str] = ...) -> None: ...

class LoadEpisodeResponse(_message.Message):
    __slots__ = ("version", "anime_id", "pages", "corrections", "list_entries")
    VERSION_FIELD_NUMBER: _ClassVar[int]
    ANIME_ID_FIELD_NUMBER: _ClassVar[int]
    PAGES_FIELD_NUMBER: _ClassVar[int]
    CORRECTIONS_FIELD_NUMBER: _ClassVar[int]
    LIST_ENTRIES_FIELD_NUMBER: _ClassVar[int]
    version: int
    anime_id: str
    pages: _containers.RepeatedCompositeFieldContainer[_merge_pb2.StoredPage]
    corrections: _containers.RepeatedCompositeFieldContainer[_merge_pb2.FieldCorrection]
    list_entries: _containers.RepeatedCompositeFieldContainer[_merge_pb2.EpisodeListEntry]
    def __init__(self, version: _Optional[int] = ..., anime_id: _Optional[str] = ..., pages: _Optional[_Iterable[_Union[_merge_pb2.StoredPage, _Mapping]]] = ..., corrections: _Optional[_Iterable[_Union[_merge_pb2.FieldCorrection, _Mapping]]] = ..., list_entries: _Optional[_Iterable[_Union[_merge_pb2.EpisodeListEntry, _Mapping]]] = ...) -> None: ...

class SaveEpisodeRequest(_message.Message):
    __slots__ = ("episode_id", "expected_version", "episode", "merge_version")
    EPISODE_ID_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_VERSION_FIELD_NUMBER: _ClassVar[int]
    EPISODE_FIELD_NUMBER: _ClassVar[int]
    MERGE_VERSION_FIELD_NUMBER: _ClassVar[int]
    episode_id: str
    expected_version: int
    episode: _anime_pb2.Episode
    merge_version: int
    def __init__(self, episode_id: _Optional[str] = ..., expected_version: _Optional[int] = ..., episode: _Optional[_Union[_anime_pb2.Episode, _Mapping]] = ..., merge_version: _Optional[int] = ...) -> None: ...

class SaveEpisodeResponse(_message.Message):
    __slots__ = ("has_changed", "version")
    HAS_CHANGED_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    has_changed: bool
    version: int
    def __init__(self, has_changed: _Optional[bool] = ..., version: _Optional[int] = ...) -> None: ...

class MarkEpisodeAiredRequest(_message.Message):
    __slots__ = ("episode_id",)
    EPISODE_ID_FIELD_NUMBER: _ClassVar[int]
    episode_id: str
    def __init__(self, episode_id: _Optional[str] = ...) -> None: ...

class MarkEpisodeAiredResponse(_message.Message):
    __slots__ = ("has_notified",)
    HAS_NOTIFIED_FIELD_NUMBER: _ClassVar[int]
    has_notified: bool
    def __init__(self, has_notified: _Optional[bool] = ...) -> None: ...
