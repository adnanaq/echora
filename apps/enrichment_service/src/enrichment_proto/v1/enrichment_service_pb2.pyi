import datetime

from google.protobuf import duration_pb2 as _duration_pb2
from shared_proto.v1 import anime_pb2 as _anime_pb2
from shared_proto.v1 import error_pb2 as _error_pb2
from shared_proto.v1 import merge_pb2 as _merge_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class HealthRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthResponse(_message.Message):
    __slots__ = ("healthy", "service", "details_json", "error")
    HEALTHY_FIELD_NUMBER: _ClassVar[int]
    SERVICE_FIELD_NUMBER: _ClassVar[int]
    DETAILS_JSON_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    healthy: bool
    service: str
    details_json: str
    error: _error_pb2.ErrorDetails
    def __init__(self, healthy: _Optional[bool] = ..., service: _Optional[str] = ..., details_json: _Optional[str] = ..., error: _Optional[_Union[_error_pb2.ErrorDetails, _Mapping]] = ...) -> None: ...

class RunPipelineRequest(_message.Message):
    __slots__ = ("file_path", "index", "title", "agent_dir", "skip_services", "only_services", "skip_characters", "skip_episodes")
    FILE_PATH_FIELD_NUMBER: _ClassVar[int]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    AGENT_DIR_FIELD_NUMBER: _ClassVar[int]
    SKIP_SERVICES_FIELD_NUMBER: _ClassVar[int]
    ONLY_SERVICES_FIELD_NUMBER: _ClassVar[int]
    SKIP_CHARACTERS_FIELD_NUMBER: _ClassVar[int]
    SKIP_EPISODES_FIELD_NUMBER: _ClassVar[int]
    file_path: str
    index: int
    title: str
    agent_dir: str
    skip_services: _containers.RepeatedScalarFieldContainer[str]
    only_services: _containers.RepeatedScalarFieldContainer[str]
    skip_characters: bool
    skip_episodes: bool
    def __init__(self, file_path: _Optional[str] = ..., index: _Optional[int] = ..., title: _Optional[str] = ..., agent_dir: _Optional[str] = ..., skip_services: _Optional[_Iterable[str]] = ..., only_services: _Optional[_Iterable[str]] = ..., skip_characters: _Optional[bool] = ..., skip_episodes: _Optional[bool] = ...) -> None: ...

class RunPipelineResponse(_message.Message):
    __slots__ = ("success", "output_path", "result_json", "error")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_PATH_FIELD_NUMBER: _ClassVar[int]
    RESULT_JSON_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    success: bool
    output_path: str
    result_json: str
    error: _error_pb2.ErrorDetails
    def __init__(self, success: _Optional[bool] = ..., output_path: _Optional[str] = ..., result_json: _Optional[str] = ..., error: _Optional[_Union[_error_pb2.ErrorDetails, _Mapping]] = ...) -> None: ...

class FetchPageRequest(_message.Message):
    __slots__ = ("page_key", "max_age")
    PAGE_KEY_FIELD_NUMBER: _ClassVar[int]
    MAX_AGE_FIELD_NUMBER: _ClassVar[int]
    page_key: _merge_pb2.PageKey
    max_age: _duration_pb2.Duration
    def __init__(self, page_key: _Optional[_Union[_merge_pb2.PageKey, _Mapping]] = ..., max_age: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ...) -> None: ...

class FetchPageResponse(_message.Message):
    __slots__ = ("page",)
    PAGE_FIELD_NUMBER: _ClassVar[int]
    page: _merge_pb2.StoredPage
    def __init__(self, page: _Optional[_Union[_merge_pb2.StoredPage, _Mapping]] = ...) -> None: ...

class MergeAnimeRequest(_message.Message):
    __slots__ = ("anime_id", "source_pages", "partial_source_pages", "corrections", "generated_texts", "known_episodes")
    ANIME_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_PAGES_FIELD_NUMBER: _ClassVar[int]
    PARTIAL_SOURCE_PAGES_FIELD_NUMBER: _ClassVar[int]
    CORRECTIONS_FIELD_NUMBER: _ClassVar[int]
    GENERATED_TEXTS_FIELD_NUMBER: _ClassVar[int]
    KNOWN_EPISODES_FIELD_NUMBER: _ClassVar[int]
    anime_id: str
    source_pages: _containers.RepeatedCompositeFieldContainer[_merge_pb2.StoredPage]
    partial_source_pages: _containers.RepeatedCompositeFieldContainer[_merge_pb2.PageKey]
    corrections: _containers.RepeatedCompositeFieldContainer[_merge_pb2.FieldCorrection]
    generated_texts: _containers.RepeatedCompositeFieldContainer[_merge_pb2.GeneratedText]
    known_episodes: _containers.RepeatedCompositeFieldContainer[_merge_pb2.EpisodeReference]
    def __init__(self, anime_id: _Optional[str] = ..., source_pages: _Optional[_Iterable[_Union[_merge_pb2.StoredPage, _Mapping]]] = ..., partial_source_pages: _Optional[_Iterable[_Union[_merge_pb2.PageKey, _Mapping]]] = ..., corrections: _Optional[_Iterable[_Union[_merge_pb2.FieldCorrection, _Mapping]]] = ..., generated_texts: _Optional[_Iterable[_Union[_merge_pb2.GeneratedText, _Mapping]]] = ..., known_episodes: _Optional[_Iterable[_Union[_merge_pb2.EpisodeReference, _Mapping]]] = ...) -> None: ...

class MergeAnimeResponse(_message.Message):
    __slots__ = ("anime", "episodes", "relation_claims", "provider_synopses_hash", "page_sources")
    ANIME_FIELD_NUMBER: _ClassVar[int]
    EPISODES_FIELD_NUMBER: _ClassVar[int]
    RELATION_CLAIMS_FIELD_NUMBER: _ClassVar[int]
    PROVIDER_SYNOPSES_HASH_FIELD_NUMBER: _ClassVar[int]
    PAGE_SOURCES_FIELD_NUMBER: _ClassVar[int]
    anime: _anime_pb2.Anime
    episodes: _containers.RepeatedCompositeFieldContainer[_merge_pb2.EpisodeReference]
    relation_claims: _containers.RepeatedCompositeFieldContainer[_merge_pb2.RelationClaim]
    provider_synopses_hash: str
    page_sources: _containers.RepeatedCompositeFieldContainer[_merge_pb2.PageSource]
    def __init__(self, anime: _Optional[_Union[_anime_pb2.Anime, _Mapping]] = ..., episodes: _Optional[_Iterable[_Union[_merge_pb2.EpisodeReference, _Mapping]]] = ..., relation_claims: _Optional[_Iterable[_Union[_merge_pb2.RelationClaim, _Mapping]]] = ..., provider_synopses_hash: _Optional[str] = ..., page_sources: _Optional[_Iterable[_Union[_merge_pb2.PageSource, _Mapping]]] = ...) -> None: ...

class MergeEpisodeRequest(_message.Message):
    __slots__ = ("episode_id", "anime_id", "pages", "corrections", "list_entries", "episode_number")
    EPISODE_ID_FIELD_NUMBER: _ClassVar[int]
    ANIME_ID_FIELD_NUMBER: _ClassVar[int]
    PAGES_FIELD_NUMBER: _ClassVar[int]
    CORRECTIONS_FIELD_NUMBER: _ClassVar[int]
    LIST_ENTRIES_FIELD_NUMBER: _ClassVar[int]
    EPISODE_NUMBER_FIELD_NUMBER: _ClassVar[int]
    episode_id: str
    anime_id: str
    pages: _containers.RepeatedCompositeFieldContainer[_merge_pb2.StoredPage]
    corrections: _containers.RepeatedCompositeFieldContainer[_merge_pb2.FieldCorrection]
    list_entries: _containers.RepeatedCompositeFieldContainer[_merge_pb2.EpisodeListEntry]
    episode_number: int
    def __init__(self, episode_id: _Optional[str] = ..., anime_id: _Optional[str] = ..., pages: _Optional[_Iterable[_Union[_merge_pb2.StoredPage, _Mapping]]] = ..., corrections: _Optional[_Iterable[_Union[_merge_pb2.FieldCorrection, _Mapping]]] = ..., list_entries: _Optional[_Iterable[_Union[_merge_pb2.EpisodeListEntry, _Mapping]]] = ..., episode_number: _Optional[int] = ...) -> None: ...

class MergeEpisodeResponse(_message.Message):
    __slots__ = ("episode", "page_sources")
    EPISODE_FIELD_NUMBER: _ClassVar[int]
    PAGE_SOURCES_FIELD_NUMBER: _ClassVar[int]
    episode: _anime_pb2.Episode
    page_sources: _containers.RepeatedCompositeFieldContainer[_merge_pb2.PageSource]
    def __init__(self, episode: _Optional[_Union[_anime_pb2.Episode, _Mapping]] = ..., page_sources: _Optional[_Iterable[_Union[_merge_pb2.PageSource, _Mapping]]] = ...) -> None: ...

class MergeCharacterRequest(_message.Message):
    __slots__ = ("character_id", "pages", "corrections")
    CHARACTER_ID_FIELD_NUMBER: _ClassVar[int]
    PAGES_FIELD_NUMBER: _ClassVar[int]
    CORRECTIONS_FIELD_NUMBER: _ClassVar[int]
    character_id: str
    pages: _containers.RepeatedCompositeFieldContainer[_merge_pb2.StoredPage]
    corrections: _containers.RepeatedCompositeFieldContainer[_merge_pb2.FieldCorrection]
    def __init__(self, character_id: _Optional[str] = ..., pages: _Optional[_Iterable[_Union[_merge_pb2.StoredPage, _Mapping]]] = ..., corrections: _Optional[_Iterable[_Union[_merge_pb2.FieldCorrection, _Mapping]]] = ...) -> None: ...

class MergeCharacterResponse(_message.Message):
    __slots__ = ("character", "page_sources")
    CHARACTER_FIELD_NUMBER: _ClassVar[int]
    PAGE_SOURCES_FIELD_NUMBER: _ClassVar[int]
    character: _anime_pb2.Character
    page_sources: _containers.RepeatedCompositeFieldContainer[_merge_pb2.PageSource]
    def __init__(self, character: _Optional[_Union[_anime_pb2.Character, _Mapping]] = ..., page_sources: _Optional[_Iterable[_Union[_merge_pb2.PageSource, _Mapping]]] = ...) -> None: ...
