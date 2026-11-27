import datetime

from google.protobuf import duration_pb2 as _duration_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from shared_proto.v1 import anime_pb2 as _anime_pb2
from shared_proto.v1 import merge_pb2 as _merge_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class RefreshOutcome(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    REFRESH_OUTCOME_UNSPECIFIED: _ClassVar[RefreshOutcome]
    REFRESH_OUTCOME_SUCCEEDED: _ClassVar[RefreshOutcome]
    REFRESH_OUTCOME_PAGES_FAILED: _ClassVar[RefreshOutcome]
    REFRESH_OUTCOME_FAILED: _ClassVar[RefreshOutcome]
REFRESH_OUTCOME_UNSPECIFIED: RefreshOutcome
REFRESH_OUTCOME_SUCCEEDED: RefreshOutcome
REFRESH_OUTCOME_PAGES_FAILED: RefreshOutcome
REFRESH_OUTCOME_FAILED: RefreshOutcome

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
    __slots__ = ("anime_id", "expected_version", "anime", "merge_version", "relation_claims", "episodes", "page_sources")
    ANIME_ID_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_VERSION_FIELD_NUMBER: _ClassVar[int]
    ANIME_FIELD_NUMBER: _ClassVar[int]
    MERGE_VERSION_FIELD_NUMBER: _ClassVar[int]
    RELATION_CLAIMS_FIELD_NUMBER: _ClassVar[int]
    EPISODES_FIELD_NUMBER: _ClassVar[int]
    PAGE_SOURCES_FIELD_NUMBER: _ClassVar[int]
    anime_id: str
    expected_version: int
    anime: _anime_pb2.Anime
    merge_version: int
    relation_claims: _containers.RepeatedCompositeFieldContainer[_merge_pb2.RelationClaim]
    episodes: _containers.RepeatedCompositeFieldContainer[_merge_pb2.EpisodeReference]
    page_sources: _containers.RepeatedCompositeFieldContainer[_merge_pb2.PageSource]
    def __init__(self, anime_id: _Optional[str] = ..., expected_version: _Optional[int] = ..., anime: _Optional[_Union[_anime_pb2.Anime, _Mapping]] = ..., merge_version: _Optional[int] = ..., relation_claims: _Optional[_Iterable[_Union[_merge_pb2.RelationClaim, _Mapping]]] = ..., episodes: _Optional[_Iterable[_Union[_merge_pb2.EpisodeReference, _Mapping]]] = ..., page_sources: _Optional[_Iterable[_Union[_merge_pb2.PageSource, _Mapping]]] = ...) -> None: ...

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
    __slots__ = ("version", "anime_id", "pages", "corrections", "list_entries", "episode_number")
    VERSION_FIELD_NUMBER: _ClassVar[int]
    ANIME_ID_FIELD_NUMBER: _ClassVar[int]
    PAGES_FIELD_NUMBER: _ClassVar[int]
    CORRECTIONS_FIELD_NUMBER: _ClassVar[int]
    LIST_ENTRIES_FIELD_NUMBER: _ClassVar[int]
    EPISODE_NUMBER_FIELD_NUMBER: _ClassVar[int]
    version: int
    anime_id: str
    pages: _containers.RepeatedCompositeFieldContainer[_merge_pb2.StoredPage]
    corrections: _containers.RepeatedCompositeFieldContainer[_merge_pb2.FieldCorrection]
    list_entries: _containers.RepeatedCompositeFieldContainer[_merge_pb2.EpisodeListEntry]
    episode_number: int
    def __init__(self, version: _Optional[int] = ..., anime_id: _Optional[str] = ..., pages: _Optional[_Iterable[_Union[_merge_pb2.StoredPage, _Mapping]]] = ..., corrections: _Optional[_Iterable[_Union[_merge_pb2.FieldCorrection, _Mapping]]] = ..., list_entries: _Optional[_Iterable[_Union[_merge_pb2.EpisodeListEntry, _Mapping]]] = ..., episode_number: _Optional[int] = ...) -> None: ...

class SaveEpisodeRequest(_message.Message):
    __slots__ = ("episode_id", "expected_version", "episode", "merge_version", "page_sources")
    EPISODE_ID_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_VERSION_FIELD_NUMBER: _ClassVar[int]
    EPISODE_FIELD_NUMBER: _ClassVar[int]
    MERGE_VERSION_FIELD_NUMBER: _ClassVar[int]
    PAGE_SOURCES_FIELD_NUMBER: _ClassVar[int]
    episode_id: str
    expected_version: int
    episode: _anime_pb2.Episode
    merge_version: int
    page_sources: _containers.RepeatedCompositeFieldContainer[_merge_pb2.PageSource]
    def __init__(self, episode_id: _Optional[str] = ..., expected_version: _Optional[int] = ..., episode: _Optional[_Union[_anime_pb2.Episode, _Mapping]] = ..., merge_version: _Optional[int] = ..., page_sources: _Optional[_Iterable[_Union[_merge_pb2.PageSource, _Mapping]]] = ...) -> None: ...

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

class LoadCharacterRequest(_message.Message):
    __slots__ = ("character_id",)
    CHARACTER_ID_FIELD_NUMBER: _ClassVar[int]
    character_id: str
    def __init__(self, character_id: _Optional[str] = ...) -> None: ...

class LoadCharacterResponse(_message.Message):
    __slots__ = ("version", "pages", "corrections")
    VERSION_FIELD_NUMBER: _ClassVar[int]
    PAGES_FIELD_NUMBER: _ClassVar[int]
    CORRECTIONS_FIELD_NUMBER: _ClassVar[int]
    version: int
    pages: _containers.RepeatedCompositeFieldContainer[_merge_pb2.StoredPage]
    corrections: _containers.RepeatedCompositeFieldContainer[_merge_pb2.FieldCorrection]
    def __init__(self, version: _Optional[int] = ..., pages: _Optional[_Iterable[_Union[_merge_pb2.StoredPage, _Mapping]]] = ..., corrections: _Optional[_Iterable[_Union[_merge_pb2.FieldCorrection, _Mapping]]] = ...) -> None: ...

class SaveCharacterRequest(_message.Message):
    __slots__ = ("character_id", "expected_version", "character", "merge_version", "page_sources")
    CHARACTER_ID_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_VERSION_FIELD_NUMBER: _ClassVar[int]
    CHARACTER_FIELD_NUMBER: _ClassVar[int]
    MERGE_VERSION_FIELD_NUMBER: _ClassVar[int]
    PAGE_SOURCES_FIELD_NUMBER: _ClassVar[int]
    character_id: str
    expected_version: int
    character: _anime_pb2.Character
    merge_version: int
    page_sources: _containers.RepeatedCompositeFieldContainer[_merge_pb2.PageSource]
    def __init__(self, character_id: _Optional[str] = ..., expected_version: _Optional[int] = ..., character: _Optional[_Union[_anime_pb2.Character, _Mapping]] = ..., merge_version: _Optional[int] = ..., page_sources: _Optional[_Iterable[_Union[_merge_pb2.PageSource, _Mapping]]] = ...) -> None: ...

class SaveCharacterResponse(_message.Message):
    __slots__ = ("has_changed", "version")
    HAS_CHANGED_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    has_changed: bool
    version: int
    def __init__(self, has_changed: _Optional[bool] = ..., version: _Optional[int] = ...) -> None: ...

class ListEntityPagesRequest(_message.Message):
    __slots__ = ("entity_type", "entity_id")
    ENTITY_TYPE_FIELD_NUMBER: _ClassVar[int]
    ENTITY_ID_FIELD_NUMBER: _ClassVar[int]
    entity_type: _anime_pb2.EntityType
    entity_id: str
    def __init__(self, entity_type: _Optional[_Union[_anime_pb2.EntityType, str]] = ..., entity_id: _Optional[str] = ...) -> None: ...

class ListEntityPagesResponse(_message.Message):
    __slots__ = ("pages",)
    PAGES_FIELD_NUMBER: _ClassVar[int]
    pages: _containers.RepeatedCompositeFieldContainer[EntityPage]
    def __init__(self, pages: _Optional[_Iterable[_Union[EntityPage, _Mapping]]] = ...) -> None: ...

class EntityPage(_message.Message):
    __slots__ = ("page_key", "is_partial_source", "fetched_at", "is_dead", "fetch_failed_at", "fetch_failures")
    PAGE_KEY_FIELD_NUMBER: _ClassVar[int]
    IS_PARTIAL_SOURCE_FIELD_NUMBER: _ClassVar[int]
    FETCHED_AT_FIELD_NUMBER: _ClassVar[int]
    IS_DEAD_FIELD_NUMBER: _ClassVar[int]
    FETCH_FAILED_AT_FIELD_NUMBER: _ClassVar[int]
    FETCH_FAILURES_FIELD_NUMBER: _ClassVar[int]
    page_key: _merge_pb2.PageKey
    is_partial_source: bool
    fetched_at: _timestamp_pb2.Timestamp
    is_dead: bool
    fetch_failed_at: _timestamp_pb2.Timestamp
    fetch_failures: int
    def __init__(self, page_key: _Optional[_Union[_merge_pb2.PageKey, _Mapping]] = ..., is_partial_source: _Optional[bool] = ..., fetched_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., is_dead: _Optional[bool] = ..., fetch_failed_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., fetch_failures: _Optional[int] = ...) -> None: ...

class RecordRefreshRequest(_message.Message):
    __slots__ = ("entity_type", "entity_id", "outcome", "refresh_after")
    ENTITY_TYPE_FIELD_NUMBER: _ClassVar[int]
    ENTITY_ID_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_FIELD_NUMBER: _ClassVar[int]
    REFRESH_AFTER_FIELD_NUMBER: _ClassVar[int]
    entity_type: _anime_pb2.EntityType
    entity_id: str
    outcome: RefreshOutcome
    refresh_after: _duration_pb2.Duration
    def __init__(self, entity_type: _Optional[_Union[_anime_pb2.EntityType, str]] = ..., entity_id: _Optional[str] = ..., outcome: _Optional[_Union[RefreshOutcome, str]] = ..., refresh_after: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ...) -> None: ...

class RecordRefreshResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ListDueRequest(_message.Message):
    __slots__ = ("due_before", "older_after_years", "page_size", "after_next_refresh_at", "after_id", "merge_version")
    DUE_BEFORE_FIELD_NUMBER: _ClassVar[int]
    OLDER_AFTER_YEARS_FIELD_NUMBER: _ClassVar[int]
    PAGE_SIZE_FIELD_NUMBER: _ClassVar[int]
    AFTER_NEXT_REFRESH_AT_FIELD_NUMBER: _ClassVar[int]
    AFTER_ID_FIELD_NUMBER: _ClassVar[int]
    MERGE_VERSION_FIELD_NUMBER: _ClassVar[int]
    due_before: _timestamp_pb2.Timestamp
    older_after_years: int
    page_size: int
    after_next_refresh_at: _timestamp_pb2.Timestamp
    after_id: str
    merge_version: int
    def __init__(self, due_before: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., older_after_years: _Optional[int] = ..., page_size: _Optional[int] = ..., after_next_refresh_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., after_id: _Optional[str] = ..., merge_version: _Optional[int] = ...) -> None: ...

class ListDueResponse(_message.Message):
    __slots__ = ("entities",)
    ENTITIES_FIELD_NUMBER: _ClassVar[int]
    entities: _containers.RepeatedCompositeFieldContainer[DueEntity]
    def __init__(self, entities: _Optional[_Iterable[_Union[DueEntity, _Mapping]]] = ...) -> None: ...

class DueEntity(_message.Message):
    __slots__ = ("entity_id", "next_refresh_at")
    ENTITY_ID_FIELD_NUMBER: _ClassVar[int]
    NEXT_REFRESH_AT_FIELD_NUMBER: _ClassVar[int]
    entity_id: str
    next_refresh_at: _timestamp_pb2.Timestamp
    def __init__(self, entity_id: _Optional[str] = ..., next_refresh_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class ListUpcomingEpisodesRequest(_message.Message):
    __slots__ = ("window_days",)
    WINDOW_DAYS_FIELD_NUMBER: _ClassVar[int]
    window_days: int
    def __init__(self, window_days: _Optional[int] = ...) -> None: ...

class ListUpcomingEpisodesResponse(_message.Message):
    __slots__ = ("episodes",)
    EPISODES_FIELD_NUMBER: _ClassVar[int]
    episodes: _containers.RepeatedCompositeFieldContainer[UpcomingEpisode]
    def __init__(self, episodes: _Optional[_Iterable[_Union[UpcomingEpisode, _Mapping]]] = ...) -> None: ...

class UpcomingEpisode(_message.Message):
    __slots__ = ("anime_id", "episode_id", "episode_number", "expected_air_at")
    ANIME_ID_FIELD_NUMBER: _ClassVar[int]
    EPISODE_ID_FIELD_NUMBER: _ClassVar[int]
    EPISODE_NUMBER_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_AIR_AT_FIELD_NUMBER: _ClassVar[int]
    anime_id: str
    episode_id: str
    episode_number: int
    expected_air_at: _timestamp_pb2.Timestamp
    def __init__(self, anime_id: _Optional[str] = ..., episode_id: _Optional[str] = ..., episode_number: _Optional[int] = ..., expected_air_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class ListScoreAnimeRequest(_message.Message):
    __slots__ = ("lookback_days", "page_size", "after_id")
    LOOKBACK_DAYS_FIELD_NUMBER: _ClassVar[int]
    PAGE_SIZE_FIELD_NUMBER: _ClassVar[int]
    AFTER_ID_FIELD_NUMBER: _ClassVar[int]
    lookback_days: int
    page_size: int
    after_id: str
    def __init__(self, lookback_days: _Optional[int] = ..., page_size: _Optional[int] = ..., after_id: _Optional[str] = ...) -> None: ...

class ListScoreAnimeResponse(_message.Message):
    __slots__ = ("anime_ids",)
    ANIME_IDS_FIELD_NUMBER: _ClassVar[int]
    anime_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, anime_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class ListRequestedMergesRequest(_message.Message):
    __slots__ = ("entity_type", "limit")
    ENTITY_TYPE_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    entity_type: _anime_pb2.EntityType
    limit: int
    def __init__(self, entity_type: _Optional[_Union[_anime_pb2.EntityType, str]] = ..., limit: _Optional[int] = ...) -> None: ...

class ListRequestedMergesResponse(_message.Message):
    __slots__ = ("entity_ids",)
    ENTITY_IDS_FIELD_NUMBER: _ClassVar[int]
    entity_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, entity_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class GetEntityRequest(_message.Message):
    __slots__ = ("entity_id",)
    ENTITY_ID_FIELD_NUMBER: _ClassVar[int]
    entity_id: str
    def __init__(self, entity_id: _Optional[str] = ...) -> None: ...

class GetAnimeResponse(_message.Message):
    __slots__ = ("anime", "version", "next_refresh_at")
    ANIME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    NEXT_REFRESH_AT_FIELD_NUMBER: _ClassVar[int]
    anime: _anime_pb2.Anime
    version: int
    next_refresh_at: _timestamp_pb2.Timestamp
    def __init__(self, anime: _Optional[_Union[_anime_pb2.Anime, _Mapping]] = ..., version: _Optional[int] = ..., next_refresh_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class GetEpisodeResponse(_message.Message):
    __slots__ = ("episode", "version", "next_refresh_at")
    EPISODE_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    NEXT_REFRESH_AT_FIELD_NUMBER: _ClassVar[int]
    episode: _anime_pb2.Episode
    version: int
    next_refresh_at: _timestamp_pb2.Timestamp
    def __init__(self, episode: _Optional[_Union[_anime_pb2.Episode, _Mapping]] = ..., version: _Optional[int] = ..., next_refresh_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class GetCharacterResponse(_message.Message):
    __slots__ = ("character", "version", "next_refresh_at")
    CHARACTER_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    NEXT_REFRESH_AT_FIELD_NUMBER: _ClassVar[int]
    character: _anime_pb2.Character
    version: int
    next_refresh_at: _timestamp_pb2.Timestamp
    def __init__(self, character: _Optional[_Union[_anime_pb2.Character, _Mapping]] = ..., version: _Optional[int] = ..., next_refresh_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class ListEpisodesResponse(_message.Message):
    __slots__ = ("episodes",)
    EPISODES_FIELD_NUMBER: _ClassVar[int]
    episodes: _containers.RepeatedCompositeFieldContainer[GetEpisodeResponse]
    def __init__(self, episodes: _Optional[_Iterable[_Union[GetEpisodeResponse, _Mapping]]] = ...) -> None: ...

class ListEntityVersionsRequest(_message.Message):
    __slots__ = ("entity_type", "page_size", "after_id")
    ENTITY_TYPE_FIELD_NUMBER: _ClassVar[int]
    PAGE_SIZE_FIELD_NUMBER: _ClassVar[int]
    AFTER_ID_FIELD_NUMBER: _ClassVar[int]
    entity_type: _anime_pb2.EntityType
    page_size: int
    after_id: str
    def __init__(self, entity_type: _Optional[_Union[_anime_pb2.EntityType, str]] = ..., page_size: _Optional[int] = ..., after_id: _Optional[str] = ...) -> None: ...

class ListEntityVersionsResponse(_message.Message):
    __slots__ = ("entities",)
    ENTITIES_FIELD_NUMBER: _ClassVar[int]
    entities: _containers.RepeatedCompositeFieldContainer[EntityVersion]
    def __init__(self, entities: _Optional[_Iterable[_Union[EntityVersion, _Mapping]]] = ...) -> None: ...

class EntityVersion(_message.Message):
    __slots__ = ("entity_id", "version")
    ENTITY_ID_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    entity_id: str
    version: int
    def __init__(self, entity_id: _Optional[str] = ..., version: _Optional[int] = ...) -> None: ...

class ListDuePagesRequest(_message.Message):
    __slots__ = ("provider", "page_size")
    PROVIDER_FIELD_NUMBER: _ClassVar[int]
    PAGE_SIZE_FIELD_NUMBER: _ClassVar[int]
    provider: str
    page_size: int
    def __init__(self, provider: _Optional[str] = ..., page_size: _Optional[int] = ...) -> None: ...

class ListDuePagesResponse(_message.Message):
    __slots__ = ("pages",)
    PAGES_FIELD_NUMBER: _ClassVar[int]
    pages: _containers.RepeatedCompositeFieldContainer[DuePage]
    def __init__(self, pages: _Optional[_Iterable[_Union[DuePage, _Mapping]]] = ...) -> None: ...

class DuePage(_message.Message):
    __slots__ = ("page_key", "anime_id")
    PAGE_KEY_FIELD_NUMBER: _ClassVar[int]
    ANIME_ID_FIELD_NUMBER: _ClassVar[int]
    page_key: _merge_pb2.PageKey
    anime_id: str
    def __init__(self, page_key: _Optional[_Union[_merge_pb2.PageKey, _Mapping]] = ..., anime_id: _Optional[str] = ...) -> None: ...
