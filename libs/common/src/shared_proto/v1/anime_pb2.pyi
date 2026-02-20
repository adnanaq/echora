import datetime

from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AnimeStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ANIME_STATUS_UNSPECIFIED: _ClassVar[AnimeStatus]
    ANIME_STATUS_FINISHED: _ClassVar[AnimeStatus]
    ANIME_STATUS_UPCOMING: _ClassVar[AnimeStatus]
    ANIME_STATUS_ONGOING: _ClassVar[AnimeStatus]
    ANIME_STATUS_UNKNOWN: _ClassVar[AnimeStatus]

class EntityType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ENTITY_TYPE_UNSPECIFIED: _ClassVar[EntityType]
    ENTITY_TYPE_ANIME: _ClassVar[EntityType]
    ENTITY_TYPE_CHARACTER: _ClassVar[EntityType]
    ENTITY_TYPE_EPISODE: _ClassVar[EntityType]

class AnimeType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ANIME_TYPE_UNSPECIFIED: _ClassVar[AnimeType]
    ANIME_TYPE_MOVIE: _ClassVar[AnimeType]
    ANIME_TYPE_ONA: _ClassVar[AnimeType]
    ANIME_TYPE_OVA: _ClassVar[AnimeType]
    ANIME_TYPE_SPECIAL: _ClassVar[AnimeType]
    ANIME_TYPE_TV: _ClassVar[AnimeType]
    ANIME_TYPE_UNKNOWN: _ClassVar[AnimeType]
    ANIME_TYPE_MUSIC: _ClassVar[AnimeType]
    ANIME_TYPE_PV: _ClassVar[AnimeType]

class SourceMaterialType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SOURCE_MATERIAL_TYPE_UNSPECIFIED: _ClassVar[SourceMaterialType]
    SOURCE_MATERIAL_TYPE_MANGA: _ClassVar[SourceMaterialType]
    SOURCE_MATERIAL_TYPE_LIGHT_NOVEL: _ClassVar[SourceMaterialType]
    SOURCE_MATERIAL_TYPE_NOVEL: _ClassVar[SourceMaterialType]
    SOURCE_MATERIAL_TYPE_VISUAL_NOVEL: _ClassVar[SourceMaterialType]
    SOURCE_MATERIAL_TYPE_GAME: _ClassVar[SourceMaterialType]
    SOURCE_MATERIAL_TYPE_WEB_MANGA: _ClassVar[SourceMaterialType]
    SOURCE_MATERIAL_TYPE_WEB_NOVEL: _ClassVar[SourceMaterialType]
    SOURCE_MATERIAL_TYPE_OTHER: _ClassVar[SourceMaterialType]
    SOURCE_MATERIAL_TYPE_ORIGINAL: _ClassVar[SourceMaterialType]
    SOURCE_MATERIAL_TYPE_MIXED_MEDIA: _ClassVar[SourceMaterialType]
    SOURCE_MATERIAL_TYPE_UNKNOWN: _ClassVar[SourceMaterialType]

class AnimeRating(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ANIME_RATING_UNSPECIFIED: _ClassVar[AnimeRating]
    ANIME_RATING_G: _ClassVar[AnimeRating]
    ANIME_RATING_PG: _ClassVar[AnimeRating]
    ANIME_RATING_PG13: _ClassVar[AnimeRating]
    ANIME_RATING_R: _ClassVar[AnimeRating]
    ANIME_RATING_RPLUS: _ClassVar[AnimeRating]
    ANIME_RATING_RX: _ClassVar[AnimeRating]

class AnimeSeason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ANIME_SEASON_UNSPECIFIED: _ClassVar[AnimeSeason]
    ANIME_SEASON_SPRING: _ClassVar[AnimeSeason]
    ANIME_SEASON_SUMMER: _ClassVar[AnimeSeason]
    ANIME_SEASON_FALL: _ClassVar[AnimeSeason]
    ANIME_SEASON_WINTER: _ClassVar[AnimeSeason]

class CharacterRole(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CHARACTER_ROLE_UNSPECIFIED: _ClassVar[CharacterRole]
    CHARACTER_ROLE_MAIN: _ClassVar[CharacterRole]
    CHARACTER_ROLE_SUPPORTING: _ClassVar[CharacterRole]
    CHARACTER_ROLE_BACKGROUND: _ClassVar[CharacterRole]

class AnimeRelationType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ANIME_RELATION_TYPE_UNSPECIFIED: _ClassVar[AnimeRelationType]
    ANIME_RELATION_TYPE_SEQUEL: _ClassVar[AnimeRelationType]
    ANIME_RELATION_TYPE_PREQUEL: _ClassVar[AnimeRelationType]
    ANIME_RELATION_TYPE_SIDE_STORY: _ClassVar[AnimeRelationType]
    ANIME_RELATION_TYPE_ALTERNATIVE: _ClassVar[AnimeRelationType]
    ANIME_RELATION_TYPE_SUMMARY: _ClassVar[AnimeRelationType]
    ANIME_RELATION_TYPE_FULL_STORY: _ClassVar[AnimeRelationType]
    ANIME_RELATION_TYPE_CHARACTER: _ClassVar[AnimeRelationType]
    ANIME_RELATION_TYPE_SPIN_OFF: _ClassVar[AnimeRelationType]
    ANIME_RELATION_TYPE_ADAPTATION: _ClassVar[AnimeRelationType]
    ANIME_RELATION_TYPE_OTHER: _ClassVar[AnimeRelationType]

class SourceMaterialRelationType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SOURCE_MATERIAL_RELATION_TYPE_UNSPECIFIED: _ClassVar[SourceMaterialRelationType]
    SOURCE_MATERIAL_RELATION_TYPE_ADAPTATION: _ClassVar[SourceMaterialRelationType]
    SOURCE_MATERIAL_RELATION_TYPE_SOURCE: _ClassVar[SourceMaterialRelationType]
    SOURCE_MATERIAL_RELATION_TYPE_ALTERNATIVE: _ClassVar[SourceMaterialRelationType]
    SOURCE_MATERIAL_RELATION_TYPE_SPIN_OFF: _ClassVar[SourceMaterialRelationType]
    SOURCE_MATERIAL_RELATION_TYPE_OTHER: _ClassVar[SourceMaterialRelationType]
ANIME_STATUS_UNSPECIFIED: AnimeStatus
ANIME_STATUS_FINISHED: AnimeStatus
ANIME_STATUS_UPCOMING: AnimeStatus
ANIME_STATUS_ONGOING: AnimeStatus
ANIME_STATUS_UNKNOWN: AnimeStatus
ENTITY_TYPE_UNSPECIFIED: EntityType
ENTITY_TYPE_ANIME: EntityType
ENTITY_TYPE_CHARACTER: EntityType
ENTITY_TYPE_EPISODE: EntityType
ANIME_TYPE_UNSPECIFIED: AnimeType
ANIME_TYPE_MOVIE: AnimeType
ANIME_TYPE_ONA: AnimeType
ANIME_TYPE_OVA: AnimeType
ANIME_TYPE_SPECIAL: AnimeType
ANIME_TYPE_TV: AnimeType
ANIME_TYPE_UNKNOWN: AnimeType
ANIME_TYPE_MUSIC: AnimeType
ANIME_TYPE_PV: AnimeType
SOURCE_MATERIAL_TYPE_UNSPECIFIED: SourceMaterialType
SOURCE_MATERIAL_TYPE_MANGA: SourceMaterialType
SOURCE_MATERIAL_TYPE_LIGHT_NOVEL: SourceMaterialType
SOURCE_MATERIAL_TYPE_NOVEL: SourceMaterialType
SOURCE_MATERIAL_TYPE_VISUAL_NOVEL: SourceMaterialType
SOURCE_MATERIAL_TYPE_GAME: SourceMaterialType
SOURCE_MATERIAL_TYPE_WEB_MANGA: SourceMaterialType
SOURCE_MATERIAL_TYPE_WEB_NOVEL: SourceMaterialType
SOURCE_MATERIAL_TYPE_OTHER: SourceMaterialType
SOURCE_MATERIAL_TYPE_ORIGINAL: SourceMaterialType
SOURCE_MATERIAL_TYPE_MIXED_MEDIA: SourceMaterialType
SOURCE_MATERIAL_TYPE_UNKNOWN: SourceMaterialType
ANIME_RATING_UNSPECIFIED: AnimeRating
ANIME_RATING_G: AnimeRating
ANIME_RATING_PG: AnimeRating
ANIME_RATING_PG13: AnimeRating
ANIME_RATING_R: AnimeRating
ANIME_RATING_RPLUS: AnimeRating
ANIME_RATING_RX: AnimeRating
ANIME_SEASON_UNSPECIFIED: AnimeSeason
ANIME_SEASON_SPRING: AnimeSeason
ANIME_SEASON_SUMMER: AnimeSeason
ANIME_SEASON_FALL: AnimeSeason
ANIME_SEASON_WINTER: AnimeSeason
CHARACTER_ROLE_UNSPECIFIED: CharacterRole
CHARACTER_ROLE_MAIN: CharacterRole
CHARACTER_ROLE_SUPPORTING: CharacterRole
CHARACTER_ROLE_BACKGROUND: CharacterRole
ANIME_RELATION_TYPE_UNSPECIFIED: AnimeRelationType
ANIME_RELATION_TYPE_SEQUEL: AnimeRelationType
ANIME_RELATION_TYPE_PREQUEL: AnimeRelationType
ANIME_RELATION_TYPE_SIDE_STORY: AnimeRelationType
ANIME_RELATION_TYPE_ALTERNATIVE: AnimeRelationType
ANIME_RELATION_TYPE_SUMMARY: AnimeRelationType
ANIME_RELATION_TYPE_FULL_STORY: AnimeRelationType
ANIME_RELATION_TYPE_CHARACTER: AnimeRelationType
ANIME_RELATION_TYPE_SPIN_OFF: AnimeRelationType
ANIME_RELATION_TYPE_ADAPTATION: AnimeRelationType
ANIME_RELATION_TYPE_OTHER: AnimeRelationType
SOURCE_MATERIAL_RELATION_TYPE_UNSPECIFIED: SourceMaterialRelationType
SOURCE_MATERIAL_RELATION_TYPE_ADAPTATION: SourceMaterialRelationType
SOURCE_MATERIAL_RELATION_TYPE_SOURCE: SourceMaterialRelationType
SOURCE_MATERIAL_RELATION_TYPE_ALTERNATIVE: SourceMaterialRelationType
SOURCE_MATERIAL_RELATION_TYPE_SPIN_OFF: SourceMaterialRelationType
SOURCE_MATERIAL_RELATION_TYPE_OTHER: SourceMaterialRelationType

class StringList(_message.Message):
    __slots__ = ("values",)
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, values: _Optional[_Iterable[str]] = ...) -> None: ...

class TrailerEntry(_message.Message):
    __slots__ = ("url", "title", "thumbnail_url")
    URL_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    THUMBNAIL_URL_FIELD_NUMBER: _ClassVar[int]
    url: str
    title: str
    thumbnail_url: str
    def __init__(self, url: _Optional[str] = ..., title: _Optional[str] = ..., thumbnail_url: _Optional[str] = ...) -> None: ...

class AiredDates(_message.Message):
    __slots__ = ("from_date", "to", "string")
    FROM_DATE_FIELD_NUMBER: _ClassVar[int]
    TO_FIELD_NUMBER: _ClassVar[int]
    STRING_FIELD_NUMBER: _ClassVar[int]
    from_date: _timestamp_pb2.Timestamp
    to: _timestamp_pb2.Timestamp
    string: str
    def __init__(self, from_date: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., to: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., string: _Optional[str] = ...) -> None: ...

class Broadcast(_message.Message):
    __slots__ = ("day", "time", "timezone", "jp_time", "sub_time", "dub_time", "sub_delay_days", "dub_delay_days", "premiere_jp", "premiere_sub", "premiere_dub")
    DAY_FIELD_NUMBER: _ClassVar[int]
    TIME_FIELD_NUMBER: _ClassVar[int]
    TIMEZONE_FIELD_NUMBER: _ClassVar[int]
    JP_TIME_FIELD_NUMBER: _ClassVar[int]
    SUB_TIME_FIELD_NUMBER: _ClassVar[int]
    DUB_TIME_FIELD_NUMBER: _ClassVar[int]
    SUB_DELAY_DAYS_FIELD_NUMBER: _ClassVar[int]
    DUB_DELAY_DAYS_FIELD_NUMBER: _ClassVar[int]
    PREMIERE_JP_FIELD_NUMBER: _ClassVar[int]
    PREMIERE_SUB_FIELD_NUMBER: _ClassVar[int]
    PREMIERE_DUB_FIELD_NUMBER: _ClassVar[int]
    day: str
    time: str
    timezone: str
    jp_time: str
    sub_time: str
    dub_time: str
    sub_delay_days: int
    dub_delay_days: int
    premiere_jp: str
    premiere_sub: str
    premiere_dub: str
    def __init__(self, day: _Optional[str] = ..., time: _Optional[str] = ..., timezone: _Optional[str] = ..., jp_time: _Optional[str] = ..., sub_time: _Optional[str] = ..., dub_time: _Optional[str] = ..., sub_delay_days: _Optional[int] = ..., dub_delay_days: _Optional[int] = ..., premiere_jp: _Optional[str] = ..., premiere_sub: _Optional[str] = ..., premiere_dub: _Optional[str] = ...) -> None: ...

class AnimeHiatus(_message.Message):
    __slots__ = ("reason", "hiatus_from", "hiatus_until")
    REASON_FIELD_NUMBER: _ClassVar[int]
    HIATUS_FROM_FIELD_NUMBER: _ClassVar[int]
    HIATUS_UNTIL_FIELD_NUMBER: _ClassVar[int]
    reason: str
    hiatus_from: str
    hiatus_until: str
    def __init__(self, reason: _Optional[str] = ..., hiatus_from: _Optional[str] = ..., hiatus_until: _Optional[str] = ...) -> None: ...

class EnrichmentMetadata(_message.Message):
    __slots__ = ("source", "enriched_at", "success", "error_message")
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    ENRICHED_AT_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    source: str
    enriched_at: _timestamp_pb2.Timestamp
    success: bool
    error_message: str
    def __init__(self, source: _Optional[str] = ..., enriched_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., success: bool = ..., error_message: _Optional[str] = ...) -> None: ...

class SourceMaterial(_message.Message):
    __slots__ = ("title", "relation_type", "url", "type")
    TITLE_FIELD_NUMBER: _ClassVar[int]
    RELATION_TYPE_FIELD_NUMBER: _ClassVar[int]
    URL_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    title: str
    relation_type: SourceMaterialRelationType
    url: str
    type: SourceMaterialType
    def __init__(self, title: _Optional[str] = ..., relation_type: _Optional[_Union[SourceMaterialRelationType, str]] = ..., url: _Optional[str] = ..., type: _Optional[_Union[SourceMaterialType, str]] = ...) -> None: ...

class RelatedAnimeEntry(_message.Message):
    __slots__ = ("title", "relation_type", "url", "type")
    TITLE_FIELD_NUMBER: _ClassVar[int]
    RELATION_TYPE_FIELD_NUMBER: _ClassVar[int]
    URL_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    title: str
    relation_type: AnimeRelationType
    url: str
    type: AnimeType
    def __init__(self, title: _Optional[str] = ..., relation_type: _Optional[_Union[AnimeRelationType, str]] = ..., url: _Optional[str] = ..., type: _Optional[_Union[AnimeType, str]] = ...) -> None: ...

class StreamingEntry(_message.Message):
    __slots__ = ("platform", "url", "region", "free", "premium_required", "dub_available", "subtitle_languages")
    PLATFORM_FIELD_NUMBER: _ClassVar[int]
    URL_FIELD_NUMBER: _ClassVar[int]
    REGION_FIELD_NUMBER: _ClassVar[int]
    FREE_FIELD_NUMBER: _ClassVar[int]
    PREMIUM_REQUIRED_FIELD_NUMBER: _ClassVar[int]
    DUB_AVAILABLE_FIELD_NUMBER: _ClassVar[int]
    SUBTITLE_LANGUAGES_FIELD_NUMBER: _ClassVar[int]
    platform: str
    url: str
    region: str
    free: bool
    premium_required: bool
    dub_available: bool
    subtitle_languages: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, platform: _Optional[str] = ..., url: _Optional[str] = ..., region: _Optional[str] = ..., free: bool = ..., premium_required: bool = ..., dub_available: bool = ..., subtitle_languages: _Optional[_Iterable[str]] = ...) -> None: ...

class ThemeEntry(_message.Message):
    __slots__ = ("name", "description")
    NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    name: str
    description: str
    def __init__(self, name: _Optional[str] = ..., description: _Optional[str] = ...) -> None: ...

class ThemeSong(_message.Message):
    __slots__ = ("title", "artist", "episodes")
    TITLE_FIELD_NUMBER: _ClassVar[int]
    ARTIST_FIELD_NUMBER: _ClassVar[int]
    EPISODES_FIELD_NUMBER: _ClassVar[int]
    title: str
    artist: str
    episodes: str
    def __init__(self, title: _Optional[str] = ..., artist: _Optional[str] = ..., episodes: _Optional[str] = ...) -> None: ...

class SimpleVoiceActor(_message.Message):
    __slots__ = ("name", "language")
    NAME_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    name: str
    language: str
    def __init__(self, name: _Optional[str] = ..., language: _Optional[str] = ...) -> None: ...

class VoiceActorEntry(_message.Message):
    __slots__ = ("name", "native_name", "language", "image")
    NAME_FIELD_NUMBER: _ClassVar[int]
    NATIVE_NAME_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    name: str
    native_name: str
    language: str
    image: str
    def __init__(self, name: _Optional[str] = ..., native_name: _Optional[str] = ..., language: _Optional[str] = ..., image: _Optional[str] = ...) -> None: ...

class StaffMember(_message.Message):
    __slots__ = ("staff_ids", "name", "native_name", "role", "image", "biography", "birth_date", "hometown", "primary_occupations", "years_active", "gender", "blood_type", "community_favorites", "enhancement_status")
    class StaffIdsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    STAFF_IDS_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    NATIVE_NAME_FIELD_NUMBER: _ClassVar[int]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    BIOGRAPHY_FIELD_NUMBER: _ClassVar[int]
    BIRTH_DATE_FIELD_NUMBER: _ClassVar[int]
    HOMETOWN_FIELD_NUMBER: _ClassVar[int]
    PRIMARY_OCCUPATIONS_FIELD_NUMBER: _ClassVar[int]
    YEARS_ACTIVE_FIELD_NUMBER: _ClassVar[int]
    GENDER_FIELD_NUMBER: _ClassVar[int]
    BLOOD_TYPE_FIELD_NUMBER: _ClassVar[int]
    COMMUNITY_FAVORITES_FIELD_NUMBER: _ClassVar[int]
    ENHANCEMENT_STATUS_FIELD_NUMBER: _ClassVar[int]
    staff_ids: _containers.ScalarMap[str, int]
    name: str
    native_name: str
    role: str
    image: str
    biography: str
    birth_date: str
    hometown: str
    primary_occupations: _containers.RepeatedScalarFieldContainer[str]
    years_active: _containers.RepeatedScalarFieldContainer[int]
    gender: str
    blood_type: str
    community_favorites: int
    enhancement_status: str
    def __init__(self, staff_ids: _Optional[_Mapping[str, int]] = ..., name: _Optional[str] = ..., native_name: _Optional[str] = ..., role: _Optional[str] = ..., image: _Optional[str] = ..., biography: _Optional[str] = ..., birth_date: _Optional[str] = ..., hometown: _Optional[str] = ..., primary_occupations: _Optional[_Iterable[str]] = ..., years_active: _Optional[_Iterable[int]] = ..., gender: _Optional[str] = ..., blood_type: _Optional[str] = ..., community_favorites: _Optional[int] = ..., enhancement_status: _Optional[str] = ...) -> None: ...

class VoiceActor(_message.Message):
    __slots__ = ("staff_ids", "name", "native_name", "character_assignments", "image", "biography", "birth_date", "blood_type")
    class StaffIdsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    STAFF_IDS_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    NATIVE_NAME_FIELD_NUMBER: _ClassVar[int]
    CHARACTER_ASSIGNMENTS_FIELD_NUMBER: _ClassVar[int]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    BIOGRAPHY_FIELD_NUMBER: _ClassVar[int]
    BIRTH_DATE_FIELD_NUMBER: _ClassVar[int]
    BLOOD_TYPE_FIELD_NUMBER: _ClassVar[int]
    staff_ids: _containers.ScalarMap[str, int]
    name: str
    native_name: str
    character_assignments: _containers.RepeatedScalarFieldContainer[str]
    image: str
    biography: str
    birth_date: str
    blood_type: str
    def __init__(self, staff_ids: _Optional[_Mapping[str, int]] = ..., name: _Optional[str] = ..., native_name: _Optional[str] = ..., character_assignments: _Optional[_Iterable[str]] = ..., image: _Optional[str] = ..., biography: _Optional[str] = ..., birth_date: _Optional[str] = ..., blood_type: _Optional[str] = ...) -> None: ...

class CompanyEntry(_message.Message):
    __slots__ = ("name", "sources", "description")
    NAME_FIELD_NUMBER: _ClassVar[int]
    SOURCES_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    name: str
    sources: _containers.RepeatedScalarFieldContainer[str]
    description: str
    def __init__(self, name: _Optional[str] = ..., sources: _Optional[_Iterable[str]] = ..., description: _Optional[str] = ...) -> None: ...

class StaffMemberList(_message.Message):
    __slots__ = ("items",)
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    items: _containers.RepeatedCompositeFieldContainer[StaffMember]
    def __init__(self, items: _Optional[_Iterable[_Union[StaffMember, _Mapping]]] = ...) -> None: ...

class ProductionStaff(_message.Message):
    __slots__ = ("roles",)
    class RolesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: StaffMemberList
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[StaffMemberList, _Mapping]] = ...) -> None: ...
    ROLES_FIELD_NUMBER: _ClassVar[int]
    roles: _containers.MessageMap[str, StaffMemberList]
    def __init__(self, roles: _Optional[_Mapping[str, StaffMemberList]] = ...) -> None: ...

class VoiceActors(_message.Message):
    __slots__ = ("japanese",)
    JAPANESE_FIELD_NUMBER: _ClassVar[int]
    japanese: _containers.RepeatedCompositeFieldContainer[VoiceActor]
    def __init__(self, japanese: _Optional[_Iterable[_Union[VoiceActor, _Mapping]]] = ...) -> None: ...

class StaffData(_message.Message):
    __slots__ = ("production_staff", "licensors", "voice_actors")
    PRODUCTION_STAFF_FIELD_NUMBER: _ClassVar[int]
    LICENSORS_FIELD_NUMBER: _ClassVar[int]
    VOICE_ACTORS_FIELD_NUMBER: _ClassVar[int]
    production_staff: ProductionStaff
    licensors: _containers.RepeatedCompositeFieldContainer[CompanyEntry]
    voice_actors: VoiceActors
    def __init__(self, production_staff: _Optional[_Union[ProductionStaff, _Mapping]] = ..., licensors: _Optional[_Iterable[_Union[CompanyEntry, _Mapping]]] = ..., voice_actors: _Optional[_Union[VoiceActors, _Mapping]] = ...) -> None: ...

class ContextualRank(_message.Message):
    __slots__ = ("rank", "type", "format", "year", "season", "all_time")
    RANK_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    YEAR_FIELD_NUMBER: _ClassVar[int]
    SEASON_FIELD_NUMBER: _ClassVar[int]
    ALL_TIME_FIELD_NUMBER: _ClassVar[int]
    rank: int
    type: str
    format: str
    year: int
    season: str
    all_time: bool
    def __init__(self, rank: _Optional[int] = ..., type: _Optional[str] = ..., format: _Optional[str] = ..., year: _Optional[int] = ..., season: _Optional[str] = ..., all_time: bool = ...) -> None: ...

class Statistics(_message.Message):
    __slots__ = ("score", "scored_by", "rank", "popularity_rank", "members", "favorites", "contextual_ranks")
    SCORE_FIELD_NUMBER: _ClassVar[int]
    SCORED_BY_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    POPULARITY_RANK_FIELD_NUMBER: _ClassVar[int]
    MEMBERS_FIELD_NUMBER: _ClassVar[int]
    FAVORITES_FIELD_NUMBER: _ClassVar[int]
    CONTEXTUAL_RANKS_FIELD_NUMBER: _ClassVar[int]
    score: float
    scored_by: int
    rank: int
    popularity_rank: int
    members: int
    favorites: int
    contextual_ranks: _containers.RepeatedCompositeFieldContainer[ContextualRank]
    def __init__(self, score: _Optional[float] = ..., scored_by: _Optional[int] = ..., rank: _Optional[int] = ..., popularity_rank: _Optional[int] = ..., members: _Optional[int] = ..., favorites: _Optional[int] = ..., contextual_ranks: _Optional[_Iterable[_Union[ContextualRank, _Mapping]]] = ...) -> None: ...

class ScoreCalculations(_message.Message):
    __slots__ = ("arithmetic_geometric_mean", "arithmetic_mean", "median")
    ARITHMETIC_GEOMETRIC_MEAN_FIELD_NUMBER: _ClassVar[int]
    ARITHMETIC_MEAN_FIELD_NUMBER: _ClassVar[int]
    MEDIAN_FIELD_NUMBER: _ClassVar[int]
    arithmetic_geometric_mean: float
    arithmetic_mean: float
    median: float
    def __init__(self, arithmetic_geometric_mean: _Optional[float] = ..., arithmetic_mean: _Optional[float] = ..., median: _Optional[float] = ...) -> None: ...

class Character(_message.Message):
    __slots__ = ("age", "description", "eye_color", "favorites", "gender", "hair_color", "id", "entity_type", "name", "name_native", "role", "anime_ids", "character_traits", "images", "name_variations", "nicknames", "voice_actors", "sources")
    AGE_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    EYE_COLOR_FIELD_NUMBER: _ClassVar[int]
    FAVORITES_FIELD_NUMBER: _ClassVar[int]
    GENDER_FIELD_NUMBER: _ClassVar[int]
    HAIR_COLOR_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    ENTITY_TYPE_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    NAME_NATIVE_FIELD_NUMBER: _ClassVar[int]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    ANIME_IDS_FIELD_NUMBER: _ClassVar[int]
    CHARACTER_TRAITS_FIELD_NUMBER: _ClassVar[int]
    IMAGES_FIELD_NUMBER: _ClassVar[int]
    NAME_VARIATIONS_FIELD_NUMBER: _ClassVar[int]
    NICKNAMES_FIELD_NUMBER: _ClassVar[int]
    VOICE_ACTORS_FIELD_NUMBER: _ClassVar[int]
    SOURCES_FIELD_NUMBER: _ClassVar[int]
    age: str
    description: str
    eye_color: str
    favorites: int
    gender: str
    hair_color: str
    id: str
    entity_type: EntityType
    name: str
    name_native: str
    role: str
    anime_ids: _containers.RepeatedScalarFieldContainer[str]
    character_traits: _containers.RepeatedScalarFieldContainer[str]
    images: _containers.RepeatedScalarFieldContainer[str]
    name_variations: _containers.RepeatedScalarFieldContainer[str]
    nicknames: _containers.RepeatedScalarFieldContainer[str]
    voice_actors: _containers.RepeatedCompositeFieldContainer[SimpleVoiceActor]
    sources: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, age: _Optional[str] = ..., description: _Optional[str] = ..., eye_color: _Optional[str] = ..., favorites: _Optional[int] = ..., gender: _Optional[str] = ..., hair_color: _Optional[str] = ..., id: _Optional[str] = ..., entity_type: _Optional[_Union[EntityType, str]] = ..., name: _Optional[str] = ..., name_native: _Optional[str] = ..., role: _Optional[str] = ..., anime_ids: _Optional[_Iterable[str]] = ..., character_traits: _Optional[_Iterable[str]] = ..., images: _Optional[_Iterable[str]] = ..., name_variations: _Optional[_Iterable[str]] = ..., nicknames: _Optional[_Iterable[str]] = ..., voice_actors: _Optional[_Iterable[_Union[SimpleVoiceActor, _Mapping]]] = ..., sources: _Optional[_Iterable[str]] = ...) -> None: ...

class Episode(_message.Message):
    __slots__ = ("aired", "anime_id", "description", "duration", "episode_number", "filler", "id", "entity_type", "recap", "score", "season_number", "synopsis", "title", "title_japanese", "title_romaji", "images", "streaming", "sources")
    class StreamingEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    AIRED_FIELD_NUMBER: _ClassVar[int]
    ANIME_ID_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    DURATION_FIELD_NUMBER: _ClassVar[int]
    EPISODE_NUMBER_FIELD_NUMBER: _ClassVar[int]
    FILLER_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    ENTITY_TYPE_FIELD_NUMBER: _ClassVar[int]
    RECAP_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    SEASON_NUMBER_FIELD_NUMBER: _ClassVar[int]
    SYNOPSIS_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    TITLE_JAPANESE_FIELD_NUMBER: _ClassVar[int]
    TITLE_ROMAJI_FIELD_NUMBER: _ClassVar[int]
    IMAGES_FIELD_NUMBER: _ClassVar[int]
    STREAMING_FIELD_NUMBER: _ClassVar[int]
    SOURCES_FIELD_NUMBER: _ClassVar[int]
    aired: _timestamp_pb2.Timestamp
    anime_id: str
    description: str
    duration: int
    episode_number: int
    filler: bool
    id: str
    entity_type: EntityType
    recap: bool
    score: float
    season_number: int
    synopsis: str
    title: str
    title_japanese: str
    title_romaji: str
    images: _containers.RepeatedScalarFieldContainer[str]
    streaming: _containers.ScalarMap[str, str]
    sources: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, aired: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., anime_id: _Optional[str] = ..., description: _Optional[str] = ..., duration: _Optional[int] = ..., episode_number: _Optional[int] = ..., filler: bool = ..., id: _Optional[str] = ..., entity_type: _Optional[_Union[EntityType, str]] = ..., recap: bool = ..., score: _Optional[float] = ..., season_number: _Optional[int] = ..., synopsis: _Optional[str] = ..., title: _Optional[str] = ..., title_japanese: _Optional[str] = ..., title_romaji: _Optional[str] = ..., images: _Optional[_Iterable[str]] = ..., streaming: _Optional[_Mapping[str, str]] = ..., sources: _Optional[_Iterable[str]] = ...) -> None: ...

class Anime(_message.Message):
    __slots__ = ("background", "duration", "episode_count", "id", "entity_type", "month", "nsfw", "rating", "season", "similarity_score", "source_material", "status", "synopsis", "title", "title_english", "title_japanese", "type", "year", "content_warnings", "demographics", "ending_themes", "genres", "opening_themes", "related_anime", "related_source_material", "sources", "streaming_info", "streaming_licenses", "synonyms", "tags", "themes", "trailers", "producers", "studios", "aired_dates", "broadcast", "enrichment_metadata", "external_links", "hiatus", "images", "popularity_trends", "score", "staff_data", "statistics")
    class ExternalLinksEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    class ImagesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: StringList
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[StringList, _Mapping]] = ...) -> None: ...
    class StatisticsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Statistics
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Statistics, _Mapping]] = ...) -> None: ...
    BACKGROUND_FIELD_NUMBER: _ClassVar[int]
    DURATION_FIELD_NUMBER: _ClassVar[int]
    EPISODE_COUNT_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    ENTITY_TYPE_FIELD_NUMBER: _ClassVar[int]
    MONTH_FIELD_NUMBER: _ClassVar[int]
    NSFW_FIELD_NUMBER: _ClassVar[int]
    RATING_FIELD_NUMBER: _ClassVar[int]
    SEASON_FIELD_NUMBER: _ClassVar[int]
    SIMILARITY_SCORE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_MATERIAL_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    SYNOPSIS_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    TITLE_ENGLISH_FIELD_NUMBER: _ClassVar[int]
    TITLE_JAPANESE_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    YEAR_FIELD_NUMBER: _ClassVar[int]
    CONTENT_WARNINGS_FIELD_NUMBER: _ClassVar[int]
    DEMOGRAPHICS_FIELD_NUMBER: _ClassVar[int]
    ENDING_THEMES_FIELD_NUMBER: _ClassVar[int]
    GENRES_FIELD_NUMBER: _ClassVar[int]
    OPENING_THEMES_FIELD_NUMBER: _ClassVar[int]
    RELATED_ANIME_FIELD_NUMBER: _ClassVar[int]
    RELATED_SOURCE_MATERIAL_FIELD_NUMBER: _ClassVar[int]
    SOURCES_FIELD_NUMBER: _ClassVar[int]
    STREAMING_INFO_FIELD_NUMBER: _ClassVar[int]
    STREAMING_LICENSES_FIELD_NUMBER: _ClassVar[int]
    SYNONYMS_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    THEMES_FIELD_NUMBER: _ClassVar[int]
    TRAILERS_FIELD_NUMBER: _ClassVar[int]
    PRODUCERS_FIELD_NUMBER: _ClassVar[int]
    STUDIOS_FIELD_NUMBER: _ClassVar[int]
    AIRED_DATES_FIELD_NUMBER: _ClassVar[int]
    BROADCAST_FIELD_NUMBER: _ClassVar[int]
    ENRICHMENT_METADATA_FIELD_NUMBER: _ClassVar[int]
    EXTERNAL_LINKS_FIELD_NUMBER: _ClassVar[int]
    HIATUS_FIELD_NUMBER: _ClassVar[int]
    IMAGES_FIELD_NUMBER: _ClassVar[int]
    POPULARITY_TRENDS_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    STAFF_DATA_FIELD_NUMBER: _ClassVar[int]
    STATISTICS_FIELD_NUMBER: _ClassVar[int]
    background: str
    duration: int
    episode_count: int
    id: str
    entity_type: EntityType
    month: str
    nsfw: bool
    rating: AnimeRating
    season: AnimeSeason
    similarity_score: float
    source_material: SourceMaterialType
    status: AnimeStatus
    synopsis: str
    title: str
    title_english: str
    title_japanese: str
    type: AnimeType
    year: int
    content_warnings: _containers.RepeatedScalarFieldContainer[str]
    demographics: _containers.RepeatedScalarFieldContainer[str]
    ending_themes: _containers.RepeatedCompositeFieldContainer[ThemeSong]
    genres: _containers.RepeatedScalarFieldContainer[str]
    opening_themes: _containers.RepeatedCompositeFieldContainer[ThemeSong]
    related_anime: _containers.RepeatedCompositeFieldContainer[RelatedAnimeEntry]
    related_source_material: _containers.RepeatedCompositeFieldContainer[SourceMaterial]
    sources: _containers.RepeatedScalarFieldContainer[str]
    streaming_info: _containers.RepeatedCompositeFieldContainer[StreamingEntry]
    streaming_licenses: _containers.RepeatedScalarFieldContainer[str]
    synonyms: _containers.RepeatedScalarFieldContainer[str]
    tags: _containers.RepeatedScalarFieldContainer[str]
    themes: _containers.RepeatedCompositeFieldContainer[ThemeEntry]
    trailers: _containers.RepeatedCompositeFieldContainer[TrailerEntry]
    producers: _containers.RepeatedCompositeFieldContainer[CompanyEntry]
    studios: _containers.RepeatedCompositeFieldContainer[CompanyEntry]
    aired_dates: AiredDates
    broadcast: Broadcast
    enrichment_metadata: EnrichmentMetadata
    external_links: _containers.ScalarMap[str, str]
    hiatus: AnimeHiatus
    images: _containers.MessageMap[str, StringList]
    popularity_trends: _struct_pb2.Struct
    score: ScoreCalculations
    staff_data: StaffData
    statistics: _containers.MessageMap[str, Statistics]
    def __init__(self, background: _Optional[str] = ..., duration: _Optional[int] = ..., episode_count: _Optional[int] = ..., id: _Optional[str] = ..., entity_type: _Optional[_Union[EntityType, str]] = ..., month: _Optional[str] = ..., nsfw: bool = ..., rating: _Optional[_Union[AnimeRating, str]] = ..., season: _Optional[_Union[AnimeSeason, str]] = ..., similarity_score: _Optional[float] = ..., source_material: _Optional[_Union[SourceMaterialType, str]] = ..., status: _Optional[_Union[AnimeStatus, str]] = ..., synopsis: _Optional[str] = ..., title: _Optional[str] = ..., title_english: _Optional[str] = ..., title_japanese: _Optional[str] = ..., type: _Optional[_Union[AnimeType, str]] = ..., year: _Optional[int] = ..., content_warnings: _Optional[_Iterable[str]] = ..., demographics: _Optional[_Iterable[str]] = ..., ending_themes: _Optional[_Iterable[_Union[ThemeSong, _Mapping]]] = ..., genres: _Optional[_Iterable[str]] = ..., opening_themes: _Optional[_Iterable[_Union[ThemeSong, _Mapping]]] = ..., related_anime: _Optional[_Iterable[_Union[RelatedAnimeEntry, _Mapping]]] = ..., related_source_material: _Optional[_Iterable[_Union[SourceMaterial, _Mapping]]] = ..., sources: _Optional[_Iterable[str]] = ..., streaming_info: _Optional[_Iterable[_Union[StreamingEntry, _Mapping]]] = ..., streaming_licenses: _Optional[_Iterable[str]] = ..., synonyms: _Optional[_Iterable[str]] = ..., tags: _Optional[_Iterable[str]] = ..., themes: _Optional[_Iterable[_Union[ThemeEntry, _Mapping]]] = ..., trailers: _Optional[_Iterable[_Union[TrailerEntry, _Mapping]]] = ..., producers: _Optional[_Iterable[_Union[CompanyEntry, _Mapping]]] = ..., studios: _Optional[_Iterable[_Union[CompanyEntry, _Mapping]]] = ..., aired_dates: _Optional[_Union[AiredDates, _Mapping]] = ..., broadcast: _Optional[_Union[Broadcast, _Mapping]] = ..., enrichment_metadata: _Optional[_Union[EnrichmentMetadata, _Mapping]] = ..., external_links: _Optional[_Mapping[str, str]] = ..., hiatus: _Optional[_Union[AnimeHiatus, _Mapping]] = ..., images: _Optional[_Mapping[str, StringList]] = ..., popularity_trends: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ..., score: _Optional[_Union[ScoreCalculations, _Mapping]] = ..., staff_data: _Optional[_Union[StaffData, _Mapping]] = ..., statistics: _Optional[_Mapping[str, Statistics]] = ...) -> None: ...

class AnimeRecord(_message.Message):
    __slots__ = ("anime", "characters", "episodes")
    ANIME_FIELD_NUMBER: _ClassVar[int]
    CHARACTERS_FIELD_NUMBER: _ClassVar[int]
    EPISODES_FIELD_NUMBER: _ClassVar[int]
    anime: Anime
    characters: _containers.RepeatedCompositeFieldContainer[Character]
    episodes: _containers.RepeatedCompositeFieldContainer[Episode]
    def __init__(self, anime: _Optional[_Union[Anime, _Mapping]] = ..., characters: _Optional[_Iterable[_Union[Character, _Mapping]]] = ..., episodes: _Optional[_Iterable[_Union[Episode, _Mapping]]] = ...) -> None: ...
