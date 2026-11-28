"""Canonical enum-to-protobuf lookup tables built from ``common.models.anime``."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Any, TypeVar

from common.models.anime import (
    AnimeRating,
    AnimeRelationType,
    AnimeSeason,
    AnimeStatus,
    AnimeType,
    CharacterRole,
    CompanyRole,
    EntityType,
    SourceMaterialType,
)
from shared_proto.v1 import anime_pb2

_EnumT = TypeVar("_EnumT", bound=Enum)
_ProtoEnumT = TypeVar("_ProtoEnumT")


def resolve_enum_key(value: Any) -> str:
    """Normalize an enum member or raw string into an upper-case lookup key.

    Args:
        value: Canonical enum member or raw string value.

    Returns:
        Normalized upper-case key with spaces converted to underscores.
    """
    raw = value.value if isinstance(value, Enum) else str(value)
    return raw.strip().upper().replace(" ", "_")


def build_enum_proto_map(
    enum_cls: type[_EnumT],
    proto_by_member: Mapping[_EnumT, _ProtoEnumT],
    aliases: Mapping[str, _ProtoEnumT] | None = None,
) -> dict[str, _ProtoEnumT]:
    """Build a normalized lookup map covering every member name and canonical value.

    Args:
        enum_cls: Canonical Python ``Enum`` class from ``common.models.anime``.
        proto_by_member: Complete mapping from every member of ``enum_cls`` to
            the corresponding protobuf enum value.
        aliases: Optional shorthand or legacy lookup keys to include.

    Returns:
        Dictionary mapping normalized enum keys to protobuf enum values.

    Raises:
        ValueError: If any member of ``enum_cls`` is omitted from ``proto_by_member``.
    """
    missing_members = [member for member in enum_cls if member not in proto_by_member]
    if missing_members:
        error_message = (
            f"Incomplete proto mapping for {enum_cls.__name__}: {missing_members}"
        )
        raise ValueError(error_message)
    lookup: dict[str, _ProtoEnumT] = {}
    for member, proto_value in proto_by_member.items():
        lookup[resolve_enum_key(member.name)] = proto_value
        lookup[resolve_enum_key(member.value)] = proto_value
    if aliases:
        for alias, proto_value in aliases.items():
            lookup[resolve_enum_key(alias)] = proto_value
    return lookup


ENTITY_TYPE_MAP: dict[str, anime_pb2.EntityType] = build_enum_proto_map(
    EntityType,
    {
        EntityType.ANIME: anime_pb2.ENTITY_TYPE_ANIME,
        EntityType.CHARACTER: anime_pb2.ENTITY_TYPE_CHARACTER,
        EntityType.EPISODE: anime_pb2.ENTITY_TYPE_EPISODE,
    },
)

ANIME_STATUS_MAP: dict[str, anime_pb2.AnimeStatus] = build_enum_proto_map(
    AnimeStatus,
    {
        AnimeStatus.FINISHED: anime_pb2.ANIME_STATUS_FINISHED,
        AnimeStatus.UPCOMING: anime_pb2.ANIME_STATUS_UPCOMING,
        AnimeStatus.ONGOING: anime_pb2.ANIME_STATUS_ONGOING,
        AnimeStatus.UNKNOWN: anime_pb2.ANIME_STATUS_UNKNOWN,
        AnimeStatus.CANCELLED: anime_pb2.ANIME_STATUS_CANCELLED,
    },
)

ANIME_TYPE_MAP: dict[str, anime_pb2.AnimeType] = build_enum_proto_map(
    AnimeType,
    {
        AnimeType.MOVIE: anime_pb2.ANIME_TYPE_MOVIE,
        AnimeType.ONA: anime_pb2.ANIME_TYPE_ONA,
        AnimeType.OVA: anime_pb2.ANIME_TYPE_OVA,
        AnimeType.SPECIAL: anime_pb2.ANIME_TYPE_SPECIAL,
        AnimeType.TV: anime_pb2.ANIME_TYPE_TV,
        AnimeType.UNKNOWN: anime_pb2.ANIME_TYPE_UNKNOWN,
        AnimeType.MUSIC: anime_pb2.ANIME_TYPE_MUSIC,
        AnimeType.PV: anime_pb2.ANIME_TYPE_PV,
        AnimeType.TV_SPECIAL: anime_pb2.ANIME_TYPE_TV_SPECIAL,
        AnimeType.TV_SHORT: anime_pb2.ANIME_TYPE_TV_SHORT,
        AnimeType.CM: anime_pb2.ANIME_TYPE_CM,
        AnimeType.OTHER: anime_pb2.ANIME_TYPE_OTHER,
    },
)

ANIME_RATING_MAP: dict[str, anime_pb2.AnimeRating] = build_enum_proto_map(
    AnimeRating,
    {
        AnimeRating.G: anime_pb2.ANIME_RATING_G,
        AnimeRating.PG: anime_pb2.ANIME_RATING_PG,
        AnimeRating.PG13: anime_pb2.ANIME_RATING_PG13,
        AnimeRating.R: anime_pb2.ANIME_RATING_R,
        AnimeRating.RPLUS: anime_pb2.ANIME_RATING_RPLUS,
        AnimeRating.RX: anime_pb2.ANIME_RATING_RX,
        AnimeRating.UNKNOWN: anime_pb2.ANIME_RATING_UNKNOWN,
    },
    aliases={
        "PG-13": anime_pb2.ANIME_RATING_PG13,
        "R+": anime_pb2.ANIME_RATING_RPLUS,
    },
)

ANIME_SEASON_MAP: dict[str, anime_pb2.AnimeSeason] = build_enum_proto_map(
    AnimeSeason,
    {
        AnimeSeason.SPRING: anime_pb2.ANIME_SEASON_SPRING,
        AnimeSeason.SUMMER: anime_pb2.ANIME_SEASON_SUMMER,
        AnimeSeason.FALL: anime_pb2.ANIME_SEASON_FALL,
        AnimeSeason.WINTER: anime_pb2.ANIME_SEASON_WINTER,
    },
    aliases={"AUTUMN": anime_pb2.ANIME_SEASON_FALL},
)

SOURCE_MATERIAL_MAP: dict[str, anime_pb2.SourceMaterialType] = build_enum_proto_map(
    SourceMaterialType,
    {
        SourceMaterialType.ORIGINAL: anime_pb2.SOURCE_MATERIAL_TYPE_ORIGINAL,
        SourceMaterialType.MANGA: anime_pb2.SOURCE_MATERIAL_TYPE_MANGA,
        SourceMaterialType.LIGHT_NOVEL: anime_pb2.SOURCE_MATERIAL_TYPE_LIGHT_NOVEL,
        SourceMaterialType.VISUAL_NOVEL: anime_pb2.SOURCE_MATERIAL_TYPE_VISUAL_NOVEL,
        SourceMaterialType.GAME: anime_pb2.SOURCE_MATERIAL_TYPE_GAME,
        SourceMaterialType.WEB_MANGA: anime_pb2.SOURCE_MATERIAL_TYPE_WEB_MANGA,
        SourceMaterialType.WEB_NOVEL: anime_pb2.SOURCE_MATERIAL_TYPE_WEB_NOVEL,
        SourceMaterialType.NOVEL: anime_pb2.SOURCE_MATERIAL_TYPE_NOVEL,
        SourceMaterialType.ONE_SHOT: anime_pb2.SOURCE_MATERIAL_TYPE_ONE_SHOT,
        SourceMaterialType.DOUJINSHI: anime_pb2.SOURCE_MATERIAL_TYPE_DOUJINSHI,
        SourceMaterialType.KOMA_4: anime_pb2.SOURCE_MATERIAL_TYPE_KOMA_4,
        SourceMaterialType.MIXED_MEDIA: anime_pb2.SOURCE_MATERIAL_TYPE_MIXED_MEDIA,
        SourceMaterialType.MUSIC: anime_pb2.SOURCE_MATERIAL_TYPE_MUSIC,
        SourceMaterialType.RADIO: anime_pb2.SOURCE_MATERIAL_TYPE_RADIO,
        SourceMaterialType.BOOK: anime_pb2.SOURCE_MATERIAL_TYPE_BOOK,
        SourceMaterialType.PICTURE_BOOK: anime_pb2.SOURCE_MATERIAL_TYPE_PICTURE_BOOK,
        SourceMaterialType.CARD_GAME: anime_pb2.SOURCE_MATERIAL_TYPE_CARD_GAME,
        SourceMaterialType.OTHER: anime_pb2.SOURCE_MATERIAL_TYPE_OTHER,
        SourceMaterialType.UNKNOWN: anime_pb2.SOURCE_MATERIAL_TYPE_UNKNOWN,
        SourceMaterialType.MANHWA: anime_pb2.SOURCE_MATERIAL_TYPE_MANHWA,
        SourceMaterialType.MANHUA: anime_pb2.SOURCE_MATERIAL_TYPE_MANHUA,
        SourceMaterialType.COMIC: anime_pb2.SOURCE_MATERIAL_TYPE_COMIC,
        SourceMaterialType.LIVE_ACTION: anime_pb2.SOURCE_MATERIAL_TYPE_LIVE_ACTION,
        SourceMaterialType.ILLUSTRATION: anime_pb2.SOURCE_MATERIAL_TYPE_ILLUSTRATION,
        SourceMaterialType.WESTERN_MEDIA: anime_pb2.SOURCE_MATERIAL_TYPE_WESTERN_MEDIA,
        SourceMaterialType.WEBTOON: anime_pb2.SOURCE_MATERIAL_TYPE_WEBTOON,
        SourceMaterialType.FAIRY_TALE: anime_pb2.SOURCE_MATERIAL_TYPE_FAIRY_TALE,
        SourceMaterialType.SHORT_STORY: anime_pb2.SOURCE_MATERIAL_TYPE_SHORT_STORY,
    },
)

COMPANY_ROLE_MAP: dict[str, anime_pb2.CompanyRole] = build_enum_proto_map(
    CompanyRole,
    {
        CompanyRole.STUDIO: anime_pb2.COMPANY_ROLE_STUDIO,
        CompanyRole.PRODUCER: anime_pb2.COMPANY_ROLE_PRODUCER,
        CompanyRole.LICENSOR: anime_pb2.COMPANY_ROLE_LICENSOR,
        CompanyRole.UNKNOWN: anime_pb2.COMPANY_ROLE_UNKNOWN,
    },
)

ANIME_RELATION_TYPE_MAP: dict[str, anime_pb2.AnimeRelationType] = build_enum_proto_map(
    AnimeRelationType,
    {
        AnimeRelationType.ADAPTATION: anime_pb2.ANIME_RELATION_TYPE_ADAPTATION,
        AnimeRelationType.ALTERNATIVE_VERSION: anime_pb2.ANIME_RELATION_TYPE_ALTERNATIVE_VERSION,
        AnimeRelationType.ALTERNATIVE_SETTING: anime_pb2.ANIME_RELATION_TYPE_ALTERNATIVE_SETTING,
        AnimeRelationType.CHARACTER: anime_pb2.ANIME_RELATION_TYPE_CHARACTER,
        AnimeRelationType.CROSSOVER: anime_pb2.ANIME_RELATION_TYPE_CROSSOVER,
        AnimeRelationType.FULL_STORY: anime_pb2.ANIME_RELATION_TYPE_FULL_STORY,
        AnimeRelationType.OTHER: anime_pb2.ANIME_RELATION_TYPE_OTHER,
        AnimeRelationType.PARENT_STORY: anime_pb2.ANIME_RELATION_TYPE_PARENT_STORY,
        AnimeRelationType.PREQUEL: anime_pb2.ANIME_RELATION_TYPE_PREQUEL,
        AnimeRelationType.SAME_FRANCHISE: anime_pb2.ANIME_RELATION_TYPE_SAME_FRANCHISE,
        AnimeRelationType.SAME_SETTING: anime_pb2.ANIME_RELATION_TYPE_SAME_SETTING,
        AnimeRelationType.SEQUEL: anime_pb2.ANIME_RELATION_TYPE_SEQUEL,
        AnimeRelationType.SIDE_STORY: anime_pb2.ANIME_RELATION_TYPE_SIDE_STORY,
        AnimeRelationType.SPIN_OFF: anime_pb2.ANIME_RELATION_TYPE_SPIN_OFF,
        AnimeRelationType.SUMMARY: anime_pb2.ANIME_RELATION_TYPE_SUMMARY,
    },
)

CHARACTER_ROLE_MAP: dict[str, anime_pb2.CharacterRole] = build_enum_proto_map(
    CharacterRole,
    {
        CharacterRole.MAIN: anime_pb2.CHARACTER_ROLE_MAIN,
        CharacterRole.SUPPORTING: anime_pb2.CHARACTER_ROLE_SUPPORTING,
        CharacterRole.BACKGROUND: anime_pb2.CHARACTER_ROLE_BACKGROUND,
        CharacterRole.UNKNOWN: anime_pb2.CHARACTER_ROLE_UNKNOWN,
    },
)
