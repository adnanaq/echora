from __future__ import annotations

from datetime import UTC, date, datetime, time

import pytest
from enrichment.pipeline.anime_field_corrections import apply_field_corrections
from enrichment.pipeline.anime_proto_mapper import (
    _build_aired_dates,
    _build_broadcast,
    _build_hiatus,
    _build_statistics,
    _format_date_value,
    _format_time_value,
    build_anime_proto,
    sort_anime_proto_lists,
    to_proto_timestamp,
)
from enrichment.pipeline.proto_enum_maps import build_enum_proto_map
from google.protobuf import duration_pb2, wrappers_pb2
from google.protobuf.any_pb2 import Any as AnyProto
from shared_proto.v1 import anime_pb2
from shared_proto.v1.merge_pb2 import FieldCorrection


def _pack(msg: object) -> AnyProto:
    packed = AnyProto()
    packed.Pack(msg)  # type: ignore[arg-type]
    return packed


def test_timestamp_and_date_time_formatters() -> None:
    assert to_proto_timestamp(None) is None
    assert to_proto_timestamp("") is None
    assert to_proto_timestamp("not-a-date") is None
    assert to_proto_timestamp(12345) is None  # type: ignore[arg-type]

    z_ts = to_proto_timestamp("2026-03-15T12:00:00Z")
    assert z_ts is not None
    assert z_ts.ToDatetime(tzinfo=UTC).year == 2026

    naive_ts = to_proto_timestamp(datetime(2026, 4, 1, 8, 30))
    assert naive_ts is not None
    assert naive_ts.ToDatetime(tzinfo=UTC).hour == 8

    assert _format_time_value(None) is None
    assert _format_time_value("") is None
    assert _format_time_value(time(23, 5)) == "23:05:00"
    assert _format_time_value("01:00") == "01:00"

    assert _format_date_value(None) is None
    assert _format_date_value("") is None
    assert _format_date_value(datetime(2026, 5, 2, 10, 0)) == "2026-05-02"
    assert _format_date_value(date(2026, 5, 3)) == "2026-05-03"
    assert _format_date_value("2026-05-04") == "2026-05-04"


def test_nested_schedule_builders_handle_empty_and_full_dicts() -> None:
    assert _build_aired_dates(None) is None
    assert _build_aired_dates({"aired_from": "bad", "aired_to": "bad"}) is None
    assert _build_broadcast(None) is None
    assert _build_hiatus(None) is None

    stats = _build_statistics(
        {
            "invalid": "not-a-dict",
            "mal": {
                "score": 8.72,
                "scored_by": 100000,
                "rank": 50,
                "popularity": 10,
                "members": 500000,
                "favorites": 25000,
                "contextual_ranks": [
                    {
                        "rank": 1,
                        "context": "Spring 2026",
                        "format": "TV",
                        "year": 2026,
                        "season": "SPRING",
                        "is_all_time": False,
                    }
                ],
            },
        }
    )
    assert stats["mal"].rank == 50
    assert stats["mal"].contextual_ranks[0].year == 2026
    assert stats["mal"].contextual_ranks[0].is_all_time is False


def test_build_anime_proto_populates_all_fields_and_sorts() -> None:
    record = {
        "id": "0195b42a-0000-7000-8000-000000000010",
        "entity_type": "ANIME",
        "title": "Full Test Anime",
        "title_english": "Full Test",
        "title_japanese": "テスト",
        "background": "Original work.",
        "country_of_origin": "JP",
        "synopsis": "A summary.",
        "duration_seconds": 1440,
        "episode_count": 24,
        "month": 4,
        "year": 2026,
        "is_nsfw": False,
        "rating": "PG13",
        "season": "SPRING",
        "source_material": "MANGA",
        "status": "FINISHED",
        "type": "TV",
        "content_warnings": ["Violence"],
        "demographics": ["Shounen"],
        "genres": ["Drama", "Action"],
        "sources": ["https://myanimelist.net/anime/1"],
        "partial_sources": ["https://anidb.net/anime/2"],
        "synonyms": ["FTA"],
        "tags": ["Time Loop"],
        "opening_themes": [
            {
                "title": "Opening 1",
                "artist": "Band A",
                "episodes": [{"range_start": 1, "range_end": 12}],
            }
        ],
        "ending_themes": [
            {
                "title": "Ending 1",
                "episodes": [{"range_start": 1, "range_end": None}],
            }
        ],
        "themes": [{"name": "School", "description": "Set in high school"}],
        "trailers": [
            {
                "source": "https://youtube.com/watch?v=1",
                "title": "PV 1",
                "thumbnail_source": "https://img.youtube.com/1.jpg",
            }
        ],
        "external_sources": [
            {
                "platform": "Official Site",
                "source": "https://example.com",
                "label": "Official",
                "language": "ja",
            }
        ],
        "streaming_sources": [
            {
                "platform": "Crunchyroll",
                "source": "https://crunchyroll.com/series/1",
                "region": "US",
                "is_free": False,
                "is_premium_required": True,
                "has_dub": True,
                "subtitle_languages": ["en", "es"],
            }
        ],
        "companies": [
            {
                "name": "Studio B",
                "description": "Animation",
                "sources": ["https://anilist.co/studio/1"],
                "roles": ["STUDIO"],
            }
        ],
        "images": {
            "covers": [
                "https://cdn.example.com/c2.jpg",
                "https://cdn.example.com/c1.jpg",
            ],
            "posters": ["https://cdn.example.com/p1.jpg"],
            "banners": ["https://cdn.example.com/b1.jpg"],
        },
        "aired_dates": {
            "aired_from": "2026-04-01T00:00:00Z",
            "aired_to": "2026-09-23T00:00:00Z",
        },
        "broadcast": {
            "day": 5,
            "timezone": "Asia/Tokyo",
            "jp_time": time(23, 0),
            "sub_time": time(16, 0),
            "dub_time": time(18, 0),
            "sub_delay_days": 0,
            "dub_delay_days": 14,
            "premiere_jp": "2026-04-01T14:00:00Z",
            "premiere_sub": "2026-04-01T15:00:00Z",
            "premiere_dub": "2026-04-15T15:00:00Z",
            "next_episode_at": "2026-04-08T14:00:00Z",
            "next_episode_number": 2,
        },
        "hiatus": {
            "reason": "Production",
            "hiatus_from": date(2026, 6, 1),
            "hiatus_until": date(2026, 6, 15),
        },
        "score": {"weighted": 8.5, "mean": 8.6, "median": 8.7},
        "titles": {"en": "Full Test", "ja": "テスト"},
        "statistics": {"mal": {"score": 8.5}},
        "related_anime": {
            "SEQUEL": [
                {
                    "title": "Sequel",
                    "type": "TV",
                    "sources": ["https://anilist.co/anime/2"],
                    "status": "UPCOMING",
                    "year": 2027,
                    "score": 8.1,
                    "episode_count": 12,
                }
            ]
        },
        "related_source_material": {
            "ADAPTATION": [
                {
                    "title": "Manga Z",
                    "type": "MANGA",
                    "sources": ["https://anilist.co/manga/99"],
                    "status": "ONGOING",
                    "score": 8.9,
                    "chapters": 120,
                    "volumes": 14,
                },
                {
                    "title": "Manga A",
                    "type": "MANGA",
                    "sources": ["https://anilist.co/manga/98"],
                },
            ]
        },
    }
    proto = build_anime_proto(record)

    assert proto.duration_seconds == 1440
    assert proto.season == anime_pb2.ANIME_SEASON_SPRING
    assert proto.rating == anime_pb2.ANIME_RATING_PG13
    assert proto.source_material == anime_pb2.SOURCE_MATERIAL_TYPE_MANGA
    assert list(proto.genres) == ["Action", "Drama"]
    assert list(proto.images.covers) == [
        "https://cdn.example.com/c1.jpg",
        "https://cdn.example.com/c2.jpg",
    ]
    assert proto.broadcast.jp_time == "23:00:00"
    assert proto.hiatus.hiatus_from == "2026-06-01"
    assert proto.score.weighted == 8.5
    assert proto.companies[0].roles == [anime_pb2.COMPANY_ROLE_STUDIO]
    assert [
        item.title for item in proto.related_source_material["ADAPTATION"].items
    ] == ["Manga A", "Manga Z"]


def test_apply_field_corrections_handles_nested_enum_message_and_invalid_paths() -> (
    None
):
    anime = build_anime_proto({"id": "work-1", "title": "Initial", "status": "ONGOING"})

    apply_field_corrections(
        anime,
        [
            FieldCorrection(
                field_path="", value=_pack(wrappers_pb2.StringValue(value="x"))
            ),
            FieldCorrection(field_path="title"),
            FieldCorrection(
                field_path="title",
                value=_pack(duration_pb2.Duration(seconds=10)),
            ),
            FieldCorrection(
                field_path=".",
                value=_pack(wrappers_pb2.StringValue(value="ignored")),
            ),
            FieldCorrection(
                field_path="missing_nested.field",
                value=_pack(wrappers_pb2.StringValue(value="ignored")),
            ),
            FieldCorrection(
                field_path="unknown_leaf",
                value=_pack(wrappers_pb2.StringValue(value="ignored")),
            ),
            FieldCorrection(
                field_path="status",
                value=_pack(wrappers_pb2.StringValue(value="finished")),
            ),
            FieldCorrection(
                field_path="broadcast.jp_time",
                value=_pack(wrappers_pb2.StringValue(value="22:30:00")),
            ),
            FieldCorrection(
                field_path="hiatus",
                value=_pack(anime_pb2.AnimeHiatus(reason="Special broadcast")),
            ),
        ],
    )

    assert anime.status == anime_pb2.ANIME_STATUS_FINISHED
    assert anime.broadcast.jp_time == "22:30:00"
    assert anime.hiatus.reason == "Special broadcast"


def test_apply_field_corrections_accepts_bare_type_url_without_prefix() -> None:
    anime = build_anime_proto({"id": "work-2", "title": "Initial", "status": "ONGOING"})
    raw_string_any = AnyProto(
        type_url="google.protobuf.StringValue",
        value=wrappers_pb2.StringValue(value="Overridden Title").SerializeToString(),
    )
    raw_hiatus_any = AnyProto(
        type_url=anime_pb2.AnimeHiatus.DESCRIPTOR.full_name,
        value=anime_pb2.AnimeHiatus(reason="Bare message type").SerializeToString(),
    )

    apply_field_corrections(
        anime,
        [
            FieldCorrection(field_path="title", value=raw_string_any),
            FieldCorrection(field_path="hiatus", value=raw_hiatus_any),
        ],
    )

    assert anime.title == "Overridden Title"
    assert anime.hiatus.reason == "Bare message type"


def test_enum_proto_maps_cover_every_canonical_enum_member() -> None:
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
    from enrichment.pipeline.proto_enum_maps import (
        ANIME_RATING_MAP,
        ANIME_RELATION_TYPE_MAP,
        ANIME_SEASON_MAP,
        ANIME_STATUS_MAP,
        ANIME_TYPE_MAP,
        CHARACTER_ROLE_MAP,
        COMPANY_ROLE_MAP,
        ENTITY_TYPE_MAP,
        SOURCE_MATERIAL_MAP,
        resolve_enum_key,
    )

    mapped_enums = [
        (EntityType, ENTITY_TYPE_MAP),
        (AnimeStatus, ANIME_STATUS_MAP),
        (AnimeType, ANIME_TYPE_MAP),
        (AnimeRating, ANIME_RATING_MAP),
        (AnimeSeason, ANIME_SEASON_MAP),
        (SourceMaterialType, SOURCE_MATERIAL_MAP),
        (CompanyRole, COMPANY_ROLE_MAP),
        (AnimeRelationType, ANIME_RELATION_TYPE_MAP),
        (CharacterRole, CHARACTER_ROLE_MAP),
    ]
    for enum_cls, proto_map in mapped_enums:
        for member in enum_cls:
            by_value = proto_map.get(resolve_enum_key(member.value))
            by_name = proto_map.get(resolve_enum_key(member.name))
            assert by_value is not None, (
                f"{enum_cls.__name__}.{member.name} value missing"
            )
            assert by_name is not None, (
                f"{enum_cls.__name__}.{member.name} name missing"
            )
            assert by_value == by_name
            assert by_value != 0

    pg13_anime = build_anime_proto(
        {"id": "work-pg13", "title": "One Piece", "rating": AnimeRating.PG13.value}
    )
    assert pg13_anime.rating == anime_pb2.ANIME_RATING_PG13


def test_apply_field_corrections_guards_invalid_paths_types_and_protected_fields() -> (
    None
):
    anime = build_anime_proto(
        {
            "id": "anime-safe",
            "title": "Original",
            "episode_count": 12,
            "genres": ["Action"],
            "status": "ONGOING",
        }
    )
    episode = anime_pb2.Episode(
        id="ep-safe",
        anime_id="anime-safe",
        entity_type=anime_pb2.ENTITY_TYPE_EPISODE,
        episode_number=1,
        title="Ep 1",
    )
    character = anime_pb2.Character(
        id="char-safe",
        entity_type=anime_pb2.ENTITY_TYPE_CHARACTER,
        name="Luffy",
        roles=[anime_pb2.CHARACTER_ROLE_MAIN],
        anime_ids=["anime-safe"],
    )

    apply_field_corrections(
        anime,
        [
            FieldCorrection(
                field_path="title.child",
                value=_pack(wrappers_pb2.StringValue(value="x")),
            ),
            FieldCorrection(
                field_path="opening_themes.title",
                value=_pack(wrappers_pb2.StringValue(value="x")),
            ),
            FieldCorrection(
                field_path="related_anime.SEQUEL",
                value=_pack(wrappers_pb2.StringValue(value="x")),
            ),
            FieldCorrection(
                field_path="episode_count",
                value=_pack(wrappers_pb2.StringValue(value="not-int")),
            ),
            FieldCorrection(
                field_path="genres",
                value=_pack(wrappers_pb2.StringValue(value="Comedy")),
            ),
            FieldCorrection(
                field_path="hiatus",
                value=_pack(wrappers_pb2.StringValue(value="bad-msg")),
            ),
            FieldCorrection(
                field_path="title",
                value=_pack(anime_pb2.AnimeHiatus(reason="wrong-type")),
            ),
            FieldCorrection(
                field_path="status",
                value=_pack(wrappers_pb2.StringValue(value="NOT_A_STATUS")),
            ),
            FieldCorrection(
                field_path="id",
                value=_pack(wrappers_pb2.StringValue(value="hijacked")),
            ),
            FieldCorrection(
                field_path="entity_type",
                value=_pack(wrappers_pb2.StringValue(value="CHARACTER")),
            ),
            FieldCorrection(
                field_path="titles.en",
                value=_pack(wrappers_pb2.StringValue(value="English Title")),
            ),
            FieldCorrection(
                field_path="titles.ja",
                value=_pack(wrappers_pb2.Int32Value(value=123)),
            ),
        ],
    )
    assert anime.id == "anime-safe"
    assert anime.entity_type == anime_pb2.ENTITY_TYPE_ANIME
    assert anime.title == "Original"
    assert anime.episode_count == 12
    assert list(anime.genres) == ["Action"]
    assert anime.status == anime_pb2.ANIME_STATUS_ONGOING
    assert anime.titles["en"] == "English Title"
    assert "ja" not in anime.titles

    apply_field_corrections(
        episode,
        [
            FieldCorrection(
                field_path="id",
                value=_pack(wrappers_pb2.StringValue(value="hijacked-ep")),
            ),
            FieldCorrection(
                field_path="anime_id",
                value=_pack(wrappers_pb2.StringValue(value="hijacked-anime")),
            ),
            FieldCorrection(
                field_path="entity_type",
                value=_pack(wrappers_pb2.StringValue(value="ANIME")),
            ),
            FieldCorrection(
                field_path="streaming.crunchyroll",
                value=_pack(wrappers_pb2.StringValue(value="https://cr.example/1")),
            ),
        ],
    )
    assert episode.id == "ep-safe"
    assert episode.anime_id == "anime-safe"
    assert episode.entity_type == anime_pb2.ENTITY_TYPE_EPISODE
    assert episode.streaming["crunchyroll"] == "https://cr.example/1"

    apply_field_corrections(
        character,
        [
            FieldCorrection(
                field_path="id",
                value=_pack(wrappers_pb2.StringValue(value="hijacked-char")),
            ),
            FieldCorrection(
                field_path="entity_type",
                value=_pack(wrappers_pb2.StringValue(value="ANIME")),
            ),
            FieldCorrection(
                field_path="roles",
                value=_pack(wrappers_pb2.StringValue(value="BACKGROUND")),
            ),
            FieldCorrection(
                field_path="anime_ids",
                value=_pack(wrappers_pb2.StringValue(value="other")),
            ),
            FieldCorrection(
                field_path="attributes.age",
                value=_pack(wrappers_pb2.StringValue(value="19")),
            ),
            FieldCorrection(
                field_path="spoilers.description",
                value=_pack(wrappers_pb2.StringValue(value="Awakened form")),
            ),
        ],
    )
    assert character.id == "char-safe"
    assert character.entity_type == anime_pb2.ENTITY_TYPE_CHARACTER
    assert list(character.roles) == [anime_pb2.CHARACTER_ROLE_MAIN]
    assert list(character.anime_ids) == ["anime-safe"]
    assert character.attributes["age"] == "19"
    assert character.spoilers["description"] == "Awakened form"


def test_apply_field_corrections_newest_approved_at_wins_regardless_of_input_order() -> (
    None
):
    from google.protobuf.timestamp_pb2 import Timestamp

    anime = build_anime_proto({"id": "work-order", "title": "Initial"})
    newer = FieldCorrection(
        field_path="title",
        value=_pack(wrappers_pb2.StringValue(value="Newer Title")),
        approved_at=Timestamp(seconds=200, nanos=0),
    )
    older = FieldCorrection(
        field_path="title",
        value=_pack(wrappers_pb2.StringValue(value="Older Title")),
        approved_at=Timestamp(seconds=100, nanos=0),
    )

    apply_field_corrections(anime, [newer, older])

    assert anime.title == "Newer Title"


def test_build_enum_proto_map_missing_member_raises_value_error() -> None:
    from common.models.anime import EntityType

    with pytest.raises(ValueError, match="Incomplete proto mapping for EntityType"):
        build_enum_proto_map(
            EntityType,
            {EntityType.ANIME: anime_pb2.ENTITY_TYPE_ANIME},
        )


def test_apply_field_corrections_enum_numeric_bool_and_float_values() -> None:
    anime = build_anime_proto({"id": "work-enum", "status": "ONGOING"})

    apply_field_corrections(
        anime,
        [
            FieldCorrection(
                field_path="status",
                value=_pack(wrappers_pb2.BoolValue(value=True)),
            ),
            FieldCorrection(
                field_path="status",
                value=_pack(wrappers_pb2.Int32Value(value=9999)),
            ),
            FieldCorrection(
                field_path="status",
                value=_pack(wrappers_pb2.FloatValue(value=1.5)),
            ),
        ],
    )
    assert anime.status == anime_pb2.ANIME_STATUS_ONGOING

    apply_field_corrections(
        anime,
        [
            FieldCorrection(
                field_path="status",
                value=_pack(
                    wrappers_pb2.Int32Value(value=anime_pb2.ANIME_STATUS_FINISHED)
                ),
            )
        ],
    )
    assert anime.status == anime_pb2.ANIME_STATUS_FINISHED


def test_sort_anime_proto_lists_reversed_nested_collections_produce_identical_bytes() -> (
    None
):
    forward = anime_pb2.Anime(
        id="work-sort",
        opening_themes=[
            anime_pb2.ThemeSong(
                title="OP 1",
                artist="Band",
                episodes=[
                    anime_pb2.EpisodeRange(range_start=1, range_end=12),
                    anime_pb2.EpisodeRange(range_start=13, range_end=24),
                ],
            ),
            anime_pb2.ThemeSong(
                title="OP 1",
                artist="Band",
                episodes=[anime_pb2.EpisodeRange(range_start=25, range_end=36)],
            ),
        ],
        trailers=[
            anime_pb2.TrailerEntry(
                source="https://youtube.com/watch?v=a",
                title="PV",
                thumbnail_source="https://img/a.jpg",
            ),
            anime_pb2.TrailerEntry(
                source="https://youtube.com/watch?v=b",
                title="PV",
                thumbnail_source="https://img/b.jpg",
            ),
        ],
        streaming_sources=[
            anime_pb2.StreamingEntry(
                platform="Crunchyroll",
                source="https://cr/1",
                region="US",
                subtitle_languages=["en", "es"],
            )
        ],
        companies=[
            anime_pb2.CompanyEntry(
                name="Toei",
                roles=[anime_pb2.COMPANY_ROLE_STUDIO, anime_pb2.COMPANY_ROLE_PRODUCER],
                sources=[
                    "https://anilist.co/studio/1",
                    "https://myanimelist.net/anime/producer/18",
                ],
            )
        ],
        statistics={
            "mal": anime_pb2.Statistics(
                contextual_ranks=[
                    anime_pb2.ContextualRank(rank=1, context="Top", year=2025),
                    anime_pb2.ContextualRank(rank=2, context="Top", year=2026),
                ]
            )
        },
        related_anime={
            "SEQUEL": anime_pb2.RelatedAnimeList(
                items=[
                    anime_pb2.RelatedAnime(
                        title="Seq",
                        sources=["https://a/1", "https://a/2"],
                    )
                ]
            )
        },
        related_source_material={
            "ADAPTATION": anime_pb2.RelatedSourceMaterialList(
                items=[
                    anime_pb2.RelatedSourceMaterial(
                        title="Manga",
                        sources=["https://m/1", "https://m/2"],
                    )
                ]
            )
        },
    )
    reversed_proto = anime_pb2.Anime(
        id="work-sort",
        opening_themes=[
            anime_pb2.ThemeSong(
                title="OP 1",
                artist="Band",
                episodes=[anime_pb2.EpisodeRange(range_start=25, range_end=36)],
            ),
            anime_pb2.ThemeSong(
                title="OP 1",
                artist="Band",
                episodes=[
                    anime_pb2.EpisodeRange(range_start=13, range_end=24),
                    anime_pb2.EpisodeRange(range_start=1, range_end=12),
                ],
            ),
        ],
        trailers=[
            anime_pb2.TrailerEntry(
                source="https://youtube.com/watch?v=b",
                title="PV",
                thumbnail_source="https://img/b.jpg",
            ),
            anime_pb2.TrailerEntry(
                source="https://youtube.com/watch?v=a",
                title="PV",
                thumbnail_source="https://img/a.jpg",
            ),
        ],
        streaming_sources=[
            anime_pb2.StreamingEntry(
                platform="Crunchyroll",
                source="https://cr/1",
                region="US",
                subtitle_languages=["es", "en"],
            )
        ],
        companies=[
            anime_pb2.CompanyEntry(
                name="Toei",
                roles=[anime_pb2.COMPANY_ROLE_PRODUCER, anime_pb2.COMPANY_ROLE_STUDIO],
                sources=[
                    "https://myanimelist.net/anime/producer/18",
                    "https://anilist.co/studio/1",
                ],
            )
        ],
        statistics={
            "mal": anime_pb2.Statistics(
                contextual_ranks=[
                    anime_pb2.ContextualRank(rank=2, context="Top", year=2026),
                    anime_pb2.ContextualRank(rank=1, context="Top", year=2025),
                ]
            )
        },
        related_anime={
            "SEQUEL": anime_pb2.RelatedAnimeList(
                items=[
                    anime_pb2.RelatedAnime(
                        title="Seq",
                        sources=["https://a/2", "https://a/1"],
                    )
                ]
            )
        },
        related_source_material={
            "ADAPTATION": anime_pb2.RelatedSourceMaterialList(
                items=[
                    anime_pb2.RelatedSourceMaterial(
                        title="Manga",
                        sources=["https://m/2", "https://m/1"],
                    )
                ]
            )
        },
    )

    sort_anime_proto_lists(forward)
    sort_anime_proto_lists(reversed_proto)

    assert forward.SerializeToString(deterministic=True) == (
        reversed_proto.SerializeToString(deterministic=True)
    )
