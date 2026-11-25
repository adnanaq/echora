"""Unit tests for Anime proto conversion and FieldCorrection application."""

from __future__ import annotations

from datetime import UTC, date, datetime, time

from enrichment.pipeline.anime_field_corrections import apply_field_corrections
from enrichment.pipeline.anime_proto_mapper import (
    _build_aired_dates,
    _build_broadcast,
    _build_hiatus,
    _build_statistics,
    _format_date_value,
    _format_time_value,
    build_anime_proto,
    to_proto_timestamp,
)
from google.protobuf import duration_pb2, wrappers_pb2
from google.protobuf.any_pb2 import Any as AnyProto
from shared_proto.v1 import anime_pb2
from shared_proto.v1.merge_pb2 import FieldCorrection


def _pack(msg: object) -> AnyProto:
    packed = AnyProto()
    packed.Pack(msg)  # type: ignore[arg-type]
    return packed


def test_timestamp_and_date_time_formatters() -> None:
    """Timestamp, date, and time helpers handle empty, naive, aware, and invalid values."""
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
    """AiredDates, Broadcast, AnimeHiatus, and Statistics builders cover all fields."""
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
    """build_anime_proto maps every scalar, nested object, collection, and relation."""
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
    """Field corrections handle scalar, dotted, enum, message, and invalid inputs."""
    anime = build_anime_proto({"id": "work-1", "title": "Initial", "status": "ONGOING"})

    apply_field_corrections(
        anime,
        [
            FieldCorrection(
                field_path="", value=_pack(wrappers_pb2.StringValue(value="x"))
            ),
            FieldCorrection(field_path="title"),  # no value
            FieldCorrection(
                field_path="title",
                value=_pack(duration_pb2.Duration(seconds=10)),  # unsupported Any
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
