"""Unit tests for MergeCharacter stored-page merging and character mappers (ECHO-110)."""

from __future__ import annotations

import json

import pytest
from enrichment.pipeline.character_merge import merge_character_pages
from enrichment.sources.anilist.anilist_mapper import (
    map_anilist_character_page_responses,
)
from enrichment.sources.kitsu.kitsu_mapper import map_kitsu_character_page_responses
from enrichment_proto.v1.enrichment_service_pb2 import MergeCharacterRequest
from google.protobuf import wrappers_pb2
from google.protobuf.any_pb2 import Any as AnyProto  # ty: ignore[unresolved-import]
from shared_proto.v1 import anime_pb2
from shared_proto.v1.merge_pb2 import (
    FieldCorrection,
    PageKey,
    PageResponse,
    StoredPage,
)


def _pack_string(value: str) -> AnyProto:
    packed = AnyProto()
    string_cls = getattr(wrappers_pb2, "StringValue")
    packed.Pack(string_cls(value=value))
    return packed


def _pack_int64(value: int) -> AnyProto:
    packed = AnyProto()
    int64_cls = getattr(wrappers_pb2, "Int64Value")
    packed.Pack(int64_cls(value=value))
    return packed


def _anilist_character_page() -> StoredPage:
    main_payload = {
        "data": {
            "Character": {
                "id": 40,
                "name": {
                    "full": "Monkey D. Luffy",
                    "native": "モンキー・D・ルフィ",
                    "alternative": ["Straw Hat", "Lucy"],
                    "alternativeSpoiler": ["Joy Boy"],
                },
                "image": {"large": "https://cdn.anilist.co/luffy.png"},
                "description": "AniList description.",
                "gender": "Male",
                "dateOfBirth": {"year": None, "month": 5, "day": 5},
                "age": "19",
                "bloodType": "F",
                "favourites": 150000,
                "siteUrl": "https://anilist.co/character/40",
                "media": {
                    "pageInfo": {"hasNextPage": True},
                    "edges": [
                        {
                            "characterRole": "MAIN",
                            "node": {
                                "id": 21,
                                "type": "ANIME",
                                "title": {"romaji": "One Piece"},
                            },
                        }
                    ],
                },
            }
        }
    }
    media_page_2 = {
        "data": {
            "Character": {
                "id": 40,
                "media": {
                    "pageInfo": {"hasNextPage": False},
                    "edges": [
                        {
                            "characterRole": "MAIN",
                            "node": {
                                "id": 30013,
                                "type": "MANGA",
                                "title": {"romaji": "One Piece"},
                            },
                        }
                    ],
                },
            }
        }
    }
    return StoredPage(
        page_key=PageKey(
            provider="anilist",
            entity_type=anime_pb2.ENTITY_TYPE_CHARACTER,
            address="40",
        ),
        responses=[
            PageResponse(
                request_name="main",
                content_type="application/json",
                body=json.dumps(main_payload).encode("utf-8"),
            ),
            PageResponse(
                request_name="media_page_2",
                content_type="application/json",
                body=json.dumps(media_page_2).encode("utf-8"),
            ),
        ],
    )


def _kitsu_character_page() -> StoredPage:
    main_payload = {
        "data": {
            "id": "101",
            "type": "characters",
            "attributes": {
                "slug": "monkey-d-luffy",
                "canonicalName": "Monkey D. Luffy",
                "names": {"ja_jp": "モンキー・D・ルフィ"},
                "otherNames": ["Straw Hat Luffy"],
                "description": "Kitsu description.",
                "image": {"original": "https://cdn.kitsu.app/luffy.jpg"},
            },
        }
    }
    media_chars_payload = {
        "data": [
            {
                "id": "501",
                "type": "mediaCharacters",
                "attributes": {"role": "main"},
                "relationships": {"media": {"data": {"id": "12", "type": "anime"}}},
            }
        ],
        "included": [
            {
                "id": "12",
                "type": "anime",
                "attributes": {"canonicalTitle": "One Piece"},
            }
        ],
    }
    voices_payload = {
        "data": [
            {
                "id": "901",
                "type": "castings",
                "attributes": {"locale": "ja"},
                "relationships": {"person": {"data": {"id": "77", "type": "people"}}},
            }
        ],
        "included": [
            {
                "id": "77",
                "type": "people",
                "attributes": {"name": "Mayumi Tanaka"},
            }
        ],
    }
    return StoredPage(
        page_key=PageKey(
            provider="kitsu",
            entity_type=anime_pb2.ENTITY_TYPE_CHARACTER,
            address="101",
        ),
        responses=[
            PageResponse(
                request_name="main",
                content_type="application/json",
                body=json.dumps(main_payload).encode("utf-8"),
            ),
            PageResponse(
                request_name="media_characters_page_1",
                content_type="application/json",
                body=json.dumps(media_chars_payload).encode("utf-8"),
            ),
            PageResponse(
                request_name="voices_501_page_1",
                content_type="application/json",
                body=json.dumps(voices_payload).encode("utf-8"),
            ),
        ],
    )


def _anisearch_character_page() -> StoredPage:
    html = b"""<html><head>
<link rel="canonical" href="https://www.anisearch.com/character/100,monkey-d-luffy"/>
</head><body>
<div id="content">
  <h1>Monkey D. Luffy</h1>
</div>
</body></html>"""
    anime_ography_html = b"""<html><body>
<table class="animeography">
  <tr><td><a href="/anime/2227,one-piece">One Piece</a></td></tr>
</table>
</body></html>"""
    return StoredPage(
        page_key=PageKey(
            provider="anisearch",
            entity_type=anime_pb2.ENTITY_TYPE_CHARACTER,
            address="100",
        ),
        responses=[
            PageResponse(request_name="main", content_type="text/html", body=html),
            PageResponse(
                request_name="anime_ography",
                content_type="text/html",
                body=anime_ography_html,
            ),
        ],
    )


def test_merge_character_across_providers_and_apply_corrections() -> None:
    """MergeCharacter merges pages, deduplicates animeography/roles, and applies corrections."""
    from pathlib import Path

    anidb_html = (
        Path(__file__).resolve().parents[1]
        / "sources"
        / "anidb"
        / "fixtures"
        / "anidb_character_474.html"
    ).read_bytes()
    anidb_page = StoredPage(
        page_key=PageKey(
            provider="anidb",
            entity_type=anime_pb2.ENTITY_TYPE_CHARACTER,
            address="474",
        ),
        responses=[
            PageResponse(
                request_name="main",
                content_type="text/html",
                body=anidb_html,
            )
        ],
    )
    dead_page = StoredPage(
        page_key=PageKey(
            provider="mal",
            entity_type=anime_pb2.ENTITY_TYPE_CHARACTER,
            address="40",
        ),
        is_dead=True,
    )

    char_id = "0195b42a-0000-7000-8000-000000000474"
    resp = merge_character_pages(
        MergeCharacterRequest(
            character_id=char_id,
            pages=[
                _anilist_character_page(),
                _kitsu_character_page(),
                anidb_page,
                _anisearch_character_page(),
                dead_page,
            ],
            corrections=[
                FieldCorrection(
                    field_path="name",
                    value=_pack_string("Monkey D. Luffy (Corrected)"),
                ),
                FieldCorrection(
                    field_path="favorites",
                    value=_pack_int64(200000),
                ),
            ],
        )
    )
    char = resp.character
    assert char.id == char_id
    assert char.entity_type == anime_pb2.ENTITY_TYPE_CHARACTER
    assert char.name == "Monkey D. Luffy (Corrected)"
    assert char.favorites == 200000
    assert char.attributes["blood_type"] == "F"
    assert char.attributes["height"] == "174 cm"
    assert "Joy Boy" in char.spoilers.values()
    assert list(char.images) == sorted(char.images)
    assert list(char.traits) == sorted(char.traits)
    assert list(char.sources) == sorted(char.sources)
    page_sources_by_address = {
        (page.page_key.provider, page.page_key.address): page.source
        for page in resp.page_sources
    }
    assert len(resp.page_sources) == 5
    assert page_sources_by_address[("anidb", "474")] == (
        "https://anidb.net/character/474"
    )
    assert page_sources_by_address[("mal", "40")] == (
        "https://myanimelist.net/character/40"
    )
    assert page_sources_by_address[("anisearch", "100")] == (
        "https://www.anisearch.com/character/100,monkey-d-luffy"
    )
    assert page_sources_by_address[("kitsu", "101")] == (
        "https://kitsu.app/characters/monkey-d-luffy"
    )
    assert all(page.source in char.sources for page in resp.page_sources)

    # All providers' One Piece URLs merged into a single animeography entry with MAIN role
    assert len(char.animeography) >= 1
    one_piece = next(e for e in char.animeography if e.title == "One Piece")
    assert one_piece.role == anime_pb2.CHARACTER_ROLE_MAIN
    assert "https://anilist.co/anime/21" in one_piece.sources
    assert "https://anidb.net/anime/69" in one_piece.sources
    assert "https://kitsu.app/anime/12" in one_piece.sources
    assert len(char.mangaography) == 1
    assert "https://anilist.co/manga/30013" in char.mangaography[0].sources


def test_character_page_mapper_error_branches() -> None:
    """AniList and Kitsu character page mappers validate required responses."""
    with pytest.raises(ValueError, match="missing required 'main'"):
        map_anilist_character_page_responses([])
    with pytest.raises(TypeError, match="missing 'Character'"):
        map_anilist_character_page_responses(
            [
                PageResponse(
                    request_name="main",
                    content_type="application/json",
                    body=b'{"data": {"Character": null}}',
                )
            ]
        )
    with pytest.raises(ValueError, match="missing required 'main'"):
        map_kitsu_character_page_responses([])
    with pytest.raises(TypeError, match="missing 'data' object"):
        map_kitsu_character_page_responses(
            [
                PageResponse(
                    request_name="main",
                    content_type="application/json",
                    body=b'{"data": null}',
                )
            ]
        )
