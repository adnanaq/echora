from __future__ import annotations

import json

import pytest
from enrichment.sources.anilist.anilist_character_page_mapper import (
    map_anilist_character_page_responses,
)
from shared_proto.v1.merge_pb2 import PageResponse


def _json_response(request_name: str, payload: object) -> PageResponse:
    return PageResponse(
        request_name=request_name,
        content_type="application/json",
        body=json.dumps(payload).encode("utf-8"),
    )


def test_map_anilist_character_page_responses_maps_full_paginated_payload() -> None:
    main_payload = {
        "data": {
            "Character": {
                "id": 40,
                "name": {
                    "full": "Monkey D. Luffy",
                    "native": "モンキー・D・ルフィ",
                    "alternative": ["Straw Hat", "Lucy", ""],
                    "alternativeSpoiler": ["Joy Boy", ""],
                },
                "image": {
                    "large": "https://cdn.anilist.co/luffy-large.png",
                    "medium": "https://cdn.anilist.co/luffy-medium.png",
                },
                "description": "Captain of the Straw Hat Pirates.",
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
                            "voiceActorRoles": [
                                {
                                    "voiceActor": {
                                        "id": 95075,
                                        "name": {
                                            "full": "Mayumi Tanaka",
                                            "native": "田中真弓",
                                        },
                                        "languageV2": "Japanese",
                                        "image": {
                                            "large": "https://cdn.anilist.co/va/95075.jpg"
                                        },
                                        "siteUrl": "https://anilist.co/staff/95075",
                                    }
                                },
                                {
                                    "voiceActor": {
                                        "id": 95075,
                                        "name": {"full": "Mayumi Tanaka"},
                                        "languageV2": "Japanese",
                                        "siteUrl": "https://anilist.co/staff/95075",
                                    }
                                },
                            ],
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
                            "characterRole": "SUPPORTING",
                            "node": {
                                "id": 30013,
                                "type": "MANGA",
                                "title": {"english": "One Piece Manga"},
                            },
                            "voiceActorRoles": [
                                {
                                    "voiceActor": {
                                        "id": 1001,
                                        "name": {"full": "Colleen Clinkenbeard"},
                                        "languageV2": "English",
                                    }
                                }
                            ],
                        }
                    ],
                },
            }
        }
    }

    mapped = map_anilist_character_page_responses(
        [
            _json_response("main", main_payload),
            _json_response("media_page_2", media_page_2),
        ]
    )

    assert mapped["name"] == "Monkey D. Luffy"
    assert mapped["name_native"] == "モンキー・D・ルフィ"
    assert mapped["name_variations"] == ["Straw Hat", "Lucy"]
    assert mapped["description"] == "Captain of the Straw Hat Pirates."
    assert mapped["favorites"] == 150000
    assert mapped["images"] == ["https://cdn.anilist.co/luffy-large.png"]
    assert mapped["sources"] == ["https://anilist.co/character/40"]
    assert mapped["attributes"] == {
        "gender": "Male",
        "age": "19",
        "blood_type": "F",
        "date_of_birth": "5-5",
    }
    assert mapped["spoilers"] == {"name_spoiler_0": "Joy Boy"}
    assert mapped["animeography"] == [
        {
            "title": "One Piece",
            "role": "MAIN",
            "sources": ["https://anilist.co/anime/21"],
        }
    ]
    assert mapped["mangaography"] == [
        {
            "title": "One Piece Manga",
            "role": "SUPPORTING",
            "sources": ["https://anilist.co/manga/30013"],
        }
    ]
    assert mapped["voice_actors"] == [
        {
            "name": "Mayumi Tanaka",
            "native_name": "田中真弓",
            "language": "ja",
            "image": "https://cdn.anilist.co/va/95075.jpg",
            "character_assignments": [],
            "sources": ["https://anilist.co/staff/95075"],
        },
        {
            "name": "Colleen Clinkenbeard",
            "language": "en",
            "character_assignments": [],
            "sources": ["https://anilist.co/staff/1001"],
        },
    ]


def test_map_anilist_character_page_responses_skips_incomplete_media_and_voice_actors() -> (
    None
):
    payload = {
        "data": {
            "Character": {
                "id": 40,
                "name": {"full": "Monkey D. Luffy"},
                "media": {
                    "edges": [
                        {
                            "characterRole": "MAIN",
                            "node": {"id": None, "type": "ANIME", "title": {}},
                            "voiceActorRoles": [
                                "not-a-dict",
                                {"voiceActor": {"name": {"full": "   "}}},
                            ],
                        }
                    ]
                },
            }
        }
    }

    mapped = map_anilist_character_page_responses([_json_response("main", payload)])

    assert mapped["animeography"] == []
    assert mapped["mangaography"] == []
    assert mapped.get("voice_actors", []) == []


def test_map_anilist_character_page_responses_missing_main_raises_value_error() -> None:
    with pytest.raises(ValueError, match="missing required 'main'"):
        map_anilist_character_page_responses([])


@pytest.mark.parametrize(
    ("payload", "error_pattern"),
    [
        ({"data": "bad"}, "AniList character 'data'"),
        ({"data": {"Character": None}}, "missing 'Character'"),
        ({"data": {"Character": {"name": "bad"}}}, "AniList character 'name'"),
        (
            {"data": {"Character": {"name": {"full": "Luffy"}, "media": "bad"}}},
            "AniList character 'media'",
        ),
        (
            {"data": {"Character": {"name": {"full": "Luffy"}, "image": "bad"}}},
            "AniList character 'image'",
        ),
        (
            {"data": {"Character": {"name": {"full": "Luffy"}, "dateOfBirth": "bad"}}},
            "AniList character 'dateOfBirth'",
        ),
    ],
)
def test_map_anilist_character_page_responses_invalid_objects_raise_type_error(
    payload: dict[str, object],
    error_pattern: str,
) -> None:
    with pytest.raises(TypeError, match=error_pattern):
        map_anilist_character_page_responses([_json_response("main", payload)])


def test_map_anilist_character_page_responses_invalid_paginated_media_raises_type_error() -> (
    None
):
    valid_main = _json_response(
        "main",
        {"data": {"Character": {"id": 40, "name": {"full": "Monkey D. Luffy"}}}},
    )
    bad_page = _json_response("media_page_2", {"data": "not-an-object"})

    with pytest.raises(TypeError, match="AniList media_page 'data'"):
        map_anilist_character_page_responses([valid_main, bad_page])
