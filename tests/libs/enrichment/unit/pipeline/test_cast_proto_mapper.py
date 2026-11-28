from __future__ import annotations

import pytest
from common.models.anime import CharacterRole
from enrichment.pipeline.cast_proto_mapper import (
    collect_voice_actors,
    merge_episode_characters,
    merge_episode_staff,
    merge_voice_actors,
    resolve_character_role,
)
from shared_proto.v1 import anime_pb2


@pytest.mark.parametrize(
    ("raw_role", "expected"),
    [
        (CharacterRole.MAIN, anime_pb2.CHARACTER_ROLE_MAIN),
        (CharacterRole.SUPPORTING, anime_pb2.CHARACTER_ROLE_SUPPORTING),
        ("background", anime_pb2.CHARACTER_ROLE_BACKGROUND),
        ("  Main ", anime_pb2.CHARACTER_ROLE_MAIN),
        ("not-a-role", anime_pb2.CHARACTER_ROLE_UNKNOWN),
        (None, anime_pb2.CHARACTER_ROLE_UNKNOWN),
    ],
)
def test_resolve_character_role_maps_enum_string_and_unknown_values(
    raw_role: object,
    expected: anime_pb2.CharacterRole,
) -> None:
    assert resolve_character_role(raw_role) == expected


def test_merge_voice_actors_deduplicates_by_url_and_swapped_name_tokens() -> None:
    merged = merge_voice_actors(
        [
            {"name": "   ", "sources": ["https://example.com/blank"]},
            {
                "name": "Tanaka Mayumi",
                "native_name": "田中真弓",
                "language": "ja",
                "character_assignments": ["Monkey D. Luffy", "   "],
                "sources": ["https://anidb.net/creator/322/"],
            },
            {
                "name": "Mayumi Tanaka",
                "image": "https://cdn.anilist.co/va/95075.jpg",
                "language": "ja",
                "character_assignments": ["Krillin"],
                "sources": ["https://anilist.co/staff/95075"],
            },
            {
                "name": "Different Alias",
                "biography": "Veteran voice actor",
                "sources": ["https://anidb.net/creator/322"],
            },
            {
                "name": "Mayumi Tanaka",
                "language": "en",
                "sources": ["https://example.com/en-dub"],
            },
            {
                "name": "---",
                "sources": ["https://example.com/punct-1"],
            },
            {
                "name": "---",
                "sources": ["https://example.com/punct-2"],
            },
        ]
    )

    assert len(merged) == 4
    assert [
        (actor.name, actor.language, tuple(actor.sources)) for actor in merged
    ] == sorted((actor.name, actor.language, tuple(actor.sources)) for actor in merged)

    tanaka_ja = next(
        actor
        for actor in merged
        if actor.name == "Tanaka Mayumi" and actor.language == "ja"
    )
    assert tanaka_ja.native_name == "田中真弓"
    assert tanaka_ja.image == "https://cdn.anilist.co/va/95075.jpg"
    assert tanaka_ja.biography == "Veteran voice actor"
    assert list(tanaka_ja.character_assignments) == ["Krillin", "Monkey D. Luffy"]
    assert list(tanaka_ja.sources) == [
        "https://anidb.net/creator/322",
        "https://anidb.net/creator/322/",
        "https://anilist.co/staff/95075",
    ]
    assert any(
        actor.name == "Mayumi Tanaka" and actor.language == "en" for actor in merged
    )
    assert len([actor for actor in merged if actor.name == "---"]) == 2


def test_collect_voice_actors_skips_non_mapping_entries_and_merges_ranked_records() -> (
    None
):
    collected = collect_voice_actors(
        [
            (
                "mal",
                {
                    "voice_actors": [
                        "invalid-entry",
                        {
                            "name": "Mayumi Tanaka",
                            "language": "ja",
                            "sources": ["https://myanimelist.net/people/75"],
                        },
                    ]
                },
            ),
            (
                "anilist",
                {
                    "voice_actors": [
                        {
                            "name": "Tanaka Mayumi",
                            "native_name": "田中真弓",
                            "sources": ["https://anilist.co/staff/95075"],
                        }
                    ]
                },
            ),
        ]
    )

    assert len(collected) == 1
    assert collected[0].name == "Mayumi Tanaka"
    assert collected[0].native_name == "田中真弓"
    assert list(collected[0].sources) == [
        "https://anilist.co/staff/95075",
        "https://myanimelist.net/people/75",
    ]


def test_merge_episode_characters_deduplicates_upgrades_unknown_role_and_merges_cast() -> (
    None
):
    merged = merge_episode_characters(
        [
            (
                "mal",
                {
                    "characters": [
                        "not-a-dict",
                        {"name": "   ", "role": "MAIN"},
                        {
                            "name": "Monkey D. Luffy",
                            "role": "UNKNOWN",
                            "sources": ["https://myanimelist.net/character/40"],
                            "voice_actors": [
                                "skip-me",
                                {
                                    "name": "Mayumi Tanaka",
                                    "language": "ja",
                                    "sources": ["https://myanimelist.net/people/75"],
                                },
                            ],
                        },
                        {
                            "name": "Roronoa Zoro",
                            "role": "MAIN",
                            "sources": ["https://myanimelist.net/character/62"],
                        },
                    ]
                },
            ),
            (
                "anidb",
                {
                    "characters": [
                        {
                            "name": "Monkey D. Luffy",
                            "role": "MAIN",
                            "sources": ["https://anidb.net/character/474"],
                            "voice_actors": [
                                {
                                    "name": "Tanaka Mayumi",
                                    "language": "ja",
                                    "sources": ["https://anidb.net/creator/322"],
                                }
                            ],
                        },
                        {
                            "name": "Roronoa Zoro",
                            "role": "SUPPORTING",
                            "sources": ["https://anidb.net/character/475"],
                        },
                    ]
                },
            ),
        ]
    )

    assert [character.name for character in merged] == [
        "Monkey D. Luffy",
        "Roronoa Zoro",
    ]
    luffy, zoro = merged
    assert luffy.role == anime_pb2.CHARACTER_ROLE_MAIN
    assert list(luffy.sources) == [
        "https://anidb.net/character/474",
        "https://myanimelist.net/character/40",
    ]
    assert len(luffy.voice_actors) == 1
    assert list(luffy.voice_actors[0].sources) == [
        "https://anidb.net/creator/322",
        "https://myanimelist.net/people/75",
    ]
    assert zoro.role == anime_pb2.CHARACTER_ROLE_MAIN
    assert list(zoro.sources) == [
        "https://anidb.net/character/475",
        "https://myanimelist.net/character/62",
    ]


def test_merge_episode_staff_deduplicates_by_name_and_role_and_skips_invalid_entries() -> (
    None
):
    merged = merge_episode_staff(
        [
            (
                "mal",
                {
                    "staff": [
                        "invalid",
                        {"name": "   ", "role": "Director"},
                        {
                            "name": "Konosuke Uda",
                            "role": "Director",
                            "sources": ["https://myanimelist.net/people/5259"],
                        },
                        {
                            "name": "Konosuke Uda",
                            "role": "Storyboard",
                            "sources": ["https://myanimelist.net/people/5259"],
                        },
                    ]
                },
            ),
            (
                "anidb",
                {
                    "staff": [
                        {
                            "name": "Konosuke Uda",
                            "role": "director",
                            "sources": ["https://anidb.net/creator/123"],
                        }
                    ]
                },
            ),
        ]
    )

    assert [(member.name, member.role) for member in merged] == [
        ("Konosuke Uda", "Director"),
        ("Konosuke Uda", "Storyboard"),
    ]
    assert list(merged[0].sources) == [
        "https://anidb.net/creator/123",
        "https://myanimelist.net/people/5259",
    ]
