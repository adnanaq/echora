from __future__ import annotations

import json
from pathlib import Path

import pytest
from enrichment.pipeline.character_merge import merge_character_pages
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
    fixture_dir = (
        Path(__file__).resolve().parents[1]
        / "sources"
        / "anisearch"
        / "test_anisearch_character_crawler"
    )
    return StoredPage(
        page_key=PageKey(
            provider="anisearch",
            entity_type=anime_pb2.ENTITY_TYPE_CHARACTER,
            address="4852",
        ),
        responses=[
            PageResponse(
                request_name="main",
                content_type="text/html",
                body=(fixture_dir / "luffy_char.html").read_bytes(),
            ),
            PageResponse(
                request_name="anime",
                content_type="text/html",
                body=(fixture_dir / "luffy_anime_ography.html").read_bytes(),
            ),
        ],
    )


def test_merge_character_across_providers_and_apply_corrections() -> None:
    anidb_html = (
        Path(__file__).resolve().parents[1]
        / "sources"
        / "anidb"
        / "test_anidb_character_luffy_regression"
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
    assert char.attributes["height"] == "172-174"
    assert "Joy Boy" in char.spoilers.values()
    assert list(char.images) == sorted(char.images)
    assert list(char.traits) == sorted(char.traits)
    assert list(char.sources) == sorted(char.sources)
    page_sources_by_address = {
        (page.page_key.provider, page.page_key.address): page.source
        for page in resp.page_sources
    }
    assert len(resp.page_sources) == 4
    assert ("mal", "40") not in page_sources_by_address
    assert "https://myanimelist.net/character/40" not in char.sources
    assert page_sources_by_address[("anidb", "474")] == (
        "https://anidb.net/character/474"
    )
    assert page_sources_by_address[("anisearch", "4852")] == (
        "https://www.anisearch.com/character/4852,monkey-d-luffy"
    )
    assert page_sources_by_address[("kitsu", "101")] == (
        "https://kitsu.app/characters/monkey-d-luffy"
    )
    assert list(char.sources) == sorted({page.source for page in resp.page_sources})

    assert len(char.animeography) > 20
    one_piece = next(entry for entry in char.animeography if entry.title == "One Piece")
    assert one_piece.role == anime_pb2.CHARACTER_ROLE_MAIN
    assert "https://anilist.co/anime/21" in one_piece.sources
    assert "https://anidb.net/anime/69" in one_piece.sources
    assert "https://kitsu.app/anime/12" in one_piece.sources
    assert any(
        source.startswith("https://www.anisearch.com/anime/2227")
        for source in one_piece.sources
    )
    assert any(
        entry.role == anime_pb2.CHARACTER_ROLE_UNKNOWN
        and any("anisearch.com/anime/" in source for source in entry.sources)
        for entry in char.animeography
    )
    assert len(char.mangaography) == 1
    assert "https://anilist.co/manga/30013" in char.mangaography[0].sources


def test_character_page_mapper_error_branches() -> None:
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


def test_merge_character_pages_not_found_with_stored_responses_still_merges() -> None:
    mal_html = Path(
        "tests/libs/enrichment/unit/sources/mal/fixtures/mal_character_40.html"
    ).read_bytes()
    response = merge_character_pages(
        MergeCharacterRequest(
            character_id="0195b42a-0000-7000-8000-000000000801",
            pages=[
                StoredPage(
                    page_key=PageKey(
                        provider="mal",
                        entity_type=anime_pb2.ENTITY_TYPE_CHARACTER,
                        address="40",
                    ),
                    is_not_found=True,
                    responses=[
                        PageResponse(
                            request_name="main",
                            content_type="text/html",
                            body=mal_html,
                        )
                    ],
                )
            ],
        )
    )
    assert response.character.name == "Luffy Monkey D."
    assert len(response.character.animeography) > 0
    assert (
        "https://myanimelist.net/character/40/Luffy_Monkey_D"
        in response.character.sources
    )


def test_merge_character_pages_merges_and_deduplicates_voice_actors_including_anilist() -> (
    None
):
    mal_html = Path(
        "tests/libs/enrichment/unit/sources/mal/fixtures/mal_character_40.html"
    ).read_bytes()
    ap_html = Path(
        "tests/libs/enrichment/unit/sources/anime_planet/fixtures/ap_char_monkey-d-luffy.html"
    ).read_bytes()
    anilist_body = json.dumps(
        {
            "data": {
                "Character": {
                    "id": 40,
                    "name": {"full": "Monkey D. Luffy"},
                    "siteUrl": "https://anilist.co/character/40",
                    "media": {
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
                                    }
                                ],
                            }
                        ]
                    },
                }
            }
        }
    ).encode("utf-8")

    response = merge_character_pages(
        MergeCharacterRequest(
            character_id="0195b42a-0000-7000-8000-000000000802",
            pages=[
                StoredPage(
                    page_key=PageKey(
                        provider="mal",
                        entity_type=anime_pb2.ENTITY_TYPE_CHARACTER,
                        address="40",
                    ),
                    responses=[
                        PageResponse(
                            request_name="main",
                            content_type="text/html",
                            body=mal_html,
                        )
                    ],
                ),
                StoredPage(
                    page_key=PageKey(
                        provider="anilist",
                        entity_type=anime_pb2.ENTITY_TYPE_CHARACTER,
                        address="40",
                    ),
                    responses=[
                        PageResponse(
                            request_name="main",
                            content_type="application/json",
                            body=anilist_body,
                        )
                    ],
                ),
                StoredPage(
                    page_key=PageKey(
                        provider="anime_planet",
                        entity_type=anime_pb2.ENTITY_TYPE_CHARACTER,
                        address="monkey-d-luffy",
                    ),
                    responses=[
                        PageResponse(
                            request_name="main",
                            content_type="text/html",
                            body=ap_html,
                        )
                    ],
                ),
            ],
        )
    )
    voice_actors = response.character.voice_actors
    assert len(voice_actors) >= 28
    assert [
        (actor.name, actor.language, tuple(actor.sources)) for actor in voice_actors
    ] == sorted(
        (actor.name, actor.language, tuple(actor.sources)) for actor in voice_actors
    )
    tanaka_matches = [
        actor
        for actor in voice_actors
        if "tanaka" in actor.name.casefold() and "mayumi" in actor.name.casefold()
    ]
    assert len(tanaka_matches) == 1
    assert tanaka_matches[0].language == "ja"
    assert "https://anilist.co/staff/95075" in tanaka_matches[0].sources
    assert any(
        "myanimelist.net/people/" in source for source in tanaka_matches[0].sources
    )
    assert tanaka_matches[0].native_name == "田中真弓"


def test_merge_character_pages_normalizes_attribute_keys_and_drops_whitespace_values() -> (
    None
):
    mal_html = Path(
        "tests/libs/enrichment/unit/sources/mal/fixtures/mal_character_40.html"
    ).read_bytes()
    ap_html = Path(
        "tests/libs/enrichment/unit/sources/anime_planet/fixtures/ap_char_monkey-d-luffy.html"
    ).read_bytes()
    anisearch_html = Path(
        "tests/libs/enrichment/unit/sources/anisearch/test_anisearch_character_crawler/luffy_char.html"
    ).read_bytes()
    anilist_whitespace = json.dumps(
        {
            "data": {
                "Character": {
                    "id": 40,
                    "name": {
                        "full": "Monkey D. Luffy",
                        "alternativeSpoiler": ["   "],
                    },
                    "gender": "   ",
                    "siteUrl": "https://anilist.co/character/40",
                    "media": {"edges": []},
                }
            }
        }
    ).encode("utf-8")

    response = merge_character_pages(
        MergeCharacterRequest(
            character_id="0195b42a-0000-7000-8000-000000000803",
            pages=[
                StoredPage(
                    page_key=PageKey(
                        provider="mal",
                        entity_type=anime_pb2.ENTITY_TYPE_CHARACTER,
                        address="40",
                    ),
                    responses=[
                        PageResponse(
                            request_name="main",
                            content_type="text/html",
                            body=mal_html,
                        )
                    ],
                ),
                StoredPage(
                    page_key=PageKey(
                        provider="anilist",
                        entity_type=anime_pb2.ENTITY_TYPE_CHARACTER,
                        address="40",
                    ),
                    responses=[
                        PageResponse(
                            request_name="main",
                            content_type="application/json",
                            body=anilist_whitespace,
                        )
                    ],
                ),
                StoredPage(
                    page_key=PageKey(
                        provider="anime_planet",
                        entity_type=anime_pb2.ENTITY_TYPE_CHARACTER,
                        address="monkey-d-luffy",
                    ),
                    responses=[
                        PageResponse(
                            request_name="main",
                            content_type="text/html",
                            body=ap_html,
                        )
                    ],
                ),
                StoredPage(
                    page_key=PageKey(
                        provider="anisearch",
                        entity_type=anime_pb2.ENTITY_TYPE_CHARACTER,
                        address="4852",
                    ),
                    responses=[
                        PageResponse(
                            request_name="main",
                            content_type="text/html",
                            body=anisearch_html,
                        )
                    ],
                ),
            ],
        )
    )
    attributes = dict(response.character.attributes)
    spoilers = dict(response.character.spoilers)
    assert "blood-type" not in attributes
    assert "hair-color" not in attributes
    assert "eye-color" not in attributes
    assert "birthdate" not in attributes
    assert "birthday" not in attributes
    assert "blood_type" in attributes
    assert "date_of_birth" in attributes
    assert all(k.strip() and v.strip() for k, v in attributes.items())
    assert all(k.strip() and v.strip() for k, v in spoilers.items())


def test_group_ography_entries_bridges_chains_and_separates_same_title_same_provider() -> (
    None
):
    from enrichment.pipeline.character_merge import _group_ography_entries

    bridged = _group_ography_entries(
        [
            (
                "mal",
                {
                    "animeography": [
                        {
                            "title": "Alpha",
                            "role": "MAIN",
                            "sources": ["https://myanimelist.net/anime/1"],
                        }
                    ]
                },
            ),
            (
                "anilist",
                {
                    "animeography": [
                        {
                            "title": "Beta",
                            "role": "SUPPORTING",
                            "sources": ["https://anilist.co/anime/2"],
                        }
                    ]
                },
            ),
            (
                "kitsu",
                {
                    "animeography": [
                        {
                            "title": "Gamma",
                            "role": "UNKNOWN",
                            "sources": [
                                "https://myanimelist.net/anime/1",
                                "https://anilist.co/anime/2",
                            ],
                        }
                    ]
                },
            ),
        ],
        "animeography",
    )
    assert len(bridged) == 1
    assert len(bridged[0]) == 3

    separated = _group_ography_entries(
        [
            (
                "mal",
                {
                    "animeography": [
                        {
                            "title": "Hunter x Hunter",
                            "role": "MAIN",
                            "sources": ["https://myanimelist.net/anime/136"],
                        },
                        {
                            "title": "Hunter x Hunter",
                            "role": "MAIN",
                            "sources": ["https://myanimelist.net/anime/11061"],
                        },
                    ]
                },
            )
        ],
        "animeography",
    )
    assert len(separated) == 2


def test_character_mappers_without_canonical_url_emit_empty_sources() -> None:
    from enrichment.sources.anime_planet.anime_planet_character_models import (
        AnimePlanetCharacter,
    )
    from enrichment.sources.anime_planet.animeplanet_mapper import (
        character_from_animeplanet,
    )
    from enrichment.sources.anisearch.anisearch_anime_models import (
        AniSearchCharacter,
    )
    from enrichment.sources.anisearch.anisearch_mapper import (
        character_from_anisearch,
    )

    ap_record = character_from_animeplanet(
        AnimePlanetCharacter(name="Luffy", slug="monkey-d-luffy", url="")
    )
    anisearch_record = character_from_anisearch(
        AniSearchCharacter(name="Luffy", source="")
    )
    assert ap_record["sources"] == []
    assert anisearch_record["sources"] == []


def test_group_ography_entries_skips_blank_title_and_keeps_existing_cluster() -> None:
    from enrichment.pipeline.character_merge import _group_ography_entries

    groups = _group_ography_entries(
        [
            (
                "mal",
                {
                    "animeography": [
                        {
                            "title": "",
                            "role": "MAIN",
                            "sources": ["https://myanimelist.net/anime/21"],
                        },
                        {
                            "title": "One Piece",
                            "role": "MAIN",
                            "sources": ["https://myanimelist.net/anime/21"],
                        },
                    ]
                },
            ),
            (
                "anilist",
                {
                    "animeography": [
                        {
                            "title": "One Piece",
                            "role": "MAIN",
                            "sources": ["https://myanimelist.net/anime/21"],
                        }
                    ]
                },
            ),
        ],
        "animeography",
    )
    assert len(groups) == 1
    assert len(groups[0]) == 3


def test_merge_character_pages_unknown_provider_and_provider_priority_conflict() -> (
    None
):
    empty_response = merge_character_pages(
        MergeCharacterRequest(
            character_id="0195b42a-0000-7000-8000-000000000809",
            pages=[
                StoredPage(
                    page_key=PageKey(
                        provider="unknown_site",
                        entity_type=anime_pb2.ENTITY_TYPE_CHARACTER,
                        address="1",
                    ),
                    responses=[
                        PageResponse(
                            request_name="main",
                            content_type="text/html",
                            body=b"<html></html>",
                        )
                    ],
                )
            ],
        )
    )
    assert empty_response.character.name == ""
    assert list(empty_response.character.sources) == []
    assert list(empty_response.page_sources) == []

    priority_response = merge_character_pages(
        MergeCharacterRequest(
            character_id="0195b42a-0000-7000-8000-000000000810",
            pages=[_kitsu_character_page(), _anilist_character_page()],
        )
    )
    assert priority_response.character.name == "Monkey D. Luffy"
    assert priority_response.character.description == "AniList description."


def test_merge_character_pages_repeated_page_provider_raises_value_error() -> None:
    request = MergeCharacterRequest(
        character_id="0195b42a-0000-7000-8000-000000000201",
        pages=[_anilist_character_page(), _anilist_character_page()],
    )
    with pytest.raises(ValueError, match=r"^pages .* anilist$"):
        merge_character_pages(request)
