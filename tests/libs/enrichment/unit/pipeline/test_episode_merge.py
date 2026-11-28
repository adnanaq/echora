from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from enrichment.pipeline.episode_merge import merge_episode_pages
from enrichment_proto.v1.enrichment_service_pb2 import MergeEpisodeRequest
from google.protobuf import wrappers_pb2
from google.protobuf.any_pb2 import Any as AnyProto  # ty: ignore[unresolved-import]
from google.protobuf.timestamp_pb2 import Timestamp  # ty: ignore[unresolved-import]
from shared_proto.v1 import anime_pb2
from shared_proto.v1.merge_pb2 import (
    EpisodeListEntry,
    FieldCorrection,
    PageKey,
    PageResponse,
    PageSource,
    StoredPage,
)


def _ts(year: int, month: int, day: int) -> Timestamp:
    ts = Timestamp()
    ts.FromDatetime(datetime(year, month, day, tzinfo=UTC))
    return ts


def _pack_string(value: str) -> AnyProto:
    packed = AnyProto()
    string_cls = getattr(wrappers_pb2, "StringValue")
    packed.Pack(string_cls(value=value))
    return packed


def _pack_bool(value: bool) -> AnyProto:
    packed = AnyProto()
    bool_cls = getattr(wrappers_pb2, "BoolValue")
    packed.Pack(bool_cls(value=value))
    return packed


def test_merge_episode_from_list_entries_without_pages() -> None:
    req = MergeEpisodeRequest(
        episode_id="0195b42a-0000-7000-8000-000000000101",
        anime_id="0195b42a-0000-7000-8000-000000000001",
        episode_number=4,
        list_entries=[
            EpisodeListEntry(
                provider="kitsu",
                title="Kitsu Title",
                aired_at=_ts(1999, 10, 20),
                duration_seconds=1440,
                is_filler=False,
                is_recap=False,
            ),
            EpisodeListEntry(
                provider="anidb",
                title="AniDB Preferred Title",
                duration_seconds=1450,
                is_filler=True,
            ),
        ],
    )
    resp = merge_episode_pages(req)
    ep = resp.episode

    assert ep.id == "0195b42a-0000-7000-8000-000000000101"
    assert ep.anime_id == "0195b42a-0000-7000-8000-000000000001"
    assert ep.entity_type == anime_pb2.ENTITY_TYPE_EPISODE
    assert ep.episode_number == 4
    assert ep.title == "AniDB Preferred Title"
    assert ep.duration_seconds == 1450
    assert ep.is_filler is True
    assert ep.is_recap is False
    assert ep.aired_at.seconds > 0
    assert list(ep.sources) == []
    assert list(resp.page_sources) == []


def test_merge_episode_with_mal_and_kitsu_pages_and_corrections() -> None:
    mal_html = b"""<html><head>
<meta property="og:url" content="https://myanimelist.net/anime/21/One_Piece/episode/1"/>
</head><body>
<h2 class="fs18"><span class="fw-n">#1 - </span>I'm Luffy!</h2>
<div class="pt8 pb8">Romance Dawn</div>
<table><tr><td class="td-title">Synopsis</td></tr></table>
<div><p class="synopsis">Luffy sets sail.</p></div>
</body></html>"""
    kitsu_json = {
        "data": {
            "id": "501",
            "type": "episodes",
            "attributes": {
                "number": 1,
                "seasonNumber": 1,
                "canonicalTitle": "I'm Luffy!",
                "titles": {
                    "en": "I'm Luffy!",
                    "ja_jp": "俺はルフィ!",
                    "en_jp": "Ore wa Luffy!",
                },
                "synopsis": "Kitsu synopsis.",
                "description": "Kitsu description.",
                "airdate": "1999-10-20",
                "length": 24,
                "thumbnail": {"original": "https://cdn.kitsu.app/ep1.jpg"},
            },
        },
        "included": [
            {"id": "12", "type": "anime", "attributes": {"slug": "one-piece"}}
        ],
    }
    req = MergeEpisodeRequest(
        episode_id="0195b42a-0000-7000-8000-000000000102",
        anime_id="0195b42a-0000-7000-8000-000000000001",
        episode_number=1,
        pages=[
            StoredPage(
                page_key=PageKey(
                    provider="mal",
                    entity_type=anime_pb2.ENTITY_TYPE_EPISODE,
                    address="21:1",
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
                    provider="kitsu",
                    entity_type=anime_pb2.ENTITY_TYPE_EPISODE,
                    address="501",
                ),
                responses=[
                    PageResponse(
                        request_name="main",
                        content_type="application/json",
                        body=json.dumps(kitsu_json).encode("utf-8"),
                    )
                ],
            ),
            StoredPage(
                page_key=PageKey(
                    provider="unknown",
                    entity_type=anime_pb2.ENTITY_TYPE_EPISODE,
                    address="999",
                ),
                is_dead=True,
            ),
        ],
        list_entries=[
            EpisodeListEntry(
                provider="anisearch",
                title="AniSearch Title",
                is_filler=False,
                is_recap=True,
            )
        ],
        corrections=[
            FieldCorrection(
                field_path="title",
                value=_pack_string("Corrected Episode Title"),
            ),
            FieldCorrection(
                field_path="is_filler",
                value=_pack_bool(True),
            ),
        ],
    )
    resp = merge_episode_pages(req)
    ep = resp.episode

    assert ep.episode_number == 1
    assert ep.season_number == 1
    assert ep.title == "Corrected Episode Title"
    assert ep.title_japanese == "俺はルフィ!"
    assert ep.title_romaji == "Ore wa Luffy!"
    assert ep.synopsis == "Kitsu synopsis."
    assert ep.is_filler is True
    assert ep.is_recap is False
    assert list(ep.images) == ["https://cdn.kitsu.app/ep1.jpg"]
    assert list(ep.sources) == [
        "https://kitsu.app/anime/one-piece/episodes/1",
        "https://myanimelist.net/anime/21/One_Piece/episode/1",
    ]
    assert list(resp.page_sources) == [
        PageSource(
            page_key=PageKey(
                provider="mal",
                entity_type=anime_pb2.ENTITY_TYPE_EPISODE,
                address="21:1",
            ),
            source="https://myanimelist.net/anime/21/One_Piece/episode/1",
            title="I'm Luffy!",
        ),
        PageSource(
            page_key=PageKey(
                provider="kitsu",
                entity_type=anime_pb2.ENTITY_TYPE_EPISODE,
                address="501",
            ),
            source="https://kitsu.app/anime/one-piece/episodes/1",
            title="I'm Luffy!",
        ),
    ]


def test_merge_episode_score_titles_streaming_and_unknown_provider() -> None:
    from unittest.mock import patch

    req = MergeEpisodeRequest(
        episode_id="ep-3",
        anime_id="anime-3",
        episode_number=3,
        pages=[
            StoredPage(
                page_key=PageKey(
                    provider="custom_provider",
                    entity_type=anime_pb2.ENTITY_TYPE_EPISODE,
                    address="1",
                ),
                responses=[
                    PageResponse(
                        request_name="main",
                        content_type="application/json",
                        body=b"{}",
                    )
                ],
            )
        ],
        list_entries=[
            EpisodeListEntry(provider="anisearch", title="Ep 3", is_recap=True)
        ],
    )
    with patch(
        "enrichment.pipeline.episode_merge._map_stored_episode_page",
        return_value={
            "episode_number": 30,
            "score": 9.1,
            "titles": {"de": "Folge 3"},
            "streaming": {"crunchyroll": "https://crunchyroll.com/ep3"},
        },
    ):
        resp = merge_episode_pages(req)

    ep = resp.episode
    assert ep.episode_number == 3
    assert ep.score == pytest.approx(9.1, abs=1e-5)
    assert ep.titles["de"] == "Folge 3"
    assert ep.streaming["crunchyroll"] == "https://crunchyroll.com/ep3"
    assert ep.is_recap is True
    assert list(resp.page_sources) == []


def test_merge_episode_mal_page_stores_og_url_with_slug() -> None:
    mal_html = b"""<html><head>
<meta property="og:url" content="https://myanimelist.net/anime/21/One_Piece/episode/7"/>
</head><body>
<h2 class="fs18"><span class="fw-n">#7 - </span>Ep 7</h2>
<div class="pt8 pb8">Sub 7</div>
<table><tr><td class="td-title">Synopsis</td></tr></table>
<div><p class="synopsis">Synopsis 7.</p></div>
</body></html>"""
    req = MergeEpisodeRequest(
        episode_id="ep-7",
        anime_id="anime-21",
        episode_number=7,
        pages=[
            StoredPage(
                page_key=PageKey(
                    provider="mal",
                    entity_type=anime_pb2.ENTITY_TYPE_EPISODE,
                    address="21:7",
                ),
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
    resp = merge_episode_pages(req)
    assert resp.episode.episode_number == 7
    assert (
        "https://myanimelist.net/anime/21/One_Piece/episode/7" in resp.episode.sources
    )
    assert list(resp.page_sources) == [
        PageSource(
            page_key=PageKey(
                provider="mal",
                entity_type=anime_pb2.ENTITY_TYPE_EPISODE,
                address="21:7",
            ),
            source="https://myanimelist.net/anime/21/One_Piece/episode/7",
            title="Ep 7",
        )
    ]


def test_merge_episode_pages_not_found_with_responses_and_dead_anidb_excluded() -> None:
    mal_html = Path(
        "tests/libs/enrichment/unit/sources/mal/fixtures/mal_episode_1.html"
    ).read_bytes()
    response = merge_episode_pages(
        MergeEpisodeRequest(
            episode_id="ep-1-not-found",
            anime_id="anime-21",
            episode_number=1,
            pages=[
                StoredPage(
                    page_key=PageKey(
                        provider="mal",
                        entity_type=anime_pb2.ENTITY_TYPE_EPISODE,
                        address="21:1",
                    ),
                    is_not_found=True,
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
                        provider="anidb",
                        entity_type=anime_pb2.ENTITY_TYPE_EPISODE,
                        address="999",
                    ),
                    is_dead=True,
                ),
            ],
        )
    )
    assert response.episode.title
    assert "https://anidb.net/episode/999" not in response.episode.sources
    assert [source.page_key.provider for source in response.page_sources] == ["mal"]


def test_merge_episode_pages_real_mal_episode_populates_sorted_characters_and_staff() -> (
    None
):
    mal_html = Path(
        "tests/libs/enrichment/unit/sources/mal/fixtures/mal_episode_1.html"
    ).read_bytes()
    response = merge_episode_pages(
        MergeEpisodeRequest(
            episode_id="ep-1-cast",
            anime_id="anime-21",
            episode_number=1,
            pages=[
                StoredPage(
                    page_key=PageKey(
                        provider="mal",
                        entity_type=anime_pb2.ENTITY_TYPE_EPISODE,
                        address="21:1",
                    ),
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
    episode = response.episode
    assert len(episode.characters) == 10
    assert len(episode.staff) == 13
    assert [
        (character.name, character.role, tuple(character.sources))
        for character in episode.characters
    ] == sorted(
        (character.name, character.role, tuple(character.sources))
        for character in episode.characters
    )
    assert [
        (member.name, member.role, tuple(member.sources)) for member in episode.staff
    ] == sorted(
        (member.name, member.role, tuple(member.sources)) for member in episode.staff
    )
    luffy = next(
        character for character in episode.characters if "Luffy" in character.name
    )
    assert luffy.role == anime_pb2.CHARACTER_ROLE_MAIN
    assert len(luffy.voice_actors) >= 1


def test_mal_and_kitsu_episode_response_mapper_errors() -> None:
    from enrichment.sources.kitsu.kitsu_mapper import map_kitsu_episode_responses
    from enrichment.sources.mal.mal_mapper import map_mal_episode_responses

    with pytest.raises(ValueError, match="missing required 'main'"):
        map_mal_episode_responses([])

    with pytest.raises(ValueError, match="yielded no episode data"):
        map_mal_episode_responses(
            [
                PageResponse(
                    request_name="main",
                    content_type="text/html",
                    body=b"<html><body><div>No episode</div></body></html>",
                )
            ]
        )

    with pytest.raises(ValueError, match="missing required 'main'"):
        map_kitsu_episode_responses([])

    with pytest.raises(TypeError, match="missing 'data' object"):
        map_kitsu_episode_responses(
            [
                PageResponse(
                    request_name="main",
                    content_type="application/json",
                    body=b'{"data": null}',
                )
            ]
        )


def test_map_mal_episode_responses_header_fallback_and_missing_number_error() -> None:
    from enrichment.sources.mal.mal_mapper import map_mal_episode_responses

    header_only_html = b"""<html><body>
<h2 class="fs18"><span class="fw-n">#42 - </span>Header Number</h2>
<div class="pt8 pb8">Sub 42</div>
<table><tr><td class="td-title">Synopsis</td></tr></table>
<div><p class="synopsis">Synopsis 42.</p></div>
</body></html>"""
    mapped_header = map_mal_episode_responses(
        [
            PageResponse(
                request_name="main",
                content_type="text/html",
                body=header_only_html,
            )
        ]
    )
    assert mapped_header["episode_number"] == 42

    swapped_og_html = b"""<html><head>
<meta content="https://myanimelist.net/anime/21/One_Piece/episode/9" property="og:url" />
</head><body>
<h2 class="fs18"><span class="fw-n">#9 - </span>Swapped Meta</h2>
</body></html>"""
    mapped_swapped = map_mal_episode_responses(
        [
            PageResponse(
                request_name="main",
                content_type="text/html",
                body=swapped_og_html,
            )
        ]
    )
    assert mapped_swapped["episode_number"] == 9
    assert mapped_swapped["sources"] == [
        "https://myanimelist.net/anime/21/One_Piece/episode/9"
    ]

    special_html = b"""<html><body>
<h2 class="fs18">#Special</h2>
<div class="pt8 pb8">Sub</div>
</body></html>"""
    with pytest.raises(ValueError, match="yielded no episode number"):
        map_mal_episode_responses(
            [
                PageResponse(
                    request_name="main",
                    content_type="text/html",
                    body=special_html,
                )
            ]
        )


def test_merge_episode_pages_kitsu_page_preserves_list_entry_filler_and_recap_flags() -> (
    None
):
    kitsu_body = json.dumps(
        {
            "data": {
                "id": "103482",
                "type": "episodes",
                "attributes": {
                    "number": 50,
                    "canonicalTitle": "Usopp vs. Daddy",
                    "synopsis": "Filler episode.",
                    "length": 24,
                },
            }
        }
    ).encode("utf-8")
    response = merge_episode_pages(
        MergeEpisodeRequest(
            episode_id="ep-50-kitsu",
            anime_id="anime-21",
            episode_number=50,
            pages=[
                StoredPage(
                    page_key=PageKey(
                        provider="kitsu",
                        entity_type=anime_pb2.ENTITY_TYPE_EPISODE,
                        address="103482",
                    ),
                    responses=[
                        PageResponse(
                            request_name="main",
                            content_type="application/json",
                            body=kitsu_body,
                        )
                    ],
                )
            ],
            list_entries=[
                EpisodeListEntry(
                    provider="kitsu",
                    title="Usopp vs. Daddy",
                    is_filler=True,
                    is_recap=True,
                )
            ],
        )
    )
    assert response.episode.is_filler is True
    assert response.episode.is_recap is True


def test_merge_episode_pages_repeated_page_provider_raises_value_error() -> None:
    mal_page = StoredPage(
        page_key=PageKey(
            provider="mal",
            entity_type=anime_pb2.ENTITY_TYPE_EPISODE,
            address="21:1",
        ),
        is_dead=True,
    )
    request = MergeEpisodeRequest(
        episode_id="0195b42a-0000-7000-8000-000000000101",
        anime_id="0195b42a-0000-7000-8000-000000000001",
        episode_number=1,
        pages=[mal_page, mal_page],
    )
    with pytest.raises(ValueError, match=r"^pages .* mal$"):
        merge_episode_pages(request)


def test_merge_episode_pages_repeated_list_entry_provider_raises_value_error() -> None:
    request = MergeEpisodeRequest(
        episode_id="0195b42a-0000-7000-8000-000000000101",
        anime_id="0195b42a-0000-7000-8000-000000000001",
        episode_number=1,
        list_entries=[
            EpisodeListEntry(provider="kitsu", title="First"),
            EpisodeListEntry(provider="kitsu", title="Second"),
        ],
    )
    with pytest.raises(ValueError, match=r"^list_entries .* kitsu$"):
        merge_episode_pages(request)
