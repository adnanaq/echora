"""Unit tests for stored-page anime merging (ECHO-105)."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from enrichment.pipeline.anime_merge import merge_anime_pages, page_key_to_anime_url
from enrichment.pipeline.episode_reference_merger import (
    _find_episode_page_sources,
    _list_provider_episodes,
    _ListedEpisode,
    _pick_episode_title,
)
from enrichment_proto.v1.enrichment_service_pb2 import MergeAnimeRequest
from google.protobuf import wrappers_pb2
from google.protobuf.any_pb2 import Any as AnyProto
from google.protobuf.timestamp_pb2 import Timestamp
from shared_proto.v1 import anime_pb2
from shared_proto.v1.merge_pb2 import (
    EpisodeListEntry,
    EpisodeReference,
    FieldCorrection,
    GeneratedText,
    PageKey,
    PageResponse,
    PageSource,
    RelationClaim,
    StoredPage,
)


def _anilist_page(
    address: str = "21",
    *,
    synopsis: str = "Provider AniList synopsis.",
    relations: list[dict[str, object]] | None = None,
    site_url: str | None = None,
) -> StoredPage:
    payload = {
        "data": {
            "Media": {
                "id": int(address),
                "title": {
                    "romaji": "One Piece",
                    "english": "One Piece",
                    "native": "ワンピース",
                },
                "format": "TV",
                "status": "RELEASING",
                "description": synopsis,
                "episodes": 1100,
                "genres": ["Action", "Adventure"],
                "relations": {"edges": relations or []},
                "siteUrl": site_url or f"https://anilist.co/anime/{address}",
            }
        }
    }
    return StoredPage(
        page_key=PageKey(
            provider="anilist",
            entity_type=anime_pb2.ENTITY_TYPE_ANIME,
            address=address,
        ),
        responses=[
            PageResponse(
                request_name="main",
                content_type="application/json",
                body=json.dumps(payload).encode("utf-8"),
            )
        ],
    )


def _kitsu_page(
    address: str = "12",
    *,
    synopsis: str = "Provider Kitsu synopsis.",
    episodes: list[dict[str, object]] | None = None,
) -> StoredPage:
    main_payload = {
        "data": {
            "id": address,
            "type": "anime",
            "attributes": {
                "slug": "one-piece",
                "canonicalTitle": "One Piece",
                "titles": {"en": "One Piece", "ja_jp": "ワンピース"},
                "synopsis": synopsis,
                "subtype": "TV",
                "status": "current",
                "episodeCount": 1100,
            },
        }
    }
    ep_payload = {"data": episodes or []}
    return StoredPage(
        page_key=PageKey(
            provider="kitsu",
            entity_type=anime_pb2.ENTITY_TYPE_ANIME,
            address=address,
        ),
        responses=[
            PageResponse(
                request_name="main",
                content_type="application/json",
                body=json.dumps(main_payload).encode("utf-8"),
            ),
            PageResponse(
                request_name="episodes_page_1",
                content_type="application/json",
                body=json.dumps(ep_payload).encode("utf-8"),
            ),
        ],
    )


def _pack_string(value: str) -> AnyProto:
    packed = AnyProto()
    packed.Pack(wrappers_pb2.StringValue(value=value))
    return packed


def _pack_int32(value: int) -> AnyProto:
    packed = AnyProto()
    packed.Pack(wrappers_pb2.Int32Value(value=value))
    return packed


def _ts(year: int, month: int, day: int) -> Timestamp:
    ts = Timestamp()
    ts.FromDatetime(datetime(year, month, day, tzinfo=UTC))
    return ts


def test_merge_uses_passed_id_and_page_link_sources() -> None:
    """Merge must use request.anime_id and derive sources/partial_sources from links."""
    work_id = "0195b42a-0000-7000-8000-000000000001"
    req = MergeAnimeRequest(
        anime_id=work_id,
        source_pages=[_anilist_page("21")],
        partial_source_pages=[
            PageKey(
                provider="anidb",
                entity_type=anime_pb2.ENTITY_TYPE_ANIME,
                address="69",
            )
        ],
    )
    resp = merge_anime_pages(req)

    assert resp.anime.id == work_id
    assert list(resp.anime.sources) == ["https://anilist.co/anime/21"]
    assert list(resp.anime.partial_sources) == ["https://anidb.net/anime/69"]
    assert resp.anime.synopsis == ""
    assert list(resp.page_sources) == [
        PageSource(
            page_key=PageKey(
                provider="anilist",
                entity_type=anime_pb2.ENTITY_TYPE_ANIME,
                address="21",
            ),
            source="https://anilist.co/anime/21",
            title="One Piece",
        ),
    ]


def test_synopsis_precedence_correction_then_latest_generated_text() -> None:
    """Synopsis uses FieldCorrection, else latest GeneratedText, never providers."""
    base_pages = [_anilist_page(synopsis="Provider text should be ignored.")]

    older = GeneratedText(
        field_path="synopsis",
        text="Older AI synopsis.",
        written_at=_ts(2026, 1, 1),
    )
    newer = GeneratedText(
        field_path="synopsis",
        text="Newest AI synopsis.",
        written_at=_ts(2026, 2, 1),
    )
    resp_ai = merge_anime_pages(
        MergeAnimeRequest(
            anime_id="0195b42a-0000-7000-8000-000000000002",
            source_pages=base_pages,
            generated_texts=[older, newer],
        )
    )
    assert resp_ai.anime.synopsis == "Newest AI synopsis."

    resp_corr = merge_anime_pages(
        MergeAnimeRequest(
            anime_id="0195b42a-0000-7000-8000-000000000002",
            source_pages=base_pages,
            generated_texts=[older, newer],
            corrections=[
                FieldCorrection(
                    field_path="synopsis",
                    value=_pack_string("Human-corrected synopsis."),
                ),
                FieldCorrection(
                    field_path="episode_count",
                    value=_pack_int32(1200),
                ),
            ],
        )
    )
    assert resp_corr.anime.synopsis == "Human-corrected synopsis."
    assert resp_corr.anime.episode_count == 1200


def test_provider_synopses_hash_detects_changes_deterministically() -> None:
    """provider_synopses_hash is order-independent and changes when synopsis text changes."""
    req_a = MergeAnimeRequest(
        anime_id="0195b42a-0000-7000-8000-000000000003",
        source_pages=[
            _anilist_page(synopsis="Alpha"),
            _kitsu_page(synopsis="Beta"),
        ],
    )
    req_b = MergeAnimeRequest(
        anime_id="0195b42a-0000-7000-8000-000000000003",
        source_pages=[
            _kitsu_page(synopsis="Beta"),
            _anilist_page(synopsis="Alpha"),
        ],
    )
    req_changed = MergeAnimeRequest(
        anime_id="0195b42a-0000-7000-8000-000000000003",
        source_pages=[
            _anilist_page(synopsis="Alpha updated"),
            _kitsu_page(synopsis="Beta"),
        ],
    )
    resp_a = merge_anime_pages(req_a)
    resp_b = merge_anime_pages(req_b)
    resp_changed = merge_anime_pages(req_changed)

    assert resp_a.provider_synopses_hash
    assert resp_a.provider_synopses_hash == resp_b.provider_synopses_hash
    assert resp_a.provider_synopses_hash != resp_changed.provider_synopses_hash


def test_episode_id_preservation_and_uuidv7_for_new_episodes() -> None:
    """Known episodes keep their IDs; newly listed episodes receive a UUIDv7."""
    known_ep_id = "0195b42a-0000-7000-8000-000000000101"
    kitsu_episodes = [
        {
            "id": "501",
            "type": "episodes",
            "attributes": {
                "number": 1,
                "canonicalTitle": "I'm Luffy!",
                "airdate": "1999-10-20",
                "length": 24,
            },
        },
        {
            "id": "502",
            "type": "episodes",
            "attributes": {
                "number": 2,
                "canonicalTitle": "Enter Zoro!",
                "airdate": "1999-11-17",
                "length": 25,
            },
        },
    ]
    req = MergeAnimeRequest(
        anime_id="0195b42a-0000-7000-8000-000000000004",
        source_pages=[_kitsu_page(episodes=kitsu_episodes)],
        known_episodes=[
            EpisodeReference(
                episode_id=known_ep_id,
                episode_number=1,
                title="Old Title",
            )
        ],
    )
    resp = merge_anime_pages(req)

    by_num = {ep.episode_number: ep for ep in resp.episodes}
    assert set(by_num) == {1, 2}
    assert by_num[1].episode_id == known_ep_id
    assert by_num[1].title == "I'm Luffy!"
    assert len(by_num[1].list_entries) == 1
    assert by_num[1].list_entries[0].provider == "kitsu"
    assert by_num[1].list_entries[0].duration_seconds == 1440
    assert [
        (
            page_source.page_key.provider,
            page_source.page_key.address,
            page_source.source,
        )
        for episode_number in (1, 2)
        for page_source in by_num[episode_number].page_sources
    ] == [
        ("kitsu", "501", "https://kitsu.app/anime/one-piece/episodes/1"),
        ("kitsu", "502", "https://kitsu.app/anime/one-piece/episodes/2"),
    ]

    new_uuid = uuid.UUID(by_num[2].episode_id)
    assert new_uuid.version == 7
    assert by_num[2].title == "Enter Zoro!"


def test_relation_claims_and_sorted_lists() -> None:
    """Relation claims preserve provider types and all output lists are sorted."""
    relations = [
        {
            "relationType": "SEQUEL",
            "node": {
                "id": 999,
                "type": "ANIME",
                "title": {"romaji": "Sequel Work"},
                "format": "TV",
                "status": "FINISHED",
                "siteUrl": "https://anilist.co/anime/999",
            },
        },
        {
            "relationType": "PREQUEL",
            "node": {
                "id": 100,
                "type": "ANIME",
                "title": {"romaji": "Prequel Work"},
                "format": "TV",
                "status": "FINISHED",
                "siteUrl": "https://anilist.co/anime/100",
            },
        },
    ]
    resp = merge_anime_pages(
        MergeAnimeRequest(
            anime_id="0195b42a-0000-7000-8000-000000000005",
            source_pages=[_anilist_page(relations=relations), _kitsu_page()],
        )
    )

    assert list(resp.anime.genres) == sorted(resp.anime.genres)
    assert list(resp.anime.sources) == sorted(resp.anime.sources)
    assert len(resp.relation_claims) == 2
    assert [c.target_page.address for c in resp.relation_claims] == ["100", "999"]
    assert (
        resp.relation_claims[0].relation_type == anime_pb2.ANIME_RELATION_TYPE_PREQUEL
    )
    assert resp.relation_claims[1].relation_type == anime_pb2.ANIME_RELATION_TYPE_SEQUEL


def test_dead_not_found_unknown_provider_and_non_anime_claims() -> None:
    """Dead, not-found, unknown-provider pages and non-anime claims are skipped safely."""
    unknown_key = PageKey(
        provider="unknown_site",
        entity_type=anime_pb2.ENTITY_TYPE_ANIME,
        address="1",
    )
    assert page_key_to_anime_url(unknown_key) == ""
    assert (
        _find_episode_page_sources({"sources": ["https://anilist.co/anime/21"]}) == []
    )
    assert _find_episode_page_sources(
        {"sources": ["https://anidb.net/episode/412"]}
    ) == [
        PageSource(
            page_key=PageKey(
                provider="anidb",
                entity_type=anime_pb2.ENTITY_TYPE_EPISODE,
                address="412",
            ),
            source="https://anidb.net/episode/412",
        )
    ]

    manga_rel = [
        {
            "relationType": "ADAPTATION",
            "node": {
                "id": 13,
                "type": "MANGA",
                "title": {"romaji": "One Piece"},
                "format": "MANGA",
                "status": "RELEASING",
                "siteUrl": "https://anilist.co/manga/30013",
            },
        }
    ]
    req = MergeAnimeRequest(
        anime_id="0195b42a-0000-7000-8000-000000000006",
        source_pages=[
            _anilist_page(relations=manga_rel),
            StoredPage(
                page_key=PageKey(
                    provider="mal",
                    entity_type=anime_pb2.ENTITY_TYPE_ANIME,
                    address="21",
                ),
                is_dead=True,
            ),
            StoredPage(
                page_key=unknown_key,
                responses=[
                    PageResponse(
                        request_name="main",
                        content_type="text/html",
                        body=b"<html></html>",
                    )
                ],
            ),
        ],
    )
    resp = merge_anime_pages(req)
    assert resp.relation_claims == []
    assert "https://myanimelist.net/anime/21" in resp.anime.sources
    page_sources_by_provider = {
        page.page_key.provider: page.source for page in resp.page_sources
    }
    assert page_sources_by_provider == {
        "anilist": "https://anilist.co/anime/21",
        "mal": "https://myanimelist.net/anime/21",
    }


def test_anidb_and_anisearch_episode_extraction_and_page_key_matching() -> None:
    """AniDB XML and AniSearch HTML episode lists merge and match by PageKey."""
    anidb_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<anime id="69" restricted="false">
  <type>TV Series</type>
  <episodecount>1</episodecount>
  <titles>
    <title xml:lang="x-jat" type="main">One Piece</title>
  </titles>
  <episodes>
    <episode id="1001" update="2020-01-01">
      <epno type="1">1</epno>
      <length>24</length>
      <airdate>1999-10-20</airdate>
      <title xml:lang="en">I'm Luffy!</title>
    </episode>
  </episodes>
</anime>"""
    anisearch_main = b"""<html><head>
<link rel="canonical" href="https://www.anisearch.com/anime/2227,one-piece"/>
</head><body><h1>One Piece</h1></body></html>"""
    anisearch_eps = b"""<html><body>
<table class="episodes">
  <tr data-episode="true">
    <th itemprop="episodeNumber">1</th>
    <td data-title="Runtime"><div lang="ja">24 min</div></td>
    <td data-title="Date of Original Release"><div lang="ja">20.10.1999</div></td>
    <td data-title="Title"><div lang="en"><span itemprop="name">I'm Luffy!</span></div></td>
  </tr>
  <tr data-episode="true">
    <th itemprop="episodeNumber"></th>
    <td data-title="Title"><div lang="en"><span itemprop="name">Missing number</span></div></td>
  </tr>
</table>
</body></html>"""

    known_id = "0195b42a-0000-7000-8000-000000000555"
    req = MergeAnimeRequest(
        anime_id="0195b42a-0000-7000-8000-000000000007",
        source_pages=[
            StoredPage(
                page_key=PageKey(
                    provider="anidb",
                    entity_type=anime_pb2.ENTITY_TYPE_ANIME,
                    address="69",
                ),
                responses=[
                    PageResponse(
                        request_name="main",
                        content_type="application/xml",
                        body=anidb_xml,
                    )
                ],
            ),
            StoredPage(
                page_key=PageKey(
                    provider="anisearch",
                    entity_type=anime_pb2.ENTITY_TYPE_ANIME,
                    address="2227",
                ),
                responses=[
                    PageResponse(
                        request_name="main",
                        content_type="text/html",
                        body=anisearch_main,
                    ),
                    PageResponse(
                        request_name="episodes",
                        content_type="text/html",
                        body=anisearch_eps,
                    ),
                ],
            ),
        ],
        known_episodes=[
            EpisodeReference(
                episode_id=known_id,
                episode_number=999,
                title="Known Fallback",
                page_sources=[
                    PageSource(
                        page_key=PageKey(
                            provider="anidb",
                            entity_type=anime_pb2.ENTITY_TYPE_EPISODE,
                            address="1001",
                        )
                    )
                ],
            )
        ],
    )
    resp = merge_anime_pages(req)
    assert len(resp.episodes) == 1
    ep = resp.episodes[0]
    assert ep.episode_id == known_id
    assert ep.episode_number == 1
    assert ep.title == "I'm Luffy!"
    assert [entry.provider for entry in ep.list_entries] == ["anidb", "anisearch"]

    fallback_title = _pick_episode_title(
        [EpisodeListEntry(provider="custom", title="Custom Title")],
        None,
    )
    assert fallback_title == "Custom Title"
    known_title = _pick_episode_title(
        [EpisodeListEntry(provider="anidb")],
        EpisodeReference(episode_id=known_id, episode_number=1, title="Saved Title"),
    )
    assert known_title == "Saved Title"


def test_merge_helper_edge_cases_non_anime_claim_empty_source_and_missing_ep_num() -> (
    None
):
    """Cover non-anime relation URLs, empty source URLs, and missing episode numbers."""
    from unittest.mock import patch

    from enrichment.pipeline.anime_merge import (
        _build_linked_page_sources,
        _extract_relation_claims,
    )
    from enrichment.pipeline.episode_reference_merger import _collect_provider_episodes

    claims = _extract_relation_claims(
        {
            "anilist": {
                "related_anime": {
                    "OTHER": [{"sources": ["https://anilist.co/character/1"]}]
                }
            }
        }
    )
    assert claims == []

    source_page_sources, partial_page_sources = _build_linked_page_sources(
        [],
        [],
        {"anilist": {"sources": ["", "https://anilist.co/anime/21"]}},
    )
    assert source_page_sources == []
    assert partial_page_sources == []

    with patch(
        "enrichment.pipeline.episode_reference_merger._list_provider_episodes",
        return_value=[
            _ListedEpisode(
                page_sources=[], episode={"episode_number": None, "title": "Skip me"}
            )
        ],
    ):
        entries, keys = _collect_provider_episodes([_kitsu_page()], {})
    assert entries == {}
    assert keys == {}


def test_build_linked_page_sources_choose_reported_or_canonical_url_per_linked_page() -> (
    None
):
    from enrichment.pipeline.anime_merge import _build_linked_page_sources

    def build_anime_page_key(provider: str, address: str) -> PageKey:
        return PageKey(
            provider=provider,
            entity_type=anime_pb2.ENTITY_TYPE_ANIME,
            address=address,
        )

    records = {
        "mal": {
            "sources": [
                "https://myanimelist.net/anime/21",
                "https://myanimelist.net/anime/21/One_Piece",
                "https://myanimelist.net/anime/999/Unlinked_Work",
            ]
        },
        "kitsu": {"sources": ["https://kitsu.app/anime/one-piece-film/extra"]},
    }
    source_page_sources, partial_page_sources = _build_linked_page_sources(
        [
            StoredPage(page_key=build_anime_page_key("mal", "21")),
            StoredPage(page_key=build_anime_page_key("anilist", "21")),
            StoredPage(page_key=build_anime_page_key("unknown_site", "1")),
        ],
        [build_anime_page_key("kitsu", "one-piece-film")],
        records,
    )

    assert source_page_sources == [
        PageSource(
            page_key=build_anime_page_key("mal", "21"),
            source="https://myanimelist.net/anime/21/One_Piece",
        ),
        PageSource(
            page_key=build_anime_page_key("anilist", "21"),
            source="https://anilist.co/anime/21",
        ),
    ]
    assert partial_page_sources == [
        PageSource(
            page_key=build_anime_page_key("kitsu", "one-piece-film"),
            source="https://kitsu.app/anime/one-piece-film/extra",
        )
    ]


def test_extract_relation_claims_anime_planet_target_uses_stored_provider_name() -> (
    None
):
    from enrichment.pipeline.anime_merge import _extract_relation_claims

    claims = _extract_relation_claims(
        {
            "anime_planet": {
                "related_anime": {
                    "SEQUEL": [
                        {
                            "sources": [
                                "https://www.anime-planet.com/anime/one-piece-film-z"
                            ]
                        }
                    ]
                }
            }
        }
    )

    assert claims == [
        RelationClaim(
            provider="anime_planet",
            relation_type=anime_pb2.ANIME_RELATION_TYPE_SEQUEL,
            target_page=PageKey(
                provider="anime_planet",
                entity_type=anime_pb2.ENTITY_TYPE_ANIME,
                address="one-piece-film-z",
            ),
        )
    ]


def test_list_provider_episodes_mal_episode_list_gives_mal_episode_pages() -> None:
    saved_list_page = (
        Path(__file__).resolve().parents[1]
        / "sources"
        / "mal"
        / "test_mal_episode_crawler"
        / "one_piece_episodes.html"
    )
    mal_page = StoredPage(
        page_key=PageKey(
            provider="mal", entity_type=anime_pb2.ENTITY_TYPE_ANIME, address="21"
        ),
        responses=[
            PageResponse(
                request_name="episodes_page_1",
                content_type="text/html",
                body=saved_list_page.read_bytes(),
            )
        ],
    )

    listed_episodes = _list_provider_episodes(mal_page, "")

    assert len(listed_episodes) == 100
    assert listed_episodes[0].page_sources == [
        PageSource(
            page_key=PageKey(
                provider="mal",
                entity_type=anime_pb2.ENTITY_TYPE_EPISODE,
                address="21:1",
            ),
            source="https://myanimelist.net/anime/21/One_Piece/episode/1",
            title="I'm Luffy! The Man Who's Gonna Be King of the Pirates!",
        )
    ]
    assert listed_episodes[49].episode["is_filler"] is True
