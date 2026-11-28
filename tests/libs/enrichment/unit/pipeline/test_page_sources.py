import pytest
from enrichment.pipeline.page_sources import (
    build_page_sources,
    build_stored_page_key,
    collect_live_page_keys,
    is_mappable_stored_page,
    rank_provider_priority,
    read_page_urls,
    validate_one_page_per_provider,
)
from shared_proto.v1 import anime_pb2
from shared_proto.v1.merge_pb2 import PageKey, PageResponse, PageSource, StoredPage


def _build_mal_character_key(address: str) -> PageKey:
    return PageKey(
        provider="mal",
        entity_type=anime_pb2.ENTITY_TYPE_CHARACTER,
        address=address,
    )


def _build_template_url(page_key: PageKey) -> str:
    if page_key.provider != "mal":
        return ""
    return f"https://myanimelist.net/character/{page_key.address}"


@pytest.mark.parametrize(
    ("page_key", "stored_page_key"),
    [
        (
            "anime_planet:anime:one-piece",
            PageKey(
                provider="anime_planet",
                entity_type=anime_pb2.ENTITY_TYPE_ANIME,
                address="one-piece",
            ),
        ),
        (
            "kitsu:character:monkey-d-luffy",
            PageKey(
                provider="kitsu",
                entity_type=anime_pb2.ENTITY_TYPE_CHARACTER,
                address="monkey-d-luffy",
            ),
        ),
        ("mal:character:40", _build_mal_character_key("40")),
        (
            "mal:episode:21:1",
            PageKey(
                provider="mal",
                entity_type=anime_pb2.ENTITY_TYPE_EPISODE,
                address="21:1",
            ),
        ),
        ("anilist:manga:30013", None),
        ("https://example.com/show", None),
    ],
)
def test_build_stored_page_key_converts_provider_and_kind(
    page_key: str, stored_page_key: PageKey | None
) -> None:
    assert build_stored_page_key(page_key) == stored_page_key


def test_build_page_sources_reported_slug_url_beats_template() -> None:
    page_sources = build_page_sources(
        [_build_mal_character_key("40")],
        {
            "mal": {
                "sources": ["https://myanimelist.net/character/40"],
                "title": "Page of mal",
            },
            "kitsu": {
                "sources": ["https://myanimelist.net/character/40/Luffy_Monkey_D"],
                "title": "Page of kitsu",
            },
        },
        _build_template_url,
        "title",
    )

    assert page_sources == [
        PageSource(
            page_key=_build_mal_character_key("40"),
            source="https://myanimelist.net/character/40/Luffy_Monkey_D",
            title="Page of mal",
        )
    ]


def test_build_page_sources_other_provider_url_of_other_page_leaves_template() -> None:
    page_sources = build_page_sources(
        [_build_mal_character_key("40")],
        {
            "mal": {"sources": ["", "not a url"], "title": "Page of mal"},
            "kitsu": {
                "sources": ["https://myanimelist.net/character/400/Other"],
                "title": "Page of kitsu",
            },
        },
        _build_template_url,
        "title",
    )

    assert page_sources == [
        PageSource(
            page_key=_build_mal_character_key("40"),
            source="https://myanimelist.net/character/40",
            title="Page of mal",
        )
    ]


def test_build_page_sources_page_without_url_or_template_left_out() -> None:
    kitsu_page_key = PageKey(
        provider="kitsu",
        entity_type=anime_pb2.ENTITY_TYPE_CHARACTER,
        address="2",
    )

    page_sources = build_page_sources(
        [kitsu_page_key, _build_mal_character_key("40")],
        {
            "mal": {
                "sources": ["https://kitsu.app/characters/monkey-d-luffy"],
                "title": "Page of mal",
            }
        },
        _build_template_url,
        "title",
    )

    assert page_sources == [
        PageSource(
            page_key=_build_mal_character_key("40"),
            source="https://myanimelist.net/character/40",
            title="Page of mal",
        )
    ]


def test_build_page_sources_page_with_title_and_no_url_keeps_empty_url() -> None:
    kitsu_page_key = PageKey(
        provider="kitsu",
        entity_type=anime_pb2.ENTITY_TYPE_CHARACTER,
        address="2",
    )

    page_sources = build_page_sources(
        [kitsu_page_key],
        {"kitsu": {"sources": [], "title": "Page of kitsu"}},
        lambda page_key: "",
        "title",
    )

    assert page_sources == [
        PageSource(page_key=kitsu_page_key, source="", title="Page of kitsu")
    ]


def test_read_page_urls_skips_empty_and_repeated_urls() -> None:
    page_sources = [
        PageSource(
            page_key=_build_mal_character_key("40"),
            source="https://myanimelist.net/character/40/Luffy",
        ),
        PageSource(page_key=_build_mal_character_key("41"), source=""),
        PageSource(
            page_key=_build_mal_character_key("39"),
            source="https://myanimelist.net/character/39",
        ),
        PageSource(
            page_key=_build_mal_character_key("40"),
            source="https://myanimelist.net/character/40/Luffy",
        ),
    ]

    assert read_page_urls(page_sources) == [
        "https://myanimelist.net/character/39",
        "https://myanimelist.net/character/40/Luffy",
    ]


def test_build_page_sources_anime_planet_slug_url_names_stored_page() -> None:
    anime_planet_page_key = PageKey(
        provider="anime_planet",
        entity_type=anime_pb2.ENTITY_TYPE_ANIME,
        address="one-piece",
    )

    page_sources = build_page_sources(
        [anime_planet_page_key],
        {
            "anime_planet": {
                "sources": ["https://www.anime-planet.com/anime/one-piece"],
                "title": "Page of anime_planet",
            }
        },
        lambda page_key: "",
        "title",
    )

    assert page_sources == [
        PageSource(
            page_key=anime_planet_page_key,
            source="https://www.anime-planet.com/anime/one-piece",
            title="Page of anime_planet",
        )
    ]


def test_build_page_sources_page_key_url_lacks_takes_own_record_url() -> None:
    kitsu_episode_key = PageKey(
        provider="kitsu",
        entity_type=anime_pb2.ENTITY_TYPE_EPISODE,
        address="103482",
    )
    kitsu_character_key = PageKey(
        provider="kitsu",
        entity_type=anime_pb2.ENTITY_TYPE_CHARACTER,
        address="101",
    )

    episode_sources = build_page_sources(
        [kitsu_episode_key],
        {
            "kitsu": {
                "sources": [
                    "https://kitsu.app/anime/one-piece/episodes/1",
                    "https://kitsu.app/anime/one-piece",
                ],
                "title": "Page of kitsu",
            }
        },
        lambda page_key: "",
        "title",
    )
    character_sources = build_page_sources(
        [kitsu_character_key],
        {
            "kitsu": {
                "sources": [
                    "https://kitsu.app/characters/monkey-d-luffy",
                    "https://myanimelist.net/character/40",
                ],
                "title": "Page of kitsu",
            }
        },
        lambda page_key: "",
        "title",
    )

    assert episode_sources == [
        PageSource(
            page_key=kitsu_episode_key,
            source="https://kitsu.app/anime/one-piece/episodes/1",
            title="Page of kitsu",
        )
    ]
    assert character_sources == [
        PageSource(
            page_key=kitsu_character_key,
            source="https://kitsu.app/characters/monkey-d-luffy",
            title="Page of kitsu",
        )
    ]


def test_is_mappable_stored_page_accepts_not_found_with_responses() -> None:
    response = PageResponse(
        request_name="main", content_type="text/html", body=b"<html/>"
    )
    not_found_with_body = StoredPage(
        page_key=_build_mal_character_key("40"),
        is_not_found=True,
        responses=[response],
    )
    not_found_empty = StoredPage(
        page_key=_build_mal_character_key("40"),
        is_not_found=True,
    )
    dead_with_body = StoredPage(
        page_key=_build_mal_character_key("40"),
        is_dead=True,
        responses=[response],
    )

    assert is_mappable_stored_page(not_found_with_body) is True
    assert is_mappable_stored_page(not_found_empty) is False
    assert is_mappable_stored_page(dead_with_body) is False


def test_collect_live_page_keys_omits_dead_pages() -> None:
    live_key = _build_mal_character_key("40")
    dead_key = _build_mal_character_key("41")

    live_keys = collect_live_page_keys(
        [
            StoredPage(page_key=live_key, is_not_found=True),
            StoredPage(page_key=dead_key, is_dead=True),
        ]
    )

    assert live_keys == [live_key]


def test_rank_provider_priority_places_known_before_unknown() -> None:
    assert rank_provider_priority("mal") < rank_provider_priority("kitsu")
    assert rank_provider_priority("kitsu") < rank_provider_priority("unknown_source")


def test_validate_one_page_per_provider_distinct_providers_passes() -> None:
    validate_one_page_per_provider(["mal", "anilist", "kitsu"], "pages")


def test_validate_one_page_per_provider_repeated_providers_raises_value_error() -> None:
    with pytest.raises(
        ValueError,
        match=r"^pages holds more than one entry for provider\(s\) kitsu, mal$",
    ):
        validate_one_page_per_provider(
            ["mal", "kitsu", "mal", "anilist", "kitsu"], "pages"
        )
