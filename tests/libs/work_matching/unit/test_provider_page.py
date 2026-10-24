from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from work_matching.provider_page import ProviderPage, index_pages_by_current_key

KITSU = "https://kitsu.app/anime/12"
MAL = "https://myanimelist.net/anime/21"
ANIME_PLANET_OLD = (
    "https://anime-planet.com/anime/cocoon-aru-natsu-no-shoujo-tachi-yori"
)
ANISEARCH = "https://anisearch.com/anime/13751"


def test_provider_page_from_canonical_record_reads_matching_fields(
    canonical_records,
) -> None:
    page = ProviderPage.from_canonical_record(KITSU, canonical_records[KITSU])
    assert (page.key, page.address, page.type, page.year, page.season) == (
        "kitsu:anime:12",
        KITSU,
        "TV",
        1999,
        "FALL",
    )
    assert page.titles[0] == "One Piece"


def test_provider_page_from_canonical_record_keeps_only_other_provider_pages_as_links(
    canonical_records,
) -> None:
    page = ProviderPage.from_canonical_record(KITSU, canonical_records[KITSU])
    assert page.linked_page_keys == (
        "anidb:anime:69",
        "anilist:anime:21",
        "mal:anime:21",
    )


def test_provider_page_from_canonical_record_reads_old_anidb_and_ann_addresses(
    canonical_records,
) -> None:
    page = ProviderPage.from_canonical_record(MAL, canonical_records[MAL])
    assert {"anidb:anime:69", "ann:anime:836"} <= set(page.linked_page_keys)
    assert page.renamed_to is None


def test_provider_page_from_canonical_record_renamed_page_names_new_key(
    canonical_records,
) -> None:
    page = ProviderPage.from_canonical_record(
        ANIME_PLANET_OLD, canonical_records[ANIME_PLANET_OLD]
    )
    assert page.renamed_to == "animeplanet:anime:cocoon-one-summer-of-girlhood"
    assert page.linked_page_keys == ()


def test_provider_page_from_canonical_record_collects_titles_of_every_language(
    canonical_records,
) -> None:
    page = ProviderPage.from_canonical_record(ANISEARCH, canonical_records[ANISEARCH])
    assert page.titles[0] == "Mo Dao Zu Shi: Qianchen Pian"
    assert {"魔道祖师 前尘篇", "Madou Soshi: Zenjin Hen"} <= set(page.titles)


def test_provider_page_from_canonical_record_reads_episode_count_and_air_dates() -> (
    None
):
    page = ProviderPage.from_canonical_record(
        "https://anilist.co/anime/163794",
        {
            "title": "The Boxer",
            "episode_count": 12,
            "aired_dates": {
                "aired_from": "2026-04-05T15:00:00Z",
                "aired_to": "2026-06-21T15:00:00Z",
            },
        },
    )
    assert (page.episode_count, page.aired_from, page.aired_to) == (
        12,
        datetime(2026, 4, 5, 15, tzinfo=UTC),
        datetime(2026, 6, 21, 15, tzinfo=UTC),
    )


def test_provider_page_from_canonical_record_zero_episode_count_read_as_unknown() -> (
    None
):
    page = ProviderPage.from_canonical_record(
        "https://anilist.co/anime/21",
        {
            "title": "ONE PIECE",
            "episode_count": 0,
            "aired_dates": {"aired_from": "1999-10-19T15:00:00Z"},
        },
    )
    assert (page.episode_count, page.aired_to) == (None, None)


def test_provider_page_stored_line_without_episode_fields_loads() -> None:
    page = ProviderPage.model_validate_json(
        '{"key":"mal:anime:21","address":"https://myanimelist.net/anime/21",'
        '"titles":["One Piece"],"type":"TV","year":1999,"season":"FALL","linked_page_keys":[],'
        '"searched_by":[],"is_dead":false,"renamed_to":null}'
    )
    assert (page.episode_count, page.aired_from, page.aired_to) == (None, None, None)


def test_provider_page_current_key_renamed_page_gives_new_key(
    canonical_records,
) -> None:
    renamed = ProviderPage.from_canonical_record(
        ANIME_PLANET_OLD, canonical_records[ANIME_PLANET_OLD]
    )
    plain = ProviderPage.from_canonical_record(MAL, canonical_records[MAL])
    assert (renamed.current_key, plain.current_key) == (
        "animeplanet:anime:cocoon-one-summer-of-girlhood",
        "mal:anime:21",
    )


def test_provider_page_from_canonical_record_kitsu_slug_page_renamed_to_number() -> (
    None
):
    page = ProviderPage.from_canonical_record(
        "https://kitsu.app/anime/cue",
        {"title": "CUE!", "sources": ["https://kitsu.app/anime/44979"]},
    )
    assert (page.key, page.current_key) == ("kitsu:anime:cue", "kitsu:anime:44979")


def test_provider_page_from_canonical_record_repeated_title_kept_once() -> None:
    page = ProviderPage.from_canonical_record(
        MAL, {"title": "One Piece", "title_english": "One Piece", "titles": {}}
    )
    assert page.titles == ("One Piece",)


def test_provider_page_from_dead_address_has_no_details() -> None:
    page = ProviderPage.from_dead_address(MAL)
    assert (page.key, page.is_dead, page.titles, page.linked_page_keys) == (
        "mal:anime:21",
        True,
        (),
        (),
    )


def test_provider_page_from_canonical_record_searched_page_keeps_link_to_searched_by_page() -> (
    None
):
    page = ProviderPage.from_canonical_record(
        "https://animeschedule.net/anime/one-piece",
        {
            "title": "One Piece",
            "sources": [
                "https://animeschedule.net/anime/one-piece",
                "https://myanimelist.net/anime/21/One_Piece",
                "https://anilist.co/anime/21",
            ],
        },
        searched_by_address=MAL,
    )
    assert (page.searched_by, page.linked_page_keys) == (
        ("mal:anime:21",),
        ("anilist:anime:21", "mal:anime:21"),
    )


def test_provider_page_from_dead_address_searched_by_page_records_key() -> None:
    assert ProviderPage.from_dead_address(
        MAL, searched_by_address=KITSU
    ).searched_by == ("kitsu:anime:12",)


def test_provider_page_json_round_trip_gives_equal_page(canonical_records) -> None:
    page = ProviderPage.from_canonical_record(MAL, canonical_records[MAL])
    assert ProviderPage.model_validate_json(page.model_dump_json()) == page


def test_index_pages_by_current_key_renamed_page_found_under_new_key(
    canonical_records,
) -> None:
    renamed = ProviderPage.from_canonical_record(
        ANIME_PLANET_OLD, canonical_records[ANIME_PLANET_OLD]
    )
    plain = ProviderPage.from_canonical_record(MAL, canonical_records[MAL])
    assert index_pages_by_current_key([renamed, plain]) == {
        "animeplanet:anime:cocoon-one-summer-of-girlhood": renamed,
        "mal:anime:21": plain,
    }


def test_index_pages_by_current_key_page_fetched_at_new_key_wins(
    canonical_records,
) -> None:
    renamed = ProviderPage.from_canonical_record(
        ANIME_PLANET_OLD, canonical_records[ANIME_PLANET_OLD]
    )
    current = renamed.model_copy(update={"key": renamed.renamed_to, "renamed_to": None})
    assert index_pages_by_current_key([current, renamed])[current.key] is current


def test_provider_page_unknown_field_rejected() -> None:
    with pytest.raises(ValidationError, match="lands_on"):
        ProviderPage.model_validate(
            {
                "key": "animeplanet:anime:old",
                "address": "https://www.anime-planet.com/anime/old",
                "lands_on": "animeplanet:anime:new",
            }
        )
