import pytest
from pydantic import ValidationError
from work_matching.provider_page import ProviderPage, pages_by_current_key

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
    assert page.links == ("anidb:anime:69", "anilist:anime:21", "mal:anime:21")


def test_provider_page_from_canonical_record_reads_old_anidb_and_ann_addresses(
    canonical_records,
) -> None:
    page = ProviderPage.from_canonical_record(MAL, canonical_records[MAL])
    assert {"anidb:anime:69", "ann:anime:836"} <= set(page.links)
    assert page.renamed_to is None


def test_provider_page_from_canonical_record_renamed_page_names_new_key(
    canonical_records,
) -> None:
    page = ProviderPage.from_canonical_record(
        ANIME_PLANET_OLD, canonical_records[ANIME_PLANET_OLD]
    )
    assert page.renamed_to == "animeplanet:anime:cocoon-one-summer-of-girlhood"
    assert page.links == ()


def test_provider_page_from_canonical_record_collects_titles_of_every_language(
    canonical_records,
) -> None:
    page = ProviderPage.from_canonical_record(ANISEARCH, canonical_records[ANISEARCH])
    assert page.titles[0] == "Mo Dao Zu Shi: Qianchen Pian"
    assert {"魔道祖师 前尘篇", "Madou Soshi: Zenjin Hen"} <= set(page.titles)


def test_provider_page_from_canonical_record_repeated_title_kept_once() -> None:
    page = ProviderPage.from_canonical_record(
        MAL, {"title": "One Piece", "title_english": "One Piece", "titles": {}}
    )
    assert page.titles == ("One Piece",)


def test_provider_page_dead_page_has_no_details() -> None:
    page = ProviderPage.dead_page(MAL)
    assert (page.key, page.dead, page.titles, page.links) == (
        "mal:anime:21",
        True,
        (),
        (),
    )


def test_provider_page_from_canonical_record_found_page_keeps_link_to_lookup_page() -> (
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
        found_by=MAL,
    )
    assert (page.found_by, page.links) == (
        ("mal:anime:21",),
        ("anilist:anime:21", "mal:anime:21"),
    )


def test_provider_page_dead_page_found_by_lookup_records_it() -> None:
    assert ProviderPage.dead_page(MAL, found_by=KITSU).found_by == ("kitsu:anime:12",)


def test_provider_page_json_round_trip_gives_equal_page(canonical_records) -> None:
    page = ProviderPage.from_canonical_record(MAL, canonical_records[MAL])
    assert ProviderPage.model_validate_json(page.model_dump_json()) == page


def test_pages_by_current_key_renamed_page_found_under_new_key(
    canonical_records,
) -> None:
    renamed = ProviderPage.from_canonical_record(
        ANIME_PLANET_OLD, canonical_records[ANIME_PLANET_OLD]
    )
    plain = ProviderPage.from_canonical_record(MAL, canonical_records[MAL])
    assert pages_by_current_key([renamed, plain]) == {
        "animeplanet:anime:cocoon-one-summer-of-girlhood": renamed,
        "mal:anime:21": plain,
    }


def test_pages_by_current_key_page_fetched_at_new_key_wins(canonical_records) -> None:
    renamed = ProviderPage.from_canonical_record(
        ANIME_PLANET_OLD, canonical_records[ANIME_PLANET_OLD]
    )
    current = renamed.model_copy(update={"key": renamed.renamed_to, "renamed_to": None})
    assert pages_by_current_key([current, renamed])[current.key] is current


def test_provider_page_unknown_field_rejected() -> None:
    with pytest.raises(ValidationError, match="lands_on"):
        ProviderPage.model_validate(
            {
                "key": "animeplanet:anime:old",
                "address": "https://www.anime-planet.com/anime/old",
                "lands_on": "animeplanet:anime:new",
            }
        )
