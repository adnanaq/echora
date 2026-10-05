import pytest
from enrichment.sources.anime_planet.anime_planet_anime_crawler import (
    _build_anime_from_raw,
)
from enrichment.sources.anime_planet.anime_planet_character_models import (
    AnimePlanetCharacter,
    AnimePlanetCharacterAnimeRole,
    AnimePlanetCharacterMangaRole,
    AnimePlanetVoiceActor,
)
from enrichment.sources.anime_planet.anime_planet_models import (
    AnimePlanetAggregateRating,
    AnimePlanetAnime,
    AnimePlanetMangaEntry,
    AnimePlanetRelatedEntry,
    AnimePlanetStudio,
)
from enrichment.sources.anime_planet.animeplanet_mapper import (
    anime_from_animeplanet,
    character_from_animeplanet,
)

LUFFY_URL = "https://www.anime-planet.com/characters/monkey-d-luffy"


def _anime(**fields) -> AnimePlanetAnime:
    return AnimePlanetAnime(**{"name": "Test Anime", "slug": "test-anime", **fields})


def _character(**fields) -> AnimePlanetCharacter:
    return AnimePlanetCharacter(
        **{
            "name": "Monkey D. Luffy",
            "slug": "monkey-d-luffy",
            "url": LUFFY_URL,
            **fields,
        }
    )


def _related(title: str, **fields) -> AnimePlanetRelatedEntry:
    return AnimePlanetRelatedEntry(
        **{
            "title": title,
            "url": f"/anime/{title.lower()}",
            "slug": title.lower(),
            **fields,
        }
    )


def _manga(title: str, **fields) -> AnimePlanetMangaEntry:
    return AnimePlanetMangaEntry(
        **{
            "title": title,
            "url": f"/manga/{title.lower()}",
            "slug": title.lower(),
            **fields,
        }
    )


def _all_source_material(mapped: dict) -> dict[str, dict]:
    return {
        entry["title"]: entry
        for entries in mapped["related_source_material"].values()
        for entry in entries
    }


def test_anime_from_animeplanet_one_piece_page_maps_main_fields(
    ap_anime_extracted: dict,
) -> None:
    mapped = anime_from_animeplanet(_build_anime_from_raw(ap_anime_extracted))
    assert mapped["title"] == "One Piece"
    assert (mapped["year"], mapped["season"], mapped["status"]) == (
        1999,
        "FALL",
        "ONGOING",
    )
    assert mapped["episode_count"] == 1165
    assert mapped["title_japanese"] == "ワンピース"


def test_anime_from_animeplanet_one_piece_page_maps_studio_and_statistics(
    ap_anime_extracted: dict,
) -> None:
    mapped = anime_from_animeplanet(_build_anime_from_raw(ap_anime_extracted))
    toei = next(c for c in mapped["companies"] if c["name"] == "Toei Animation")
    assert toei["roles"] == ["STUDIO"]
    assert "studios" not in mapped
    statistics = mapped["statistics"]["anime_planet"]
    assert statistics["score"] == pytest.approx(8.63)
    assert (statistics["scored_by"], statistics["rank"]) == (64986, 161)


def test_anime_from_animeplanet_one_piece_page_maps_every_related_manga(
    ap_anime_extracted: dict,
) -> None:
    mapped = anime_from_animeplanet(_build_anime_from_raw(ap_anime_extracted))
    manga = _all_source_material(mapped)
    assert len(manga) == 24
    romance_dawn = next(
        entry for title, entry in manga.items() if "Romance Dawn" in title
    )
    assert romance_dawn["type"] == "ONE SHOT"


def test_anime_from_animeplanet_description_and_alt_title_become_synopsis_and_japanese_title() -> (
    None
):
    mapped = anime_from_animeplanet(
        _anime(description="Pirates.", alt_title="ワンピース")
    )
    assert (mapped["synopsis"], mapped["title_japanese"]) == ("Pirates.", "ワンピース")


@pytest.mark.parametrize(
    ("schema_type", "expected"),
    [("TVSeries", "TV"), ("Movie", "MOVIE"), (None, "UNKNOWN")],
)
def test_anime_from_animeplanet_schema_type_gives_anime_type(
    schema_type: str | None, expected: str
) -> None:
    assert anime_from_animeplanet(_anime(schema_type=schema_type))["type"] == expected


def test_anime_from_animeplanet_stated_season_wins_over_start_date() -> None:
    mapped = anime_from_animeplanet(_anime(season="fall", start_date="2024-07-10"))
    assert mapped["season"] == "FALL"


def test_anime_from_animeplanet_without_stated_season_derives_season_from_start_date() -> (
    None
):
    assert anime_from_animeplanet(_anime(start_date="2024-04-05"))["season"] == "SPRING"


def test_anime_from_animeplanet_without_season_or_start_date_omits_season() -> None:
    assert "season" not in anime_from_animeplanet(_anime())


def test_anime_from_animeplanet_full_start_date_takes_year_from_date() -> None:
    mapped = anime_from_animeplanet(_anime(start_date="1999-10-20", start_year=1999))
    assert mapped["year"] == 1999


def test_anime_from_animeplanet_without_start_date_takes_year_from_entry_bar() -> None:
    assert anime_from_animeplanet(_anime(start_year=2002))["year"] == 2002


def test_anime_from_animeplanet_without_any_year_omits_year() -> None:
    assert "year" not in anime_from_animeplanet(_anime())


@pytest.mark.parametrize(
    ("start_date", "end_date", "expected"),
    [
        ("2024-01-01", "2024-03-31", "FINISHED"),
        ("1999-10-20", None, "ONGOING"),
        ("2099-01-01", None, "UPCOMING"),
        (None, None, "UNKNOWN"),
    ],
)
def test_anime_from_animeplanet_dates_give_status(
    start_date: str | None, end_date: str | None, expected: str
) -> None:
    mapped = anime_from_animeplanet(_anime(start_date=start_date, end_date=end_date))
    assert mapped["status"] == expected


def test_anime_from_animeplanet_start_and_end_dates_give_aired_dates_in_utc() -> None:
    mapped = anime_from_animeplanet(
        _anime(start_date="2024-01-01", end_date="2024-03-31")
    )
    assert mapped["aired_dates"] == {
        "aired_from": "2023-12-31T15:00:00Z",
        "aired_to": "2024-03-30T15:00:00Z",
    }


def test_anime_from_animeplanet_end_date_alone_gives_aired_to_only() -> None:
    mapped = anime_from_animeplanet(_anime(end_date="2024-03-31"))
    assert mapped["aired_dates"] == {"aired_to": "2024-03-30T15:00:00Z"}


def test_anime_from_animeplanet_without_dates_omits_aired_dates() -> None:
    assert "aired_dates" not in anime_from_animeplanet(_anime())


def test_anime_from_animeplanet_rating_out_of_five_gives_score_out_of_ten() -> None:
    mapped = anime_from_animeplanet(
        _anime(
            aggregate_rating=AnimePlanetAggregateRating(
                rating_value=4.3, rating_count=500
            ),
            rank=12,
        )
    )
    statistics = mapped["statistics"]["anime_planet"]
    assert statistics["score"] == pytest.approx(8.6)
    assert (statistics["scored_by"], statistics["rank"]) == (500, 12)


def test_anime_from_animeplanet_rank_alone_gives_rank_statistics() -> None:
    statistics = anime_from_animeplanet(_anime(rank=12))["statistics"]["anime_planet"]
    assert statistics["rank"] == 12
    assert "score" not in statistics


def test_anime_from_animeplanet_empty_rating_without_rank_gives_no_statistics() -> None:
    mapped = anime_from_animeplanet(
        _anime(aggregate_rating=AnimePlanetAggregateRating())
    )
    assert not mapped.get("statistics")


def test_anime_from_animeplanet_genres_and_tags_merged_without_repeats_or_blanks() -> (
    None
):
    mapped = anime_from_animeplanet(
        _anime(genres=["Action", "Comedy"], tags=["Comedy", "", "Pirates"])
    )
    assert mapped["tags"] == ["Action", "Comedy", "Pirates"]


def test_anime_from_animeplanet_cover_becomes_cover_image() -> None:
    mapped = anime_from_animeplanet(_anime(cover="https://cdn/cover.jpg"))
    assert mapped["images"]["covers"] == ["https://cdn/cover.jpg"]


def test_anime_from_animeplanet_without_cover_gives_no_cover_images() -> None:
    assert not anime_from_animeplanet(_anime())["images"].get("covers")


def test_anime_from_animeplanet_page_url_becomes_only_source() -> None:
    url = "https://www.anime-planet.com/anime/test-anime"
    assert anime_from_animeplanet(_anime(url=url))["sources"] == [url]


def test_anime_from_animeplanet_without_page_url_gives_no_sources() -> None:
    assert not anime_from_animeplanet(_anime()).get("sources")


def test_anime_from_animeplanet_studios_become_studio_companies() -> None:
    studio_url = "https://www.anime-planet.com/anime/studios/toei-animation"
    mapped = anime_from_animeplanet(
        _anime(
            studios=[
                AnimePlanetStudio(name="Toei Animation", url=studio_url),
                AnimePlanetStudio(name="Unlinked Studio"),
            ]
        )
    )
    by_name = {company["name"]: company for company in mapped["companies"]}
    assert by_name["Toei Animation"]["sources"] == [studio_url]
    assert by_name["Unlinked Studio"]["roles"] == ["STUDIO"]
    assert not by_name["Unlinked Studio"].get("sources")


def test_anime_from_animeplanet_without_episode_count_gives_zero_episodes() -> None:
    assert anime_from_animeplanet(_anime())["episode_count"] == 0


def test_anime_from_animeplanet_related_anime_grouped_by_relation_with_full_urls() -> (
    None
):
    mapped = anime_from_animeplanet(
        _anime(
            related_anime=[
                _related("Sequel", relation_subtype="Sequel", type="TV"),
                _related(
                    "Special", relation_subtype="Same Franchise", type="TV Special"
                ),
            ],
            related_anime_other=[
                _related(
                    "Crossover",
                    url="https://www.anime-planet.com/anime/crossover",
                    relation_subtype="Other Franchise",
                )
            ],
        )
    )
    related = mapped["related_anime"]
    assert related["SEQUEL"][0]["sources"] == [
        "https://www.anime-planet.com/anime/sequel"
    ]
    assert related["SIDE_STORY"][0]["type"] == "TV SPECIAL"
    assert related["OTHER"][0]["sources"] == [
        "https://www.anime-planet.com/anime/crossover"
    ]


def test_anime_from_animeplanet_related_anime_without_relation_counts_as_side_story() -> (
    None
):
    mapped = anime_from_animeplanet(_anime(related_anime=[_related("Film")]))
    assert mapped["related_anime"]["SIDE_STORY"][0]["title"] == "Film"


def test_anime_from_animeplanet_related_anime_episode_count_kept_when_stated() -> None:
    mapped = anime_from_animeplanet(
        _anime(
            related_anime=[
                _related("Special", relation_subtype="Same Franchise", episode_count=3),
                _related("Film", relation_subtype="Same Franchise"),
            ]
        )
    )
    by_title = {
        entry["title"]: entry for entry in mapped["related_anime"]["SIDE_STORY"]
    }
    assert by_title["Special"]["episode_count"] == 3
    assert "episode_count" not in by_title["Film"]


def test_anime_from_animeplanet_original_manga_becomes_adaptation_source() -> None:
    mapped = anime_from_animeplanet(
        _anime(related_manga=[_manga("Origin", relation_subtype="Original Manga")])
    )
    assert mapped["related_source_material"]["ADAPTATION"][0]["title"] == "Origin"


def test_anime_from_animeplanet_other_manga_becomes_other_source() -> None:
    mapped = anime_from_animeplanet(
        _anime(
            related_manga=[
                _manga("Spinoff", relation_subtype="Spin-off"),
                _manga("Unlabelled"),
            ]
        )
    )
    titles = [entry["title"] for entry in mapped["related_source_material"]["OTHER"]]
    assert titles == ["Spinoff", "Unlabelled"]


def test_anime_from_animeplanet_related_manga_keeps_type_volumes_chapters_and_full_url() -> (
    None
):
    mapped = anime_from_animeplanet(
        _anime(
            related_manga=[
                _manga("Plain", volumes=7, chapters=62),
                _manga("Romance Dawn", type="One Shot", chapters=1),
                _manga(
                    "Linked",
                    url="https://www.anime-planet.com/manga/linked",
                ),
            ]
        )
    )
    manga = _all_source_material(mapped)
    assert (
        manga["Plain"]["type"],
        manga["Plain"]["volumes"],
        manga["Plain"]["chapters"],
    ) == (
        "UNKNOWN",
        7,
        62,
    )
    assert manga["Plain"]["sources"] == ["https://www.anime-planet.com/manga/plain"]
    assert manga["Romance Dawn"]["type"] == "ONE SHOT"
    assert manga["Linked"]["sources"] == ["https://www.anime-planet.com/manga/linked"]
    assert not {"volumes", "chapters"} & manga["Linked"].keys()


def test_character_from_animeplanet_name_and_page_only_gives_name_and_source() -> None:
    mapped = character_from_animeplanet(_character())
    assert mapped["name"] == "Monkey D. Luffy"
    assert mapped["sources"] == [LUFFY_URL]
    assert not {
        "favorites",
        "description",
        "traits",
        "nicknames",
        "images",
        "attributes",
        "roles",
        "animeography",
        "mangaography",
        "voice_actors",
    } & {key for key, value in mapped.items() if value}


def test_character_from_animeplanet_profile_fields_map_to_character_fields() -> None:
    mapped = character_from_animeplanet(
        _character(
            loved_count=36485,
            description="Captain of the Straw Hats.",
            tags=["Pirate"],
            alt_names=["Straw Hat"],
            image="https://cdn/luffy.jpg",
        )
    )
    assert mapped["favorites"] == 36485
    assert mapped["description"] == "Captain of the Straw Hats."
    assert (mapped["traits"], mapped["nicknames"], mapped["images"]) == (
        ["Pirate"],
        ["Straw Hat"],
        ["https://cdn/luffy.jpg"],
    )


def test_character_from_animeplanet_attributes_merge_page_values_and_ranks() -> None:
    mapped = character_from_animeplanet(
        _character(
            attributes={"Eye Color": "Black"},
            gender="Male",
            hair_color="Black",
            loved_rank=3,
            hated_rank=900,
        )
    )
    assert mapped["attributes"] == {
        "eye_color": "Black",
        "gender": "Male",
        "hair_color": "Black",
        "loved_rank": "3",
        "hated_rank": "900",
    }


def test_character_from_animeplanet_roles_keep_page_order_without_repeats() -> None:
    mapped = character_from_animeplanet(
        _character(
            anime_roles=[
                AnimePlanetCharacterAnimeRole(title="A", url="/anime/a", role="Minor"),
                AnimePlanetCharacterAnimeRole(title="B", url="/anime/b", role="Main"),
                AnimePlanetCharacterAnimeRole(title="C", url="/anime/c", role="Minor"),
            ],
            manga_roles=[
                AnimePlanetCharacterMangaRole(
                    title="D", url="/manga/d", role="Secondary"
                ),
                AnimePlanetCharacterMangaRole(title="E", url="/manga/e", role="Main"),
            ],
        )
    )
    assert mapped["roles"] == ["BACKGROUND", "MAIN", "SUPPORTING"]


def test_character_from_animeplanet_appearances_give_animeography_and_mangaography() -> (
    None
):
    mapped = character_from_animeplanet(
        _character(
            anime_roles=[
                AnimePlanetCharacterAnimeRole(
                    title="One Piece", url="/anime/one-piece", role="Main"
                ),
                AnimePlanetCharacterAnimeRole(title="Cameo", url="/anime/cameo"),
            ],
            manga_roles=[
                AnimePlanetCharacterMangaRole(
                    title="One Piece", url="/manga/one-piece", role="Main"
                )
            ],
        )
    )
    assert mapped["animeography"] == [
        {
            "title": "One Piece",
            "role": "MAIN",
            "sources": ["https://www.anime-planet.com/anime/one-piece"],
        },
        {
            "title": "Cameo",
            "role": "UNKNOWN",
            "sources": ["https://www.anime-planet.com/anime/cameo"],
        },
    ]
    assert mapped["mangaography"] == [
        {
            "title": "One Piece",
            "role": "MAIN",
            "sources": ["https://www.anime-planet.com/manga/one-piece"],
        }
    ]
    assert mapped["roles"] == ["MAIN"]


def test_character_from_animeplanet_voice_actors_named_by_language_without_repeats() -> (
    None
):
    tanaka = AnimePlanetVoiceActor(name="Mayumi Tanaka", url="/people/mayumi-tanaka")
    mapped = character_from_animeplanet(
        _character(
            anime_roles=[
                AnimePlanetCharacterAnimeRole(
                    title="One Piece",
                    url="/anime/one-piece",
                    voice_actors={
                        "jp": [tanaka],
                        "us": [
                            AnimePlanetVoiceActor(
                                name="Colleen Clinkenbeard",
                                url="/people/colleen-clinkenbeard",
                            )
                        ],
                        "pt": [
                            AnimePlanetVoiceActor(name="Some Actor", url="/people/a")
                        ],
                    },
                ),
                AnimePlanetCharacterAnimeRole(
                    title="One Piece Film",
                    url="/anime/one-piece-film",
                    voice_actors={"jp": [tanaka]},
                ),
            ]
        )
    )
    assert [(actor["name"], actor["language"]) for actor in mapped["voice_actors"]] == [
        ("Mayumi Tanaka", "Japanese"),
        ("Colleen Clinkenbeard", "English"),
        ("Some Actor", "pt"),
    ]
    assert mapped["voice_actors"][0]["sources"] == [
        "https://www.anime-planet.com/people/mayumi-tanaka"
    ]
