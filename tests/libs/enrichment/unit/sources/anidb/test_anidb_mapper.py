import pytest
from common.models.anime import Anime
from enrichment.sources.anidb.anidb_mapper import (
    anime_from_anidb,
    character_from_anidb,
    episode_from_anidb,
)
from enrichment.sources.anidb.anidb_models import (
    AniDBAnime,
    AniDBCategory,
    AniDBCharacter,
    AniDBCharacterPage,
    AniDBCreator,
    AniDBEpisode,
    AniDBExternalResource,
    AniDBRatings,
    AniDBRelatedAnime,
    AniDBSeiyuu,
)

CDN_BASE = "https://cdn-eu.anidb.net/images/main"
ANIDB_URL = "https://anidb.net/anime/69"


def _anime(**fields) -> AniDBAnime:
    return AniDBAnime(id=69, **fields)


def _map(**fields) -> dict:
    return anime_from_anidb(_anime(**fields), anidb_url=ANIDB_URL)


def _links(result: dict) -> dict[str, str]:
    return {link["platform"]: link["source"] for link in result["external_sources"]}


def _resources(*resources: AniDBExternalResource) -> dict:
    return _map(resources=list(resources))


def _company_roles(result: dict) -> dict[str, list[str]]:
    return {
        company["name"]: company["roles"] for company in result.get("companies") or []
    }


def test_anime_from_anidb_animation_work_and_work_creators_become_companies() -> None:
    result = _map(
        creators=[
            AniDBCreator(id=412, name="Toei Animation", role="Animation Work"),
            AniDBCreator(name="Fuji TV", role="Work"),
            AniDBCreator(name="Toei Animation", role="Work"),
            AniDBCreator(name="", role="Work"),
            AniDBCreator(name="Oda Eiichirou", role="Original Work"),
            AniDBCreator(name="Tanaka Kouhei", role="Music"),
            AniDBCreator(name="Shinkai Makoto", role="Animation Production"),
            AniDBCreator(name="Tezuka Osamu", role="Original Plan"),
        ]
    )
    assert _company_roles(result) == {
        "Toei Animation": ["STUDIO", "PRODUCER"],
        "Fuji TV": ["PRODUCER"],
    }
    toei = next(c for c in result["companies"] if c["name"] == "Toei Animation")
    assert "https://anidb.net/creator/412" in toei["sources"]
    assert not {"studios", "producers", "licensors"} & result.keys()


def test_anime_from_anidb_without_creators_gives_no_companies() -> None:
    assert _company_roles(_map()) == {}


def test_anime_from_anidb_main_title_wins_over_english_title() -> None:
    assert _map(title="One Piece", title_english="OP EN")["title"] == "One Piece"


def test_anime_from_anidb_without_main_title_takes_english_title() -> None:
    assert _map(title=None, title_english="One Piece EN")["title"] == "One Piece EN"


def test_anime_from_anidb_without_any_title_gives_empty_title() -> None:
    assert _map(title=None, title_english=None)["title"] == ""


def test_anime_from_anidb_scalar_fields_mapped() -> None:
    result = _map(
        title_english="One Piece EN",
        title_japanese="ワンピース",
        description="Pirates.",
        episode_count=1100,
        synonyms=["OP"],
    )
    assert (result["title_english"], result["title_japanese"]) == (
        "One Piece EN",
        "ワンピース",
    )
    assert (result["synopsis"], result["episode_count"], result["synonyms"]) == (
        "Pirates.",
        1100,
        ["OP"],
    )


@pytest.mark.parametrize(
    ("raw_type", "expected"),
    [("TV Series", "TV"), ("Movie", "MOVIE"), (None, "UNKNOWN")],
)
def test_anime_from_anidb_type_gives_anime_type(
    raw_type: str | None, expected: str
) -> None:
    assert _map(type=raw_type)["type"] == expected


def test_anime_from_anidb_seed_url_becomes_only_source() -> None:
    assert _map()["sources"] == [ANIDB_URL]


def test_anime_from_anidb_restricted_anime_marked_nsfw() -> None:
    assert _map(restricted=True)["nsfw"] is True


def test_anime_from_anidb_unrestricted_anime_omits_nsfw() -> None:
    assert "nsfw" not in _map(restricted=False)


def test_anime_from_anidb_picture_becomes_cdn_cover() -> None:
    assert _map(picture="anime.jpg")["images"]["covers"] == [f"{CDN_BASE}/anime.jpg"]


def test_anime_from_anidb_without_picture_gives_no_covers() -> None:
    assert _map(picture=None)["images"]["covers"] == []


def test_anime_from_anidb_non_hentai_categories_added_to_tags_once() -> None:
    result = _map(
        tags=["Action"],
        categories=[
            AniDBCategory(name="Action", hentai=False),
            AniDBCategory(name="Pirates", hentai=False),
            AniDBCategory(name="Ecchi", hentai=True),
        ],
    )
    assert result["tags"] == ["Action", "Pirates"]


def test_anime_from_anidb_other_language_titles_become_titles() -> None:
    result = _map(title_others={"de": "Wan Piisu", "ko": "원피스"})
    assert result["titles"] == {"de": "Wan Piisu", "ko": "원피스"}


def test_anime_from_anidb_permanent_rating_gives_score_and_votes() -> None:
    result = _map(ratings=AniDBRatings(permanent=8.33, permanent_count=9547))
    assert result["statistics"]["anidb"] == {"score": 8.33, "scored_by": 9547}


def test_anime_from_anidb_permanent_rating_without_votes_gives_score_only() -> None:
    result = _map(ratings=AniDBRatings(permanent=8.33, permanent_count=0))
    assert result["statistics"]["anidb"] == {"score": 8.33}


@pytest.mark.parametrize("ratings", [None, AniDBRatings(permanent=None)])
def test_anime_from_anidb_without_permanent_rating_gives_empty_statistics(
    ratings: AniDBRatings | None,
) -> None:
    assert _map(ratings=ratings)["statistics"] == {}


def test_anime_from_anidb_start_and_end_dates_give_utc_aired_dates_year_and_season() -> (
    None
):
    result = _map(start_date="1999-10-20", end_date="2030-06-01")
    assert result["aired_dates"] == {
        "aired_from": "1999-10-19T15:00:00Z",
        "aired_to": "2030-05-31T15:00:00Z",
    }
    assert (result["year"], result["season"], result["status"]) == (
        1999,
        "FALL",
        "ONGOING",
    )


def test_anime_from_anidb_month_only_start_gives_year_and_season_without_date() -> None:
    result = _map(start_date="1977-10")
    assert (result["year"], result["season"]) == (1977, "FALL")
    assert "aired_dates" not in result


def test_anime_from_anidb_without_dates_gives_no_aired_dates_and_unknown_status() -> (
    None
):
    result = _map(start_date=None, end_date=None)
    assert "aired_dates" not in result
    assert result["status"] == "UNKNOWN"


def test_anime_from_anidb_unknown_date_placeholder_gives_no_date_year_or_season() -> (
    None
):
    result = _map(type="Movie", start_date="1970-01-01")
    assert not {"aired_dates", "year", "season"} & result.keys()
    assert result["status"] == "UNKNOWN"


def test_anime_from_anidb_related_anime_grouped_by_relation_with_anidb_links() -> None:
    result = _map(
        related_anime=[
            AniDBRelatedAnime(
                id=522, title="One Piece Movie 1", relation_type="Side Story"
            ),
            AniDBRelatedAnime(
                id=523, title="One Piece Movie 2", relation_type="Side Story"
            ),
            AniDBRelatedAnime(id=9999, relation_type="Sequel"),
        ]
    )
    related = result["related_anime"]
    assert [entry["title"] for entry in related["SIDE_STORY"]] == [
        "One Piece Movie 1",
        "One Piece Movie 2",
    ]
    assert related["SIDE_STORY"][0]["sources"] == ["https://anidb.net/anime/522"]
    assert related["SEQUEL"][0]["title"] == ""
    assert related["SEQUEL"][0]["type"] == "UNKNOWN"


def test_anime_from_anidb_url_field_becomes_japanese_official_site() -> None:
    result = _map(url="http://onepiece.toei-anim.co.jp")
    assert result["external_sources"] == [
        {
            "platform": "official_site",
            "source": "http://onepiece.toei-anim.co.jp",
            "language": "Japanese",
        }
    ]


def test_anime_from_anidb_url_field_without_host_gives_no_link() -> None:
    assert _map(url="not a link")["external_sources"] == []


@pytest.mark.parametrize(
    ("resource_type", "identifier", "platform", "expected"),
    [
        ("2", "21", "myanimelist", "https://myanimelist.net/anime/21"),
        (
            "1",
            "149",
            "anime_news_network",
            "https://www.animenewsnetwork.com/encyclopedia/anime.php?id=149",
        ),
        ("9", "162790", "allcinema", "https://www.allcinema.net/cinema/162790"),
        ("10", "3270", "anison", "http://anison.info/data/program/3270.html"),
        (
            "28",
            "GRMG8ZQZR",
            "crunchyroll",
            "https://www.crunchyroll.com/series/GRMG8ZQZR",
        ),
        ("43", "tt0388629", "imdb", "https://www.imdb.com/title/tt0388629"),
    ],
)
def test_anime_from_anidb_single_identifier_resource_gives_platform_link(
    resource_type: str, identifier: str, platform: str, expected: str
) -> None:
    result = _resources(
        AniDBExternalResource(type=resource_type, identifiers=[identifier])
    )
    assert _links(result)[platform] == expected


def test_anime_from_anidb_type_45_resource_gives_funimation_link_not_hulu() -> None:
    result = _resources(AniDBExternalResource(type="45", identifiers=["one-piece/"]))
    assert _links(result) == {
        "funimation": "https://www.funimation.com/shows/one-piece/"
    }


def test_anime_from_anidb_syoboi_resource_gives_https_time_page() -> None:
    result = _resources(AniDBExternalResource(type="8", identifiers=["350"]))
    assert _links(result)["syoboi"] == "https://cal.syoboi.jp/tid/350/time"


def test_anime_from_anidb_vndb_resource_joins_number_and_entry_letter() -> None:
    result = _resources(AniDBExternalResource(type="14", identifiers=["7721", "v"]))
    assert _links(result)["vndb"] == "https://vndb.org/v7721"


def test_anime_from_anidb_vndb_resource_with_one_identifier_gives_no_link() -> None:
    result = _resources(AniDBExternalResource(type="14", identifiers=["7721"]))
    assert result["external_sources"] == []


def test_anime_from_anidb_tmdb_resource_joins_media_type_and_number() -> None:
    result = _resources(AniDBExternalResource(type="44", identifiers=["37854", "tv"]))
    assert _links(result)["themoviedb"] == "https://www.themoviedb.org/tv/37854"


def test_anime_from_anidb_tmdb_resource_with_one_identifier_gives_no_link() -> None:
    result = _resources(AniDBExternalResource(type="44", identifiers=["37854"]))
    assert "themoviedb" not in _links(result)


def test_anime_from_anidb_baidu_resource_drops_query_string() -> None:
    result = _resources(
        AniDBExternalResource(type="33", identifiers=["海贼王?fromModule=lemma"])
    )
    assert [link["source"] for link in result["external_sources"]] == [
        "https://baike.baidu.com/item/海贼王"
    ]


def test_anime_from_anidb_baidu_resource_without_identifier_gives_no_link() -> None:
    assert _resources(AniDBExternalResource(type="33"))["external_sources"] == []


def test_anime_from_anidb_full_address_resources_kept_in_order() -> None:
    result = _resources(
        AniDBExternalResource(type="5", urls=["https://gkids.com/"]),
        AniDBExternalResource(type="34", urls=["https://wetv.vip/en/play/x"]),
        AniDBExternalResource(type="35", urls=["http://blog.naver.com/fh"]),
    )
    assert [
        (link["platform"], link["source"]) for link in result["external_sources"]
    ] == [
        ("official_site", "https://gkids.com/"),
        ("wetv", "https://wetv.vip/en/play/x"),
        ("official_site", "http://blog.naver.com/fh"),
    ]


def test_anime_from_anidb_official_site_resource_reads_address_with_language() -> None:
    result = _resources(
        AniDBExternalResource(type="4", urls=["http://resource-site.jp"])
    )
    assert result["external_sources"] == [
        {
            "platform": "official_site",
            "source": "http://resource-site.jp",
            "language": "Japanese",
        }
    ]


def test_anime_from_anidb_parked_or_moved_resource_types_give_no_link() -> None:
    result = _resources(
        AniDBExternalResource(type="15", identifiers=["1996/akaboku"]),
        AniDBExternalResource(type="31", identifiers=["12519"]),
    )
    assert result["external_sources"] == []


def test_anime_from_anidb_unmapped_resource_type_gives_no_link() -> None:
    result = _resources(AniDBExternalResource(type="999", identifiers=["abc"]))
    assert result["external_sources"] == []


def test_anime_from_anidb_resource_with_several_identifiers_skipped_as_ambiguous() -> (
    None
):
    result = _resources(
        AniDBExternalResource(type="2", identifiers=["62593", "60022", "21"]),
        AniDBExternalResource(type="1", identifiers=["836", "3709"]),
    )
    assert result["external_sources"] == []


def test_anime_from_anidb_resource_without_identifier_or_address_gives_no_link() -> (
    None
):
    assert _resources(AniDBExternalResource(type="2"))["external_sources"] == []


def test_anime_from_anidb_output_keys_are_anime_fields() -> None:
    result = _map(title="One Piece", type="TV Series")
    assert set(result) <= set(Anime.model_fields)


def test_anime_from_anidb_one_piece_xml_maps_title_type_year_season_and_source(
    onepiece_anime: AniDBAnime,
) -> None:
    result = anime_from_anidb(onepiece_anime, anidb_url=ANIDB_URL)
    assert (result["title"], result["type"], result["year"], result["season"]) == (
        "One Piece",
        "TV",
        1999,
        "FALL",
    )
    assert result["sources"] == [ANIDB_URL]


def test_anime_from_anidb_one_piece_xml_keeps_end_date_and_score(
    onepiece_anime: AniDBAnime,
) -> None:
    result = anime_from_anidb(onepiece_anime, anidb_url=ANIDB_URL)
    assert result["aired_dates"]["aired_to"].startswith("2030")
    assert result["statistics"]["anidb"]["score"] > 0


def test_anime_from_anidb_one_piece_xml_gives_crunchyroll_and_tmdb_links(
    onepiece_anime: AniDBAnime,
) -> None:
    links = _links(anime_from_anidb(onepiece_anime, anidb_url=ANIDB_URL))
    assert links["crunchyroll"] == "https://www.crunchyroll.com/series/GRMG8ZQZR"
    assert links["themoviedb"] == "https://www.themoviedb.org/tv/37854"


def test_episode_from_anidb_regular_episode_maps_number_title_and_duration() -> None:
    episode = AniDBEpisode(
        id=1001,
        episode_number=1,
        episode_type=1,
        length=24,
        airdate="1999-10-20",
        summary="Luffy sets sail.",
        titles={"en": "Romance Dawn", "romaji": "Romance Dawn", "ja": "ロマンス"},
    )
    result = episode_from_anidb(episode)
    assert (result["episode_number"], result["title"], result["duration"]) == (
        1,
        "Romance Dawn",
        1440,
    )
    assert (result["title_romaji"], result["title_japanese"]) == (
        "Romance Dawn",
        "ロマンス",
    )
    assert (result["synopsis"], result["filler"], result["recap"]) == (
        "Luffy sets sail.",
        False,
        False,
    )


@pytest.mark.parametrize(
    ("episode_type", "episode_number"),
    [(2, 1), (1, "S1"), (None, 1)],
)
def test_episode_from_anidb_non_regular_episode_returns_none(
    episode_type: int | None, episode_number: int | str
) -> None:
    episode = AniDBEpisode(episode_type=episode_type, episode_number=episode_number)
    assert episode_from_anidb(episode) is None


def test_episode_from_anidb_english_title_wins_over_romaji() -> None:
    episode = AniDBEpisode(
        episode_type=1, episode_number=1, titles={"en": "English", "romaji": "Romaji"}
    )
    assert episode_from_anidb(episode)["title"] == "English"


def test_episode_from_anidb_without_english_title_takes_romaji() -> None:
    episode = AniDBEpisode(
        episode_type=1, episode_number=1, titles={"romaji": "Romaji"}
    )
    assert episode_from_anidb(episode)["title"] == "Romaji"


def test_episode_from_anidb_without_titles_gives_empty_title() -> None:
    episode = AniDBEpisode(episode_type=1, episode_number=1, titles={})
    assert episode_from_anidb(episode)["title"] == ""


def test_episode_from_anidb_zero_length_omits_duration() -> None:
    episode = AniDBEpisode(episode_type=1, episode_number=1, length=0)
    assert "duration" not in episode_from_anidb(episode)


def test_episode_from_anidb_episode_id_becomes_anidb_link() -> None:
    episode = AniDBEpisode(id=1001, episode_type=1, episode_number=1)
    assert episode_from_anidb(episode)["sources"] == ["https://anidb.net/episode/1001"]


def test_episode_from_anidb_without_episode_id_gives_no_sources() -> None:
    episode = AniDBEpisode(id=None, episode_type=1, episode_number=1)
    assert episode_from_anidb(episode)["sources"] == []


def test_episode_from_anidb_anime_id_added_when_given() -> None:
    episode = AniDBEpisode(id=1001, episode_type=1, episode_number=1)
    assert episode_from_anidb(episode, anime_id="uuid-abc-123")["anime_id"] == (
        "uuid-abc-123"
    )


def test_episode_from_anidb_other_language_titles_go_to_titles() -> None:
    episode = AniDBEpisode(
        episode_type=1, episode_number=1, titles={"en": "Title", "de": "Titel"}
    )
    assert episode_from_anidb(episode)["titles"] == {"de": "Titel"}


def test_episode_from_anidb_streaming_links_kept() -> None:
    episode = AniDBEpisode(
        episode_type=1,
        episode_number=1,
        streaming={"crunchyroll": "https://crunchyroll.com/watch/G6NQ5DWZ6"},
    )
    assert episode_from_anidb(episode)["streaming"] == {
        "crunchyroll": "https://crunchyroll.com/watch/G6NQ5DWZ6"
    }


def test_episode_from_anidb_one_piece_xml_maps_every_regular_episode(
    onepiece_anime: AniDBAnime,
) -> None:
    regular = [
        mapped
        for episode in onepiece_anime.episodes
        if (mapped := episode_from_anidb(episode)) is not None
    ]
    assert len(regular) > 1000
    first = next(mapped for mapped in regular if mapped["episode_number"] == 1)
    assert first["duration"] > 0


def test_character_from_anidb_name_only_gives_name_and_empty_lists() -> None:
    result = character_from_anidb(AniDBCharacter(name="Luffy"))
    assert result["name"] == "Luffy"
    assert (result["sources"], result["images"], result["roles"]) == ([], [], [])
    assert result["voice_actors"] == []


def test_character_from_anidb_without_name_gives_empty_name() -> None:
    assert character_from_anidb(AniDBCharacter())["name"] == ""


def test_character_from_anidb_character_id_becomes_anidb_link() -> None:
    result = character_from_anidb(AniDBCharacter(id=40, name="Luffy"))
    assert result["sources"] == ["https://anidb.net/character/40"]


def test_character_from_anidb_description_and_picture_mapped() -> None:
    result = character_from_anidb(
        AniDBCharacter(name="Luffy", description="Captain.", picture="luffy.jpg")
    )
    assert result["description"] == "Captain."
    assert result["images"] == [f"{CDN_BASE}/luffy.jpg"]


@pytest.mark.parametrize(
    ("character_type", "expected"),
    [("main character in", "MAIN"), ("secondary cast in", "SUPPORTING")],
)
def test_character_from_anidb_cast_type_gives_role(
    character_type: str, expected: str
) -> None:
    result = character_from_anidb(AniDBCharacter(name="Nami", type=character_type))
    assert result["roles"] == [expected]


def test_character_from_anidb_xml_gender_kept_without_page() -> None:
    character = AniDBCharacter(id=474, name="Luffy", gender="male")
    assert character_from_anidb(character)["attributes"] == {"gender": "male"}


def test_character_from_anidb_seiyuu_become_voice_actors_with_links_and_images() -> (
    None
):
    character = AniDBCharacter(
        id=40,
        name="Luffy",
        seiyuu=[
            AniDBSeiyuu(id=95, name="Mayumi Tanaka", picture="95.jpg"),
            AniDBSeiyuu(name="Unlinked Actor"),
            AniDBSeiyuu(),
        ],
    )
    actors = character_from_anidb(character)["voice_actors"]
    assert [(actor["name"], actor["sources"]) for actor in actors] == [
        ("Mayumi Tanaka", ["https://anidb.net/creator/95"]),
        ("Unlinked Actor", []),
        ("", []),
    ]
    assert actors[0]["image"] == f"{CDN_BASE}/95.jpg"
    assert "image" not in actors[1]


def test_character_from_anidb_page_names_and_description_mapped() -> None:
    page = AniDBCharacterPage(
        name_kanji="モンキー・D・ルフィ",
        description="From the page.",
        nicknames=["Straw Hat"],
        official_names=["Monkey D. Luffy"],
    )
    result = character_from_anidb(AniDBCharacter(id=40, name="Luffy"), page_data=page)
    assert result["name_native"] == "モンキー・D・ルフィ"
    assert result["description"] == "From the page."
    assert (result["nicknames"], result["name_variations"]) == (
        ["Straw Hat"],
        ["Monkey D. Luffy"],
    )


def test_character_from_anidb_xml_description_wins_over_page_description() -> None:
    page = AniDBCharacterPage(description="From the page.")
    result = character_from_anidb(
        AniDBCharacter(id=40, name="Luffy", description="From the XML."),
        page_data=page,
    )
    assert result["description"] == "From the XML."


def test_character_from_anidb_page_trait_groups_joined_into_traits() -> None:
    page = AniDBCharacterPage(
        abilities=["Gomu Gomu"],
        looks=["Scar under left eye"],
        personality=["Cheerful"],
        role=["Captain"],
        supernatural_abilities=["Haki"],
    )
    result = character_from_anidb(AniDBCharacter(id=40, name="Luffy"), page_data=page)
    assert result["traits"] == [
        "Gomu Gomu",
        "Scar under left eye",
        "Cheerful",
        "Captain",
        "Haki",
    ]


def test_character_from_anidb_page_appearances_give_animeography_and_roles() -> None:
    page = AniDBCharacterPage(
        animeography=[
            {
                "title": "One Piece Film: Red",
                "role": "main character in",
                "url": "https://anidb.net/anime/16537",
            },
            {"title": "Cameo Special", "role": "appears in"},
            {"title": "", "role": "main character in"},
            {"title": "Strong World", "role": "secondary cast in"},
        ]
    )
    result = character_from_anidb(
        AniDBCharacter(id=40, name="Luffy", type="secondary cast in"), page_data=page
    )
    assert [(entry["title"], entry["role"]) for entry in result["animeography"]] == [
        ("One Piece Film: Red", "MAIN"),
        ("Cameo Special", "BACKGROUND"),
        ("Strong World", "SUPPORTING"),
    ]
    assert result["animeography"][0]["sources"] == ["https://anidb.net/anime/16537"]
    assert result["roles"] == ["SUPPORTING", "MAIN", "BACKGROUND"]


def test_character_from_anidb_page_appearances_without_known_roles_leave_roles_empty() -> (
    None
):
    page = AniDBCharacterPage(animeography=[{"title": "One Piece"}])
    result = character_from_anidb(AniDBCharacter(id=40, name="Luffy"), page_data=page)
    assert result["animeography"][0]["role"] == "UNKNOWN"
    assert result["roles"] == []


def test_character_from_anidb_page_gender_wins_over_xml_gender() -> None:
    character = AniDBCharacter(id=474, name="Luffy", gender="male")
    page = AniDBCharacterPage(name_main="Luffy", gender="female")
    assert character_from_anidb(character, page)["attributes"] == {"gender": "female"}


def test_character_from_anidb_empty_page_adds_nothing() -> None:
    result = character_from_anidb(
        AniDBCharacter(id=40, name="Luffy"), page_data=AniDBCharacterPage()
    )
    assert "name_native" not in result
    assert (result["nicknames"], result["traits"]) == ([], [])


def test_character_from_anidb_one_piece_xml_main_character_has_role_and_voice_actors(
    onepiece_anime: AniDBAnime,
) -> None:
    main_character = next(
        character
        for character in onepiece_anime.characters
        if character.type == "main character in" and character.seiyuu
    )
    result = character_from_anidb(main_character)
    assert "MAIN" in result["roles"]
    assert result["voice_actors"]
