"""Unit tests for anilist_mapper.py — anime_from_anilist and character_from_anilist."""

from enrichment.api_helpers.anilist.anilist_anime_models import AniListAnime
from enrichment.api_helpers.anilist.anilist_character_models import (
    AniListCharacterEdge,
    AniListFuzzyDate,
)
from enrichment.mappers.anilist_mapper import (
    _fuzzy_date_str,
    anime_from_anilist,
    character_from_anilist,
)


# =============================================================================
# Fixtures
# =============================================================================

def _make_anime(**overrides) -> AniListAnime:
    """Minimal valid AniListAnime with sensible defaults."""
    data = {
        "id": 21,
        "idMal": 21,
        "title": {"romaji": "ONE PIECE", "english": "ONE PIECE", "native": "ONE PIECE"},
        "format": "TV",
        "status": "RELEASING",
        "source": "MANGA",
        "episodes": 1000,
        "duration": 24,
        "isAdult": False,
        "seasonYear": 1999,
        "season": "FALL",
        "countryOfOrigin": "JP",
        "description": "Monkey D. Luffy sets sail.",
        "averageScore": 87,
        "popularity": 673293,
        "favourites": 98448,
        "genres": ["Action", "Adventure"],
        "synonyms": ["OP"],
        "tags": [],
        "studios": {"edges": []},
        "relations": {"edges": []},
        "externalLinks": [],
        "rankings": [],
    }
    data.update(overrides)
    return AniListAnime.model_validate(data)


def _make_character_edge(**overrides) -> AniListCharacterEdge:
    """Minimal valid AniListCharacterEdge."""
    data = {
        "node": {
            "id": 40,
            "name": {"full": "Monkey D. Luffy", "native": "モンキー・D・ルフィ", "alternative": ["Luffy"], "alternativeSpoiler": []},
            "image": {"large": "https://anilist.co/img/luffy.jpg"},
            "description": None,
            "gender": "Male",
            "age": "19",
            "bloodType": "F",
            "favourites": 50000,
            "siteUrl": "https://anilist.co/character/40",
        },
        "role": "MAIN",
        "voiceActorRoles": [
            {
                "voiceActor": {
                    "id": 95,
                    "name": {"full": "Mayumi Tanaka", "native": "田中真弓"},
                    "languageV2": "Japanese",
                    "image": {"large": "https://anilist.co/img/tanaka.jpg"},
                    "siteUrl": "https://anilist.co/staff/95",
                },
            }
        ],
    }
    data.update(overrides)
    return AniListCharacterEdge.model_validate(data)


# =============================================================================
# _fuzzy_date_str
# =============================================================================

def test_fuzzy_date_full() -> None:
    d = AniListFuzzyDate(year=1999, month=10, day=20)
    assert _fuzzy_date_str(d) == "1999-10-20"


def test_fuzzy_date_year_month() -> None:
    d = AniListFuzzyDate(year=1999, month=10)
    assert _fuzzy_date_str(d) == "1999-10"


def test_fuzzy_date_year_only() -> None:
    d = AniListFuzzyDate(year=1999)
    assert _fuzzy_date_str(d) == "1999"


def test_fuzzy_date_none() -> None:
    assert _fuzzy_date_str(None) is None


def test_fuzzy_date_no_year() -> None:
    d = AniListFuzzyDate(month=10, day=20)
    assert _fuzzy_date_str(d) is None


# =============================================================================
# anime_from_anilist — scalars
# =============================================================================

def test_anime_title() -> None:
    result = anime_from_anilist(_make_anime())
    assert result["title"] == "ONE PIECE"


def test_anime_title_english() -> None:
    result = anime_from_anilist(_make_anime())
    assert result["title_english"] == "ONE PIECE"


def test_anime_title_japanese() -> None:
    result = anime_from_anilist(_make_anime())
    assert result["title_japanese"] == "ONE PIECE"


def test_anime_title_fallback_to_empty_string() -> None:
    anime = _make_anime(title=None)
    result = anime_from_anilist(anime)
    assert result["title"] == ""


def test_anime_type_tv() -> None:
    result = anime_from_anilist(_make_anime())
    assert result["type"] == "TV"


def test_anime_type_unknown_format() -> None:
    result = anime_from_anilist(_make_anime(format=None))
    assert result["type"] == "UNKNOWN"


def test_anime_status_releasing_maps_to_ongoing() -> None:
    result = anime_from_anilist(_make_anime(status="RELEASING"))
    assert result["status"] == "ONGOING"


def test_anime_status_finished() -> None:
    result = anime_from_anilist(_make_anime(status="FINISHED"))
    assert result["status"] == "FINISHED"


def test_anime_source_material_manga() -> None:
    result = anime_from_anilist(_make_anime())
    assert result["source_material"] == "MANGA"


def test_anime_source_material_video_game() -> None:
    result = anime_from_anilist(_make_anime(source="VIDEO_GAME"))
    assert result["source_material"] == "GAME"


def test_anime_source_material_none() -> None:
    result = anime_from_anilist(_make_anime(source=None))
    assert "source_material" not in result


def test_anime_episode_count() -> None:
    result = anime_from_anilist(_make_anime())
    assert result["episode_count"] == 1000


def test_anime_episode_count_none_defaults_to_zero() -> None:
    result = anime_from_anilist(_make_anime(episodes=None))
    assert result["episode_count"] == 0


def test_anime_duration_converted_to_seconds() -> None:
    result = anime_from_anilist(_make_anime(duration=24))
    assert result["duration"] == 1440


def test_anime_duration_none_excluded() -> None:
    result = anime_from_anilist(_make_anime(duration=None))
    assert "duration" not in result


def test_anime_nsfw_false() -> None:
    result = anime_from_anilist(_make_anime(isAdult=False))
    assert result["nsfw"] is False


def test_anime_nsfw_true() -> None:
    result = anime_from_anilist(_make_anime(isAdult=True))
    assert result["nsfw"] is True


def test_anime_year() -> None:
    result = anime_from_anilist(_make_anime())
    assert result["year"] == 1999


def test_anime_season() -> None:
    result = anime_from_anilist(_make_anime(season="SPRING"))
    assert result["season"] == "SPRING"


def test_anime_season_none_excluded() -> None:
    result = anime_from_anilist(_make_anime(season=None))
    assert "season" not in result


def test_anime_country_of_origin() -> None:
    result = anime_from_anilist(_make_anime(countryOfOrigin="CN"))
    assert result["country_of_origin"] == "CN"


def test_anime_country_of_origin_none_excluded() -> None:
    result = anime_from_anilist(_make_anime(countryOfOrigin=None))
    assert "country_of_origin" not in result


def test_anime_synopsis() -> None:
    result = anime_from_anilist(_make_anime())
    assert result["synopsis"] == "Monkey D. Luffy sets sail."


# =============================================================================
# anime_from_anilist — sources
# =============================================================================

def test_anime_sources_anilist_url() -> None:
    result = anime_from_anilist(_make_anime(id=21))
    assert "https://anilist.co/anime/21" in result["sources"]


def test_anime_sources_mal_url_when_id_mal_set() -> None:
    result = anime_from_anilist(_make_anime(idMal=21))
    assert "https://myanimelist.net/anime/21" in result["sources"]


def test_anime_sources_no_mal_url_when_id_mal_none() -> None:
    result = anime_from_anilist(_make_anime(idMal=None))
    assert all("myanimelist" not in s for s in result["sources"])


# =============================================================================
# anime_from_anilist — genres, synonyms, tags
# =============================================================================

def test_anime_genres() -> None:
    result = anime_from_anilist(_make_anime())
    assert result["genres"] == ["Action", "Adventure"]


def test_anime_synonyms() -> None:
    result = anime_from_anilist(_make_anime())
    assert "OP" in result["synonyms"]


def test_anime_tag_demographic() -> None:
    anime = _make_anime(tags=[{"name": "Shounen", "category": "Demographic", "isAdult": False}])
    result = anime_from_anilist(anime)
    assert "Shounen" in result["demographics"]
    assert "Shounen" not in result.get("tags", [])


def test_anime_tag_theme() -> None:
    anime = _make_anime(tags=[{"name": "Travel", "description": "Moving around", "category": "Theme-Action", "isAdult": False}])
    result = anime_from_anilist(anime)
    themes = result["themes"]
    assert any(t["name"] == "Travel" for t in themes)


def test_anime_tag_flat() -> None:
    anime = _make_anime(tags=[{"name": "Pirates", "category": "Cast-Main Cast", "isAdult": False}])
    result = anime_from_anilist(anime)
    assert "Pirates" in result["tags"]


def test_anime_tag_adult_goes_to_content_warnings() -> None:
    anime = _make_anime(tags=[{"name": "Nudity", "category": "Theme-Ecchi", "isAdult": True}])
    result = anime_from_anilist(anime)
    assert "Nudity" in result["content_warnings"]
    assert "Nudity" not in result.get("tags", [])
    assert "Nudity" not in result.get("themes", [])


# =============================================================================
# anime_from_anilist — studios & producers
# =============================================================================

def test_anime_studios_split_from_producers() -> None:
    anime = _make_anime(studios={"edges": [
        {"node": {"id": 18, "name": "Toei Animation", "isAnimationStudio": True}},
        {"node": {"id": 102, "name": "Funimation", "isAnimationStudio": False}},
    ]})
    result = anime_from_anilist(anime)
    assert any(s["name"] == "Toei Animation" for s in result["studios"])
    assert any(p["name"] == "Funimation" for p in result["producers"])


def test_anime_studio_source_url() -> None:
    anime = _make_anime(studios={"edges": [
        {"node": {"id": 18, "name": "Toei Animation", "isAnimationStudio": True}},
    ]})
    result = anime_from_anilist(anime)
    assert result["studios"][0]["sources"] == ["https://anilist.co/studio/18"]


# =============================================================================
# anime_from_anilist — streaming & external links
# =============================================================================

def test_anime_streaming_sources() -> None:
    anime = _make_anime(externalLinks=[
        {"id": 1, "url": "https://crunchyroll.com/one-piece", "site": "Crunchyroll", "type": "STREAMING"},
    ])
    result = anime_from_anilist(anime)
    assert any(s["platform"] == "Crunchyroll" for s in result["streaming_sources"])


def test_anime_external_sources_info() -> None:
    anime = _make_anime(externalLinks=[
        {"id": 2, "url": "https://one-piece.com", "site": "Official Site", "type": "INFO"},
    ])
    result = anime_from_anilist(anime)
    assert result["external_sources"]["official site"] == "https://one-piece.com"


def test_anime_external_sources_social() -> None:
    anime = _make_anime(externalLinks=[
        {"id": 3, "url": "https://twitter.com/onepiece", "site": "Twitter", "type": "SOCIAL"},
    ])
    result = anime_from_anilist(anime)
    assert "twitter" in result["external_sources"]


def test_anime_external_links_skips_missing_url_or_site() -> None:
    anime = _make_anime(externalLinks=[
        {"id": 4, "url": None, "site": "Broken", "type": "INFO"},
        {"id": 5, "url": "https://example.com", "site": None, "type": "INFO"},
    ])
    result = anime_from_anilist(anime)
    assert result.get("external_sources", {}) == {}


# =============================================================================
# anime_from_anilist — trailer
# =============================================================================

def test_anime_youtube_trailer_mapped() -> None:
    anime = _make_anime(trailer={"id": "abc123", "site": "youtube", "thumbnail": "https://img.youtube.com/abc123.jpg"})
    result = anime_from_anilist(anime)
    assert result["trailers"][0]["source"] == "https://youtu.be/abc123"
    assert result["trailers"][0]["thumbnail"] == "https://img.youtube.com/abc123.jpg"


def test_anime_non_youtube_trailer_skipped() -> None:
    anime = _make_anime(trailer={"id": "abc", "site": "dailymotion", "thumbnail": None})
    result = anime_from_anilist(anime)
    assert result.get("trailers", []) == []


def test_anime_no_trailer_excluded() -> None:
    result = anime_from_anilist(_make_anime(trailer=None))
    assert result.get("trailers", []) == []


# =============================================================================
# anime_from_anilist — images
# =============================================================================

def test_anime_cover_extra_large_preferred() -> None:
    anime = _make_anime(coverImage={"extraLarge": "https://xl.jpg", "large": "https://l.jpg"})
    result = anime_from_anilist(anime)
    assert result["images"]["covers"][0] == "https://xl.jpg"


def test_anime_cover_falls_back_to_large() -> None:
    anime = _make_anime(coverImage={"extraLarge": None, "large": "https://l.jpg"})
    result = anime_from_anilist(anime)
    assert result["images"]["covers"][0] == "https://l.jpg"


def test_anime_banner_image_mapped() -> None:
    anime = _make_anime(bannerImage="https://banner.jpg")
    result = anime_from_anilist(anime)
    assert result["images"]["banners"] == ["https://banner.jpg"]


# =============================================================================
# anime_from_anilist — statistics
# =============================================================================

def test_anime_statistics_score_normalized() -> None:
    result = anime_from_anilist(_make_anime(averageScore=87))
    assert result["statistics"]["anilist"]["score"] == 8.7


def test_anime_statistics_score_none_excluded() -> None:
    result = anime_from_anilist(_make_anime(averageScore=None))
    assert "score" not in result["statistics"]["anilist"]


def test_anime_statistics_members_and_favorites() -> None:
    result = anime_from_anilist(_make_anime())
    stats = result["statistics"]["anilist"]
    assert stats["members"] == 673293
    assert stats["favorites"] == 98448


def test_anime_contextual_ranks_mapped() -> None:
    anime = _make_anime(rankings=[
        {"rank": 22, "context": "highest rated all time", "format": "TV", "allTime": True},
    ])
    result = anime_from_anilist(anime)
    ranks = result["statistics"]["anilist"]["contextual_ranks"]
    assert ranks[0]["rank"] == 22
    assert ranks[0]["context"] == "highest rated all time"
    assert ranks[0]["all_time"] is True


def test_anime_no_rankings_excludes_contextual_ranks() -> None:
    result = anime_from_anilist(_make_anime(rankings=[]))
    assert "contextual_ranks" not in result["statistics"]["anilist"]


# =============================================================================
# anime_from_anilist — broadcast
# =============================================================================

def test_anime_broadcast_next_episode_at() -> None:
    anime = _make_anime(nextAiringEpisode={"episode": 1110, "airingAt": 1743865200, "timeUntilAiring": 600})
    result = anime_from_anilist(anime)
    assert "broadcast" in result
    assert "next_episode_at" in result["broadcast"]


def test_anime_no_next_airing_episode_excludes_broadcast() -> None:
    result = anime_from_anilist(_make_anime(nextAiringEpisode=None))
    assert "broadcast" not in result


# =============================================================================
# anime_from_anilist — relations
# =============================================================================

def test_anime_relations_anime_bucket() -> None:
    anime = _make_anime(relations={"edges": [
        {
            "node": {"id": 100, "format": "MOVIE", "status": "FINISHED", "title": {"romaji": "OP Film"}, "seasonYear": 2000, "averageScore": 80, "coverImage": None, "episodes": 1, "chapters": None, "volumes": None},
            "relationType": "SIDE_STORY",
        }
    ]})
    result = anime_from_anilist(anime)
    assert "SIDE_STORY" in result["related_anime"]
    assert result["related_anime"]["SIDE_STORY"][0]["title"] == "OP Film"
    assert result["related_anime"]["SIDE_STORY"][0]["sources"] == ["https://anilist.co/anime/100"]


def test_anime_relations_source_material_bucket() -> None:
    anime = _make_anime(relations={"edges": [
        {
            "node": {"id": 200, "format": "MANGA", "status": "ONGOING", "title": {"romaji": "OP Manga"}, "seasonYear": None, "averageScore": None, "coverImage": None, "episodes": None, "chapters": 1100, "volumes": 105},
            "relationType": "ADAPTATION",
        }
    ]})
    result = anime_from_anilist(anime)
    assert "ADAPTATION" in result["related_source_material"]
    entry = result["related_source_material"]["ADAPTATION"][0]
    assert entry["title"] == "OP Manga"
    assert entry["sources"] == ["https://anilist.co/manga/200"]
    assert entry["chapters"] == 1100


def test_anime_relation_source_material_cover_uses_extra_large() -> None:
    anime = _make_anime(relations={"edges": [
        {
            "node": {"id": 200, "format": "MANGA", "status": "ONGOING", "title": {"romaji": "OP Manga"}, "seasonYear": None, "averageScore": None, "coverImage": {"extraLarge": "https://xl.jpg"}, "episodes": None, "chapters": None, "volumes": None},
            "relationType": "ADAPTATION",
        }
    ]})
    result = anime_from_anilist(anime)
    entry = result["related_source_material"]["ADAPTATION"][0]
    assert entry["images"] == ["https://xl.jpg"]


def test_anime_relation_anime_score_normalized() -> None:
    anime = _make_anime(relations={"edges": [
        {
            "node": {"id": 300, "format": "TV", "status": "FINISHED", "title": {"romaji": "Sequel"}, "seasonYear": 2022, "averageScore": 90, "coverImage": None, "episodes": 12, "chapters": None, "volumes": None},
            "relationType": "SEQUEL",
        }
    ]})
    result = anime_from_anilist(anime)
    assert result["related_anime"]["SEQUEL"][0]["score"] == 9.0


# =============================================================================
# character_from_anilist — scalars
# =============================================================================

def test_character_name() -> None:
    result = character_from_anilist(_make_character_edge())
    assert result["name"] == "Monkey D. Luffy"


def test_character_name_native() -> None:
    result = character_from_anilist(_make_character_edge())
    assert result["name_native"] == "モンキー・D・ルフィ"


def test_character_name_variations() -> None:
    result = character_from_anilist(_make_character_edge())
    assert "Luffy" in result["name_variations"]


def test_character_role_main() -> None:
    result = character_from_anilist(_make_character_edge())
    assert "MAIN" in result["roles"]


def test_character_role_supporting() -> None:
    edge = _make_character_edge(role="SUPPORTING")
    result = character_from_anilist(edge)
    assert "SUPPORTING" in result["roles"]


def test_character_favorites() -> None:
    result = character_from_anilist(_make_character_edge())
    assert result["favorites"] == 50000


def test_character_image() -> None:
    result = character_from_anilist(_make_character_edge())
    assert result["images"] == ["https://anilist.co/img/luffy.jpg"]


def test_character_no_image_excluded() -> None:
    edge = _make_character_edge()
    edge.node.image = None
    result = character_from_anilist(edge)
    assert result.get("images", []) == []


def test_character_source_url_from_site_url() -> None:
    result = character_from_anilist(_make_character_edge())
    assert result["sources"] == ["https://anilist.co/character/40"]


def test_character_source_url_fallback_to_id() -> None:
    edge = _make_character_edge()
    edge.node.site_url = None
    result = character_from_anilist(edge)
    assert result["sources"] == ["https://anilist.co/character/40"]


# =============================================================================
# character_from_anilist — attributes
# =============================================================================

def test_character_gender_attribute() -> None:
    result = character_from_anilist(_make_character_edge())
    assert result["attributes"]["gender"] == "Male"


def test_character_age_attribute() -> None:
    result = character_from_anilist(_make_character_edge())
    assert result["attributes"]["age"] == "19"


def test_character_blood_type_attribute() -> None:
    result = character_from_anilist(_make_character_edge())
    assert result["attributes"]["blood_type"] == "F"


def test_character_date_of_birth_full() -> None:
    edge = _make_character_edge()
    edge.node.date_of_birth = AniListFuzzyDate(year=1980, month=5, day=5)
    result = character_from_anilist(edge)
    assert result["attributes"]["date_of_birth"] == "1980-05-05"


def test_character_no_date_of_birth_excluded() -> None:
    result = character_from_anilist(_make_character_edge())
    assert "date_of_birth" not in result["attributes"]


# =============================================================================
# character_from_anilist — description parsing
# =============================================================================

def test_character_description_parsed_from_structured_lines() -> None:
    # Pass description through model_validate so the @model_validator runs on it
    edge = _make_character_edge(node={
        "id": 40,
        "name": {"full": "Usopp", "native": None, "alternative": [], "alternativeSpoiler": []},
        "description": "__Height:__ 174 cm\n\nUsopp is a liar.",
    })
    result = character_from_anilist(edge)
    assert result["description"] == "Usopp is a liar."
    assert result["attributes"]["height"] == "174 cm"


def test_character_spoiler_in_description_goes_to_spoilers() -> None:
    # Pass description through model_validate so the @model_validator runs on it
    edge = _make_character_edge(node={
        "id": 40,
        "name": {"full": "Usopp", "native": None, "alternative": [], "alternativeSpoiler": []},
        "description": "__Bounty:__ ~!500,000,000!~\n\nUsopp is brave.",
    })
    result = character_from_anilist(edge)
    assert result["spoilers"]["bounty"] == "500,000,000"
    assert "bounty" not in result["attributes"]


def test_character_description_multiline_prose() -> None:
    """Multi-line prose: second+ lines must hit the in_prose=True continuation path."""
    edge = _make_character_edge(node={
        "id": 40,
        "name": {"full": "Usopp", "native": None, "alternative": [], "alternativeSpoiler": []},
        "description": "__Height:__ 174 cm\n\nFirst prose line.\nSecond prose line.",
    })
    result = character_from_anilist(edge)
    assert "First prose line." in result["description"]
    assert "Second prose line." in result["description"]
    assert result["attributes"]["height"] == "174 cm"


def test_character_description_none_excluded() -> None:
    edge = _make_character_edge()  # default has description=None
    result = character_from_anilist(edge)
    assert "description" not in result


# =============================================================================
# character_from_anilist — nicknames (alternativeSpoiler)
# =============================================================================

def test_character_nicknames_from_alternative_spoiler() -> None:
    edge = _make_character_edge()
    edge.node.name.alternative_spoiler = ["God Usopp", "King of Snipers"]
    result = character_from_anilist(edge)
    assert "God Usopp" in result["nicknames"]
    assert "King of Snipers" in result["nicknames"]


# =============================================================================
# character_from_anilist — voice actors
# =============================================================================

def test_character_voice_actor_mapped() -> None:
    result = character_from_anilist(_make_character_edge())
    va = result["voice_actors"][0]
    assert va["name"] == "Mayumi Tanaka"
    assert va["native_name"] == "田中真弓"
    assert va["language"] == "Japanese"
    assert va["image"] == "https://anilist.co/img/tanaka.jpg"
    assert va["sources"] == ["https://anilist.co/staff/95"]


def test_character_voice_actor_source_fallback_to_id() -> None:
    edge = _make_character_edge()
    edge.voice_actor_roles[0].voice_actor.site_url = None
    result = character_from_anilist(edge)
    assert result["voice_actors"][0]["sources"] == ["https://anilist.co/staff/95"]


def test_character_voice_actor_skipped_when_no_name() -> None:
    edge = _make_character_edge()
    edge.voice_actor_roles[0].voice_actor.name = None
    result = character_from_anilist(edge)
    assert result.get("voice_actors", []) == []


def test_character_voice_actor_skipped_when_no_va() -> None:
    edge = _make_character_edge()
    edge.voice_actor_roles[0].voice_actor = None
    result = character_from_anilist(edge)
    assert result.get("voice_actors", []) == []


def test_character_multiple_voice_actors() -> None:
    edge = _make_character_edge()
    edge.voice_actor_roles = [
        type(edge.voice_actor_roles[0]).model_validate({
            "voiceActor": {"id": 95, "name": {"full": "Mayumi Tanaka"}, "languageV2": "Japanese", "siteUrl": "https://anilist.co/staff/95"},
        }),
        type(edge.voice_actor_roles[0]).model_validate({
            "voiceActor": {"id": 360, "name": {"full": "Sonny Strait"}, "languageV2": "English", "siteUrl": "https://anilist.co/staff/360"},
        }),
    ]
    result = character_from_anilist(edge)
    assert len(result["voice_actors"]) == 2
    languages = [va["language"] for va in result["voice_actors"]]
    assert "Japanese" in languages
    assert "English" in languages
