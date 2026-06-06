"""Unit tests for anidb_mapper.py — anime_from_anidb, episode_from_anidb, character_from_anidb."""

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
    AniDBEpisode,
    AniDBExternalResource,
    AniDBRatings,
    AniDBRelatedAnime,
    AniDBSeiyuu,
)

_CDN_BASE = "https://cdn-eu.anidb.net/images/main"
_ANIDB_URL = "https://anidb.net/anime/69"


# =============================================================================
# HELPERS
# =============================================================================


def _anime(**kwargs) -> AniDBAnime:
    """Minimal AniDBAnime factory with controlled fields."""
    return AniDBAnime(id=69, **kwargs)


# =============================================================================
# anime_from_anidb — scalars
# =============================================================================


def test_anime_from_anidb_title_from_main() -> None:
    result = anime_from_anidb(_anime(title="One Piece", title_english="OP EN"), anidb_url=_ANIDB_URL)
    assert result["title"] == "One Piece"


def test_anime_from_anidb_title_fallback_to_english() -> None:
    result = anime_from_anidb(_anime(title=None, title_english="One Piece EN"), anidb_url=_ANIDB_URL)
    assert result["title"] == "One Piece EN"


def test_anime_from_anidb_title_empty_when_nothing_available() -> None:
    result = anime_from_anidb(_anime(title=None, title_english=None), anidb_url=_ANIDB_URL)
    assert result["title"] == ""


def test_anime_from_anidb_type_tv_series() -> None:
    result = anime_from_anidb(_anime(type="TV Series"), anidb_url=_ANIDB_URL)
    assert result["type"] == "TV"


def test_anime_from_anidb_type_movie() -> None:
    result = anime_from_anidb(_anime(type="Movie"), anidb_url=_ANIDB_URL)
    assert result["type"] == "MOVIE"


def test_anime_from_anidb_type_unknown_for_missing() -> None:
    result = anime_from_anidb(_anime(type=None), anidb_url=_ANIDB_URL)
    assert result["type"] == "UNKNOWN"


def test_anime_from_anidb_sources_contains_anidb_url() -> None:
    result = anime_from_anidb(_anime(), anidb_url=_ANIDB_URL)
    assert result["sources"] == [_ANIDB_URL]


def test_anime_from_anidb_nsfw_true_when_restricted() -> None:
    result = anime_from_anidb(_anime(restricted=True), anidb_url=_ANIDB_URL)
    assert result["nsfw"] is True


def test_anime_from_anidb_nsfw_absent_when_not_restricted() -> None:
    """restricted=False → nsfw=None → excluded by model_dump(exclude_none=True)."""
    result = anime_from_anidb(_anime(restricted=False), anidb_url=_ANIDB_URL)
    assert "nsfw" not in result


# =============================================================================
# anime_from_anidb — images
# =============================================================================


def test_anime_from_anidb_images_cover_cdn_prefixed() -> None:
    result = anime_from_anidb(_anime(picture="anime.jpg"), anidb_url=_ANIDB_URL)
    assert result["images"]["covers"] == [f"{_CDN_BASE}/anime.jpg"]


def test_anime_from_anidb_images_empty_when_no_picture() -> None:
    result = anime_from_anidb(_anime(picture=None), anidb_url=_ANIDB_URL)
    assert result["images"]["covers"] == []


# =============================================================================
# anime_from_anidb — tags / categories
# =============================================================================


def test_anime_from_anidb_categories_appended_to_tags() -> None:
    result = anime_from_anidb(
        _anime(categories=[AniDBCategory(name="Action", hentai=False)]),
        anidb_url=_ANIDB_URL,
    )
    assert "Action" in result["tags"]


def test_anime_from_anidb_hentai_categories_excluded_from_tags() -> None:
    result = anime_from_anidb(
        _anime(categories=[AniDBCategory(name="Ecchi", hentai=True)]),
        anidb_url=_ANIDB_URL,
    )
    assert "Ecchi" not in result["tags"]


def test_anime_from_anidb_tags_deduplicated() -> None:
    """Category name already in tags list is not appended twice."""
    result = anime_from_anidb(
        _anime(
            tags=["Action"],
            categories=[AniDBCategory(name="Action", hentai=False)],
        ),
        anidb_url=_ANIDB_URL,
    )
    assert result["tags"].count("Action") == 1


# =============================================================================
# anime_from_anidb — titles (title_others → canonical titles)
# =============================================================================


def test_anime_from_anidb_title_others_mapped_to_titles() -> None:
    result = anime_from_anidb(
        _anime(title_others={"de": "Wan Piisu", "ko": "원피스"}),
        anidb_url=_ANIDB_URL,
    )
    assert result["titles"] == {"de": "Wan Piisu", "ko": "원피스"}


# =============================================================================
# anime_from_anidb — statistics
# =============================================================================


def test_anime_from_anidb_statistics_built_from_ratings() -> None:
    result = anime_from_anidb(
        _anime(ratings=AniDBRatings(permanent=8.33, permanent_count=9547)),
        anidb_url=_ANIDB_URL,
    )
    assert "statistics" in result
    assert result["statistics"]["anidb"]["score"] == 8.33
    assert result["statistics"]["anidb"]["scored_by"] == 9547


def test_anime_from_anidb_statistics_empty_when_no_ratings() -> None:
    """statistics field is always present but empty dict when no ratings exist."""
    result = anime_from_anidb(_anime(ratings=None), anidb_url=_ANIDB_URL)
    assert result["statistics"] == {}


def test_anime_from_anidb_statistics_empty_when_permanent_none() -> None:
    result = anime_from_anidb(
        _anime(ratings=AniDBRatings(permanent=None)),
        anidb_url=_ANIDB_URL,
    )
    assert result["statistics"] == {}


# =============================================================================
# anime_from_anidb — aired dates
# =============================================================================


def test_anime_from_anidb_aired_dates_built_from_dates() -> None:
    result = anime_from_anidb(
        _anime(start_date="1999-10-20", end_date="2030-06-01"),
        anidb_url=_ANIDB_URL,
    )
    assert "aired_dates" in result
    assert result["aired_dates"]["aired_from"] is not None
    assert result["aired_dates"]["aired_to"] is not None


def test_anime_from_anidb_no_aired_dates_when_no_dates() -> None:
    result = anime_from_anidb(_anime(start_date=None, end_date=None), anidb_url=_ANIDB_URL)
    assert "aired_dates" not in result


# =============================================================================
# anime_from_anidb — related anime
# =============================================================================


def test_anime_from_anidb_related_anime_grouped_by_type() -> None:
    result = anime_from_anidb(
        _anime(
            related_anime=[
                AniDBRelatedAnime(id=522, title="One Piece Movie 1", relation_type="Side Story"),
                AniDBRelatedAnime(id=9999, title="Season 2", relation_type="Sequel"),
            ]
        ),
        anidb_url=_ANIDB_URL,
    )
    assert "SIDE_STORY" in result["related_anime"]
    assert "SEQUEL" in result["related_anime"]
    assert result["related_anime"]["SIDE_STORY"][0]["title"] == "One Piece Movie 1"


def test_anime_from_anidb_related_anime_source_url_built_from_id() -> None:
    result = anime_from_anidb(
        _anime(related_anime=[AniDBRelatedAnime(id=522, relation_type="Side Story")]),
        anidb_url=_ANIDB_URL,
    )
    sources = result["related_anime"]["SIDE_STORY"][0]["sources"]
    assert "https://anidb.net/anime/522" in sources


# =============================================================================
# anime_from_anidb — external sources
# =============================================================================


def test_anime_from_anidb_url_field_to_official_website() -> None:
    result = anime_from_anidb(_anime(url="http://onepiece.toei-anim.co.jp"), anidb_url=_ANIDB_URL)
    assert result["external_sources"]["official_website"] == "http://onepiece.toei-anim.co.jp"


def test_anime_from_anidb_mal_resource_mapped() -> None:
    result = anime_from_anidb(
        _anime(resources=[AniDBExternalResource(type="2", identifiers=["21"])]),
        anidb_url=_ANIDB_URL,
    )
    assert result["external_sources"]["myanimelist"] == "https://myanimelist.net/anime/21"


def test_anime_from_anidb_ann_resource_mapped() -> None:
    result = anime_from_anidb(
        _anime(resources=[AniDBExternalResource(type="1", identifiers=["149"])]),
        anidb_url=_ANIDB_URL,
    )
    assert "anime_news_network" in result["external_sources"]
    assert "149" in result["external_sources"]["anime_news_network"]


def test_anime_from_anidb_type4_resource_uses_url_directly() -> None:
    """Type 4 (official website in resources) reads from urls list, not identifiers."""
    result = anime_from_anidb(
        _anime(resources=[AniDBExternalResource(type="4", urls=["http://resource-site.jp"])]),
        anidb_url=_ANIDB_URL,
    )
    assert result["external_sources"]["official_website"] == "http://resource-site.jp"


def test_anime_from_anidb_unknown_resource_type_skipped() -> None:
    result = anime_from_anidb(
        _anime(resources=[AniDBExternalResource(type="999", identifiers=["abc"])]),
        anidb_url=_ANIDB_URL,
    )
    assert "external_sources" not in result or len(result.get("external_sources", {})) == 0


def test_anime_from_anidb_crunchyroll_resource_mapped() -> None:
    result = anime_from_anidb(
        _anime(resources=[AniDBExternalResource(type="28", identifiers=["GRMG8ZQZR"])]),
        anidb_url=_ANIDB_URL,
    )
    assert result["external_sources"]["crunchyroll"] == "https://www.crunchyroll.com/series/GRMG8ZQZR"


def test_anime_from_anidb_tmdb_resource_mapped() -> None:
    result = anime_from_anidb(
        _anime(resources=[AniDBExternalResource(type="44", identifiers=["37854", "tv"])]),
        anidb_url=_ANIDB_URL,
    )
    assert result["external_sources"]["themoviedb"] == "https://www.themoviedb.org/tv/37854"


def test_anime_from_anidb_tmdb_single_identifier_skipped() -> None:
    """TMDB requires both id and media type — single identifier must be skipped."""
    result = anime_from_anidb(
        _anime(resources=[AniDBExternalResource(type="44", identifiers=["37854"])]),
        anidb_url=_ANIDB_URL,
    )
    assert "themoviedb" not in result.get("external_sources", {})


def test_anime_from_anidb_onepiece_has_crunchyroll_and_tmdb(onepiece_anime: AniDBAnime) -> None:
    result = anime_from_anidb(onepiece_anime, anidb_url=_ANIDB_URL)
    assert result["external_sources"]["crunchyroll"] == "https://www.crunchyroll.com/series/GRMG8ZQZR"
    assert result["external_sources"]["themoviedb"] == "https://www.themoviedb.org/tv/37854"


# =============================================================================
# anime_from_anidb — field validity
# =============================================================================


def test_anime_from_anidb_field_names_valid() -> None:
    """All output keys must be valid Anime model field names."""
    from common.models.anime import Anime

    result = anime_from_anidb(_anime(title="One Piece", type="TV Series"), anidb_url=_ANIDB_URL)
    valid_fields = set(Anime.model_fields.keys())
    for key in result:
        assert key in valid_fields, f"Mapper output key '{key}' not in Anime model"


# =============================================================================
# anime_from_anidb — real-fixture smoke tests
# =============================================================================


def test_anime_from_anidb_onepiece_title(onepiece_anime) -> None:
    result = anime_from_anidb(onepiece_anime, anidb_url=_ANIDB_URL)
    assert result["title"] == "One Piece"


def test_anime_from_anidb_onepiece_type(onepiece_anime) -> None:
    result = anime_from_anidb(onepiece_anime, anidb_url=_ANIDB_URL)
    assert result["type"] == "TV"


def test_anime_from_anidb_onepiece_year(onepiece_anime) -> None:
    result = anime_from_anidb(onepiece_anime, anidb_url=_ANIDB_URL)
    assert result["year"] == 1999


def test_anime_from_anidb_onepiece_season(onepiece_anime) -> None:
    result = anime_from_anidb(onepiece_anime, anidb_url=_ANIDB_URL)
    assert result["season"] == "FALL"


def test_anime_from_anidb_onepiece_sources(onepiece_anime) -> None:
    result = anime_from_anidb(onepiece_anime, anidb_url=_ANIDB_URL)
    assert result["sources"] == [_ANIDB_URL]


def test_anime_from_anidb_onepiece_injected_end_date(onepiece_anime) -> None:
    """Injected fake end date 2030-06-01 survives through the mapper."""
    result = anime_from_anidb(onepiece_anime, anidb_url=_ANIDB_URL)
    aired_to = result.get("aired_dates", {}).get("aired_to")
    assert aired_to is not None
    assert "2030" in str(aired_to)


def test_anime_from_anidb_onepiece_statistics_present(onepiece_anime) -> None:
    result = anime_from_anidb(onepiece_anime, anidb_url=_ANIDB_URL)
    assert "statistics" in result
    assert result["statistics"]["anidb"]["score"] > 0


# =============================================================================
# episode_from_anidb — scalar mapping
# =============================================================================


def test_episode_from_anidb_regular_maps_fields() -> None:
    ep = AniDBEpisode(
        id=1001,
        episode_number=1,
        episode_type=1,
        length=24,
        airdate="1999-10-20",
        titles={"en": "Romance Dawn", "romaji": "Romance Dawn"},
    )
    result = episode_from_anidb(ep)
    assert result is not None
    assert result["episode_number"] == 1
    assert result["title"] == "Romance Dawn"
    assert result["duration"] == 1440  # 24 min × 60 s


def test_episode_from_anidb_non_regular_returns_none() -> None:
    ep = AniDBEpisode(episode_type=2, episode_number=1)
    assert episode_from_anidb(ep) is None


def test_episode_from_anidb_string_episode_number_returns_none() -> None:
    ep = AniDBEpisode(episode_type=1, episode_number="S1")
    assert episode_from_anidb(ep) is None


def test_episode_from_anidb_none_episode_type_returns_none() -> None:
    ep = AniDBEpisode(episode_type=None, episode_number=1)
    assert episode_from_anidb(ep) is None


# =============================================================================
# episode_from_anidb — title fallback chain
# =============================================================================


def test_episode_from_anidb_title_en_preferred() -> None:
    ep = AniDBEpisode(
        episode_type=1,
        episode_number=1,
        titles={"en": "English Title", "romaji": "Romaji Title"},
    )
    assert episode_from_anidb(ep)["title"] == "English Title"


def test_episode_from_anidb_title_fallback_romaji() -> None:
    ep = AniDBEpisode(episode_type=1, episode_number=1, titles={"romaji": "Romaji Title"})
    assert episode_from_anidb(ep)["title"] == "Romaji Title"


def test_episode_from_anidb_title_empty_when_no_titles() -> None:
    ep = AniDBEpisode(episode_type=1, episode_number=1, titles={})
    assert episode_from_anidb(ep)["title"] == ""


# =============================================================================
# episode_from_anidb — duration + sources
# =============================================================================


def test_episode_from_anidb_duration_minutes_to_seconds() -> None:
    ep = AniDBEpisode(episode_type=1, episode_number=1, length=24)
    assert episode_from_anidb(ep)["duration"] == 1440


def test_episode_from_anidb_no_duration_when_zero_length() -> None:
    ep = AniDBEpisode(episode_type=1, episode_number=1, length=0)
    assert "duration" not in episode_from_anidb(ep)


def test_episode_from_anidb_source_url_built_from_id() -> None:
    ep = AniDBEpisode(id=1001, episode_type=1, episode_number=1)
    assert episode_from_anidb(ep)["sources"] == ["https://anidb.net/episode/1001"]


def test_episode_from_anidb_no_source_when_no_id() -> None:
    ep = AniDBEpisode(id=None, episode_type=1, episode_number=1)
    assert episode_from_anidb(ep)["sources"] == []


def test_episode_from_anidb_anime_id_injected() -> None:
    ep = AniDBEpisode(id=1001, episode_type=1, episode_number=1)
    result = episode_from_anidb(ep, anime_id="uuid-abc-123")
    assert result["anime_id"] == "uuid-abc-123"


# =============================================================================
# episode_from_anidb — titles dict + streaming
# =============================================================================


def test_episode_from_anidb_japanese_title_extracted() -> None:
    ep = AniDBEpisode(
        episode_type=1,
        episode_number=1,
        titles={"en": "Title", "ja": "タイトル"},
    )
    result = episode_from_anidb(ep)
    assert result["title_japanese"] == "タイトル"


def test_episode_from_anidb_extra_lang_in_titles_dict() -> None:
    """Non-standard lang codes (de, fr, etc.) go into episode.titles dict."""
    ep = AniDBEpisode(
        episode_type=1,
        episode_number=1,
        titles={"en": "Title", "de": "Titel"},
    )
    result = episode_from_anidb(ep)
    assert result.get("titles", {}).get("de") == "Titel"
    assert "en" not in result.get("titles", {})


def test_episode_from_anidb_streaming_passed_through() -> None:
    ep = AniDBEpisode(
        episode_type=1,
        episode_number=1,
        streaming={"crunchyroll": "https://crunchyroll.com/watch/G6NQ5DWZ6"},
    )
    result = episode_from_anidb(ep)
    assert result["streaming"]["crunchyroll"] == "https://crunchyroll.com/watch/G6NQ5DWZ6"


# =============================================================================
# episode_from_anidb — real-fixture smoke tests
# =============================================================================


def test_episode_from_anidb_onepiece_first_episode(onepiece_anime) -> None:
    """Episode 1 of real One Piece fixture maps without errors."""
    ep1 = next(
        (e for e in onepiece_anime.episodes if e.episode_type == 1 and e.episode_number == 1),
        None,
    )
    assert ep1 is not None
    result = episode_from_anidb(ep1)
    assert result is not None
    assert result["episode_number"] == 1
    assert result["duration"] is not None


def test_episode_from_anidb_onepiece_regular_episode_count(onepiece_anime) -> None:
    """All regular (type 1, integer-numbered) One Piece episodes map successfully."""
    regular = [ep for e in onepiece_anime.episodes if (ep := episode_from_anidb(e)) is not None]
    assert len(regular) > 1000
    assert all(isinstance(ep["episode_number"], int) for ep in regular)


# =============================================================================
# character_from_anidb — minimal + scalars
# =============================================================================


def test_character_from_anidb_minimal() -> None:
    result = character_from_anidb(AniDBCharacter(name="Luffy"))
    assert result["name"] == "Luffy"


def test_character_from_anidb_source_url_from_id() -> None:
    result = character_from_anidb(AniDBCharacter(id=40, name="Luffy"))
    assert result["sources"] == ["https://anidb.net/character/40"]


def test_character_from_anidb_sources_empty_when_no_id() -> None:
    """sources field is always present; empty list when no character id exists."""
    result = character_from_anidb(AniDBCharacter(name="Unknown"))
    assert result["sources"] == []


def test_character_from_anidb_image_cdn_prefixed() -> None:
    result = character_from_anidb(AniDBCharacter(name="Luffy", picture="luffy.jpg"))
    assert f"{_CDN_BASE}/luffy.jpg" in result["images"]


def test_character_from_anidb_images_empty_when_no_picture() -> None:
    """images field is always present; empty list when no picture exists."""
    result = character_from_anidb(AniDBCharacter(name="Luffy"))
    assert result["images"] == []


# =============================================================================
# character_from_anidb — role mapping
# =============================================================================


def test_character_from_anidb_role_main_mapped() -> None:
    result = character_from_anidb(AniDBCharacter(name="Luffy", type="main character in"))
    assert "MAIN" in result["roles"]


def test_character_from_anidb_role_supporting_mapped() -> None:
    # AniDB uses "secondary cast in" for supporting characters
    result = character_from_anidb(AniDBCharacter(name="Nami", type="secondary cast in"))
    assert "SUPPORTING" in result["roles"]


def test_character_from_anidb_roles_empty_when_no_type() -> None:
    """roles field is always present; empty list when type is None (mapper skips nil type)."""
    result = character_from_anidb(AniDBCharacter(name="Nami", type=None))
    assert result["roles"] == []


# =============================================================================
# character_from_anidb — voice actors
# =============================================================================


def test_character_from_anidb_with_seiyuu() -> None:
    char = AniDBCharacter(
        id=40,
        name="Luffy",
        seiyuu=[AniDBSeiyuu(id=95, name="Mayumi Tanaka", picture="95.jpg")],
    )
    result = character_from_anidb(char)
    assert len(result["voice_actors"]) == 1
    va = result["voice_actors"][0]
    assert va["name"] == "Mayumi Tanaka"
    assert va["image"] == f"{_CDN_BASE}/95.jpg"
    assert va["sources"] == ["https://anidb.net/creator/95"]


def test_character_from_anidb_multiple_seiyuu() -> None:
    char = AniDBCharacter(
        id=40,
        name="Luffy",
        seiyuu=[
            AniDBSeiyuu(id=95, name="Mayumi Tanaka"),
            AniDBSeiyuu(id=200, name="Colleen Clinkenbeard"),
        ],
    )
    result = character_from_anidb(char)
    assert len(result["voice_actors"]) == 2


def test_character_from_anidb_seiyuu_no_source_when_no_id() -> None:
    char = AniDBCharacter(name="Luffy", seiyuu=[AniDBSeiyuu(name="Unknown VA")])
    va = character_from_anidb(char)["voice_actors"][0]
    assert va["sources"] == []
    assert "image" not in va


def test_character_from_anidb_voice_actors_empty_when_no_seiyuu() -> None:
    """voice_actors field is always present; empty list when character has no seiyuu."""
    result = character_from_anidb(AniDBCharacter(id=40, name="Luffy"))
    assert result["voice_actors"] == []


# =============================================================================
# character_from_anidb — page data enrichment
# =============================================================================


def test_character_from_anidb_page_data_name_kanji_mapped() -> None:
    page = AniDBCharacterPage(name_kanji="モンキー・D・ルフィ")
    result = character_from_anidb(AniDBCharacter(id=40, name="Luffy"), page_data=page)
    assert result["name_native"] == "モンキー・D・ルフィ"


def test_character_from_anidb_page_data_nicknames_mapped() -> None:
    page = AniDBCharacterPage(nicknames=["Straw Hat", "Mugiwara"])
    result = character_from_anidb(AniDBCharacter(id=40, name="Luffy"), page_data=page)
    assert result["nicknames"] == ["Straw Hat", "Mugiwara"]


def test_character_from_anidb_page_data_traits_merged() -> None:
    page = AniDBCharacterPage(
        abilities=["Gomu Gomu"],
        looks=["Scar under left eye"],
        personality=["Cheerful"],
        role=[],
        supernatural_abilities=[],
    )
    result = character_from_anidb(AniDBCharacter(id=40, name="Luffy"), page_data=page)
    assert "Gomu Gomu" in result["traits"]
    assert "Scar under left eye" in result["traits"]
    assert "Cheerful" in result["traits"]


def test_character_from_anidb_page_data_none_no_enrichment() -> None:
    """With no page_data, enriched fields are absent or at their empty defaults."""
    result = character_from_anidb(AniDBCharacter(id=40, name="Luffy"), page_data=None)
    # name_native is str | None — excluded by exclude_none=True when not set
    assert "name_native" not in result
    # list fields default to [] — always present, just empty
    assert result["nicknames"] == []
    assert result["traits"] == []


# =============================================================================
# character_from_anidb — real-fixture smoke tests
# =============================================================================


def test_character_from_anidb_onepiece_main_character(onepiece_anime) -> None:
    """First main character from real One Piece fixture maps with voice actors."""
    main_char = next(
        (c for c in onepiece_anime.characters if c.type == "main character in" and c.seiyuu),
        None,
    )
    assert main_char is not None
    result = character_from_anidb(main_char)
    assert result["name"] is not None
    assert "MAIN" in result["roles"]
    assert len(result["voice_actors"]) > 0
