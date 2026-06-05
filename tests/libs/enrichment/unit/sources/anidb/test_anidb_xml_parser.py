"""Unit tests for anidb_xml_parser.py — parse_anime_xml and all private parsers."""

import pytest

from enrichment.sources.anidb.anidb_models import AniDBAnime
from enrichment.sources.anidb.anidb_xml_parser import parse_anime_xml


# =============================================================================
# Helpers
# =============================================================================


def _anime(body: str, aid: str = "69", restricted: str = "false") -> str:
    """Wrap body XML in a minimal <anime> root element."""
    return f'<anime id="{aid}" restricted="{restricted}">{body}</anime>'


# =============================================================================
# parse_anime_xml — error paths
# =============================================================================


def test_parse_invalid_xml_raises() -> None:
    with pytest.raises(ValueError, match="XML parse error"):
        parse_anime_xml("not xml at all <<<")


def test_parse_wrong_root_element_raises() -> None:
    with pytest.raises(ValueError, match="Expected <anime>"):
        parse_anime_xml("<root id='1'></root>")


def test_parse_missing_id_raises() -> None:
    with pytest.raises(ValueError, match="Missing or invalid"):
        parse_anime_xml("<anime></anime>")


def test_parse_non_numeric_id_raises() -> None:
    with pytest.raises(ValueError, match="Missing or invalid"):
        parse_anime_xml("<anime id='abc'></anime>")


# =============================================================================
# parse_anime_xml — scalar fields (real fixture)
# =============================================================================


def test_parse_id(onepiece_anime: AniDBAnime) -> None:
    assert onepiece_anime.id == 69


def test_parse_restricted_false(onepiece_anime: AniDBAnime) -> None:
    assert onepiece_anime.restricted is False


def test_parse_restricted_true() -> None:
    result = parse_anime_xml(_anime("", restricted="true"))
    assert result.restricted is True


def test_parse_type(onepiece_anime: AniDBAnime) -> None:
    assert onepiece_anime.type == "TV Series"


def test_parse_episode_count(onepiece_anime: AniDBAnime) -> None:
    assert onepiece_anime.episode_count == 1168


def test_parse_start_date(onepiece_anime: AniDBAnime) -> None:
    assert onepiece_anime.start_date == "1999-10-20"


def test_parse_end_date(onepiece_anime: AniDBAnime) -> None:
    # Fixture has injected fake end date for testing
    assert onepiece_anime.end_date == "2030-06-01"


def test_parse_description(onepiece_anime: AniDBAnime) -> None:
    assert onepiece_anime.description is not None
    assert "pirates" in onepiece_anime.description.lower()


def test_parse_picture(onepiece_anime: AniDBAnime) -> None:
    assert onepiece_anime.picture is not None
    assert onepiece_anime.picture.endswith(".jpg")


def test_parse_url(onepiece_anime: AniDBAnime) -> None:
    assert onepiece_anime.url is not None
    assert "toei" in onepiece_anime.url


def test_parse_missing_optional_scalar() -> None:
    result = parse_anime_xml(_anime("<type>TV</type>"))
    assert result.url is None
    assert result.description is None
    assert result.picture is None


# =============================================================================
# parse_anime_xml — titles
# =============================================================================


def test_parse_title_main(onepiece_anime: AniDBAnime) -> None:
    assert onepiece_anime.title == "One Piece"


def test_parse_title_english(onepiece_anime: AniDBAnime) -> None:
    assert onepiece_anime.title_english == "One Piece"


def test_parse_title_japanese(onepiece_anime: AniDBAnime) -> None:
    assert onepiece_anime.title_japanese == "ONE PIECE"


def test_parse_synonyms_non_empty(onepiece_anime: AniDBAnime) -> None:
    assert len(onepiece_anime.synonyms) > 0


def test_parse_title_others_bcp47(onepiece_anime: AniDBAnime) -> None:
    # One Piece has many official titles in other languages
    assert len(onepiece_anime.title_others) > 0
    # en and ja must NOT be in title_others — they have dedicated fields
    assert "en" not in onepiece_anime.title_others
    assert "ja" not in onepiece_anime.title_others


def test_parse_xjat_lang_normalized_to_romaji() -> None:
    xml = _anime("""
        <titles>
            <title xml:lang="x-jat" type="main">One Piece</title>
        </titles>
        <episodes>
            <episode id="1">
                <epno type="1">1</epno>
                <title xml:lang="x-jat">Romance Dawn</title>
            </episode>
        </episodes>
    """)
    result = parse_anime_xml(xml)
    assert "romaji" in result.episodes[0].titles
    assert "x-jat" not in result.episodes[0].titles


def test_parse_titles_empty_when_no_titles_element() -> None:
    result = parse_anime_xml(_anime(""))
    assert result.title is None
    assert result.title_english is None
    assert result.synonyms == []


def test_parse_official_title_other_langs() -> None:
    xml = _anime("""
        <titles>
            <title xml:lang="x-jat" type="main">Dan Da Dan</title>
            <title xml:lang="de" type="official">Dandadan DE</title>
            <title xml:lang="ko" type="official">단다단</title>
        </titles>
    """, aid="18290")
    result = parse_anime_xml(xml)
    assert result.title_others["de"] == "Dandadan DE"
    assert result.title_others["ko"] == "단다단"


# =============================================================================
# parse_anime_xml — categories
# =============================================================================


def test_parse_categories_count(onepiece_anime: AniDBAnime) -> None:
    # One Piece has no categories in this fixture (empty <categories>)
    assert isinstance(onepiece_anime.categories, list)


def test_parse_categories_fields() -> None:
    xml = _anime("""
        <categories>
            <category id="1" weight="600" hentai="false">
                <name>Action</name>
            </category>
            <category id="2" weight="100" hentai="true">
                <name>Ecchi</name>
            </category>
        </categories>
    """)
    result = parse_anime_xml(xml)
    assert len(result.categories) == 2
    assert result.categories[0].name == "Action"
    assert result.categories[0].hentai is False
    assert result.categories[1].hentai is True


def test_parse_category_without_name_skipped() -> None:
    xml = _anime("""
        <categories>
            <category id="1" weight="100"></category>
            <category id="2" weight="200"><name>Valid</name></category>
        </categories>
    """)
    result = parse_anime_xml(xml)
    assert len(result.categories) == 1
    assert result.categories[0].name == "Valid"


def test_parse_categories_absent() -> None:
    result = parse_anime_xml(_anime(""))
    assert result.categories == []


# =============================================================================
# parse_anime_xml — characters
# =============================================================================


def test_parse_characters_count(onepiece_anime: AniDBAnime) -> None:
    assert len(onepiece_anime.characters) > 0


def test_parse_character_fields(onepiece_anime: AniDBAnime) -> None:
    char = onepiece_anime.characters[0]
    assert char.id is not None
    assert char.name is not None
    assert char.type is not None


def test_parse_character_multiple_seiyuu() -> None:
    xml = _anime("""
        <characters>
            <character id="40" type="main character in">
                <name>Luffy</name>
                <seiyuu id="95" picture="tanaka.jpg">Mayumi Tanaka</seiyuu>
                <seiyuu id="200" picture="colleen.jpg">Colleen Clinkenbeard</seiyuu>
            </character>
        </characters>
    """)
    result = parse_anime_xml(xml)
    assert len(result.characters[0].seiyuu) == 2
    assert result.characters[0].seiyuu[0].name == "Mayumi Tanaka"
    assert result.characters[0].seiyuu[1].name == "Colleen Clinkenbeard"


def test_parse_character_no_seiyuu() -> None:
    xml = _anime("""
        <characters>
            <character id="99" type="appears in">
                <name>Background Person</name>
            </character>
        </characters>
    """)
    result = parse_anime_xml(xml)
    assert result.characters[0].seiyuu == []


def test_parse_character_picture_raw_filename() -> None:
    xml = _anime("""
        <characters>
            <character id="40" type="main character in">
                <name>Luffy</name>
                <picture>14789.jpg</picture>
            </character>
        </characters>
    """)
    result = parse_anime_xml(xml)
    # Must be raw filename only — CDN prefix is mapper's job
    assert result.characters[0].picture == "14789.jpg"
    assert "cdn" not in (result.characters[0].picture or "")


def test_parse_character_rating() -> None:
    xml = _anime("""
        <characters>
            <character id="40" type="main character in">
                <name>Luffy</name>
                <rating votes="5000">9.1</rating>
            </character>
        </characters>
    """)
    result = parse_anime_xml(xml)
    assert result.characters[0].rating == 9.1
    assert result.characters[0].rating_votes == 5000


def test_parse_characters_absent() -> None:
    result = parse_anime_xml(_anime(""))
    assert result.characters == []


# =============================================================================
# parse_anime_xml — creators
# =============================================================================


def test_parse_creators_count(onepiece_anime: AniDBAnime) -> None:
    assert len(onepiece_anime.creators) > 0


def test_parse_creator_fields(onepiece_anime: AniDBAnime) -> None:
    creator = onepiece_anime.creators[0]
    assert creator.name is not None
    assert creator.role is not None


def test_parse_creator_missing_id() -> None:
    xml = _anime("""
        <creators>
            <name type="Director">John Doe</name>
            <name id="123" type="Writer">Jane Smith</name>
        </creators>
    """)
    result = parse_anime_xml(xml)
    assert result.creators[0].id is None
    assert result.creators[1].id == 123


def test_parse_creator_non_numeric_id() -> None:
    xml = _anime("""
        <creators>
            <name id="abc" type="Director">Bad ID</name>
        </creators>
    """)
    result = parse_anime_xml(xml)
    assert result.creators[0].id is None


def test_parse_creators_absent() -> None:
    result = parse_anime_xml(_anime(""))
    assert result.creators == []


# =============================================================================
# parse_anime_xml — episodes
# =============================================================================


def test_parse_episodes_count(onepiece_anime: AniDBAnime) -> None:
    # Fixture has 1168+ episodes (regular + specials + credits etc.)
    assert len(onepiece_anime.episodes) > 1168


def test_parse_episode_regular_int_number() -> None:
    xml = _anime("""
        <episodes>
            <episode id="1001">
                <epno type="1">1</epno>
                <length>24</length>
                <airdate>1999-10-20</airdate>
                <rating votes="100">8.5</rating>
                <summary>Romance Dawn</summary>
                <title xml:lang="en">Romance Dawn</title>
                <title xml:lang="x-jat">Romance Dawn Romaji</title>
            </episode>
        </episodes>
    """)
    result = parse_anime_xml(xml)
    ep = result.episodes[0]
    assert ep.id == 1001
    assert ep.episode_number == 1
    assert ep.episode_type == 1
    assert ep.length == 24
    assert ep.airdate == "1999-10-20"
    assert ep.rating == 8.5
    assert ep.rating_votes == 100
    assert ep.summary == "Romance Dawn"
    assert ep.titles["en"] == "Romance Dawn"
    assert ep.titles["romaji"] == "Romance Dawn Romaji"


def test_parse_episode_special_string_number() -> None:
    xml = _anime("""
        <episodes>
            <episode id="2001">
                <epno type="2">S1</epno>
            </episode>
        </episodes>
    """)
    result = parse_anime_xml(xml)
    assert result.episodes[0].episode_number == "S1"
    assert result.episodes[0].episode_type == 2


def test_parse_episode_non_numeric_length_is_none() -> None:
    xml = _anime("""
        <episodes>
            <episode id="1">
                <epno type="1">1</epno>
                <length>TBA</length>
            </episode>
        </episodes>
    """)
    result = parse_anime_xml(xml)
    assert result.episodes[0].length is None


def test_parse_episode_malformed_id_is_none() -> None:
    xml = _anime("""
        <episodes>
            <episode id="bad">
                <epno type="1">1</epno>
            </episode>
        </episodes>
    """)
    result = parse_anime_xml(xml)
    assert result.episodes[0].id is None


def test_parse_episode_malformed_type_gives_string_number() -> None:
    xml = _anime("""
        <episodes>
            <episode id="1">
                <epno type="xyz">2</epno>
            </episode>
        </episodes>
    """)
    result = parse_anime_xml(xml)
    assert result.episodes[0].episode_type is None
    assert result.episodes[0].episode_number == "2"


def test_parse_episode_crunchyroll_streaming() -> None:
    xml = _anime("""
        <episodes>
            <episode id="1">
                <epno type="1">1</epno>
                <resources>
                    <resource type="28">
                        <externalentity>
                            <identifier>G6NQ5DWZ6</identifier>
                        </externalentity>
                    </resource>
                </resources>
            </episode>
        </episodes>
    """)
    result = parse_anime_xml(xml)
    assert result.episodes[0].streaming["crunchyroll"] == "https://www.crunchyroll.com/watch/G6NQ5DWZ6"


def test_parse_episode_non_crunchyroll_resource_ignored() -> None:
    xml = _anime("""
        <episodes>
            <episode id="1">
                <epno type="1">1</epno>
                <resources>
                    <resource type="41">
                        <externalentity><identifier>12345</identifier></externalentity>
                    </resource>
                </resources>
            </episode>
        </episodes>
    """)
    result = parse_anime_xml(xml)
    assert result.episodes[0].streaming == {}


def test_parse_episodes_absent() -> None:
    result = parse_anime_xml(_anime(""))
    assert result.episodes == []


# =============================================================================
# parse_anime_xml — related anime
# =============================================================================


def test_parse_related_anime_count(onepiece_anime: AniDBAnime) -> None:
    assert len(onepiece_anime.related_anime) > 0


def test_parse_related_anime_fields(onepiece_anime: AniDBAnime) -> None:
    rel = onepiece_anime.related_anime[0]
    assert rel.id > 0
    assert rel.relation_type is not None
    assert rel.title is not None


def test_parse_related_anime_whitespace_stripped() -> None:
    xml = _anime("""
        <relatedanime>
            <anime id="100" type="Sequel">  Dan Da Dan (2025)  </anime>
        </relatedanime>
    """)
    result = parse_anime_xml(xml)
    assert result.related_anime[0].title == "Dan Da Dan (2025)"


def test_parse_related_anime_missing_type_skipped() -> None:
    xml = _anime("""
        <relatedanime>
            <anime id="100">No Type</anime>
            <anime id="101" type="Sequel">Valid</anime>
        </relatedanime>
    """)
    result = parse_anime_xml(xml)
    assert len(result.related_anime) == 1
    assert result.related_anime[0].id == 101


def test_parse_related_anime_non_numeric_id_skipped() -> None:
    xml = _anime("""
        <relatedanime>
            <anime id="abc" type="Sequel">Bad ID</anime>
        </relatedanime>
    """)
    result = parse_anime_xml(xml)
    assert result.related_anime == []


def test_parse_related_anime_absent() -> None:
    result = parse_anime_xml(_anime(""))
    assert result.related_anime == []


# =============================================================================
# parse_anime_xml — resources
# =============================================================================


def test_parse_resources_count(onepiece_anime: AniDBAnime) -> None:
    assert len(onepiece_anime.resources) > 0


def test_parse_resources_fields() -> None:
    xml = _anime("""
        <resources>
            <resource type="2">
                <externalentity>
                    <identifier>21</identifier>
                </externalentity>
            </resource>
            <resource type="4">
                <externalentity>
                    <url>http://example.com</url>
                </externalentity>
            </resource>
        </resources>
    """)
    result = parse_anime_xml(xml)
    mal_res = next(r for r in result.resources if r.type == "2")
    web_res = next(r for r in result.resources if r.type == "4")
    assert mal_res.identifiers == ["21"]
    assert web_res.urls == ["http://example.com"]


def test_parse_resources_absent() -> None:
    result = parse_anime_xml(_anime(""))
    assert result.resources == []


# =============================================================================
# parse_anime_xml — tags
# =============================================================================


def test_parse_tags_count(onepiece_anime: AniDBAnime) -> None:
    assert len(onepiece_anime.tags) > 0


def test_parse_tags_empty_filtered() -> None:
    xml = _anime("""
        <tags>
            <tag id="1" weight="100"><name>action</name></tag>
            <tag id="2" weight="0"><name></name></tag>
            <tag id="3" weight="50"><name>adventure</name></tag>
        </tags>
    """)
    result = parse_anime_xml(xml)
    assert result.tags == ["action", "adventure"]
    assert "" not in result.tags


def test_parse_tags_absent() -> None:
    result = parse_anime_xml(_anime(""))
    assert result.tags == []


# =============================================================================
# parse_anime_xml — ratings
# =============================================================================


def test_parse_ratings_present(onepiece_anime: AniDBAnime) -> None:
    assert onepiece_anime.ratings is not None
    assert onepiece_anime.ratings.permanent is not None
    assert onepiece_anime.ratings.permanent_count > 0


def test_parse_ratings_all_sub_elements() -> None:
    xml = _anime("""
        <ratings>
            <permanent count="9547">8.33</permanent>
            <temporary count="10282">8.58</temporary>
            <review count="20">8.68</review>
        </ratings>
    """)
    result = parse_anime_xml(xml)
    assert result.ratings is not None
    assert result.ratings.permanent == 8.33
    assert result.ratings.permanent_count == 9547
    assert result.ratings.temporary == 8.58
    assert result.ratings.temporary_count == 10282
    assert result.ratings.review == 8.68
    assert result.ratings.review_count == 20


def test_parse_ratings_absent() -> None:
    result = parse_anime_xml(_anime(""))
    assert result.ratings is None
