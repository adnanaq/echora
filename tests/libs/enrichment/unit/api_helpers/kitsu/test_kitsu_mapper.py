"""Unit tests for kitsu_mapper.py."""

from enrichment.api_helpers.kitsu.kitsu_models import (
    KitsuAnime,
    KitsuAnimeAttributes,
    KitsuAnimeographyAttributes,
    KitsuAnimeographyEntry,
    KitsuCharacter,
    KitsuCharacterAttributes,
    KitsuCharacterNames,
    KitsuCharacterVoice,
    KitsuCharacterVoiceAttributes,
    KitsuEpisode,
    KitsuEpisodeAttributes,
    KitsuImage,
    KitsuMediaCharacter,
    KitsuMediaCharacterAttributes,
    KitsuPerson,
    KitsuPersonAttributes,
    KitsuTitles,
)
from enrichment.api_helpers.kitsu.kitsu_mapper import (
    _strip_html,
    _strip_preamble,
    anime_from_kitsu,
    character_from_kitsu,
    episode_from_kitsu,
)
from common.models.anime import ThemeEntry


# =============================================================================
# Fixtures
# =============================================================================


def _make_anime(**overrides) -> KitsuAnime:
    """Minimal valid KitsuAnime with sensible defaults."""
    attrs = KitsuAnimeAttributes(
        slug="one-piece",
        canonicalTitle="One Piece",
        titles=KitsuTitles(en="One Piece", en_jp="One Piece", ja_jp="ONE PIECE"),
        subtype="TV",
        status="current",
        startDate="1999-10-20",
        episodeCount=1000,
        episodeLength=24,
        averageRating="83.40",
        userCount=500000,
        favoritesCount=30000,
        popularityRank=3,
        ratingRank=50,
        ageRating="PG",
        nsfw=False,
        posterImage=KitsuImage(original="https://example.com/poster.jpg"),
        coverImage=KitsuImage(original="https://example.com/cover.jpg"),
        youtubeVideoId="abc123",
        abbreviatedTitles=["OP"],
    )
    for key, val in overrides.items():
        setattr(attrs, key, val)
    return KitsuAnime(id="12", attributes=attrs)


def _make_media_char(
    char: KitsuCharacter | None = None, role: str = "main"
) -> KitsuMediaCharacter:
    mc = KitsuMediaCharacter(
        id="46066",
        attributes=KitsuMediaCharacterAttributes(role=role),
    )
    mc.character = char or _make_character()
    return mc


def _make_character() -> KitsuCharacter:
    return KitsuCharacter(
        id="411",
        attributes=KitsuCharacterAttributes(
            slug="monkey-d-luffy",
            canonicalName="Monkey D. Luffy",
            names=KitsuCharacterNames(ja_jp="モンキー・D・ルフィ"),
            malId=40,
            description="<p>Captain of the Straw Hats</p>",
            image=KitsuImage(original="https://example.com/luffy.jpg"),
            otherNames=["Luffy", "Straw Hat"],
        ),
    )


def _make_voice(locale: str = "ja_jp", person_id: str = "1") -> KitsuCharacterVoice:
    voice = KitsuCharacterVoice(
        id=f"v{person_id}",
        attributes=KitsuCharacterVoiceAttributes(locale=locale),
    )
    voice.person = KitsuPerson(
        id=person_id,
        attributes=KitsuPersonAttributes(
            name="Mayumi Tanaka",
            description="<p>Veteran voice actress</p>",
            image=KitsuImage(original="https://example.com/mayumi.jpg"),
        ),
    )
    return voice


def _make_animeography_entry(media_type: str = "anime") -> KitsuAnimeographyEntry:
    entry = KitsuAnimeographyEntry(
        id="99",
        attributes=KitsuAnimeographyAttributes(role="main"),
    )
    entry.media_type = media_type
    entry.media = KitsuAnime(
        id="12",
        type=media_type,
        attributes=KitsuAnimeAttributes(
            slug="one-piece", canonicalTitle="One Piece"
        ),
    )
    return entry


# =============================================================================
# anime_from_kitsu
# =============================================================================


def test_anime_from_kitsu_full():
    anime = _make_anime()
    genres = ["Action", "Adventure", "Comedy"]
    themes = [ThemeEntry(name="Pirate", description="High seas pirates")]

    anime.genres = genres
    anime.themes = themes
    result = anime_from_kitsu(anime)

    assert result["title"] == "One Piece"
    assert result["title_english"] == "One Piece"
    assert result["title_japanese"] == "ONE PIECE"
    assert result["type"] == "TV"
    assert result["status"] == "ONGOING"
    assert result["season"] == "FALL"
    assert result["year"] == 1999
    assert result["episode_count"] == 1000
    assert result["duration"] == 24 * 60
    assert result["genres"] == ["Action", "Adventure", "Comedy"]
    assert result["themes"][0]["name"] == "Pirate"
    assert result["synonyms"] == ["OP"]
    assert result["sources"] == ["https://kitsu.io/anime/one-piece"]
    assert result["images"]["posters"] == ["https://example.com/poster.jpg"]
    assert result["images"]["covers"] == ["https://example.com/cover.jpg"]
    assert result["statistics"]["kitsu"]["score"] == pytest.approx(8.34, abs=0.01)
    assert result["statistics"]["kitsu"]["members"] == 500000
    assert result["statistics"]["kitsu"]["rank"] == 50
    assert result["trailers"][0]["source"] == "https://www.youtube.com/watch?v=abc123"
    assert result["aired_dates"]["aired_from"].startswith("1999-10-19")  # midnight JST → UTC prev day
    assert result["rating"] == "PG - Children"


def test_anime_from_kitsu_minimal():
    """Minimal anime with only required fields — no crash, sensible defaults."""
    anime = KitsuAnime(
        id="99",
        attributes=KitsuAnimeAttributes(
            canonicalTitle="Test Anime",
            subtype="TV",
            status="current",
        ),
    )
    result = anime_from_kitsu(anime)

    assert result["title"] == "Test Anime"
    assert result["episode_count"] == 0
    assert result["genres"] == []
    assert result["trailers"] == []        # empty list, not None
    assert "aired_dates" not in result     # no startDate → excluded


def test_anime_from_kitsu_r18_rating():
    """R18+ ageRating should map to the RX canonical rating."""
    anime = _make_anime(ageRating="R18+")
    result = anime_from_kitsu(anime)
    assert result["rating"] == "Rx - Hentai"


def test_anime_from_kitsu_next_release_sets_broadcast():
    anime = _make_anime(nextRelease="2024-04-05T09:30:00.000+09:00")
    result = anime_from_kitsu(anime)
    assert result["broadcast"]["next_episode_at"] == "2024-04-05T00:30:00Z"


def test_anime_from_kitsu_season_derivation():
    """startDate month determines the anime season."""
    for month, expected_season in [("01", "WINTER"), ("04", "SPRING"), ("07", "SUMMER"), ("10", "FALL")]:
        anime = _make_anime(startDate=f"2020-{month}-01")
        result = anime_from_kitsu(anime)
        assert result["season"] == expected_season, f"month {month}"


# =============================================================================
# character_from_kitsu
# =============================================================================


import pytest


def test_character_from_kitsu():
    mc = _make_media_char()
    voices = [_make_voice("ja_jp", "1"), _make_voice("en", "2")]
    animeography = [_make_animeography_entry("anime"), _make_animeography_entry("manga")]

    mc.voices = voices
    mc.animeography = animeography
    result = character_from_kitsu(mc)

    assert result["name"] == "Monkey D. Luffy"
    assert result["name_native"] == "モンキー・D・ルフィ"
    assert result["name_variations"] == ["Luffy", "Straw Hat"]
    assert result["description"] == "Captain of the Straw Hats"
    assert result["images"] == ["https://example.com/luffy.jpg"]
    assert result["roles"] == ["MAIN"]
    assert "https://kitsu.io/characters/monkey-d-luffy" in result["sources"]
    assert "https://myanimelist.net/character/40" in result["sources"]

    assert len(result["voice_actors"]) == 2
    va_jp = result["voice_actors"][0]
    assert va_jp["name"] == "Mayumi Tanaka"
    assert va_jp["language"] == "Japanese"
    assert va_jp["image"] == "https://example.com/mayumi.jpg"
    assert va_jp["biography"] == "Veteran voice actress"
    assert va_jp["sources"] == ["https://kitsu.app/people/1"]

    assert len(result["animeography"]) == 1
    assert result["animeography"][0]["title"] == "One Piece"
    assert result["animeography"][0]["role"] == "MAIN"
    assert result["animeography"][0]["sources"] == ["https://kitsu.io/anime/one-piece"]

    assert len(result["mangaography"]) == 1
    assert result["mangaography"][0]["sources"] == ["https://kitsu.io/manga/one-piece"]


def test_character_from_kitsu_skips_voice_without_person():
    mc = _make_media_char()
    voice_no_person = KitsuCharacterVoice(
        id="v_orphan", attributes=KitsuCharacterVoiceAttributes(locale="ja_jp")
    )
    # person is None — should be skipped, not crash
    mc.voices = [voice_no_person]
    mc.animeography = []
    result = character_from_kitsu(mc)
    assert result["voice_actors"] == []


def test_character_from_kitsu_skips_animeography_without_media():
    mc = _make_media_char()
    entry_no_media = KitsuAnimeographyEntry(
        id="e_orphan", attributes=KitsuAnimeographyAttributes(role="main")
    )
    # media is None — should be skipped, not crash
    mc.voices = []
    mc.animeography = [entry_no_media]
    result = character_from_kitsu(mc)
    assert result.get("animeography", []) == []


def test_character_from_kitsu_no_character_raises():
    mc = KitsuMediaCharacter(
        id="99", attributes=KitsuMediaCharacterAttributes(role="main")
    )
    mc.character = None
    with pytest.raises(ValueError, match="no resolved character"):
        character_from_kitsu(mc)


# =============================================================================
# episode_from_kitsu
# =============================================================================


def test_episode_from_kitsu():
    ep = KitsuEpisode(
        id="ep1",
        attributes=KitsuEpisodeAttributes(
            canonicalTitle="I'm Luffy! The Man Who Will Become the Pirate King!",
            titles=KitsuTitles(ja_jp="俺はルフィ！海賊王になる男だ！", en_jp="Ore wa Rufi! Kaizoku Ou ni Naru Otoko da!"),
            synopsis="<p>Luffy sets sail.</p>",
            number=1,
            seasonNumber=1,
            airdate="1999-10-20",
            length=24,
            thumbnail=KitsuImage(original="https://example.com/ep1.jpg"),
        ),
    )

    result = episode_from_kitsu(ep, anime_slug="one-piece")

    assert result["title"] == "I'm Luffy! The Man Who Will Become the Pirate King!"
    assert result["title_japanese"] == "俺はルフィ！海賊王になる男だ！"
    assert result["title_romaji"] == "Ore wa Rufi! Kaizoku Ou ni Naru Otoko da!"
    assert result["episode_number"] == 1
    assert result["season_number"] == 1
    assert result["synopsis"] == "Luffy sets sail."
    assert result["duration"] == 24 * 60
    assert result["images"] == ["https://example.com/ep1.jpg"]
    assert result["sources"] == ["https://kitsu.app/anime/one-piece/episodes/1"]
    # airdate "1999-10-20" → midnight UTC
    assert result["aired"].startswith("1999-10-19")  # midnight JST → UTC prev day


def test_episode_from_kitsu_fallback_title():
    """Episode with no canonicalTitle falls back to en_us then en_jp."""
    ep = KitsuEpisode(
        id="ep2",
        attributes=KitsuEpisodeAttributes(
            titles=KitsuTitles(en_us="Episode Title US"),
            number=2,
        ),
    )
    result = episode_from_kitsu(ep)
    assert result["title"] == "Episode Title US"
    assert result["sources"] == []  # no slug → no valid URL


# =============================================================================
# _strip_html
# =============================================================================


def test_strip_html_removes_tags_and_decodes_entities():
    assert _strip_html("<p>Luffy &amp; Zoro</p>") == "Luffy & Zoro"


def test_strip_html_recovers_cp1252_mojibake():
    """UTF-8 bytes misread as cp1252 (e.g. Japanese katakana) are restored."""
    # ã‚¢ = cp1252 bytes E3 82 A2, which decode as UTF-8 to ア
    mojibake = "Aisa (\u00e3\u201a\u00a2\u00e3\u201a\u00a4\u00e3\u201a\u00b5) is a Shandian girl."
    result = _strip_html(mojibake)
    assert result == "Aisa (アイサ) is a Shandian girl."


def test_strip_html_keeps_text_that_cannot_be_recovered():
    """Text with characters outside cp1252 range (e.g. real CJK) is returned unchanged."""
    real_japanese = "カイドウ is one of the Yonko."
    assert _strip_html(real_japanese) == real_japanese


def test_strip_html_none_returns_none():
    assert _strip_html(None) is None


def test_strip_html_empty_returns_none():
    assert _strip_html("") is None


# =============================================================================
# _strip_preamble
# =============================================================================


def test_strip_preamble_no_preamble_returns_unchanged():
    """Plain narrative without a key-value block is returned as-is."""
    desc = "A dwarf who is no stranger to dungeons. He's also a great cook."
    assert _strip_preamble(desc) == desc


def test_strip_preamble_simple_preamble():
    """Single-chunk preamble separated from narrative by double-space (Luffy/Franky pattern)."""
    desc = (
        "Name: Monkey D. Luffy Age: 17 Birthdate: May 5 Height: 172 cm"
        "  Luffy is the captain of the Straw Hat Pirates."
    )
    assert _strip_preamble(desc) == "Luffy is the captain of the Straw Hat Pirates."


def test_strip_preamble_internal_double_space_in_preamble():
    """Preamble with internal double-spaces (Nami/Zoro pattern).

    Some preambles contain double-spaces mid-block (e.g. after a zodiac sign):
        "Age: 18 Birthdate: July 3, Cancer  Height: 169 cm  <narrative>"
    The algorithm must skip ALL key-value chunks, not just the first.
    """
    desc = (
        "Age: 18; 20 Birthdate: July 3, Cancer"
        "  Height: 169 cm Affiliation: Straw Hat Pirates Position: Navigator"
        "   She is the navigator of the Straw Hat Pirates."
    )
    assert _strip_preamble(desc) == "She is the navigator of the Straw Hat Pirates."


def test_strip_preamble_value_continuation_with_embedded_prose():
    """Double-space between key and value puts prose in same chunk (Robin pattern).

    "Bounty:  130,000,000 (previously: 79,000,000) Robin is the seventh member"
    The value chunk starts with a digit, but prose begins after ")".
    """
    desc = (
        "Age: 28 Birthdate: February 6, Aquarius Height: 188 cm"
        "  Sizes: 99-59-89 Affiliation: Straw Hat Pirates Bounty:"
        "  130,000,000 (previously: 79,000,000) Robin is the seventh member."
    )
    assert _strip_preamble(desc) == "Robin is the seventh member."


def test_strip_preamble_preamble_only_returns_none():
    """Preamble with no narrative after the separator returns None."""
    desc = "Age: 15 Birthday: November 3 Height: 163 cm  "
    assert _strip_preamble(desc) is None


def test_strip_preamble_none_input():
    assert _strip_preamble(None) is None


def test_strip_preamble_empty_string():
    assert _strip_preamble("") is None


def test_strip_preamble_preserves_full_narrative():
    """Narrative after stripping preamble is complete, not truncated mid-sentence."""
    narrative = (
        "Zoro was the first crew member to be recruited by Luffy. "
        "Zoro is a skilled swordsman who fights with santoryu."
    )
    desc = (
        "Name: Roronoa Zoro Age: 19 Birthdate: November 11, Scorpio"
        "  Height: 178 cm Affiliation: Straw Hat Pirates"
        "  " + narrative
    )
    assert _strip_preamble(desc) == narrative


def test_strip_preamble_trailing_spaces_produce_no_empty_chunks():
    """Trailing double-space after preamble (no narrative) must not crash on empty chunks."""
    # re.split on "Age: 18  " yields ["Age: 18", ""] — the empty string must be skipped
    assert _strip_preamble("Age: 18  ") is None


def test_strip_preamble_narrative_with_internal_double_spaces():
    """Narrative split into multiple chunks by its own double-spaces is joined correctly.

    The `in_narrative` branch (line 102) collects all subsequent chunks after the
    first narrative chunk is found, so multi-chunk narratives are complete.
    """
    desc = (
        "Age: 19 Height: 178 cm"
        "  First sentence of narrative.  Second sentence after double space."
    )
    result = _strip_preamble(desc)
    # Double-spaces within narrative split into parts that are joined with single space
    assert result == "First sentence of narrative. Second sentence after double space."


def test_strip_preamble_character_description_uses_strip_preamble():
    """character_from_kitsu strips preamble from description before storing."""
    preamble_desc = (
        "Name: Franky Age: 34 Height: 225 cm"
        "  Franky is the shipwright of the Straw Hat Pirates."
    )
    char = KitsuCharacter(
        id="42",
        attributes=KitsuCharacterAttributes(
            slug="franky",
            canonicalName="Franky",
            description=preamble_desc,
        ),
    )
    mc = _make_media_char(char=char)
    mc.voices = []
    mc.animeography = []
    result = character_from_kitsu(mc)
    assert result["description"] == "Franky is the shipwright of the Straw Hat Pirates."


def test_episode_from_kitsu_with_slug():
    """Episode with anime_slug produces correct kitsu.app URL."""
    ep = KitsuEpisode(
        id="ep3",
        attributes=KitsuEpisodeAttributes(number=11),
    )
    result = episode_from_kitsu(ep, anime_slug="DAN-DA-DAN")
    assert result["sources"] == ["https://kitsu.app/anime/DAN-DA-DAN/episodes/11"]
