import logging

import pytest
from common.models.anime import AnimeRating
from enrichment.sources.kitsu.kitsu_models import (
    KitsuAnimeAttributes,
    KitsuCharacterAttributes,
    KitsuEpisodeAttributes,
    KitsuPersonAttributes,
    KitsuTitles,
    _strip_preamble,
)


@pytest.mark.parametrize(
    ("raw", "expected"), [("78.34", 78.34), ("0.00", 0.0), ("", None), (None, None)]
)
def test_kitsu_anime_attributes_average_rating_parses_to_float(
    raw: str | None, expected: float | None
) -> None:
    assert (
        KitsuAnimeAttributes.model_validate({"averageRating": raw}).averageRating
        == expected
    )


def test_kitsu_anime_attributes_rating_frequencies_parse_to_counts() -> None:
    attrs = KitsuAnimeAttributes.model_validate(
        {"ratingFrequencies": {"2": "10", "20": "1000"}}
    )

    assert attrs.ratingFrequencies == {"2": 10, "20": 1000}
    assert attrs.scored_by == 1010


def test_kitsu_anime_attributes_scored_by_millions_of_ratings_sums_counts() -> None:
    attrs = KitsuAnimeAttributes.model_validate(
        {"ratingFrequencies": {"18": "1234567", "20": "2000000"}}
    )

    assert attrs.scored_by == 3234567


def test_kitsu_anime_attributes_scored_by_without_histogram_returns_none() -> None:
    assert KitsuAnimeAttributes().scored_by is None


@pytest.mark.parametrize("field", ["en", "en_jp", "en_us", "ja_jp"])
def test_kitsu_titles_empty_title_becomes_none(field: str) -> None:
    assert getattr(KitsuTitles.model_validate({field: "  "}), field) is None


def test_kitsu_anime_attributes_empty_canonical_title_becomes_none() -> None:
    assert (
        KitsuAnimeAttributes.model_validate({"canonicalTitle": ""}).canonicalTitle
        is None
    )


def test_kitsu_anime_attributes_empty_abbreviated_titles_dropped() -> None:
    attrs = KitsuAnimeAttributes.model_validate(
        {"abbreviatedTitles": ["OP", "", "  ", " Wan Piisu "]}
    )

    assert attrs.abbreviatedTitles == ["OP", "Wan Piisu"]


def test_kitsu_anime_attributes_null_abbreviated_title_dropped() -> None:
    attrs = KitsuAnimeAttributes.model_validate({"abbreviatedTitles": ["OP", None]})

    assert attrs.abbreviatedTitles == ["OP"]


def test_kitsu_anime_attributes_null_abbreviated_titles_give_empty_list() -> None:
    attrs = KitsuAnimeAttributes.model_validate({"abbreviatedTitles": None})

    assert attrs.abbreviatedTitles == []


def test_kitsu_character_attributes_null_other_name_dropped() -> None:
    attrs = KitsuCharacterAttributes.model_validate({"otherNames": [None, "Luffy"]})

    assert attrs.otherNames == ["Luffy"]


def test_kitsu_anime_attributes_synopsis_html_becomes_plain_text() -> None:
    attrs = KitsuAnimeAttributes.model_validate({"synopsis": "<p>Luffy &amp; Zoro</p>"})

    assert attrs.synopsis == "Luffy & Zoro"


def test_kitsu_person_attributes_description_html_becomes_plain_text() -> None:
    assert (
        KitsuPersonAttributes.model_validate(
            {"description": "<b>Seiyuu</b>"}
        ).description
        == "Seiyuu"
    )


def test_kitsu_character_attributes_description_preamble_and_html_removed() -> None:
    attrs = KitsuCharacterAttributes.model_validate(
        {"description": "Age: 17  <p>Luffy is the captain.</p>"}
    )

    assert attrs.description == "Luffy is the captain."


def test_kitsu_character_attributes_preamble_only_description_becomes_none() -> None:
    assert (
        KitsuCharacterAttributes.model_validate({"description": "Age: 17"}).description
        is None
    )


def test_kitsu_character_attributes_empty_other_names_dropped() -> None:
    attrs = KitsuCharacterAttributes.model_validate({"otherNames": ["Luffy", ""]})

    assert attrs.otherNames == ["Luffy"]


def test_kitsu_episode_attributes_empty_synopsis_becomes_none() -> None:
    attrs = KitsuEpisodeAttributes.model_validate(
        {"synopsis": "<p></p>", "description": "Luffy sets sail."}
    )

    assert attrs.synopsis is None
    assert attrs.description == "Luffy sets sail."


def test_strip_preamble_no_preamble_returns_unchanged():
    desc = "A dwarf who is no stranger to dungeons. He's also a great cook."
    assert _strip_preamble(desc) == desc


def test_strip_preamble_key_value_preamble_returns_narrative():
    desc = (
        "Name: Monkey D. Luffy Age: 17 Birthdate: May 5 Height: 172 cm"
        "  Luffy is the captain of the Straw Hat Pirates."
    )
    assert _strip_preamble(desc) == "Luffy is the captain of the Straw Hat Pirates."


def test_strip_preamble_double_space_inside_preamble_returns_narrative():
    desc = (
        "Age: 18; 20 Birthdate: July 3, Cancer"
        "  Height: 169 cm Affiliation: Straw Hat Pirates Position: Navigator"
        "   She is the navigator of the Straw Hat Pirates."
    )
    assert _strip_preamble(desc) == "She is the navigator of the Straw Hat Pirates."


def test_strip_preamble_prose_after_value_returns_prose():
    desc = (
        "Age: 28 Birthdate: February 6, Aquarius Height: 188 cm"
        "  Sizes: 99-59-89 Affiliation: Straw Hat Pirates Bounty:"
        "  130,000,000 (previously: 79,000,000) Robin is the seventh member."
    )
    assert _strip_preamble(desc) == "Robin is the seventh member."


def test_strip_preamble_preamble_only_returns_none():
    desc = "Age: 15 Birthday: November 3 Height: 163 cm  "
    assert _strip_preamble(desc) is None


def test_strip_preamble_none_returns_none():
    assert _strip_preamble(None) is None


def test_kitsu_character_attributes_empty_description_becomes_none() -> None:
    assert (
        KitsuCharacterAttributes.model_validate({"description": ""}).description is None
    )


@pytest.mark.parametrize(
    "description",
    [
        "Age: 17<br><br>Luffy is the captain.",
        "Age: 17<br/>Height: 174 cm<br/><br/>Luffy is the captain.",
        "<p>Age: 17</p><p>Luffy is the captain.</p>",
        "Age: 17\n\nLuffy is the captain.",
    ],
)
def test_kitsu_character_attributes_preamble_ended_by_line_break_keeps_narrative(
    description: str,
) -> None:
    assert (
        KitsuCharacterAttributes.model_validate(
            {"description": description}
        ).description
        == "Luffy is the captain."
    )


def test_strip_preamble_long_preamble_returns_full_narrative():
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
    assert _strip_preamble("Age: 18  ") is None


def test_strip_preamble_double_spaces_in_narrative_joins_sentences():
    desc = (
        "Age: 19 Height: 178 cm"
        "  First sentence of narrative.  Second sentence after double space."
    )
    result = _strip_preamble(desc)
    assert result == "First sentence of narrative. Second sentence after double space."


@pytest.mark.parametrize(
    ("age_rating", "guide", "expected"),
    [
        ("G", "All Ages", AnimeRating.G),
        ("G", "Kids", AnimeRating.G),
        ("PG", "Children", AnimeRating.PG),
        ("PG", "Teens 13 or older", AnimeRating.PG13),
        ("PG", "13 - Teens 13 or older", AnimeRating.PG13),
        ("PG", "13 and older", AnimeRating.PG13),
        ("PG", "Teens 13 or Older Violence", AnimeRating.PG13),
        ("PG", "teens or older", AnimeRating.PG13),
        ("PG", "PG-13", AnimeRating.PG13),
        ("PG", "13 Brief Nudity", AnimeRating.PG13),
        ("PG", "Some Graphic Violence", AnimeRating.PG),
        ("PG", "Partial Nudity", AnimeRating.PG),
        ("R", "17+ (violence & profanity)", AnimeRating.R),
        ("R", "17+", AnimeRating.R),
        ("R", "gore", AnimeRating.R),
        ("R", "Mild Nudity", AnimeRating.RPLUS),
        ("R", "partial nudity", AnimeRating.RPLUS),
        ("R", "Partial nuditiy", AnimeRating.RPLUS),
        ("R", "Ecchi, ", AnimeRating.RPLUS),
        ("R", "Nudity, Sex", AnimeRating.RPLUS),
        ("R", "smut", AnimeRating.RPLUS),
        ("R", "Ecchi, Erotica", AnimeRating.RPLUS),
        ("R18", "Hentai", AnimeRating.RX),
        ("R", None, AnimeRating.R),
        ("R", "", AnimeRating.R),
    ],
)
def test_kitsu_anime_attributes_content_rating_code_level_raised_by_guide(
    age_rating: str, guide: str | None, expected: AnimeRating
) -> None:
    attrs = KitsuAnimeAttributes(ageRating=age_rating, ageRatingGuide=guide)
    assert attrs.content_rating is expected


@pytest.mark.parametrize(
    ("age_rating", "guide", "expected"),
    [
        ("G", "Teens 13 or older", AnimeRating.G),
        ("G", "Children", AnimeRating.G),
        ("R", "Teens 13 or older", AnimeRating.R),
        ("PG", "All Ages", AnimeRating.PG),
        ("PG", "Mild Nudity", AnimeRating.PG),
        ("PG", "17+ (violence & profanity)", AnimeRating.PG),
        ("R", "None", AnimeRating.R),
    ],
)
def test_kitsu_anime_attributes_content_rating_guide_disagreeing_with_code_keeps_code(
    age_rating: str, guide: str, expected: AnimeRating
) -> None:
    attrs = KitsuAnimeAttributes(ageRating=age_rating, ageRatingGuide=guide)
    assert attrs.content_rating is expected


@pytest.mark.parametrize(
    ("guide", "expected"),
    [
        ("Mild Nudity", AnimeRating.RPLUS),
        ("13 teens or older", AnimeRating.PG13),
        ("13teens or older", AnimeRating.PG13),
        ("17+ (violence & profanity)", AnimeRating.R),
        ("Children", AnimeRating.PG),
        ("None", AnimeRating.UNKNOWN),
        ("", AnimeRating.UNKNOWN),
        (None, AnimeRating.UNKNOWN),
    ],
)
def test_kitsu_anime_attributes_content_rating_without_code_reads_guide(
    guide: str | None, expected: AnimeRating
) -> None:
    attrs = KitsuAnimeAttributes(ageRating=None, ageRatingGuide=guide)
    assert attrs.content_rating is expected


def test_kitsu_anime_attributes_content_rating_unread_guide_with_code_logs_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    attrs = KitsuAnimeAttributes(ageRating="R", ageRatingGuide="Mature Themes")
    with caplog.at_level(logging.WARNING):
        assert attrs.content_rating is AnimeRating.R
    assert caplog.text == ""


def test_kitsu_anime_attributes_content_rating_unread_guide_without_code_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    attrs = KitsuAnimeAttributes(ageRating=None, ageRatingGuide="Mature Themes")
    with caplog.at_level(logging.WARNING):
        assert attrs.content_rating is AnimeRating.UNKNOWN
    assert "Mature Themes" in caplog.text
