import pytest
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


def test_strip_preamble_empty_text_returns_none():
    assert _strip_preamble("") is None


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
