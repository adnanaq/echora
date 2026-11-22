"""Tests for the text a vector is built from.

This is the only consumer of the company fields, so a change there reaches the
embeddings rather than failing loudly.
"""

import pytest
from common.models.anime import (
    Anime,
    AnimeStatus,
    AnimeType,
    CompanyEntry,
    CompanyRole,
)
from vector_processing.processors.anime_field_mapper import AnimeFieldMapper


def _anime(**overrides) -> Anime:
    fields = {
        "title": "One Piece",
        "sources": ["https://myanimelist.net/anime/21"],
        "status": AnimeStatus.ONGOING,
        "type": AnimeType.TV,
    }
    fields.update(overrides)
    return Anime(**fields)


def _production_text(anime: Anime) -> str:
    text = AnimeFieldMapper().extract_anime_text(anime)
    return next(
        (
            part
            for part in text.split("\n")
            if "Studios:" in part or "Producers:" in part or "Licensors:" in part
        ),
        "",
    )


def test_each_role_is_labelled() -> None:
    text = _production_text(
        _anime(
            companies=[
                CompanyEntry(name="Toei Animation", roles=[CompanyRole.STUDIO]),
                CompanyEntry(name="Fuji TV", roles=[CompanyRole.PRODUCER]),
                CompanyEntry(name="Funimation", roles=[CompanyRole.LICENSOR]),
            ]
        )
    )
    assert "Studios: Toei Animation" in text
    assert "Producers: Fuji TV" in text
    assert "Licensors: Funimation" in text


def test_a_company_with_two_roles_appears_under_both() -> None:
    text = _production_text(
        _anime(
            companies=[
                CompanyEntry(
                    name="Madhouse",
                    roles=[CompanyRole.STUDIO, CompanyRole.PRODUCER],
                )
            ]
        )
    )
    assert "Studios: Madhouse" in text
    assert "Producers: Madhouse" in text


def test_companies_sharing_a_role_are_listed_together() -> None:
    text = _production_text(
        _anime(
            companies=[
                CompanyEntry(name="Toei Animation", roles=[CompanyRole.STUDIO]),
                CompanyEntry(name="TAP", roles=[CompanyRole.STUDIO]),
            ]
        )
    )
    assert "Studios: Toei Animation, TAP" in text


def test_unknown_roles_are_not_labelled() -> None:
    text = _production_text(
        _anime(companies=[CompanyEntry(name="Mystery Co", roles=[CompanyRole.UNKNOWN])])
    )
    assert text == ""


def test_no_companies_yields_no_production_line() -> None:
    assert _production_text(_anime()) == ""


@pytest.mark.parametrize("name", ["", None])
def test_a_company_without_a_name_is_skipped(name) -> None:
    text = _production_text(
        _anime(
            companies=[
                CompanyEntry(name=name or "", roles=[CompanyRole.STUDIO]),
                CompanyEntry(name="Bones", roles=[CompanyRole.STUDIO]),
            ]
        )
    )
    assert "Studios: Bones" in text


def test_extract_anime_text_includes_genres_temporal_and_streaming() -> None:
    from common.models.anime import (
        AiredDates,
        AnimeSeason,
        AnimeStatus,
        AnimeType,
        Episode,
        StreamingEntry,
        ThemeEntry,
        TrailerEntry,
    )

    anime = _anime(
        title="Full Example",
        title_english="Full Example EN",
        title_japanese="例",
        synonyms=["Alt"],
        synopsis="Story here",
        background="Behind the scenes",
        genres=["Shounen", "Shoujo", "Seinen", "Josei"],
        demographics=["Shounen"],
        tags=["Magic"],
        themes=[ThemeEntry(name="School", description="School life")],
        type=AnimeType.TV,
        status=AnimeStatus.FINISHED,
        year=2024,
        season=AnimeSeason.SPRING,
        aired_dates=AiredDates(aired_from="2024-04-01"),
        month=4,
        streaming_sources=[
            StreamingEntry(platform="Crunchyroll", source="crunchyroll")
        ],
        trailers=[TrailerEntry(thumbnail_source="https://example.com/thumb.jpg")],
    )
    mapper = AnimeFieldMapper()
    text = mapper.extract_anime_text(anime)

    assert "Shonen (young male)" in text
    assert "Demographics: Shounen" in text
    assert "Tags: Magic" in text
    assert "Themes: School" in text
    assert "Year: 2024" in text
    assert "Season: SPRING" in text
    assert "Aired: 2024-04-01" in text
    assert "Month: April" in text
    assert "Platform: Crunchyroll" in text
    assert "https://example.com/thumb.jpg" in mapper.extract_image_urls(anime)

    episode = Episode(episode_number=1, title="Recap Ep", is_filler=True, is_recap=True)
    ep_text = mapper.extract_episode_text(episode)
    assert "(Filler, Recap)" in ep_text

    from common.models.anime import Character

    character = Character(
        name="Hero", images=["https://example.com/c1.jpg", "https://example.com/c2.jpg"]
    )
    assert mapper.extract_character_image_urls(character) == [
        "https://example.com/c1.jpg",
        "https://example.com/c2.jpg",
    ]
    assert mapper.extract_character_image_urls(Character(name="NoImg")) == []
