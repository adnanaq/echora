from enrichment.sources.anime_planet.anime_planet_models import AnimePlanetAnime
from enrichment.sources.anime_planet.animeplanet_mapper import anime_from_animeplanet


def _anime(**fields) -> AnimePlanetAnime:
    return AnimePlanetAnime(
        name="Natsu no Shisen 1942", slug="a-gaze-in-summer-1942", **fields
    )


def test_anime_from_animeplanet_full_start_date_takes_year_from_date() -> None:
    mapped = anime_from_animeplanet(_anime(start_date="1999-10-20", start_year=1999))
    assert mapped["year"] == 1999
    assert mapped["aired_dates"]["aired_from"] == "1999-10-19T15:00:00Z"


def test_anime_from_animeplanet_without_start_date_takes_year_from_entry_bar() -> None:
    mapped = anime_from_animeplanet(_anime(start_year=2002))
    assert mapped["year"] == 2002
    assert "aired_dates" not in mapped


def test_anime_from_animeplanet_without_any_year_omits_year() -> None:
    assert "year" not in anime_from_animeplanet(_anime())
